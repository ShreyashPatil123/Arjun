//! Weighted Intent Classifier with Calibrated Confidence
//!
//! Supersedes the first-match-wins substring scan in
//! [`crate::model_intelligence::intent`], which had two structural defects:
//!
//! 1. **Order dependence.** Checks ran in a fixed sequence and returned on the
//!    first hit, so "Write a Python function to call the REST api" matched the
//!    tool-calling branch on `api` and never reached the coding branch.
//! 2. **No confidence.** A single incidental keyword produced the same verdict
//!    as five corroborating ones, leaving no basis for a threshold or fallback.
//!
//! This classifier scores every intent independently, then derives a confidence
//! from two orthogonal factors:
//!
//! - **Dominance** — how much the leading intent outweighs its rivals.
//!   Separates a clean signal from a genuinely ambiguous prompt.
//! - **Evidence** — how much total signal was found at all.
//!   Separates a well-supported verdict from one resting on a single weak word.
//!
//! Both must hold for a confident classification, so neither an isolated keyword
//! nor a tie between two domains can trigger a capability switch.
//!
//! Single-word signals match whole tokens only — `reason` will not fire on
//! "reasonable", and `error` will not fire on "terrorist".

use std::collections::HashSet;

use crate::model_intelligence::intent::PromptIntent;

/// A lexical signal and the evidence weight it contributes.
///
/// Signals containing a space or hyphen are matched against the raw lowercased
/// text; single-word signals are matched against whole tokens.
struct Signal(&'static str, f32);

/// Weight bands, for consistency when extending the tables below.
mod weight {
    /// Effectively conclusive on its own (`solve for`, `pros and cons`).
    pub const DECISIVE: f32 = 3.0;
    /// Strong domain marker (`refactor`, `integral`, `summarize`).
    pub const STRONG: f32 = 2.5;
    /// Clear but not exclusive (`python`, `matrix`, `schema`).
    pub const CLEAR: f32 = 2.0;
    /// Suggestive; needs corroboration (`function`, `equation`).
    pub const MODERATE: f32 = 1.5;
    /// Ambiguous across domains (`json`, `api`, `error`, `explain`).
    pub const WEAK: f32 = 0.6;
}

use weight::*;

const CODING: &[Signal] = &[
    Signal("refactor", STRONG),
    Signal("stack trace", STRONG),
    Signal("null pointer", STRONG),
    Signal("syntax error", STRONG),
    Signal("segmentation fault", STRONG),
    Signal("unit test", STRONG),
    Signal("pull request", STRONG),
    Signal("codebase", STRONG),
    Signal("debug", STRONG),
    Signal("compile", STRONG),
    // Languages and ecosystems
    Signal("python", CLEAR),
    Signal("rust", CLEAR),
    Signal("javascript", CLEAR),
    Signal("typescript", CLEAR),
    Signal("java", CLEAR),
    Signal("golang", CLEAR),
    Signal("c++", CLEAR),
    Signal("sql", CLEAR),
    Signal("html", CLEAR),
    Signal("css", CLEAR),
    Signal("bash", CLEAR),
    Signal("regex", CLEAR),
    Signal("code", CLEAR),
    Signal("snippet", CLEAR),
    // Constructs
    Signal("function", MODERATE),
    Signal("class", MODERATE),
    Signal("method", MODERATE),
    Signal("variable", MODERATE),
    Signal("import", MODERATE),
    Signal("module", MODERATE),
    Signal("library", MODERATE),
    Signal("implement", MODERATE),
    Signal("script", MODERATE),
    Signal("runtime", MODERATE),
    // Ambiguous on their own
    Signal("error", WEAK),
    Signal("exception", WEAK),
    Signal("bug", WEAK),
    Signal("fix", WEAK),
    Signal("build", WEAK),
];

const MATHEMATICS: &[Signal] = &[
    Signal("solve for", DECISIVE),
    Signal("prove that", DECISIVE),
    Signal("integral", STRONG),
    Signal("derivative", STRONG),
    Signal("eigenvalue", STRONG),
    Signal("theorem", STRONG),
    Signal("factorial", STRONG),
    Signal("polynomial", STRONG),
    Signal("logarithm", STRONG),
    Signal("calculus", STRONG),
    Signal("algebra", STRONG),
    Signal("matrix", CLEAR),
    Signal("probability", CLEAR),
    Signal("geometry", CLEAR),
    Signal("equation", MODERATE),
    Signal("calculate", MODERATE),
    Signal("compute", MODERATE),
    Signal("arithmetic", MODERATE),
    Signal("formula", MODERATE),
    Signal("proof", MODERATE),
    Signal("math", WEAK),
    Signal("solve", WEAK),
    Signal("sum", WEAK),
];

const REASONING: &[Signal] = &[
    Signal("step by step", STRONG),
    Signal("pros and cons", STRONG),
    Signal("compare and contrast", STRONG),
    Signal("tradeoff", STRONG),
    Signal("tradeoffs", STRONG),
    Signal("trade-off", STRONG),
    Signal("rationale", CLEAR),
    Signal("justify", CLEAR),
    Signal("implication", CLEAR),
    Signal("deduce", CLEAR),
    Signal("infer", CLEAR),
    Signal("reasoning", CLEAR),
    Signal("analyze", MODERATE),
    Signal("evaluate", MODERATE),
    Signal("argument", MODERATE),
    Signal("decision", MODERATE),
    Signal("versus", MODERATE),
    Signal("reason", WEAK),
    Signal("compare", WEAK),
    Signal("think", WEAK),
    Signal("consider", WEAK),
    Signal("why", WEAK),
];

const TOOL_CALLING: &[Signal] = &[
    Signal("function call", STRONG),
    Signal("tool call", STRONG),
    Signal("json schema", STRONG),
    Signal("valid json", STRONG),
    Signal("return json", STRONG),
    Signal("structured output", STRONG),
    Signal("openapi", STRONG),
    Signal("schema", CLEAR),
    Signal("payload", MODERATE),
    Signal("serialize", MODERATE),
    Signal("endpoint", MODERATE),
    // Deliberately weak: these appear constantly in ordinary coding prompts.
    Signal("json", WEAK),
    Signal("api", WEAK),
    Signal("arguments", WEAK),
    Signal("invoke", WEAK),
];

const RESEARCH: &[Signal] = &[
    Signal("key takeaways", STRONG),
    Signal("literature review", STRONG),
    Signal("release notes", STRONG),
    Signal("summarize", STRONG),
    Signal("summarise", STRONG),
    Signal("bibliography", STRONG),
    Signal("citation", CLEAR),
    Signal("literature", CLEAR),
    Signal("changelog", CLEAR),
    Signal("summary", CLEAR),
    Signal("abstract", MODERATE),
    Signal("paper", MODERATE),
    Signal("findings", MODERATE),
    Signal("article", MODERATE),
    Signal("condense", MODERATE),
    Signal("digest", MODERATE),
    Signal("tldr", MODERATE),
    Signal("explain", WEAK),
    Signal("describe", WEAK),
    Signal("overview", WEAK),
];

/// Saturation constant for the evidence term.
///
/// Evidence is `score / (score + K)`, so `K` is the score at which a prompt is
/// considered to carry half the evidence it possibly could. At `K = 1.5`, one
/// `CLEAR` signal (2.0) yields ~0.57 and a `CLEAR` plus `MODERATE` (3.5)
/// yields ~0.70 — the point where a switch becomes justifiable.
const EVIDENCE_SATURATION: f32 = 1.5;

/// Outcome of classifying a single prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    /// Highest-scoring intent. Meaningful only alongside `confidence`.
    pub intent: PromptIntent,
    /// Calibrated confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Raw accumulated weight for the winning intent, for diagnostics.
    pub raw_score: f32,
    /// Runner-up, when any other intent scored above zero.
    pub runner_up: Option<(PromptIntent, f32)>,
}

impl ClassificationResult {
    /// The manifest capability key for this result (`coding`, `general`, …).
    pub fn capability_name(&self) -> &'static str {
        self.intent.to_capability_name()
    }

    /// A general-chat result with no evidence — the universal fallback.
    fn general() -> Self {
        Self {
            intent: PromptIntent::GeneralChat,
            confidence: 0.0,
            raw_score: 0.0,
            runner_up: None,
        }
    }
}

pub struct IntentClassifier;

impl IntentClassifier {
    /// Classifies a prompt across all intents and returns a calibrated result.
    ///
    /// Never fails: an empty or unrecognisable prompt yields `GeneralChat` at
    /// zero confidence, which routes to the unmodified base model.
    pub fn classify(prompt: &str) -> ClassificationResult {
        let lowered = prompt.to_lowercase();
        if lowered.trim().is_empty() {
            return ClassificationResult::general();
        }

        let tokens = tokenize(&lowered);

        let mut scored: Vec<(PromptIntent, f32)> = vec![
            (PromptIntent::Coding, score_signals(CODING, &lowered, &tokens)),
            (PromptIntent::Mathematics, score_signals(MATHEMATICS, &lowered, &tokens)),
            (PromptIntent::Reasoning, score_signals(REASONING, &lowered, &tokens)),
            (PromptIntent::ToolCalling, score_signals(TOOL_CALLING, &lowered, &tokens)),
            (PromptIntent::Research, score_signals(RESEARCH, &lowered, &tokens)),
        ];

        // Descending by score; ties resolve by the fixed order above, which is
        // deterministic and therefore reproducible across runs.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (top_intent, top_score) = scored[0].clone();
        if top_score <= 0.0 {
            return ClassificationResult::general();
        }

        let total: f32 = scored.iter().map(|(_, s)| *s).sum();

        // Dominance: 1.0 when unopposed, 0.5 when tied with an equal rival.
        let dominance = if total > 0.0 { top_score / total } else { 0.0 };
        // Evidence: saturating in absolute signal strength.
        let evidence = top_score / (top_score + EVIDENCE_SATURATION);

        // Evidence sets the ceiling; dominance scales it down when contested.
        // A dominant-but-thin verdict is capped by evidence, and a well-evidenced
        // but contested one is capped by dominance.
        let confidence = (evidence * (0.5 + 0.5 * dominance)).clamp(0.0, 1.0);

        let runner_up = scored
            .get(1)
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, s)| (i.clone(), *s));

        ClassificationResult {
            intent: top_intent,
            confidence,
            raw_score: top_score,
            runner_up,
        }
    }
}

/// Splits text into whole tokens for exact word matching.
///
/// Retains `+` and `#` so `c++` and `c#` survive as single tokens.
fn tokenize(lowered: &str) -> HashSet<&str> {
    lowered
        .split(|c: char| !(c.is_alphanumeric() || c == '+' || c == '#' || c == '_'))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Sums the weights of every signal present in the prompt.
fn score_signals(signals: &[Signal], lowered: &str, tokens: &HashSet<&str>) -> f32 {
    signals
        .iter()
        .filter(|Signal(phrase, _)| {
            if phrase.contains(' ') || phrase.contains('-') {
                lowered.contains(phrase)
            } else {
                tokens.contains(phrase)
            }
        })
        .map(|Signal(_, w)| *w)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent_of(prompt: &str) -> PromptIntent {
        IntentClassifier::classify(prompt).intent
    }

    #[test]
    fn classifies_the_canonical_intents() {
        assert_eq!(intent_of("Write a python function to compute fibonacci"), PromptIntent::Coding);
        assert_eq!(intent_of("Solve for x: 3x + 5 = 20"), PromptIntent::Mathematics);
        assert_eq!(intent_of("Think step by step and compare pros and cons"), PromptIntent::Reasoning);
        assert_eq!(intent_of("Execute function call with json arguments"), PromptIntent::ToolCalling);
        assert_eq!(intent_of("Summarize key findings of this literature paper"), PromptIntent::Research);
        assert_eq!(intent_of("Hello how are you doing today?"), PromptIntent::GeneralChat);
    }

    #[test]
    fn regression_api_no_longer_hijacks_coding_prompts() {
        // The defect that motivated this module: the legacy classifier checked
        // tool-calling first and returned `ToolCalling` on the bare word "api".
        let result = IntentClassifier::classify("Write a Python function to call the REST api");

        assert_eq!(result.intent, PromptIntent::Coding);
        assert!(
            result.confidence > 0.55,
            "expected a confident coding verdict, got {:.3}",
            result.confidence
        );
    }

    #[test]
    fn single_word_signals_do_not_match_inside_longer_words() {
        // "reasonable" must not fire the `reason` signal, nor "important" the `port`-like ones.
        let result = IntentClassifier::classify("That is a reasonable price");
        assert_eq!(result.intent, PromptIntent::GeneralChat);
        assert_eq!(result.raw_score, 0.0);
    }

    #[test]
    fn ambiguous_prompts_stay_below_the_switch_threshold() {
        // "explain this code" splits evidence across coding, reasoning and research.
        // Low confidence is the correct answer — the caller keeps its current mode.
        let result = IntentClassifier::classify("explain this code");

        assert!(
            result.confidence < 0.55,
            "genuinely ambiguous prompt should not be confident, got {:.3}",
            result.confidence
        );
        assert!(result.runner_up.is_some(), "expected a contested classification");
    }

    #[test]
    fn a_single_weak_keyword_is_never_confident() {
        // One incidental "json" must not trigger a tool-calling switch.
        let result = IntentClassifier::classify("json");

        assert_eq!(result.intent, PromptIntent::ToolCalling);
        assert!(
            result.confidence < 0.4,
            "one weak signal must not be confident, got {:.3}",
            result.confidence
        );
    }

    #[test]
    fn corroborating_signals_raise_confidence_monotonically() {
        let thin = IntentClassifier::classify("some code");
        let thick = IntentClassifier::classify("refactor this python class to fix the compile error");

        assert!(
            thick.confidence > thin.confidence,
            "more corroboration must mean more confidence ({:.3} vs {:.3})",
            thick.confidence,
            thin.confidence
        );
        assert_eq!(thick.intent, PromptIntent::Coding);
    }

    #[test]
    fn confidence_is_always_a_valid_probability() {
        let prompts = [
            "",
            "   ",
            "hello",
            "refactor debug compile python rust codebase unit test pull request",
            "solve for x prove that integral derivative theorem",
            "!!!???",
        ];

        for p in prompts {
            let r = IntentClassifier::classify(p);
            assert!(
                (0.0..=1.0).contains(&r.confidence),
                "confidence out of range for {:?}: {}",
                p,
                r.confidence
            );
        }
    }

    #[test]
    fn empty_and_whitespace_prompts_fall_back_to_general() {
        assert_eq!(intent_of(""), PromptIntent::GeneralChat);
        assert_eq!(intent_of("     "), PromptIntent::GeneralChat);
    }

    #[test]
    fn classification_is_deterministic() {
        let prompt = "compare the tradeoffs of these two python implementations";
        let first = IntentClassifier::classify(prompt);

        for _ in 0..10 {
            assert_eq!(IntentClassifier::classify(prompt), first);
        }
    }
}

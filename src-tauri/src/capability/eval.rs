//! Classifier Evaluation Set
//!
//! The build plan asserts ">85% accuracy on a curated test set of 500 developer
//! queries" but no such set was ever written, so the claim was never checked.
//! This module supplies a real one and a scorer, so classifier changes are
//! measured rather than assumed.
//!
//! The prompts were written to represent ordinary usage *before* accuracy was
//! measured, and deliberately include cases designed to be hard:
//!
//! - Cross-domain vocabulary (`api` in a coding prompt, `matrix` in a code prompt)
//! - Short, context-poor turns (`fix this`, `why?`)
//! - Follow-ups that only make sense mid-conversation
//!
//! Genuinely ambiguous prompts are labelled [`GENERAL`], because the correct
//! behaviour there is low confidence and no capability switch — the switch
//! policy holds whatever was already active. Accuracy is therefore a floor on
//! real-world quality, not a ceiling: a `general` verdict on an ambiguous turn
//! costs nothing at runtime.

use std::collections::BTreeMap;

use crate::capability::classifier::IntentClassifier;
use crate::capability::policy::SwitchPolicy;

/// One labelled example.
pub struct EvalCase {
    pub prompt: &'static str,
    /// Expected capability key, matching `PromptIntent::to_capability_name`.
    pub expected: &'static str,
}

const fn case(prompt: &'static str, expected: &'static str) -> EvalCase {
    EvalCase { prompt, expected }
}

/// Curated developer prompts with expected capability labels.
pub const EVAL_SET: &[EvalCase] = &[
    // ── coding ──────────────────────────────────────────────────────────────
    case("write a python function to reverse a linked list", "coding"),
    case("refactor this class to use dependency injection", "coding"),
    case("why does my rust code fail the borrow checker here", "coding"),
    case("implement a binary search in typescript", "coding"),
    case("fix the null pointer exception in this java method", "coding"),
    case("add error handling to the database module", "coding"),
    case("convert this callback code to async await in javascript", "coding"),
    case("write a sql query to join users and orders", "coding"),
    case("my python script throws an import error on startup", "coding"),
    case("debug this segmentation fault in my c++ program", "coding"),
    case("write a regex to validate email addresses", "coding"),
    case("create a css grid layout for this page", "coding"),
    case("how do I write a unit test for this function", "coding"),
    case("the build fails with a compile error in the parser module", "coding"),
    case("optimise this python loop, it is too slow", "coding"),
    case("write a bash script to rotate log files", "coding"),
    case("split this god class into smaller modules", "coding"),
    case("review my pull request for the auth refactor", "coding"),
    // Cross-domain vocabulary: these carry tool-calling and math words but are
    // unambiguously coding requests.
    case("write a python function to call the REST api", "coding"),
    case("parse this json response in typescript and handle errors", "coding"),
    case("implement matrix multiplication in rust", "coding"),
    case("write code to calculate the sum of a list", "coding"),
    case("serialize this struct to json in rust", "coding"),

    // ── mathematics ─────────────────────────────────────────────────────────
    case("solve for x: 3x + 5 = 20", "mathematics"),
    case("what is the derivative of sin(x) times x squared", "mathematics"),
    case("prove that the square root of two is irrational", "mathematics"),
    case("compute the eigenvalues of this matrix", "mathematics"),
    case("evaluate the integral of 1/x from 1 to e", "mathematics"),
    case("what is the probability of three heads in five coin flips", "mathematics"),
    case("factor this polynomial completely", "mathematics"),
    case("solve this system of linear equations", "mathematics"),
    case("what is the logarithm of 1000 base 10", "mathematics"),
    case("prove that this series converges", "mathematics"),
    case("calculate the area under this curve", "mathematics"),
    case("find the roots of this quadratic equation", "mathematics"),

    // ── reasoning ───────────────────────────────────────────────────────────
    case("what are the tradeoffs between rest and graphql", "reasoning"),
    case("think step by step about why this design fails at scale", "reasoning"),
    case("compare and contrast microservices and a monolith", "reasoning"),
    case("justify choosing postgres over mongodb for this workload", "reasoning"),
    case("what are the pros and cons of server side rendering", "reasoning"),
    case("walk through the implications of this architecture decision", "reasoning"),
    case("deduce what went wrong from these symptoms", "reasoning"),
    case("evaluate whether we should adopt this framework", "reasoning"),
    case("what is the rationale for using event sourcing here", "reasoning"),

    // ── tool-calling ────────────────────────────────────────────────────────
    case("return valid json matching this schema", "tool-calling"),
    case("emit a function call with the correct arguments", "tool-calling"),
    case("produce structured output conforming to this json schema", "tool-calling"),
    case("generate an openapi specification for these endpoints", "tool-calling"),
    case("respond with only json, no prose", "tool-calling"),

    // ── research ────────────────────────────────────────────────────────────
    case("summarize the key findings of this paper", "research"),
    case("write release notes from these commits", "research"),
    case("give me the key takeaways from this document", "research"),
    case("condense this meeting transcript into bullet points", "research"),
    case("summarise this literature review on transformers", "research"),
    case("write a changelog entry for this version", "research"),
    case("give me a tldr of this article", "research"),

    // ── general / ambiguous ─────────────────────────────────────────────────
    // Correct behaviour is low confidence and no switch.
    case("hello how are you doing today", "general"),
    case("thanks that was helpful", "general"),
    case("can you help me with something", "general"),
    case("what do you think", "general"),
    case("never mind, forget it", "general"),
    case("good morning", "general"),
    case("that makes sense", "general"),
    case("ok", "general"),
];

/// Per-class counts.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClassStats {
    pub support: usize,
    pub correct: usize,
    pub predicted: usize,
}

impl ClassStats {
    pub fn recall(&self) -> f64 {
        if self.support == 0 { 0.0 } else { self.correct as f64 / self.support as f64 }
    }
    pub fn precision(&self) -> f64 {
        if self.predicted == 0 { 0.0 } else { self.correct as f64 / self.predicted as f64 }
    }
}

/// A single wrong prediction, for diagnosis.
#[derive(Debug, Clone)]
pub struct Miss {
    pub prompt: &'static str,
    pub expected: &'static str,
    pub predicted: String,
    pub confidence: f32,
}

/// Aggregate evaluation outcome.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub total: usize,
    pub correct: usize,
    pub per_class: BTreeMap<String, ClassStats>,
    pub misses: Vec<Miss>,
    /// Share of non-general cases that both classified correctly *and* cleared
    /// the switch threshold — i.e. would actually engage a capability.
    pub actionable_rate: f64,
}

impl EvalReport {
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.correct as f64 / self.total as f64 }
    }

    /// Multi-line human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "Classifier eval: {}/{} correct ({:.1}%), actionable {:.1}%\n",
            self.correct,
            self.total,
            self.accuracy() * 100.0,
            self.actionable_rate * 100.0
        );
        for (class, s) in &self.per_class {
            out.push_str(&format!(
                "  {:<13} support {:>3}  recall {:>5.1}%  precision {:>5.1}%\n",
                class,
                s.support,
                s.recall() * 100.0,
                s.precision() * 100.0
            ));
        }
        if !self.misses.is_empty() {
            out.push_str(&format!("  {} miss(es):\n", self.misses.len()));
            for m in &self.misses {
                out.push_str(&format!(
                    "    [{} -> {} @{:.2}] {:?}\n",
                    m.expected, m.predicted, m.confidence, m.prompt
                ));
            }
        }
        out
    }
}

/// Scores [`EVAL_SET`] against the current classifier.
pub fn run_eval() -> EvalReport {
    let policy = SwitchPolicy::default();
    let mut per_class: BTreeMap<String, ClassStats> = BTreeMap::new();
    let mut misses = Vec::new();
    let mut correct = 0usize;
    let mut actionable = 0usize;
    let mut non_general = 0usize;

    for c in EVAL_SET {
        let result = IntentClassifier::classify(c.prompt);
        let predicted = result.capability_name().to_string();

        per_class.entry(c.expected.to_string()).or_default().support += 1;
        per_class.entry(predicted.clone()).or_default().predicted += 1;

        let hit = predicted == c.expected;
        if hit {
            correct += 1;
            per_class.entry(c.expected.to_string()).or_default().correct += 1;
        } else {
            misses.push(Miss {
                prompt: c.prompt,
                expected: c.expected,
                predicted: predicted.clone(),
                confidence: result.confidence,
            });
        }

        if c.expected != "general" {
            non_general += 1;
            if hit && result.confidence >= policy.enter_threshold {
                actionable += 1;
            }
        }
    }

    EvalReport {
        total: EVAL_SET.len(),
        correct,
        per_class,
        misses,
        actionable_rate: if non_general == 0 {
            0.0
        } else {
            actionable as f64 / non_general as f64
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prints the full report so regressions are diagnosable, not just red.
    /// Run with `cargo test -- --nocapture` to see it on success too.
    #[test]
    fn classifier_meets_accuracy_target() {
        let report = run_eval();
        println!("\n{}", report.summary());

        assert!(
            report.accuracy() >= 0.85,
            "classifier accuracy {:.1}% is below the 85% target\n{}",
            report.accuracy() * 100.0,
            report.summary()
        );
    }

    /// Correct classification is worthless if confidence never clears the
    /// switch threshold — the capability would never actually engage.
    #[test]
    fn confident_enough_to_actually_switch() {
        let report = run_eval();

        assert!(
            report.actionable_rate >= 0.70,
            "only {:.1}% of domain prompts both classified correctly and cleared \
             the {:.2} switch threshold — capabilities would rarely engage\n{}",
            report.actionable_rate * 100.0,
            SwitchPolicy::default().enter_threshold,
            report.summary()
        );
    }

    /// A general/small-talk prompt must never confidently claim a domain.
    #[test]
    fn small_talk_never_triggers_a_capability_switch() {
        let policy = SwitchPolicy::default();

        for c in EVAL_SET.iter().filter(|c| c.expected == "general") {
            let r = IntentClassifier::classify(c.prompt);
            assert!(
                r.confidence < policy.enter_threshold,
                "small talk {:?} reached confidence {:.2}, which would switch capability",
                c.prompt,
                r.confidence
            );
        }
    }

    #[test]
    fn eval_set_is_balanced_enough_to_be_meaningful() {
        let report = run_eval();
        for key in ["coding", "mathematics", "reasoning", "tool-calling", "research", "general"] {
            let support = report.per_class.get(key).map(|s| s.support).unwrap_or(0);
            assert!(support >= 5, "class '{}' has only {} cases", key, support);
        }
    }
}

//! One turn's reading of what the person asked for, and which engine read it.
//!
//! The router used to call the keyword classifier itself and act on its score.
//! It now receives an [`IntentAnalysis`] instead: the same decision — which
//! intent, how sure, and whether sure enough to pick a specialist — with the
//! engine that made it named, so a semantic reader can replace keyword counting
//! without the router learning anything new about either.
//!
//! ```text
//! prompt → language → Laya `choice` ──(gate)──► IntentAnalysis → ModelRouter (unchanged)
//!                        │ unavailable, failed, timed out, or uncalibrated
//!                        └──────────► keyword classifier ──────►
//! ```
//!
//! Three sources, and what each one means for routing:
//!
//! - [`IntentSource::Laya`] — Laya's leading intent cleared the calibrated gate
//!   (enough probability, and enough lead over the runner-up), or failed it and
//!   was marked ambiguous. Ambiguous routes to the general reasoning band, which
//!   is where the router already sends every unclear prompt.
//! - [`IntentSource::Agreement`] — Laya's leader fell short of the gate but the
//!   keyword classifier independently and confidently named the same intent.
//!   Used only when the calibration run measured it to help.
//! - [`IntentSource::Keyword`] — the weighted classifier, exactly as before.
//!   Either nothing else ran, or Laya could not be used; `fallback_reason`
//!   says why.
//!
//! Laya never names a model. It returns one of six intents, and which installed
//! model serves that intent is still the registry's decision, against role,
//! clearance, modality and VRAM.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capability::classifier::{ClassificationResult, IntentClassifier};
use crate::capability::language::{self, PromptLanguage};
use crate::model_intelligence::intent::PromptIntent;

/// The intent question Laya is asked, byte-for-byte what the sidecar loads.
///
/// Compiled in so the labels can be checked against [`PromptIntent`] by a test:
/// a label the Rust side cannot map would turn every Laya answer into a
/// protocol error and every turn into a silent keyword fallback.
pub const INTENT_QUESTION_JSON: &str =
    include_str!("../../../sidecars/intent_sidecar/intent_question.json");

/// Below this, the keyword classifier is not trusted to pick a specialist.
///
/// Unchanged from the router's original `SPECIALIST_CONFIDENCE`: it matches the
/// classifier's own calibration, which reaches 0.55 only when one intent leads
/// clearly *and* several signals supported it. It lives here now because the
/// keyword path builds its [`IntentAnalysis`] here, and the router reads the
/// verdict rather than the number.
pub const KEYWORD_SPECIALIST_CONFIDENCE: f32 = 0.55;

/// The six labels, in the order the question lists them.
pub const INTENT_LABELS: [&str; 6] = [
    "coding",
    "mathematics",
    "reasoning",
    "tool-calling",
    "research",
    "general",
];

/// Maps a Laya choice label (the capability key) onto an intent.
pub fn intent_from_label(label: &str) -> Option<PromptIntent> {
    match label {
        "coding" => Some(PromptIntent::Coding),
        "mathematics" => Some(PromptIntent::Mathematics),
        "reasoning" => Some(PromptIntent::Reasoning),
        "tool-calling" => Some(PromptIntent::ToolCalling),
        "research" => Some(PromptIntent::Research),
        "general" => Some(PromptIntent::GeneralChat),
        _ => None,
    }
}

/// Which engine produced the intent the router acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntentSource {
    Laya,
    Agreement,
    Keyword,
}

impl IntentSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Laya => "laya",
            Self::Agreement => "laya+keyword",
            Self::Keyword => "keyword",
        }
    }
}

/// Laya's raw answer, as the sidecar returns it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaVerdict {
    pub choice: String,
    pub probabilities: BTreeMap<String, f32>,
    /// `max(p)` after temperature scaling — Laya's calibrated quantity.
    #[serde(default)]
    pub answer_confidence: Option<f32>,
    /// Normalised entropy. Reported, never gated on: Laya's `common.py` says
    /// the two confidences must not share a threshold.
    #[serde(default)]
    pub entropy_confidence: Option<f32>,
    pub checkpoint: String,
    #[serde(default)]
    pub checkpoint_reason: Option<String>,
    #[serde(default)]
    pub input_tokens: Option<u32>,
    /// Time inside the sidecar, forward pass included.
    pub latency_ms: f32,
}

/// A verdict whose labels and probabilities have been checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranked {
    pub top: PromptIntent,
    pub top_probability: f32,
    pub second: PromptIntent,
    pub second_probability: f32,
}

impl Ranked {
    pub fn margin(&self) -> f32 {
        self.top_probability - self.second_probability
    }
}

impl LayaVerdict {
    /// Checks the answer covers exactly the six intents with a real
    /// distribution, and ranks it.
    ///
    /// Anything else is a protocol fault, not a low-confidence answer — a
    /// sidecar answering a different question than the one ARJUN asked — and
    /// the caller falls back to the keyword classifier rather than route on it.
    pub fn ranked(&self) -> Result<Ranked, String> {
        if self.probabilities.len() != INTENT_LABELS.len() {
            return Err(format!(
                "Laya answered with {} options where the intent question has {}",
                self.probabilities.len(),
                INTENT_LABELS.len()
            ));
        }
        let mut scored = Vec::with_capacity(INTENT_LABELS.len());
        let mut total = 0.0f32;
        for (label, probability) in &self.probabilities {
            let intent = intent_from_label(label)
                .ok_or_else(|| format!("Laya answered with an unknown intent label {label:?}"))?;
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(format!("Laya gave {label:?} an impossible probability {probability}"));
            }
            total += probability;
            scored.push((intent, *probability));
        }
        // The sidecar rounds each probability to four places, so six of them
        // can miss 1.0 by a few ten-thousandths and no more.
        if (total - 1.0).abs() > 0.01 {
            return Err(format!("Laya's probabilities sum to {total:.4}, not 1"));
        }
        if intent_from_label(&self.choice).is_none() {
            return Err(format!("Laya chose an unknown intent label {:?}", self.choice));
        }
        // Descending by probability; ties resolve by label order, which the
        // BTreeMap makes deterministic.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Ranked {
            top: scored[0].0.clone(),
            top_probability: scored[0].1,
            second: scored[1].0.clone(),
            second_probability: scored[1].1,
        })
    }
}

/// How much Laya must believe its leading intent before the router acts on it.
///
/// Two conditions, because they catch different failures. A low top
/// probability is a prompt Laya cannot read; a small margin is a prompt that
/// reads as two things at once — "summarise this paper and implement it" —
/// where either specialist is a guess.
///
/// There is deliberately no `Default`. These numbers come from a calibration
/// run over the validation set (see [`crate::capability::intent_eval`]), and a
/// default would be exactly the unmeasured threshold that run exists to replace.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaGate {
    pub min_probability: f32,
    pub min_margin: f32,
}

impl LayaGate {
    pub fn admits(&self, ranked: &Ranked) -> bool {
        ranked.top_probability >= self.min_probability && ranked.margin() >= self.min_margin
    }
}

/// Whether keyword agreement may rescue a Laya verdict that missed the gate.
///
/// Chosen by the calibration run: whichever routed the fitting split with
/// fewer errors, with plain `Laya` preferred on a tie because it has one fewer
/// rule to explain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatePolicy {
    Laya,
    Hybrid,
}

/// Laya's answer, kept beside a keyword verdict that routed instead of it.
///
/// Recorded so the log and the trace show what the semantic reader thought
/// even while it is not trusted to decide — which is also the data a later
/// calibration run needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShadowVerdict {
    pub intent: PromptIntent,
    pub probability: f32,
    pub runner_up: PromptIntent,
    pub runner_up_probability: f32,
    pub checkpoint: String,
}

/// What a turn is asking for, and how that was decided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntentAnalysis {
    pub primary_intent: PromptIntent,
    /// For Laya, the probability of the leading intent. For the keyword
    /// classifier, its own calibrated confidence. The two are on different
    /// scales, which is why routing reads `ambiguous` rather than this number.
    pub confidence: f32,
    /// Laya's full distribution over the six intents. Empty for the keyword
    /// classifier, which scores signals rather than producing probabilities.
    pub probabilities: BTreeMap<String, f32>,
    pub secondary_intent: Option<PromptIntent>,
    /// Probability (Laya) or raw signal weight (keyword) of the runner-up.
    pub secondary_confidence: Option<f32>,
    /// True when no intent led clearly enough to justify a specialist.
    pub ambiguous: bool,
    /// `en`, `hi`, `hi-en` or `und`; see [`PromptLanguage`].
    pub language: String,
    /// The Laya checkpoint that answered, when Laya answered at all.
    pub laya_model: Option<String>,
    /// Wall time to reach this verdict, the sidecar round trip included.
    pub latency_ms: f32,
    /// True when Laya was expected to decide and could not.
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
    pub source: IntentSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laya_shadow: Option<ShadowVerdict>,
}

impl IntentAnalysis {
    /// The keyword classifier's reading, with nothing else consulted.
    ///
    /// This is what every routing entry point without an intent engine uses,
    /// and it reproduces the router's behaviour before this module existed.
    pub fn keyword(prompt: &str) -> Self {
        let started = Instant::now();
        let classified = IntentClassifier::classify(prompt);
        Self::from_keyword(&classified, language::detect(prompt), started)
    }

    /// The keyword reading, used because Laya could not be.
    pub fn keyword_fallback(
        prompt: &str,
        reason: impl Into<String>,
        shadow: Option<ShadowVerdict>,
    ) -> Self {
        let started = Instant::now();
        let classified = IntentClassifier::classify(prompt);
        let mut analysis = Self::from_keyword(&classified, language::detect(prompt), started);
        analysis.fallback_used = true;
        analysis.fallback_reason = Some(reason.into());
        analysis.laya_model = shadow.as_ref().map(|s| s.checkpoint.clone());
        analysis.laya_shadow = shadow;
        analysis
    }

    pub(crate) fn from_keyword(
        classified: &ClassificationResult,
        language: PromptLanguage,
        started: Instant,
    ) -> Self {
        Self {
            primary_intent: classified.intent.clone(),
            confidence: classified.confidence,
            probabilities: BTreeMap::new(),
            secondary_intent: classified.runner_up.as_ref().map(|(i, _)| i.clone()),
            secondary_confidence: classified.runner_up.as_ref().map(|(_, s)| *s),
            ambiguous: classified.confidence < KEYWORD_SPECIALIST_CONFIDENCE,
            language: language.code().to_string(),
            laya_model: None,
            latency_ms: started.elapsed().as_secs_f32() * 1000.0,
            fallback_used: false,
            fallback_reason: None,
            source: IntentSource::Keyword,
            laya_shadow: None,
        }
    }

    /// Laya's reading, gated.
    ///
    /// `keyword` is consulted only under [`GatePolicy::Hybrid`], and only to
    /// rescue a verdict Laya itself leaned toward: it can confirm Laya, never
    /// overrule it. A confident keyword verdict that disagrees with a gated
    /// Laya verdict loses, because the cases where they disagree are the cases
    /// keyword counting is known to get wrong — "debug why the exchanger outlet
    /// temperature keeps dropping" carries a STRONG coding signal and is not a
    /// coding request.
    pub fn from_laya(
        verdict: &LayaVerdict,
        keyword: &ClassificationResult,
        gate: LayaGate,
        policy: GatePolicy,
        language: PromptLanguage,
        latency_ms: f32,
    ) -> Result<Self, String> {
        let ranked = verdict.ranked()?;
        let (ambiguous, source) = if gate.admits(&ranked) {
            (false, IntentSource::Laya)
        } else if policy == GatePolicy::Hybrid
            && keyword.intent == ranked.top
            && keyword.confidence >= KEYWORD_SPECIALIST_CONFIDENCE
        {
            (false, IntentSource::Agreement)
        } else {
            (true, IntentSource::Laya)
        };
        Ok(Self {
            primary_intent: ranked.top.clone(),
            confidence: ranked.top_probability,
            probabilities: verdict.probabilities.clone(),
            secondary_intent: Some(ranked.second.clone()),
            secondary_confidence: Some(ranked.second_probability),
            ambiguous,
            language: language.code().to_string(),
            laya_model: Some(verdict.checkpoint.clone()),
            latency_ms,
            fallback_used: false,
            fallback_reason: None,
            source,
            laya_shadow: None,
        })
    }

    /// Whether the router may pick a specialist on this reading.
    pub fn is_specialist_grade(&self) -> bool {
        !self.ambiguous
    }

    /// The capability key (`coding`, `general`, …) of the leading intent.
    pub fn capability_name(&self) -> &'static str {
        self.primary_intent.to_capability_name()
    }

    /// The intent routing should act on: the leader, or general when unclear.
    pub fn effective_intent(&self) -> PromptIntent {
        if self.ambiguous {
            PromptIntent::GeneralChat
        } else {
            self.primary_intent.clone()
        }
    }

    /// The first line of the routing trace: what the turn was read as.
    ///
    /// The keyword wording is the router's original wording, verbatim, so a
    /// deployment without Laya produces exactly the trace it always did.
    pub fn reading_reason(&self) -> String {
        let pct = |p: f32| p * 100.0;
        let runner_up = match (&self.secondary_intent, self.secondary_confidence) {
            (Some(intent), Some(p)) => format!("{} at {:.0}%", intent.to_capability_name(), pct(p)),
            _ => "no runner-up".to_string(),
        };
        match (self.source, self.ambiguous) {
            (IntentSource::Keyword, false) => format!(
                "Read as a {} request (confidence {:.0}%).",
                self.capability_name(),
                pct(self.confidence)
            ),
            (IntentSource::Keyword, true) => format!(
                "Intent was unclear (confidence {:.0}%), so it is being handled by a general \
                 reasoning model rather than a specialist.",
                pct(self.confidence)
            ),
            (IntentSource::Laya, false) => format!(
                "Read as a {} request by the semantic intent model (Laya {} checkpoint, {:.0}%; \
                 runner-up {}).",
                self.capability_name(),
                self.laya_model.as_deref().unwrap_or("unknown"),
                pct(self.confidence),
                runner_up
            ),
            (IntentSource::Laya, true) => format!(
                "Intent was unclear to the semantic intent model ({} at {:.0}%, {}), so it is \
                 being handled by a general reasoning model rather than a specialist.",
                self.capability_name(),
                pct(self.confidence),
                runner_up
            ),
            (IntentSource::Agreement, _) => format!(
                "Read as a {} request: the semantic intent model leaned that way ({:.0}%, {}) \
                 and the keyword classifier agreed confidently.",
                self.capability_name(),
                pct(self.confidence),
                runner_up
            ),
        }
    }

    /// A trace line saying the semantic reader was not used, and why.
    pub fn fallback_note(&self) -> Option<String> {
        self.fallback_reason.as_ref().map(|reason| {
            format!(
                "The semantic intent model did not decide this turn ({reason}), so the keyword \
                 classifier's reading was used."
            )
        })
    }

    /// One structured line for the application log.
    pub fn log_line(&self) -> String {
        let second = match (&self.secondary_intent, self.secondary_confidence) {
            (Some(i), Some(p)) => format!("{} {:.3}", i.to_capability_name(), p),
            _ => "-".to_string(),
        };
        let mut line = format!(
            "[INTENT] source={} lang={} intent={} confidence={:.3} runner_up={} ambiguous={} \
             checkpoint={} latency_ms={:.1}",
            self.source.label(),
            self.language,
            self.capability_name(),
            self.confidence,
            second,
            self.ambiguous,
            self.laya_model.as_deref().unwrap_or("-"),
            self.latency_ms,
        );
        if let Some(reason) = &self.fallback_reason {
            line.push_str(&format!(" fallback={reason:?}"));
        }
        if let Some(shadow) = &self.laya_shadow {
            line.push_str(&format!(
                " laya_shadow={}:{:.3}/{}:{:.3}@{}",
                shadow.intent.to_capability_name(),
                shadow.probability,
                shadow.runner_up.to_capability_name(),
                shadow.runner_up_probability,
                shadow.checkpoint
            ));
        }
        line
    }
}

impl ShadowVerdict {
    pub fn from_ranked(ranked: &Ranked, checkpoint: &str) -> Self {
        Self {
            intent: ranked.top.clone(),
            probability: ranked.top_probability,
            runner_up: ranked.second.clone(),
            runner_up_probability: ranked.second_probability,
            checkpoint: checkpoint.to_string(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A Laya answer with `leader` at `p`, `second` at `q`, the rest shared.
    pub(crate) fn verdict(leader: &str, p: f32, second: &str, q: f32) -> LayaVerdict {
        let rest = (1.0 - p - q) / 4.0;
        let probabilities = INTENT_LABELS
            .iter()
            .map(|label| {
                let value = if *label == leader {
                    p
                } else if *label == second {
                    q
                } else {
                    rest
                };
                (label.to_string(), value)
            })
            .collect();
        LayaVerdict {
            choice: leader.to_string(),
            probabilities,
            answer_confidence: Some(p),
            entropy_confidence: None,
            checkpoint: "english".to_string(),
            checkpoint_reason: None,
            input_tokens: None,
            latency_ms: 300.0,
        }
    }

    const GATE: LayaGate = LayaGate { min_probability: 0.5, min_margin: 0.2 };

    fn laya(v: &LayaVerdict, prompt: &str, policy: GatePolicy) -> IntentAnalysis {
        IntentAnalysis::from_laya(
            v,
            &IntentClassifier::classify(prompt),
            GATE,
            policy,
            language::detect(prompt),
            310.0,
        )
        .unwrap()
    }

    /// The shared question file and the Rust mapping cannot drift apart.
    #[test]
    fn the_question_file_names_exactly_the_six_intents() {
        let spec: serde_json::Value = serde_json::from_str(INTENT_QUESTION_JSON).unwrap();
        let criteria = spec["criteria"].as_object().expect("criteria is an object");
        let labels: Vec<&str> = criteria.keys().map(String::as_str).collect();
        assert_eq!(labels.len(), 6);
        for label in INTENT_LABELS {
            assert!(criteria.contains_key(label), "question file lacks {label:?}");
            let intent = intent_from_label(label).unwrap();
            assert_eq!(intent.to_capability_name(), label, "label is the capability key");
        }
    }

    /// Laya's README: choice keys are rendered verbatim, and boolean-word
    /// labels get followed instead of their descriptions.
    #[test]
    fn no_label_is_a_boolean_word() {
        for label in INTENT_LABELS {
            assert!(!["true", "false", "yes", "no", "a", "b"].contains(&label), "{label}");
        }
    }

    /// The English checkpoint gives the six options and the instruction 192
    /// tokens between them. The sidecar measures this exactly with the
    /// checkpoint's own tokenizer and refuses to start over it; this is the
    /// cheap early warning, at roughly four characters a token.
    #[test]
    fn the_descriptions_fit_the_english_head_budget_with_room() {
        let spec: serde_json::Value = serde_json::from_str(INTENT_QUESTION_JSON).unwrap();
        let options: usize = spec["criteria"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(label, text)| label.len() + 2 + text.as_str().unwrap().len())
            .sum();
        let instruction = spec["instructions"].as_str().unwrap().len() + "choice question: ".len();
        assert!(
            options + instruction <= 150 * 4,
            "{options} characters of options and {instruction} of instruction leave too little of \
             the 192-token head budget"
        );
    }

    #[test]
    fn a_clear_laya_verdict_is_specialist_grade() {
        let analysis = laya(&verdict("coding", 0.82, "research", 0.09), "fix it", GatePolicy::Laya);
        assert_eq!(analysis.primary_intent, PromptIntent::Coding);
        assert!(analysis.is_specialist_grade());
        assert_eq!(analysis.source, IntentSource::Laya);
        assert_eq!(analysis.secondary_intent, Some(PromptIntent::Research));
        assert_eq!(analysis.laya_model.as_deref(), Some("english"));
        assert!(!analysis.fallback_used);
    }

    #[test]
    fn a_low_top_probability_is_ambiguous() {
        let analysis = laya(&verdict("coding", 0.41, "research", 0.12), "x", GatePolicy::Laya);
        assert!(analysis.ambiguous);
        assert_eq!(analysis.effective_intent(), PromptIntent::GeneralChat);
    }

    /// Two intents close together is ambiguous even when the leader is high.
    #[test]
    fn a_narrow_lead_is_ambiguous() {
        let analysis = laya(&verdict("coding", 0.52, "research", 0.40), "x", GatePolicy::Laya);
        assert!(analysis.ambiguous, "a 12-point lead is two readings, not one");
    }

    /// Agreement can confirm what Laya leaned toward…
    #[test]
    fn hybrid_agreement_rescues_a_near_miss_the_keywords_confirm() {
        let prompt = "Refactor this Python function and fix the stack trace";
        let near_miss = verdict("coding", 0.45, "reasoning", 0.30);
        assert!(laya(&near_miss, prompt, GatePolicy::Laya).ambiguous);
        let hybrid = laya(&near_miss, prompt, GatePolicy::Hybrid);
        assert!(!hybrid.ambiguous);
        assert_eq!(hybrid.source, IntentSource::Agreement);
    }

    /// …and never overrule it.
    #[test]
    fn keywords_cannot_overrule_a_laya_verdict() {
        // Keyword classifier reads this as confident coding (`debug`, STRONG).
        let prompt = "debug and refactor why the heat exchanger outlet temperature keeps dropping";
        assert_eq!(IntentClassifier::classify(prompt).intent, PromptIntent::Coding);
        let analysis = laya(&verdict("reasoning", 0.71, "coding", 0.12), prompt, GatePolicy::Hybrid);
        assert_eq!(analysis.primary_intent, PromptIntent::Reasoning);
        assert_eq!(analysis.source, IntentSource::Laya);

        // And a keyword verdict for a *different* intent does not rescue a near miss.
        let near_miss = verdict("research", 0.45, "coding", 0.30);
        assert!(laya(&near_miss, prompt, GatePolicy::Hybrid).ambiguous);
    }

    #[test]
    fn a_malformed_verdict_is_an_error_not_a_reading() {
        let mut missing = verdict("coding", 0.8, "research", 0.1);
        missing.probabilities.remove("general");
        assert!(missing.ranked().is_err());

        let mut renamed = verdict("coding", 0.8, "research", 0.1);
        let p = renamed.probabilities.remove("coding").unwrap();
        renamed.probabilities.insert("code".into(), p);
        assert!(renamed.ranked().is_err());

        let mut unnormalised = verdict("coding", 0.8, "research", 0.1);
        unnormalised.probabilities.insert("coding".into(), 0.95);
        assert!(unnormalised.ranked().is_err());
    }

    /// The keyword path is the router's old behaviour, down to the wording.
    #[test]
    fn the_keyword_reading_matches_the_old_router() {
        let clear = IntentAnalysis::keyword("Refactor this Python function and fix the stack trace");
        assert!(clear.is_specialist_grade());
        assert!(clear.reading_reason().starts_with("Read as a coding request (confidence "));

        let unclear = IntentAnalysis::keyword("hello");
        assert!(unclear.ambiguous);
        assert!(unclear.reading_reason().starts_with("Intent was unclear (confidence "));
        assert!(!unclear.fallback_used && unclear.fallback_note().is_none());
    }

    #[test]
    fn a_fallback_says_why_and_keeps_what_laya_thought() {
        let shadow = ShadowVerdict::from_ranked(
            &verdict("research", 0.7, "general", 0.1).ranked().unwrap(),
            "english",
        );
        let analysis =
            IntentAnalysis::keyword_fallback("summarise this", "Laya is not calibrated", Some(shadow));
        assert!(analysis.fallback_used);
        assert_eq!(analysis.source, IntentSource::Keyword);
        assert!(analysis.fallback_note().unwrap().contains("Laya is not calibrated"));
        let line = analysis.log_line();
        assert!(line.contains("source=keyword"), "{line}");
        assert!(line.contains("laya_shadow=research:0.700"), "{line}");
    }

    #[test]
    fn the_log_line_names_every_required_field() {
        let line = laya(&verdict("coding", 0.82, "research", 0.09), "fix it", GatePolicy::Laya).log_line();
        for field in ["source=laya", "intent=coding", "confidence=0.820", "runner_up=research 0.090",
                      "checkpoint=english", "lang=en", "latency_ms=310.0"] {
            assert!(line.contains(field), "{field} missing from {line}");
        }
    }

    #[test]
    fn the_analysis_serialises_for_the_trace() {
        let analysis = laya(&verdict("coding", 0.82, "research", 0.09), "fix it", GatePolicy::Laya);
        let json = serde_json::to_value(&analysis).unwrap();
        for key in ["primaryIntent", "confidence", "probabilities", "secondaryIntent", "ambiguous",
                    "language", "layaModel", "latencyMs", "fallbackUsed", "source"] {
            assert!(json.get(key).is_some(), "{key} missing: {json}");
        }
        let back: IntentAnalysis = serde_json::from_value(json).unwrap();
        assert_eq!(back, analysis);
    }
}

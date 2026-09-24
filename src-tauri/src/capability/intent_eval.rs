//! The intent validation set, three routing policies scored on it, and the
//! calibration run that fits Laya's gate.
//!
//! `capability::eval` measures the keyword classifier on developer prompts.
//! This set is about what routing gets wrong *at a plant*: engineering language
//! that reads as programming, Hindi and Hindi-English, one-word follow-ups,
//! prompts with no clear task, and prompts with two. It lives in
//! `fixtures/intent-routing/v1/validation.jsonl`, 180 cases split evenly into a
//! `fit` half, which the gate is chosen on, and a `test` half, which reports how
//! it does on prompts it was not chosen on.
//!
//! ## What is scored
//!
//! The router has one specialist role, coding; every other intent reaches the
//! general reasoning band. So the errors that change which model answers are:
//!
//! - **false specialist** — a prompt that is not coding reaches the coding
//!   model. The costly one: a coding model handles a summary badly, and the
//!   router's own doctrine is that a general model handles a coding question
//!   adequately.
//! - **missed specialist** — a coding prompt reaches the general model.
//!
//! Intent accuracy over all six labels is reported beside them, and breaks ties.
//!
//! ## How the gate is fitted
//!
//! Every gate on a grid of minimum probability × minimum margin, under both
//! [`GatePolicy`]s, is scored on the fitting half. Among those that produce no
//! more false-specialist routes than the keyword classifier does on the same
//! prompts, the one with the fewest routing errors wins; ties go to higher
//! intent accuracy, then to plain `Laya`, then to the stricter gate. The
//! constraint is always satisfiable, because a strict enough gate abstains on
//! everything.
//!
//! ## Running it
//!
//! The keyword baseline and the dataset checks run in the ordinary test suite.
//! Measuring Laya needs its weights, so that test is ignored by default and
//! fails loudly without them:
//!
//! ```text
//! ARJUN_LAYA_DIR=<convaiinnovations/laya bundle> ARJUN_PYTHON=python3 \
//!   ARJUN_LAYA_DEVICE=cpu cargo test --lib intent_eval::measure -- --ignored --nocapture
//! ```
//!
//! It writes `predictions-<device>.json` and `arjun-intent-calibration.json` to
//! `target/intent-eval/` (or `ARJUN_INTENT_OUT_DIR`) and prints the comparison.
//! Run it once with `ARJUN_LAYA_DEVICE=cuda` for the GPU row.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capability::classifier::{ClassificationResult, IntentClassifier};
use crate::capability::intent_analysis::{
    intent_from_label, GatePolicy, IntentAnalysis, LayaGate, LayaVerdict,
};
use crate::capability::language;
use crate::registry::router::ModelRouter;
use crate::registry::ModelRole;

pub const VALIDATION_JSONL: &str =
    include_str!("../../../fixtures/intent-routing/v1/validation.jsonl");
pub const VALIDATION_PATH: &str = "fixtures/intent-routing/v1/validation.jsonl";

/// One labelled prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub id: String,
    pub split: String,
    pub category: String,
    pub lang: String,
    pub prompt: String,
    /// The best label; `general` where the right answer is to abstain.
    pub intent: String,
    /// Other labels a careful reader could defend.
    #[serde(default)]
    pub accept: Vec<String>,
    /// For multi-intent prompts, the kinds of work combined.
    #[serde(default)]
    pub components: Vec<String>,
}

impl Case {
    fn acceptable(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.intent.as_str()).chain(self.accept.iter().map(String::as_str))
    }

    fn acceptable_roles(&self) -> Vec<ModelRole> {
        let mut roles: Vec<ModelRole> = self
            .acceptable()
            .filter_map(intent_from_label)
            .map(ModelRouter::role_for)
            .collect();
        roles.dedup();
        roles
    }
}

pub fn cases() -> Vec<Case> {
    VALIDATION_JSONL
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("validation.jsonl holds one case per line"))
        .collect()
}

/// What a policy did with one case.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub analysis: IntentAnalysis,
}

impl Outcome {
    fn effective(&self) -> &'static str {
        self.analysis.effective_intent().to_capability_name()
    }

    fn routed(&self) -> ModelRole {
        ModelRouter::role_for(self.analysis.effective_intent())
    }
}

/// Aggregate scores for one policy on a set of cases.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub cases: usize,
    pub intent_correct: usize,
    pub role_correct: usize,
    pub false_specialist: usize,
    pub missed_specialist: usize,
    pub abstained: usize,
    /// Multi-intent cases whose top two intents are two of their components.
    pub multi_detected: usize,
    pub multi_cases: usize,
    pub by_category: BTreeMap<String, CategoryMetrics>,
    pub latency_p50_ms: f32,
    pub latency_p95_ms: f32,
    /// Ids of the cases routed to the coding model that should not have been.
    pub false_specialist_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CategoryMetrics {
    pub cases: usize,
    pub intent_correct: usize,
    pub role_correct: usize,
    pub false_specialist: usize,
}

impl Metrics {
    pub fn routing_errors(&self) -> usize {
        self.cases - self.role_correct
    }

    fn pct(part: usize, whole: usize) -> f64 {
        if whole == 0 { 0.0 } else { 100.0 * part as f64 / whole as f64 }
    }

    pub fn row(&self, name: &str) -> String {
        format!(
            "{name:<22} n={:>3}  intent {:>5.1}%  role {:>5.1}%  false-spec {:>2}  missed-spec {:>2}  abstain {:>5.1}%  multi {}/{}  p50 {:>8.3} ms  p95 {:>8.3} ms",
            self.cases,
            Self::pct(self.intent_correct, self.cases),
            Self::pct(self.role_correct, self.cases),
            self.false_specialist,
            self.missed_specialist,
            Self::pct(self.abstained, self.cases),
            self.multi_detected,
            self.multi_cases,
            self.latency_p50_ms,
            self.latency_p95_ms,
        )
    }

    pub fn category_table(&self) -> String {
        let mut out = String::new();
        for (category, m) in &self.by_category {
            out.push_str(&format!(
                "    {category:<10} n={:>3}  intent {:>5.1}%  role {:>5.1}%  false-spec {}\n",
                m.cases,
                Self::pct(m.intent_correct, m.cases),
                Self::pct(m.role_correct, m.cases),
                m.false_specialist
            ));
        }
        out
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Scores outcomes against their cases, in order.
pub fn score(cases: &[&Case], outcomes: &[Outcome]) -> Metrics {
    assert_eq!(cases.len(), outcomes.len());
    let mut m = Metrics { cases: cases.len(), ..Metrics::default() };
    let mut latencies = Vec::with_capacity(cases.len());
    for (case, outcome) in cases.iter().zip(outcomes) {
        let effective = outcome.effective();
        let routed = outcome.routed();
        let roles = case.acceptable_roles();
        let intent_ok = case.acceptable().any(|label| label == effective);
        let role_ok = roles.contains(&routed);
        let false_spec = routed == ModelRole::Coding && !roles.contains(&ModelRole::Coding);
        let missed_spec = routed == ModelRole::Reasoning && roles == [ModelRole::Coding];

        m.intent_correct += usize::from(intent_ok);
        m.role_correct += usize::from(role_ok);
        m.false_specialist += usize::from(false_spec);
        if false_spec {
            m.false_specialist_ids.push(case.id.clone());
        }
        m.missed_specialist += usize::from(missed_spec);
        m.abstained += usize::from(outcome.analysis.ambiguous);
        if case.components.len() >= 2 {
            m.multi_cases += 1;
            let top = outcome.analysis.primary_intent.to_capability_name();
            let second = outcome.analysis.secondary_intent.as_ref().map(|i| i.to_capability_name());
            let hit = |label: &str| case.components.iter().any(|c| c == label);
            if hit(top) && second.is_some_and(hit) {
                m.multi_detected += 1;
            }
        }
        let c = m.by_category.entry(case.category.clone()).or_default();
        c.cases += 1;
        c.intent_correct += usize::from(intent_ok);
        c.role_correct += usize::from(role_ok);
        c.false_specialist += usize::from(false_spec);
        latencies.push(outcome.analysis.latency_ms);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    m.latency_p50_ms = percentile(&latencies, 50.0);
    m.latency_p95_ms = percentile(&latencies, 95.0);
    m
}

/// The keyword classifier, as routing uses it today.
pub fn keyword_outcomes(cases: &[&Case]) -> Vec<Outcome> {
    cases
        .iter()
        .map(|case| {
            let started = Instant::now();
            let mut analysis = IntentAnalysis::keyword(&case.prompt);
            analysis.latency_ms = started.elapsed().as_secs_f32() * 1000.0;
            Outcome { analysis }
        })
        .collect()
}

/// The keyword classifier with a different specialist threshold, for the sweep.
pub fn keyword_outcomes_at(cases: &[&Case], threshold: f32) -> Vec<Outcome> {
    keyword_outcomes(cases)
        .into_iter()
        .map(|mut outcome| {
            outcome.analysis.ambiguous = outcome.analysis.confidence < threshold;
            outcome
        })
        .collect()
}

/// One recorded Laya answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Prediction {
    pub id: String,
    pub verdict: LayaVerdict,
    /// Wall time as a turn sees it: request written to answer parsed.
    pub round_trip_ms: f32,
}

/// A case with its recorded Laya answer and keyword reading, computed once.
///
/// The gate search scores thousands of gates over the same prompts; the
/// keyword reading and language do not depend on the gate.
pub struct Prepared<'a> {
    case: &'a Case,
    prediction: &'a Prediction,
    keyword: ClassificationResult,
    language: language::PromptLanguage,
}

pub fn prepare<'a>(cases: &[&'a Case], predictions: &'a HashMap<String, Prediction>) -> Vec<Prepared<'a>> {
    cases
        .iter()
        .map(|case| Prepared {
            case,
            prediction: predictions
                .get(&case.id)
                .unwrap_or_else(|| panic!("no recorded Laya answer for {}", case.id)),
            keyword: IntentClassifier::classify(&case.prompt),
            language: language::detect(&case.prompt),
        })
        .collect()
}

/// Laya under a gate and policy, from recorded answers.
pub fn laya_outcomes(prepared: &[Prepared<'_>], gate: LayaGate, policy: GatePolicy) -> Vec<Outcome> {
    prepared
        .iter()
        .map(|p| {
            let analysis = IntentAnalysis::from_laya(
                &p.prediction.verdict,
                &p.keyword,
                gate,
                policy,
                p.language,
                p.prediction.round_trip_ms,
            )
            .unwrap_or_else(|reason| panic!("{}: {reason}", p.case.id));
            Outcome { analysis }
        })
        .collect()
}

/// [`laya_outcomes`] scored.
pub fn score_laya(prepared: &[Prepared<'_>], gate: LayaGate, policy: GatePolicy) -> Metrics {
    let cases: Vec<&Case> = prepared.iter().map(|p| p.case).collect();
    score(&cases, &laya_outcomes(prepared, gate, policy))
}

/// The gate chosen on the fitting half, and what it scored there.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fit {
    pub gate: LayaGate,
    pub policy: GatePolicy,
    pub fit: Metrics,
    pub keyword_false_specialist: usize,
}

/// Lexicographic preference for a gate: fewer routing errors, then more
/// intents right, then plain `Laya` over `Hybrid`, then the stricter gate.
type Preference = (usize, std::cmp::Reverse<usize>, u8, std::cmp::Reverse<u32>, std::cmp::Reverse<u32>);

/// Grid-searches the gate on `cases`; see the module note for the objective.
pub fn fit_gate(cases: &[&Case], predictions: &HashMap<String, Prediction>) -> Fit {
    let baseline = score(cases, &keyword_outcomes(cases)).false_specialist;
    let prepared = prepare(cases, predictions);
    let mut best: Option<(Fit, Preference)> = None;
    for policy in [GatePolicy::Laya, GatePolicy::Hybrid] {
        for p in 20..=95u32 {
            for margin in 0..=60u32 {
                let gate = LayaGate {
                    min_probability: p as f32 / 100.0,
                    min_margin: margin as f32 / 100.0,
                };
                let metrics = score_laya(&prepared, gate, policy);
                if metrics.false_specialist > baseline {
                    continue;
                }
                let key: Preference = (
                    metrics.routing_errors(),
                    std::cmp::Reverse(metrics.intent_correct),
                    u8::from(policy == GatePolicy::Hybrid),
                    std::cmp::Reverse(p),
                    std::cmp::Reverse(margin),
                );
                if best.as_ref().is_none_or(|(_, k)| key < *k) {
                    best = Some((
                        Fit { gate, policy, fit: metrics, keyword_false_specialist: baseline },
                        key,
                    ));
                }
            }
        }
    }
    best.expect("a strict enough gate abstains on everything and always qualifies").0
}

/// The split a case belongs to.
pub fn split<'a>(all: &'a [Case], name: &str) -> Vec<&'a Case> {
    all.iter().filter(|c| c.split == name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::intent_analysis::tests::verdict;
    use crate::capability::intent_analysis::INTENT_LABELS;
    use std::collections::HashSet;

    /// Requirement coverage: every kind of prompt the brief names is present,
    /// in both halves, and every label is one the question can answer.
    #[test]
    fn the_validation_set_covers_what_it_claims_to() {
        let all = cases();
        assert_eq!(all.len(), 180);
        let ids: HashSet<&str> = all.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), all.len(), "ids are unique");
        for case in &all {
            for label in case.acceptable().chain(case.components.iter().map(String::as_str)) {
                assert!(INTENT_LABELS.contains(&label), "{}: unknown label {label:?}", case.id);
            }
            assert!(["fit", "test"].contains(&case.split.as_str()), "{}", case.id);
        }
        for (category, minimum) in [
            ("clear", 40), ("refinery", 20), ("ambiguous", 10), ("short", 10),
            ("hindi", 15), ("hinglish", 15), ("multi", 10),
        ] {
            for half in ["fit", "test"] {
                let n = all.iter().filter(|c| c.category == category && c.split == half).count();
                assert!(n * 2 >= minimum, "{category}/{half} has only {n} cases");
            }
        }
        for label in INTENT_LABELS {
            assert!(all.iter().any(|c| c.intent == label), "no case is labelled {label}");
        }
    }

    /// The language detector is measured against the set's own labels, which
    /// were written by hand before the detector ran on them.
    #[test]
    fn language_detection_matches_the_labels() {
        let all = cases();
        let mut wrong = Vec::new();
        for case in &all {
            let detected = language::detect(&case.prompt).code();
            if detected != case.lang {
                wrong.push(format!("{} {:?}: labelled {} detected {detected}", case.id, case.prompt, case.lang));
            }
        }
        println!("language detection: {}/{} match", all.len() - wrong.len(), all.len());
        for w in &wrong {
            println!("  {w}");
        }
        assert!(wrong.is_empty(), "{} mismatches:\n{}", wrong.len(), wrong.join("\n"));
    }

    /// The keyword classifier on the new set — the baseline Laya has to beat —
    /// with its specialist threshold swept, so the 0.55 in use is measured
    /// rather than assumed. Printed with `--nocapture`.
    #[test]
    fn keyword_baseline_and_threshold_sweep() {
        let all = cases();
        let everything: Vec<&Case> = all.iter().collect();
        let baseline = score(&everything, &keyword_outcomes(&everything));
        println!("\n{}", baseline.row("keyword @0.55 (all)"));
        print!("{}", baseline.category_table());
        for id in &baseline.false_specialist_ids {
            let case = all.iter().find(|c| &c.id == id).unwrap();
            println!("    false specialist: {id} {:?}", case.prompt);
        }
        for half in ["fit", "test"] {
            let part = split(&all, half);
            println!("{}", score(&part, &keyword_outcomes(&part)).row(&format!("keyword @0.55 ({half})")));
        }
        println!("  threshold sweep (all 180):");
        let mut abstained = 0;
        for t in (30..=80).step_by(5) {
            let m = score(&everything, &keyword_outcomes_at(&everything, t as f32 / 100.0));
            println!(
                "    t={:.2}  routing errors {:>3}  false-spec {:>2}  missed-spec {:>2}  intent {:>3}/{}",
                t as f32 / 100.0,
                m.routing_errors(),
                m.false_specialist,
                m.missed_specialist,
                m.intent_correct,
                m.cases
            );
            assert!(m.abstained >= abstained, "abstaining must grow with the threshold");
            abstained = m.abstained;
        }

        // The failure mode of the keyword table on Hindi is safe: it finds no
        // signal, abstains, and the turn reaches the general model.
        let hindi: Vec<&Case> = all.iter().filter(|c| c.category == "hindi").collect();
        assert_eq!(score(&hindi, &keyword_outcomes(&hindi)).false_specialist, 0);
    }

    // The tests below drive the scoring and fitting code with *synthetic*
    // Laya answers built from the labels. They check the arithmetic, not
    // Laya; no number they produce is a measurement of anything.

    fn oracle(all: &[Case], confidence: f32) -> HashMap<String, Prediction> {
        all.iter()
            .map(|case| {
                let second = if case.intent == "general" { "reasoning" } else { "general" };
                let p = Prediction {
                    id: case.id.clone(),
                    verdict: verdict(&case.intent, confidence, second, (1.0 - confidence) / 2.0),
                    round_trip_ms: 1.0,
                };
                (case.id.clone(), p)
            })
            .collect()
    }

    #[test]
    fn a_perfect_reader_scores_perfectly() {
        let all = cases();
        let everything: Vec<&Case> = all.iter().collect();
        let gate = LayaGate { min_probability: 0.5, min_margin: 0.2 };
        let predictions = oracle(&all, 0.9);
        let m = score_laya(&prepare(&everything, &predictions), gate, GatePolicy::Laya);
        assert_eq!(m.intent_correct, m.cases);
        assert_eq!(m.routing_errors(), 0);
        assert_eq!(m.false_specialist, 0);
    }

    #[test]
    fn fitting_never_accepts_more_false_specialists_than_the_keywords() {
        let all = cases();
        // A reader that calls everything coding at 0.6: any gate that lets it
        // through sends every plant question to the coding model.
        let predictions: HashMap<String, Prediction> = all
            .iter()
            .map(|case| {
                let p = Prediction {
                    id: case.id.clone(),
                    verdict: verdict("coding", 0.6, "general", 0.1),
                    round_trip_ms: 1.0,
                };
                (case.id.clone(), p)
            })
            .collect();
        let fit_half = split(&all, "fit");
        let fit = fit_gate(&fit_half, &predictions);
        assert!(fit.fit.false_specialist <= fit.keyword_false_specialist, "{fit:?}");
        assert!(fit.gate.min_probability > 0.6 || fit.gate.min_margin > 0.5, "{:?}", fit.gate);
    }

    #[test]
    fn fitting_a_clear_reader_admits_it() {
        let all = cases();
        let fit_half = split(&all, "fit");
        let fit = fit_gate(&fit_half, &oracle(&all, 0.8));
        assert_eq!(fit.fit.routing_errors(), 0, "{fit:?}");
        assert!(fit.gate.min_probability <= 0.8);
    }

    /// Measures Laya on the validation set, fits the gate, and prints the
    /// three-way comparison. Needs the weights; see the module note.
    #[test]
    #[ignore = "needs the Laya weights: set ARJUN_LAYA_DIR (see capability::intent_eval)"]
    fn measure() {
        use crate::capability::laya_sidecar::{
            IntentEngine, LayaCalibration, LayaConfig, Launch, CALIBRATION_FILE, CALIBRATION_SCHEMA,
        };
        use std::time::Duration;

        let model_dir = std::env::var("ARJUN_LAYA_DIR").expect(
            "ARJUN_LAYA_DIR must name the convaiinnovations/laya bundle; this test measures the \
             real model and has nothing to measure without it",
        );
        let device = std::env::var("ARJUN_LAYA_DEVICE").unwrap_or_else(|_| "cpu".into());
        let out_dir = std::env::var("ARJUN_INTENT_OUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("target/intent-eval"));
        std::fs::create_dir_all(&out_dir).unwrap();

        let engine = IntentEngine::new(
            LayaConfig {
                enabled: true,
                model_dir: model_dir.clone().into(),
                calibration_path: out_dir.join("unused-during-measurement.json"),
                device: device.clone(),
                deadline: Duration::from_secs(60),
            },
            Launch::bundled(),
        );
        let loaded = engine
            .wait_until_ready(Duration::from_secs(900))
            .unwrap_or_else(|reason| panic!("Laya did not load: {reason}"));
        println!("\nloaded: {loaded:?}");

        for warm in ["hello", "write a parser", "इस रिपोर्ट का सारांश दो"] {
            let _ = engine.raw_verdict(warm);
        }

        let all = cases();
        let mut predictions = HashMap::new();
        for case in &all {
            let (verdict, round_trip_ms) = engine
                .raw_verdict(&case.prompt)
                .unwrap_or_else(|reason| panic!("{}: {reason}", case.id));
            predictions.insert(case.id.clone(), Prediction { id: case.id.clone(), verdict, round_trip_ms });
        }
        let after = engine.status();
        println!("after the run: {after:?}");

        let recorded: Vec<&Prediction> = all.iter().map(|c| &predictions[&c.id]).collect();
        let predictions_path = out_dir.join(format!("predictions-{device}.json"));
        std::fs::write(
            &predictions_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "device": device, "loaded": loaded, "after": after, "predictions": recorded,
            }))
            .unwrap(),
        )
        .unwrap();

        let fit_half = split(&all, "fit");
        let test_half = split(&all, "test");
        let everything: Vec<&Case> = all.iter().collect();
        let fit = fit_gate(&fit_half, &predictions);
        let ungated = LayaGate { min_probability: 0.0, min_margin: 0.0 };

        let mut report = String::new();
        report.push_str(&format!(
            "gate fitted on the fit half: p>={:.2} margin>={:.2} policy={:?}\n",
            fit.gate.min_probability, fit.gate.min_margin, fit.policy
        ));
        for (name, half) in [("test", &test_half), ("all", &everything)] {
            let prepared = prepare(half, &predictions);
            let k = score(half, &keyword_outcomes(half));
            let raw = score_laya(&prepared, ungated, GatePolicy::Laya);
            let laya = score_laya(&prepared, fit.gate, GatePolicy::Laya);
            let hybrid = score_laya(&prepared, fit.gate, GatePolicy::Hybrid);
            report.push_str(&format!("[{name}]\n{}\n{}\n{}\n{}\n", k.row("keyword"), raw.row("laya ungated"), laya.row("laya gated"), hybrid.row("hybrid")));
            report.push_str(&format!("  laya gated, by category:\n{}", laya.category_table()));
        }
        println!("{report}");

        let test_metrics = score_laya(&prepare(&test_half, &predictions), fit.gate, fit.policy);
        let calibration = LayaCalibration {
            schema: CALIBRATION_SCHEMA,
            question_version: loaded.question_version.clone().unwrap_or_default(),
            question_fingerprint: loaded.question_fingerprint.clone().unwrap_or_default(),
            laya_version: loaded.laya_version.clone(),
            device: device.clone(),
            checkpoints: {
                let mut used: Vec<String> = predictions.values().map(|p| p.verdict.checkpoint.clone()).collect();
                used.sort();
                used.dedup();
                used
            },
            gate: fit.gate,
            policy: fit.policy,
            dataset: {
                use sha2::{Digest, Sha256};
                format!("{VALIDATION_PATH} sha256:{}", hex::encode(Sha256::digest(VALIDATION_JSONL.as_bytes())))
            },
            fitted_at: chrono::Utc::now().to_rfc3339(),
            evidence: serde_json::json!({ "fit": fit.fit, "test": test_metrics, "keywordFalseSpecialistOnFit": fit.keyword_false_specialist, "report": report }),
        };
        let calibration_path = out_dir.join(CALIBRATION_FILE);
        std::fs::write(&calibration_path, serde_json::to_vec_pretty(&calibration).unwrap()).unwrap();
        println!(
            "wrote {} and {}.\nTo route on this gate: cp {} {}/",
            predictions_path.display(),
            calibration_path.display(),
            calibration_path.display(),
            model_dir
        );
    }
}

//! Choosing a model for a task, and being able to say why.
//!
//! PS 26117 asks for *"model auto selection across at least two different task
//! types"* — a coding request handled differently from a document summary. It
//! also asks, in step 10, that the router **record why a model was selected**,
//! and that an uncertain router fall back to something safe rather than quietly
//! picking badly.
//!
//! So the decision is a value, not a side effect: [`RoutingDecision`] carries the
//! model, the intent that led to it, and the reasons in the order they applied.
//! The task trace shows that list verbatim. A router that cannot explain itself
//! is indistinguishable from a coin toss with good luck.
//!
//! ## How it decides
//!
//! 1. **Classify the prompt.** Sarathi's weighted classifier, which scores every
//!    intent and derives confidence from how far ahead the leader is *and* how
//!    much evidence there was at all.
//! 2. **Low confidence routes to reasoning, not to nothing.** A general model
//!    handles a coding question adequately; a coding model handles a summary
//!    badly. When unsure, the cost of being wrong is lower in that direction.
//! 3. **Filter to real candidates** — enabled, right role, above the floor,
//!    cleared for this material.
//! 4. **Prefer the largest that fits.** Capability tracks size within a role, so
//!    the best model is the biggest one whose weights and KV cache fit the GPU
//!    budget that [`crate::ai_engine::vram_planner`] computes.
//! 5. **Fall back rather than fail.** If nothing fits the GPU, the smallest
//!    candidate runs partly on the CPU and the decision says so.

use serde::{Deserialize, Serialize};

use super::{ModelEntry, ModelRegistry, ModelRole, Modality};
use crate::ai_engine::vram_planner::{plan_gpu_offload, GpuOffloadPlan};
use crate::capability::classifier::IntentClassifier;
use crate::model_intelligence::intent::PromptIntent;
use crate::policy::Classification;

/// Below this, the classification is not trusted to pick a specialist.
///
/// Chosen to match the classifier's own calibration: it reaches this only when
/// one intent leads clearly *and* several signals supported it. A single
/// incidental keyword cannot get here.
const SPECIALIST_CONFIDENCE: f32 = 0.55;

/// What the router decided, and every reason that led there.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingDecision {
    pub model_id: String,
    pub model_name: String,
    pub role: ModelRole,
    /// What the prompt was taken to be asking for.
    pub intent: String,
    pub confidence: f32,
    /// True when the first choice was unavailable and something else was used.
    pub used_fallback: bool,
    /// Ordered, human-readable. Shown verbatim in the task trace.
    pub reasons: Vec<String>,
    pub gpu_plan_summary: String,
    pub fully_on_gpu: bool,
}

/// Why no model could be chosen. Each names what would fix it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingFailure {
    pub role: ModelRole,
    pub reason: String,
}

pub struct ModelRouter;

impl ModelRouter {
    /// Maps a classified intent onto the role that should handle it.
    ///
    /// Only coding gets its own specialist. Mathematics, research and reasoning
    /// all want a strong general model rather than a differently-trained one,
    /// and inventing a role per intent would produce a registry nobody can fill.
    fn role_for(intent: PromptIntent) -> ModelRole {
        match intent {
            PromptIntent::Coding => ModelRole::Coding,
            PromptIntent::Reasoning
            | PromptIntent::Mathematics
            | PromptIntent::ToolCalling
            | PromptIntent::Research
            | PromptIntent::GeneralChat => ModelRole::Reasoning,
        }
    }

    /// Routes a text prompt.
    pub fn route(
        registry: &ModelRegistry,
        prompt: &str,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Result<RoutingDecision, RoutingFailure> {
        let classified = IntentClassifier::classify(prompt);
        // Taken before `intent` is moved into `role_for` below.
        let intent_label = classified.capability_name().to_string();
        let confidence = classified.confidence;
        let mut reasons = Vec::new();

        // Step 1–2: what kind of task is this, and do we trust the answer?
        let confident = classified.confidence >= SPECIALIST_CONFIDENCE;
        let role = if confident {
            reasons.push(format!(
                "Read as a {} request (confidence {:.0}%).",
                intent_label,
                confidence * 100.0
            ));
            Self::role_for(classified.intent)
        } else {
            reasons.push(format!(
                "Intent was unclear (confidence {:.0}%), so it is being handled by a general \
                 reasoning model rather than a specialist.",
                confidence * 100.0
            ));
            ModelRole::Reasoning
        };

        Self::route_to_role(
            registry,
            role,
            classification,
            vram_total_bytes,
            reasons,
            intent_label,
            confidence,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
        )
    }

    /// Routes to a named role directly, for work whose kind is already known —
    /// OCR on a scanned page, embeddings for retrieval — where classifying the
    /// user's words would be answering the wrong question.
    pub fn route_for_role(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Result<RoutingDecision, RoutingFailure> {
        let reasons = vec![format!(
            "The task needs a {} model, so no classification of the prompt was involved.",
            role.label()
        )];
        Self::route_to_role(
            registry,
            role,
            classification,
            vram_total_bytes,
            reasons,
            role.label().to_string(),
            1.0,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
        )
    }

    fn route_to_role(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        mut reasons: Vec<String>,
        intent_label: String,
        confidence: f32,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Result<RoutingDecision, RoutingFailure> {
        // Step 3: real candidates only.
        let mut candidates = registry.candidates(
            role,
            classification,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
        );
        if candidates.is_empty() {
            return Err(RoutingFailure {
                role,
                reason: Self::explain_no_candidates(
                    registry,
                    role,
                    classification,
                    required_modality,
                    require_structured_output,
                    available_runtime_profiles,
                    allowed_licenses,
                ),
            });
        }

        if let Some(classification) = classification {
            reasons.push(format!(
                "{} of {} registered models are cleared for {} material.",
                candidates.len(),
                registry.all().len(),
                classification.label()
            ));
        }

        // Step 4: largest first, so the first that fits is the best that fits.
        // The sort key is (size desc, preferred desc, rank asc). The size
        // band is the dominant factor; the tie-breakers only matter when
        // two candidates have the same `parameters_b`. `preferred` is the
        // operator-set "use this one" knob, and `rank_within_band` is the
        // telemetry-driven ordering, so the deterministic default is
        // preserved when both are absent.
        candidates.sort_by(|a, b| {
            b.parameters_b
                .partial_cmp(&a.parameters_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.routing
                        .preferred
                        .cmp(&a.routing.preferred)
                })
                .then_with(|| {
                    a.routing
                        .rank_within_band
                        .cmp(&b.routing.rank_within_band)
                })
        });

        for entry in &candidates {
            let plan = plan_gpu_offload(vram_total_bytes, entry.weights_bytes, entry.context_length, None);
            if plan.full_offload {
                reasons.push(format!(
                    "{} is the largest cleared {} model that fits in VRAM.",
                    entry.name,
                    role.label()
                ));
                reasons.push(plan.reason.clone());
                return Ok(Self::decide(entry, role, intent_label, confidence, plan, false, reasons));
            }
        }

        // Step 5: nothing fits entirely. Take the smallest and say so, rather
        // than refusing to work on a machine that can still do the job slowly.
        let smallest = candidates
            .last()
            .expect("candidates was checked non-empty above");
        let plan = plan_gpu_offload(
            vram_total_bytes,
            smallest.weights_bytes,
            smallest.context_length,
            None,
        );

        reasons.push(format!(
            "No cleared {} model fits entirely in this machine's VRAM, so ARJUN fell back to \
             {}, the smallest one available. It will run more slowly.",
            role.label(),
            smallest.name
        ));
        reasons.push(plan.reason.clone());

        Ok(Self::decide(smallest, role, intent_label, confidence, plan, true, reasons))
    }

    /// Says which filter emptied the candidate list, so the fix is obvious.
    ///
    /// "No model available" is useless to an administrator. Whether the registry
    /// is empty, the model is disabled, everything is below the floor, or nothing
    /// is cleared for this material are four different problems with four
    /// different remedies.
    fn explain_no_candidates(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> String {
        if registry.all().is_empty() {
            return "No models are registered yet. An administrator imports one in \
                    Provisioning mode."
                .to_string();
        }

        let serving: Vec<_> = registry
            .all()
            .iter()
            .filter(|e| e.serves(role))
            .collect();

        if serving.is_empty() {
            return format!(
                "No registered model is set up for {} work. Register one, or enable the role \
                 on an existing model.",
                role.label()
            );
        }

        if !serving.iter().any(|e| e.enabled) {
            return format!(
                "Every {} model is currently disabled. An administrator re-enables one in \
                 Models.",
                role.label()
            );
        }

        if !serving.iter().any(|e| e.meets_floor(role)) {
            return format!(
                "The registered {} models are all below {:.0}B parameters, which is too small \
                 to be reliable at this kind of work. A larger model is needed.",
                role.label(),
                role.minimum_parameters_b()
            );
        }

        // Check modality filter
        if let Some(modality) = required_modality {
            if !serving.iter().any(|e| e.supports_modality(modality)) {
                return format!(
                    "No {} model supports the required modality ({}). A model with {} capability is needed.",
                    role.label(),
                    modality.label(),
                    modality.label()
                );
            }
        }

        // Check structured output filter
        if require_structured_output {
            if !serving.iter().any(|e| e.supports_structured_output()) {
                return format!(
                    "No {} model supports structured output / tool calling. A model with this capability is needed.",
                    role.label()
                );
            }
        }

        // Check runtime profile filter
        if !available_runtime_profiles.is_empty() {
            if !serving.iter().any(|e| e.runtime_profile_available(available_runtime_profiles)) {
                return format!(
                    "No {} model is compatible with the available runtime profiles ({}). A model compatible with one of these profiles is needed.",
                    role.label(),
                    available_runtime_profiles.join(", ")
                );
            }
        }

        // Check license filter
        if !allowed_licenses.is_empty() {
            if !serving.iter().any(|e| e.license_allowed(allowed_licenses)) {
                return format!(
                    "No {} model has an allowed license. Allowed licenses: {}. A model with an allowed license is needed.",
                    role.label(),
                    allowed_licenses.join(", ")
                );
            }
        }

        match classification {
            Some(c) => {
                // Name the model that needs administrator review rather
                // than asking the operator to find one. The closest
                // candidate is the one that clears every other gate
                // (role, modality, floor, runtime, license) but not the
                // classification — that is the one to inspect and
                // either clear or refuse to clear on the record.
                let closest = serving
                    .iter()
                    .find(|e| {
                        e.enabled
                            && e.meets_floor(role)
                            && (required_modality.is_none()
                                || e.supports_modality(required_modality.unwrap()))
                            && (!require_structured_output || e.supports_structured_output())
                            && (available_runtime_profiles.is_empty()
                                || e.runtime_profile_available(available_runtime_profiles))
                            && (allowed_licenses.is_empty()
                                || e.license_allowed(allowed_licenses))
                    })
                    .map(|e| e.id.as_str());
                match closest {
                    Some(model_id) => format!(
                        "No {} model is cleared for {} material. Ask an administrator to \
                         review {} and add {} to its permitted classifications.",
                        role.label(),
                        c.label(),
                        model_id,
                        c.label()
                    ),
                    None => format!(
                        "No {} model is cleared for {} material. An administrator clears one, \
                         having checked it is appropriate for data of that sensitivity.",
                        role.label(),
                        c.label()
                    ),
                }
            }
            None => format!("No {} model is available.", role.label()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decide(
        entry: &ModelEntry,
        role: ModelRole,
        intent: String,
        confidence: f32,
        plan: GpuOffloadPlan,
        used_fallback: bool,
        reasons: Vec<String>,
    ) -> RoutingDecision {
        RoutingDecision {
            model_id: entry.id.clone(),
            model_name: entry.name.clone(),
            role,
            intent,
            confidence,
            used_fallback,
            reasons,
            gpu_plan_summary: plan.reason,
            fully_on_gpu: plan.full_offload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::tests::entry;
    use crate::registry::{ModelManifest, ModelRegistry};
    use std::path::PathBuf;

    const GB: u64 = 1024 * 1024 * 1024;

    fn registry(entries: Vec<ModelEntry>) -> ModelRegistry {
        ModelRegistry::from_manifest(ModelManifest { models: entries }, PathBuf::from("registry.json"))
            .unwrap()
    }

    fn stocked() -> ModelRegistry {
        registry(vec![
            entry("qwen-coder-14b", 14.0, vec![ModelRole::Coding]),
            entry("qwen-coder-7b", 7.0, vec![ModelRole::Coding]),
            entry("qwen-32b", 32.0, vec![ModelRole::Reasoning]),
            entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]),
            entry("surya", 0.65, vec![ModelRole::DocumentOcr]),
        ])
    }

    /// The problem statement's own demo: a coding request and a document summary
    /// must reach different models, each with a reason.
    #[test]
    fn the_two_demo_task_types_reach_different_models() {
        let registry = stocked();

        let coding = ModelRouter::route(
            &registry,
            "Refactor this Python function and write a unit test for the null pointer case",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        let summary = ModelRouter::route(
            &registry,
            "Summarise the key findings in this inspection report and list them by severity",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(coding.role, ModelRole::Coding);
        assert_eq!(summary.role, ModelRole::Reasoning);
        assert_ne!(coding.model_id, summary.model_id);
    }

    #[test]
    fn an_unclear_prompt_goes_to_reasoning_rather_than_a_specialist() {
        let registry = stocked();
        let decision =
            ModelRouter::route(&registry, "hello", None, 24 * GB, None, false, &[], &[]).unwrap();
        assert_eq!(decision.role, ModelRole::Reasoning);
    }

    #[test]
    fn the_largest_model_that_fits_is_preferred() {
        let registry = stocked();
        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-32b");
        assert!(!decision.used_fallback);
    }

    /// A laptop is the case the problem statement explicitly allows for.
    #[test]
    fn a_small_gpu_falls_back_and_says_so() {
        let registry = stocked();
        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-8b", "the smallest cleared candidate");
        assert!(decision.used_fallback);
    }

    #[test]
    fn routing_to_a_role_directly_skips_classification() {
        let registry = stocked();
        let decision = ModelRouter::route_for_role(
            &registry,
            ModelRole::DocumentOcr,
            None,
            8 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "surya");
    }

    #[test]
    fn an_empty_registry_explains_what_to_do() {
        let registry = registry(vec![]);
        let failure =
            ModelRouter::route(&registry, "anything", None, 24 * GB, None, false, &[], &[])
                .unwrap_err();
        assert!(failure.reason.contains("No models are registered"), "{}", failure.reason);
    }

    #[test]
    fn every_model_below_the_floor_says_so_rather_than_blaming_availability() {
        let registry = registry(vec![entry("tiny-coder", 1.5, vec![ModelRole::Coding])]);
        let failure = ModelRouter::route(
            &registry,
            "Refactor this Python function and fix the stack trace",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("too small"), "{}", failure.reason);
    }

    #[test]
    fn a_disabled_model_produces_a_disabled_explanation() {
        let mut off = entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]);
        off.enabled = false;
        let registry = registry(vec![off]);
        let failure = ModelRouter::route(
            &registry,
            "Summarise this",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("disabled"), "{}", failure.reason);
    }

    #[test]
    fn an_uncleared_classification_names_the_material() {
        let mut restricted = entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]);
        restricted.permitted_classifications = vec![Classification::Internal];
        let registry = registry(vec![restricted]);

        let failure = ModelRouter::route(
            &registry,
            "Summarise this",
            Some(Classification::VendorNegotiation),
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("Vendor negotiation"), "{}", failure.reason);
    }

    /// The contract from the prompt: when no model is cleared for the
    /// material, the refusal must (a) name the classification, (b) name
    /// the model that needs administrator review, and (c) not look
    /// like a generic "no model" message. Each of those is what a
    /// plant safety officer needs to act on the refusal in a hurry.
    #[test]
    fn refusal_names_the_classification_and_a_candidate_for_review() {
        let mut cleared = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        cleared.permitted_classifications = vec![Classification::Internal];
        let registry = registry(vec![cleared]);

        let failure = ModelRouter::route(
            &registry,
            "Summarise the vendor's offer",
            Some(Classification::VendorNegotiation),
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        let reason = &failure.reason;
        assert!(
            reason.contains("Vendor negotiation"),
            "refusal must name the classification, got: {reason}"
        );
        assert!(
            reason.contains("qwen-7b"),
            "refusal must name a model an administrator can review, got: {reason}"
        );
    }

    /// Whatever else changes, a decision must always be explainable.
    #[test]
    fn every_successful_decision_carries_its_reasons() {
        let registry = stocked();
        for prompt in [
            "Refactor this Python function",
            "Summarise the inspection findings",
            "hello",
        ] {
            let decision =
                ModelRouter::route(&registry, prompt, None, 24 * GB, None, false, &[], &[])
                    .unwrap();
            assert!(
                !decision.reasons.is_empty(),
                "no reasons recorded for {prompt:?}"
            );
        }
    }
    // ── Routing preferences: a deterministic tie-breaker ────────────────

    /// "Always use the best model" is implemented as `routing.preferred`
    /// on a model entry. Two same-size peers, one preferred, the
    /// preferred one wins. The hard gates still run first — preferred
    /// is a tie-break, not a bypass.
    #[test]
    fn the_preferred_model_wins_a_size_tie() {
        let mut preferred = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        preferred.routing.preferred = true;
        let other = entry("mistral-7b", 7.0, vec![ModelRole::Reasoning]);
        let registry = registry(vec![preferred, other]);

        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-7b");
    }

    /// `rank_within_band` is the telemetry-driven tie-breaker. Lower
    /// rank wins. The point is to let the per-model performance sink
    /// influence routing without changing the hard gates.
    #[test]
    fn a_lower_rank_within_band_wins_among_same_size_peers() {
        let mut faster = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        faster.routing.rank_within_band = 0;
        let mut slower = entry("mistral-7b", 7.0, vec![ModelRole::Reasoning]);
        slower.routing.rank_within_band = 5;
        let registry = registry(vec![slower, faster]);

        let decision = ModelRouter::route(
            &registry,
            "Summarise the inspection findings",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-7b");
    }

    /// Preferred is *only* a tie-breaker. A larger model that is not
    /// preferred still wins over a smaller preferred one, because size
    /// is the dominant sort key.
    #[test]
    fn preferred_does_not_bypass_the_size_band() {
        let mut small_preferred = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        small_preferred.routing.preferred = true;
        let big = entry("qwen-14b", 14.0, vec![ModelRole::Reasoning]);
        let registry = registry(vec![small_preferred, big]);

        let decision = ModelRouter::route(
            &registry,
            "Summarise the inspection findings",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            decision.model_id, "qwen-14b",
            "size dominates; preferred only breaks ties"
        );
    }}

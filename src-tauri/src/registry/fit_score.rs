//! Explainable fit score for model selection.
//!
//! PS 26117 asks that a router *explain* why it chose a model, and that an
//! uncertain router fall back rather than picking badly. This module turns
//! measured local certification data into a score with named components, so
//! the decision is reproducible and auditable.
//!
//! The score is the sum of weighted components, each derived from a real
//! measurement rather than a guess:
//!
//! - **Capability match** (0.0–1.0): does the model serve the role? Does it
//!   support the required modality?
//! - **Size fit** (0.0–1.0): how well does the model fit in the available
//!   VRAM/RAM? Penalised for partial offload.
//! - **License compliance** (0.0–1.0): is the model's license in the allowed
//!   list? 0.0 if not.
//! - **Hash verification** (0.0–1.0): has the model's hash been verified?
//!   0.0 if unverified.
//! - **Classification match** (0.0–1.0): is the model cleared for this
//!   material? 0.0 if not.
//! - **Context sufficiency** (0.0–1.0): is the model's context window
//!   sufficient for the task?
//! - **Structured output** (0.0–1.0): does the model support structured
//!   output if required?
//! - **Runtime profile** (0.0–1.0): is the required runtime profile
//!   available?
//!
//! Each component is recorded with its value and a human-readable reason.
//! The total score is the weighted sum, and a model below a threshold is
//! considered unfit.
//!
//! No invented benchmark numbers. All values come from real measurements or
//! honest zeros.

use serde::{Deserialize, Serialize};

use super::{ModelEntry, Modality};
use crate::policy::Classification;

/// A single component of the fit score.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitComponent {
    /// Component name, e.g. "capability_match", "size_fit".
    pub name: String,
    /// Score value, 0.0–1.0.
    pub value: f32,
    /// Weight applied to this component.
    pub weight: f32,
    /// Human-readable reason for the score.
    pub reason: String,
}

/// The complete fit score for a model against a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitScore {
    pub model_id: String,
    pub model_name: String,
    /// Weighted total score, 0.0–1.0.
    pub total: f32,
    /// Individual components.
    pub components: Vec<FitComponent>,
    /// Why this model was rejected, if it was. Empty if it was accepted.
    pub rejection_reasons: Vec<String>,
    /// True if the model passes all hard gates.
    pub passes_hard_gates: bool,
}

/// What a task needs from a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRequirements {
    pub role: super::ModelRole,
    /// Required modality (text, image, etc.). None means text is acceptable.
    pub required_modality: Option<Modality>,
    /// Whether the task requires structured output / tool calling.
    pub require_structured_output: bool,
    /// Required context length in tokens. 0 means no minimum.
    pub required_context_length: u32,
    /// Classification of the material being processed.
    pub classification: Option<Classification>,
    /// Available VRAM in bytes.
    pub available_vram_bytes: u64,
    /// Total available RAM in bytes (for CPU offload).
    pub available_ram_bytes: u64,
    /// Available runtime profiles on this machine.
    pub available_runtime_profiles: Vec<String>,
    /// Licenses allowed by policy.
    pub allowed_licenses: Vec<String>,
}

/// Default weights for fit score components.
mod weights {
    pub const CAPABILITY_MATCH: f32 = 2.0;
    pub const SIZE_FIT: f32 = 1.5;
    pub const LICENSE: f32 = 2.0;
    pub const HASH_VERIFIED: f32 = 1.0;
    pub const CLASSIFICATION: f32 = 2.0;
    pub const CONTEXT: f32 = 1.0;
    pub const STRUCTURED_OUTPUT: f32 = 1.0;
    pub const RUNTIME_PROFILE: f32 = 1.0;
}

/// Minimum total score for a model to be considered fit.
pub const FIT_THRESHOLD: f32 = 0.5;

pub struct FitScorer;

impl FitScorer {
    /// Computes the fit score for a model against a task.
    pub fn score(entry: &ModelEntry, requirements: &TaskRequirements) -> FitScore {
        let mut components = Vec::new();
        let mut rejection_reasons = Vec::new();
        let mut passes_hard_gates = true;

        // 1. Capability match
        let serves_role = entry.serves(requirements.role);
        let supports_modality = requirements
            .required_modality
            .map(|m| entry.supports_modality(m))
            .unwrap_or(true);
        let capability_value = if serves_role && supports_modality {
            1.0
        } else {
            0.0
        };
        let capability_reason = if !serves_role {
            format!(
                "Model does not serve the {} role",
                requirements.role.label()
            )
        } else if !supports_modality {
            format!(
                "Model does not support the required modality ({})",
                requirements
                    .required_modality
                    .map(|m| m.label())
                    .unwrap_or("text")
            )
        } else {
            format!(
                "Model serves the {} role and supports the required modality",
                requirements.role.label()
            )
        };
        if capability_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(capability_reason.clone());
        }
        components.push(FitComponent {
            name: "capability_match".to_string(),
            value: capability_value,
            weight: weights::CAPABILITY_MATCH,
            reason: capability_reason,
        });

        // 2. Size fit
        let size_value = Self::compute_size_fit(entry, requirements);
        let size_reason = Self::explain_size_fit(entry, requirements);
        components.push(FitComponent {
            name: "size_fit".to_string(),
            value: size_value,
            weight: weights::SIZE_FIT,
            reason: size_reason,
        });

        // 3. License compliance
        let license_value = if requirements.allowed_licenses.is_empty() {
            1.0
        } else if entry.license_allowed(&requirements.allowed_licenses) {
            1.0
        } else {
            0.0
        };
        let license_reason = if requirements.allowed_licenses.is_empty() {
            "No license restrictions configured".to_string()
        } else if license_value == 1.0 {
            format!("License '{}' is in the allowed list", entry.license)
        } else {
            format!(
                "License '{}' is not in the allowed list ({})",
                entry.license,
                requirements.allowed_licenses.join(", ")
            )
        };
        if license_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(license_reason.clone());
        }
        components.push(FitComponent {
            name: "license_compliance".to_string(),
            value: license_value,
            weight: weights::LICENSE,
            reason: license_reason,
        });

        // 4. Hash verification
        let hash_value = if entry.hash_verified() { 1.0 } else { 0.0 };
        let hash_reason = if entry.hash_verified() {
            format!("Hash verified: {}", entry.sha256.as_deref().unwrap_or(""))
        } else {
            "Hash not verified".to_string()
        };
        if hash_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push("Hash not verified".to_string());
        }
        components.push(FitComponent {
            name: "hash_verified".to_string(),
            value: hash_value,
            weight: weights::HASH_VERIFIED,
            reason: hash_reason,
        });

        // 5. Classification match
        let class_value = match requirements.classification {
            Some(c) if entry.permits(c) => 1.0,
            Some(_) => 0.0,
            None => 1.0,
        };
        let class_reason = match requirements.classification {
            Some(c) if entry.permits(c) => {
                format!("Model is cleared for {} material", c.label())
            }
            Some(c) => {
                format!("Model is not cleared for {} material", c.label())
            }
            None => "No classification requirement".to_string(),
        };
        if class_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(class_reason.clone());
        }
        components.push(FitComponent {
            name: "classification_match".to_string(),
            value: class_value,
            weight: weights::CLASSIFICATION,
            reason: class_reason,
        });

        // 6. Context sufficiency
        let context_value = if requirements.required_context_length == 0 {
            1.0
        } else if entry.context_length >= requirements.required_context_length {
            1.0
        } else {
            0.0
        };
        let context_reason = if requirements.required_context_length == 0 {
            "No context length requirement".to_string()
        } else if context_value == 1.0 {
            format!(
                "Context window {} >= required {}",
                entry.context_length, requirements.required_context_length
            )
        } else {
            format!(
                "Context window {} < required {}",
                entry.context_length, requirements.required_context_length
            )
        };
        if context_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(context_reason.clone());
        }
        components.push(FitComponent {
            name: "context_sufficiency".to_string(),
            value: context_value,
            weight: weights::CONTEXT,
            reason: context_reason,
        });

        // 7. Structured output
        let structured_value = if requirements.require_structured_output {
            if entry.supports_structured_output() {
                1.0
            } else {
                0.0
            }
        } else {
            1.0
        };
        let structured_reason = if requirements.require_structured_output {
            if entry.supports_structured_output() {
                "Model supports structured output".to_string()
            } else {
                "Model does not support structured output".to_string()
            }
        } else {
            "Structured output not required".to_string()
        };
        if structured_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(structured_reason.clone());
        }
        components.push(FitComponent {
            name: "structured_output".to_string(),
            value: structured_value,
            weight: weights::STRUCTURED_OUTPUT,
            reason: structured_reason,
        });

        // 8. Runtime profile
        let runtime_value = if entry.runtime_profile_available(&requirements.available_runtime_profiles) {
            1.0
        } else {
            0.0
        };
        let runtime_reason = if entry.runtime_profile_available(&requirements.available_runtime_profiles) {
            if let Some(profile) = &entry.required_runtime_profile {
                format!("Runtime profile '{}' is available", profile)
            } else {
                "No specific runtime profile required".to_string()
            }
        } else {
            format!(
                "Required runtime profile not available. Available: [{}]",
                requirements.available_runtime_profiles.join(", ")
            )
        };
        if runtime_value == 0.0 {
            passes_hard_gates = false;
            rejection_reasons.push(runtime_reason.clone());
        }
        components.push(FitComponent {
            name: "runtime_profile".to_string(),
            value: runtime_value,
            weight: weights::RUNTIME_PROFILE,
            reason: runtime_reason,
        });

        // Compute weighted total
        let total_weight: f32 = components.iter().map(|c| c.weight).sum();
        let weighted_sum: f32 = components.iter().map(|c| c.value * c.weight).sum();
        let total = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        FitScore {
            model_id: entry.id.clone(),
            model_name: entry.name.clone(),
            total,
            components,
            rejection_reasons,
            passes_hard_gates,
        }
    }

    /// Computes how well the model fits in available memory.
    fn compute_size_fit(entry: &ModelEntry, requirements: &TaskRequirements) -> f32 {
        let weights = entry.weights_bytes;
        let vram = requirements.available_vram_bytes;
        let ram = requirements.available_ram_bytes;

        if weights == 0 {
            return 0.0;
        }

        if weights <= vram {
            // Full GPU offload — best case.
            1.0
        } else if weights <= vram + ram {
            // Partial offload possible — score based on how much fits on GPU.
            let gpu_fraction = vram as f32 / weights as f32;
            0.5 + 0.5 * gpu_fraction
        } else {
            // Doesn't fit even with RAM offload.
            0.0
        }
    }

    fn explain_size_fit(entry: &ModelEntry, requirements: &TaskRequirements) -> String {
        let weights = entry.weights_bytes;
        let vram = requirements.available_vram_bytes;
        let ram = requirements.available_ram_bytes;

        if weights == 0 {
            return "Model size unknown".to_string();
        }

        if weights <= vram {
            format!(
                "Model ({} MB) fits entirely in VRAM ({} MB)",
                weights / 1024 / 1024,
                vram / 1024 / 1024
            )
        } else if weights <= vram + ram {
            let gpu_fraction = vram as f32 / weights as f32;
            format!(
                "Model ({} MB) partially fits in VRAM ({} MB, {:.0}% on GPU), rest in RAM",
                weights / 1024 / 1024,
                vram / 1024 / 1024,
                gpu_fraction * 100.0
            )
        } else {
            format!(
                "Model ({} MB) does not fit in available memory (VRAM: {} MB, RAM: {} MB)",
                weights / 1024 / 1024,
                vram / 1024 / 1024,
                ram / 1024 / 1024
            )
        }
    }

    /// Scores all candidates and returns them sorted by fit score (best first).
    pub fn rank_candidates(
        candidates: &[&ModelEntry],
        requirements: &TaskRequirements,
    ) -> Vec<FitScore> {
        let mut scores: Vec<FitScore> = candidates
            .iter()
            .map(|e| Self::score(e, requirements))
            .collect();
        scores.sort_by(|a, b| {
            b.total
                .partial_cmp(&a.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{LoadSpec, ModelEntry, ModelRole, RoutingPreference, Runtime};
    use std::path::PathBuf;

    fn test_entry(id: &str, params: f32, roles: Vec<ModelRole>) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            name: id.into(),
            version: "1".into(),
            license: "apache-2.0".into(),
            sha256: Some("abc123".to_string()),
            runtime: Runtime::LlamaCpp,
            roles,
            modalities: vec![Modality::Text],
            quantization: Some("Q4_K_M".into()),
            parameters_b: params,
            active_parameters_b: None,
            context_length: 32_768,
            weights_bytes: (params * 0.6 * 1e9) as u64,
            supports_structured_output: true,
            permitted_classifications: Classification::ALL.to_vec(),
            path: PathBuf::from(format!("{id}.gguf")),
            load: Some(LoadSpec {
                provider_id: "huggingface".into(),
                model_id: id.into(),
                quantization: "Q4_K_M".into(),
            }),
            serving: None,
            required_runtime_profile: None,
            enabled: true,
        routing: RoutingPreference::default(),
        }
    }

    fn text_requirements() -> TaskRequirements {
        TaskRequirements {
            role: ModelRole::Reasoning,
            required_modality: None,
            require_structured_output: false,
            required_context_length: 0,
            classification: None,
            available_vram_bytes: 8 * 1024 * 1024 * 1024,
            available_ram_bytes: 32 * 1024 * 1024 * 1024,
            available_runtime_profiles: vec![],
            allowed_licenses: vec![],
        }
    }

    #[test]
    fn a_text_task_and_a_coding_task_choose_different_eligible_models() {
        // Both models are sized to fit in the test's VRAM
        let coder = test_entry("qwen-coder-7b", 7.0, vec![ModelRole::Coding]);
        let reasoner = test_entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);

        let text_score = FitScorer::score(
            &reasoner,
            &TaskRequirements {
                role: ModelRole::Reasoning,
                ..text_requirements()
            },
        );
        let coding_score = FitScorer::score(
            &coder,
            &TaskRequirements {
                role: ModelRole::Coding,
                ..text_requirements()
            },
        );

        // Both models should pass hard gates for their respective roles
        assert!(text_score.passes_hard_gates);
        assert!(coding_score.passes_hard_gates);

        // Coding task should prefer the coding model over the reasoning model
        let coding_score_for_coding = FitScorer::score(
            &coder,
            &TaskRequirements {
                role: ModelRole::Coding,
                ..text_requirements()
            },
        );
        let text_score_for_coding = FitScorer::score(
            &reasoner,
            &TaskRequirements {
                role: ModelRole::Coding,
                ..text_requirements()
            },
        );
        // The coding model serves the coding role (capability=1.0),
        // while the reasoning model doesn't (capability=0.0)
        assert!(coding_score_for_coding.total > text_score_for_coding.total);
    }

    #[test]
    fn an_image_task_refuses_a_text_only_model() {
        let text_only = test_entry("text-7b", 7.0, vec![ModelRole::Reasoning]);
        let mut vision_model = test_entry("vision-7b", 7.0, vec![ModelRole::Reasoning]);
        vision_model.modalities = vec![Modality::Text, Modality::Image];

        let requirements = TaskRequirements {
            role: ModelRole::Reasoning,
            required_modality: Some(Modality::Image),
            ..text_requirements()
        };

        let text_score = FitScorer::score(&text_only, &requirements);
        let vision_score = FitScorer::score(&vision_model, &requirements);

        assert!(!text_score.passes_hard_gates);
        assert!(vision_score.passes_hard_gates);
        assert!(vision_score.total > text_score.total);
    }

    #[test]
    fn model_hash_mismatch_is_rejected() {
        let mut entry = test_entry("unverified", 7.0, vec![ModelRole::Reasoning]);
        entry.sha256 = None;

        let score = FitScorer::score(&entry, &text_requirements());
        assert!(!score.passes_hard_gates);
        assert!(score
            .rejection_reasons
            .iter()
            .any(|r| r.contains("Hash not verified")));
    }

    #[test]
    fn license_mismatch_is_rejected() {
        let mut entry = test_entry("gpl-model", 7.0, vec![ModelRole::Reasoning]);
        entry.license = "gpl-3.0".to_string();

        let requirements = TaskRequirements {
            allowed_licenses: vec!["apache-2.0".to_string()],
            ..text_requirements()
        };

        let score = FitScorer::score(&entry, &requirements);
        assert!(!score.passes_hard_gates);
        assert!(score
            .rejection_reasons
            .iter()
            .any(|r| r.contains("not in the allowed list")));
    }

    #[test]
    fn low_vram_causes_partial_fit_or_refusal() {
        // 7B model with weights_bytes = 7 * 0.6 * 1e9 = 4.2 GB
        let medium_model = test_entry("medium-7b", 7.0, vec![ModelRole::Reasoning]);

        // Low VRAM but enough RAM for partial fit
        let requirements = TaskRequirements {
            available_vram_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            available_ram_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB
            ..text_requirements()
        };

        let score = FitScorer::score(&medium_model, &requirements);
        // Should still pass hard gates but have lower size_fit
        let size_component = score
            .components
            .iter()
            .find(|c| c.name == "size_fit")
            .unwrap();
        assert!(size_component.value < 1.0);
        assert!(size_component.value > 0.0);
    }

    #[test]
    fn model_too_large_for_any_memory_fails_size_fit() {
        let huge_model = test_entry("huge-100b", 100.0, vec![ModelRole::Reasoning]);

        let requirements = TaskRequirements {
            available_vram_bytes: 1024 * 1024 * 1024, // 1 GB
            available_ram_bytes: 1024 * 1024 * 1024,  // 1 GB
            ..text_requirements()
        };

        let score = FitScorer::score(&huge_model, &requirements);
        let size_component = score
            .components
            .iter()
            .find(|c| c.name == "size_fit")
            .unwrap();
        assert_eq!(size_component.value, 0.0);
    }

    #[test]
    fn context_sufficiency_is_checked() {
        let mut small_context = test_entry("small-ctx", 7.0, vec![ModelRole::Reasoning]);
        small_context.context_length = 2048;

        let requirements = TaskRequirements {
            required_context_length: 8192,
            ..text_requirements()
        };

        let score = FitScorer::score(&small_context, &requirements);
        assert!(!score.passes_hard_gates);
        assert!(score
            .rejection_reasons
            .iter()
            .any(|r| r.contains("Context window")));
    }

    #[test]
    fn structured_output_requirement_is_checked() {
        let mut no_structured = test_entry("no-structured", 7.0, vec![ModelRole::Reasoning]);
        no_structured.supports_structured_output = false;

        let requirements = TaskRequirements {
            require_structured_output: true,
            ..text_requirements()
        };

        let score = FitScorer::score(&no_structured, &requirements);
        assert!(!score.passes_hard_gates);
    }

    #[test]
    fn runtime_profile_mismatch_is_rejected() {
        let mut cuda_only = test_entry("cuda-only", 7.0, vec![ModelRole::Reasoning]);
        cuda_only.required_runtime_profile = Some("cuda".to_string());

        let requirements = TaskRequirements {
            available_runtime_profiles: vec!["vulkan".to_string()],
            ..text_requirements()
        };

        let score = FitScorer::score(&cuda_only, &requirements);
        assert!(!score.passes_hard_gates);
    }

    #[test]
    fn ranking_sorts_by_total_score() {
        let small = test_entry("small-3b", 3.0, vec![ModelRole::Reasoning]);
        let medium = test_entry("medium-7b", 7.0, vec![ModelRole::Reasoning]);
        let large = test_entry("large-14b", 14.0, vec![ModelRole::Reasoning]);
        let candidates: Vec<&ModelEntry> = vec![&small, &medium, &large];

        let scores = FitScorer::rank_candidates(&candidates, &text_requirements());

        // Scores should be in descending order
        for i in 1..scores.len() {
            assert!(scores[i - 1].total >= scores[i].total);
        }
    }
}

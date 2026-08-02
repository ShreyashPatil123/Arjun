//! Saarthi Data-Driven Certified Model Catalog & Decoupled Metadata Specs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CertificationTier {
    Certified,
    Compatible,
    Experimental,
}

impl CertificationTier {
    pub fn badge_name(&self) -> &str {
        match self {
            Self::Certified => "⭐⭐⭐⭐⭐ Saarthi Certified Package",
            Self::Compatible => "⭐⭐⭐⭐ Compatible Package",
            Self::Experimental => "⚠️ Experimental Package",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericScores {
    pub instruction_following: f64,
    pub reasoning_quality: f64,
    pub hallucination_rate: f64,
    pub coding_ability: f64,
    pub mathematical_reasoning: f64,
    pub json_reliability: f64,
    pub tool_calling_accuracy: f64,
    pub memory_engine_compatibility: f64,
    pub lora_adapter_switching: f64,
    pub context_window_retention: f64,
    pub response_stability: f64,
    pub chat_template_correctness: f64,
    pub bos_eos_stop_token_compliance: f64,
    pub reasoning_tag_leakage_filter: f64,
    pub streaming_parser_stability: f64,
    pub runtime_process_stability: f64,
    pub restart_state_persistence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub created_by: String,
    pub certified_by: String,
    pub generated_with: String,
    pub runner_version: String,
    pub profile_hash: String,
    pub signature: String,
    pub generated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageCertification {
    pub package_id: String,
    pub model_id: String,
    pub model_name: String,
    pub quant_label: String,
    pub backend: String,
    pub tier: CertificationTier,
    pub confidence_score: f64,
    pub runtime_profile_id: String,
    pub numeric_scores: NumericScores,
    pub lora_capability_matrix: HashMap<String, CertificationTier>,
    pub provenance: Provenance,
    pub quirks_and_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedVersions {
    pub sarathi_version: String,
    pub profile_schema_version: String,
    pub llamacpp_version: String,
    pub llamacpp2_rust_version: String,
    pub certification_spec_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    pub freq_base: f32,
    pub freq_scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub chat_template: String,
    pub stop_tokens: Vec<String>,
    pub context_length: u32,
    pub gpu_layers: u32,
    pub threads: u32,
    pub sampling_defaults: SamplingDefaults,
    pub rope_config: Option<RopeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub profile_id: String,
    pub name: String,
    pub pinned_versions: PinnedVersions,
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationPack {
    pub pack_id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub generated_at: String,
    pub certified_packages: Vec<PackageCertification>,
}

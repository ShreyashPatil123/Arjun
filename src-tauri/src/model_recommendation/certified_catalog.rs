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
#[serde(rename_all = "camelCase")]
pub struct NumericScores {
    #[serde(alias = "instruction_following")]
    pub instruction_following: f64,
    #[serde(alias = "reasoning_quality")]
    pub reasoning_quality: f64,
    #[serde(alias = "hallucination_rate")]
    pub hallucination_rate: f64,
    #[serde(alias = "coding_ability")]
    pub coding_ability: f64,
    #[serde(alias = "mathematical_reasoning")]
    pub mathematical_reasoning: f64,
    #[serde(alias = "json_reliability")]
    pub json_reliability: f64,
    #[serde(alias = "tool_calling_accuracy")]
    pub tool_calling_accuracy: f64,
    #[serde(alias = "memory_engine_compatibility")]
    pub memory_engine_compatibility: f64,
    #[serde(alias = "lora_adapter_switching")]
    pub lora_adapter_switching: f64,
    #[serde(alias = "context_window_retention")]
    pub context_window_retention: f64,
    #[serde(alias = "response_stability")]
    pub response_stability: f64,
    #[serde(alias = "chat_template_correctness")]
    pub chat_template_correctness: f64,
    #[serde(alias = "bos_eos_stop_token_compliance")]
    pub bos_eos_stop_token_compliance: f64,
    #[serde(alias = "reasoning_tag_leakage_filter")]
    pub reasoning_tag_leakage_filter: f64,
    #[serde(alias = "streaming_parser_stability")]
    pub streaming_parser_stability: f64,
    #[serde(alias = "runtime_process_stability")]
    pub runtime_process_stability: f64,
    #[serde(alias = "restart_state_persistence")]
    pub restart_state_persistence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    #[serde(alias = "created_by")]
    pub created_by: String,
    #[serde(alias = "certified_by")]
    pub certified_by: String,
    #[serde(alias = "generated_with")]
    pub generated_with: String,
    #[serde(alias = "runner_version")]
    pub runner_version: String,
    #[serde(alias = "profile_hash")]
    pub profile_hash: String,
    pub signature: String,
    #[serde(alias = "generated_at")]
    pub generated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageCertification {
    #[serde(alias = "package_id")]
    pub package_id: String,
    #[serde(alias = "model_id")]
    pub model_id: String,
    #[serde(alias = "model_name")]
    pub model_name: String,
    #[serde(alias = "quant_label")]
    pub quant_label: String,
    pub backend: String,
    pub tier: CertificationTier,
    #[serde(alias = "confidence_score")]
    pub confidence_score: f64,
    #[serde(alias = "runtime_profile_id")]
    pub runtime_profile_id: String,
    #[serde(alias = "numeric_scores")]
    pub numeric_scores: NumericScores,
    #[serde(alias = "lora_capability_matrix")]
    pub lora_capability_matrix: HashMap<String, CertificationTier>,
    pub provenance: Provenance,
    #[serde(alias = "quirks_and_notes")]
    pub quirks_and_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PinnedVersions {
    #[serde(alias = "sarathi_version")]
    pub sarathi_version: String,
    #[serde(alias = "profile_schema_version")]
    pub profile_schema_version: String,
    #[serde(alias = "llamacpp_version")]
    pub llamacpp_version: String,
    #[serde(alias = "llamacpp2_rust_version")]
    pub llamacpp2_rust_version: String,
    #[serde(alias = "certification_spec_version")]
    pub certification_spec_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingDefaults {
    pub temperature: f32,
    #[serde(alias = "top_p")]
    pub top_p: f32,
    #[serde(alias = "top_k")]
    pub top_k: u32,
    #[serde(alias = "min_p")]
    pub min_p: f32,
    #[serde(alias = "repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(alias = "max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RopeConfig {
    #[serde(alias = "freq_base")]
    pub freq_base: f32,
    #[serde(alias = "freq_scale")]
    pub freq_scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionConfig {
    #[serde(alias = "chat_template")]
    pub chat_template: String,
    #[serde(alias = "stop_tokens")]
    pub stop_tokens: Vec<String>,
    #[serde(alias = "context_length")]
    pub context_length: u32,
    #[serde(alias = "gpu_layers")]
    pub gpu_layers: u32,
    pub threads: u32,
    #[serde(alias = "sampling_defaults")]
    pub sampling_defaults: SamplingDefaults,
    #[serde(alias = "rope_config")]
    pub rope_config: Option<RopeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeProfile {
    #[serde(alias = "profile_id")]
    pub profile_id: String,
    pub name: String,
    #[serde(alias = "pinned_versions")]
    pub pinned_versions: PinnedVersions,
    #[serde(alias = "execution_config")]
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CertificationPack {
    #[serde(alias = "pack_id")]
    pub pack_id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    #[serde(alias = "generated_at")]
    pub generated_at: String,
    #[serde(alias = "certified_packages")]
    pub certified_packages: Vec<PackageCertification>,
}

//! Model Intelligence Profile Data Structures
//!
//! Provides versioned, source-driven profile models for local LLMs,
//! dynamic capability registration, and runtime inference settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current schema version for Model Profile
pub const CURRENT_PROFILE_VERSION: u32 = 1;

/// Supported LLM Model Families
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Llama,
    Qwen,
    Gemma,
    Mistral,
    Mixtral,
    Phi,
    DeepSeek,
    CommandR,
    Starcoder,
    GLM,
    Yi,
    Baichuan,
    Falcon,
    Granite,
    InternLM,
    SmolLM,
    TinyLlama,
    StableLM,
    OpenChat,
    CodeLlama,
    Generic,
}

impl Default for ModelFamily {
    fn default() -> Self {
        Self::Llama
    }
}

/// Token configuration and stop sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenConfig {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub stop_tokens: Vec<String>,
    pub pad_token: Option<String>,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            bos_token: Some("<|begin_of_text|>".to_string()),
            eos_token: Some("<|eot_id|>".to_string()),
            stop_tokens: vec![
                "<|eot_id|>".to_string(),
                "<|end_of_text|>".to_string(),
                "</s>".to_string(),
                "<|im_end|>".to_string(),
            ],
            pad_token: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceParameters {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    pub repeat_penalty: f32,
    #[serde(default = "default_zero")]
    pub mirostat: u32,
    pub max_tokens: u32,
    pub context_length: u32,
    pub threads: u32,
    pub gpu_layers: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_true")]
    pub flash_attn: bool,
}

fn default_min_p() -> f32 { 0.05 }
fn default_zero() -> u32 { 0 }
fn default_batch_size() -> u32 { 512 }
fn default_true() -> bool { true }

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            mirostat: 0,
            max_tokens: 2048,
            context_length: 4096,
            threads: 4,
            gpu_layers: 999,
            batch_size: 512,
            flash_attn: true,
        }
    }
}

/// Dynamic Capability Metadata Item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityItem {
    pub capability: String,
    pub supported: bool,
    pub confidence: f32, // 0.0 to 1.0 score
    pub description: String,
}

/// Extensible Dynamic Capability Registry
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityRegistry {
    pub capabilities: HashMap<String, CapabilityItem>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            capabilities: HashMap::new(),
        };
        // Register core capabilities by default
        registry.set_capability("coding", true, 1.0, "Code generation, debugging, and refactoring");
        registry.set_capability("reasoning", true, 0.9, "Multi-step logical reasoning and analysis");
        registry.set_capability("mathematics", true, 0.9, "Mathematical calculation and problem solving");
        registry.set_capability("tool_calling", true, 0.85, "Structured function and tool execution");
        registry.set_capability("research", true, 0.95, "Information synthesis and document Q&A");
        registry.set_capability("vision", false, 0.0, "Multimodal image understanding");
        registry.set_capability("audio", false, 0.0, "Speech and audio processing");
        registry.set_capability("embeddings", true, 0.8, "Semantic vector embeddings");
        registry.set_capability("rag", true, 0.9, "Retrieval-augmented generation");
        registry.set_capability("agents", true, 0.85, "Autonomous multi-step agent workflows");
        registry
    }

    pub fn set_capability(&mut self, cap: &str, supported: bool, confidence: f32, desc: &str) {
        self.capabilities.insert(
            cap.to_string(),
            CapabilityItem {
                capability: cap.to_string(),
                supported,
                confidence,
                description: desc.to_string(),
            },
        );
    }

    pub fn is_supported(&self, cap: &str) -> bool {
        self.capabilities.get(cap).map(|c| c.supported).unwrap_or(false)
    }
}

/// Metadata extraction provenance tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataProvenance {
    pub gguf_metadata_extracted: bool,
    pub generation_config_extracted: bool,
    pub tokenizer_config_extracted: bool,
    pub tokenizer_json_extracted: bool,
    pub config_extracted: bool,
    pub model_card_extracted: bool,
    pub source_summary: String,
}

impl Default for MetadataProvenance {
    fn default() -> Self {
        Self {
            gguf_metadata_extracted: false,
            generation_config_extracted: false,
            tokenizer_config_extracted: false,
            tokenizer_json_extracted: false,
            config_extracted: false,
            model_card_extracted: false,
            source_summary: "Generated from GGUF headers and default profiles".to_string(),
        }
    }
}

/// Complete Versioned Local Model Profile (`profile.json`)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfile {
    pub profile_version: u32,
    pub package_id: String,
    pub model_id: String,
    pub model_name: String,
    pub model_family: ModelFamily,
    pub architecture: String,
    pub chat_template: String,
    pub system_prompt_format: String,
    pub tokens: TokenConfig,
    pub recommended_params: InferenceParameters,
    pub active_user_params: Option<InferenceParameters>,
    pub capability_registry: CapabilityRegistry,
    pub provenance: MetadataProvenance,
    pub created_at: String,
    pub updated_at: String,
}

impl ModelProfile {
    pub fn new(package_id: &str, model_id: &str, model_name: &str) -> Self {
        Self {
            profile_version: CURRENT_PROFILE_VERSION,
            package_id: package_id.to_string(),
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            model_family: ModelFamily::Llama,
            architecture: "llama".to_string(),
            chat_template: String::new(),
            system_prompt_format: "You are a helpful, respectful, and honest assistant.".to_string(),
            tokens: TokenConfig::default(),
            recommended_params: InferenceParameters::default(),
            active_user_params: None,
            capability_registry: CapabilityRegistry::new(),
            provenance: MetadataProvenance::default(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Migrate profile if version is older than CURRENT_PROFILE_VERSION
    pub fn migrate_if_needed(&mut self) -> bool {
        if self.profile_version < CURRENT_PROFILE_VERSION {
            log::info!(
                "[PROFILE] Migrating profile '{}' from v{} to v{}",
                self.model_id, self.profile_version, CURRENT_PROFILE_VERSION
            );
            self.profile_version = CURRENT_PROFILE_VERSION;
            self.updated_at = chrono::Utc::now().to_rfc3339();
            true
        } else {
            false
        }
    }

    /// Returns active effective inference parameters (user override or recommended)
    pub fn effective_params(&self) -> &InferenceParameters {
        self.active_user_params.as_ref().unwrap_or(&self.recommended_params)
    }
}

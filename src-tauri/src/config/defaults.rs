//! Default configuration definitions

use serde::{Deserialize, Serialize};

pub const DEFAULT_ORCHESTRATOR_MODEL_ID: &str =
    "lmstudio-community/gemma-4-12B-it-QAT-GGUF";
pub const DEFAULT_ORCHESTRATOR_PROVIDER_ID: &str = "huggingface";
pub const DEFAULT_ORCHESTRATOR_QUANTIZATION: &str = "Q4_0";

/// Main configuration for Sarathi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarathiConfig {
    pub theme: String,
    pub language: String,
    pub backend_url: String,
    pub ollama_url: String,
    pub model_directory: String,
    pub download_directory: String,
    pub cache_directory: String,
    pub log_level: String,
    pub ai_settings: AiSettings,
    /// HuggingFace access token. Raises the Hub's anonymous rate limit, which
    /// is what caps the catalog to a small popular slice, and unlocks gated
    /// repositories such as `meta-llama/*` and `google/gemma-*`.
    ///
    /// Stored in this file as plain text — the same as the `HF_TOKEN` variable
    /// it substitutes for. Empty or absent means "use the environment".
    #[serde(default)]
    pub hf_token: String,
}

/// AI-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSettings {
    pub max_context_length: u32,
    pub default_temperature: f32,
    pub use_gpu: bool,
    pub gpu_layers: u32,

    /// Exact installed package coordinates selected by an administrator for
    /// the agent orchestrator at startup.
    #[serde(default = "default_orchestrator_provider_id")]
    pub orchestrator_provider_id: String,
    #[serde(default = "default_orchestrator_model_id")]
    pub orchestrator_model_id: String,
    #[serde(default = "default_orchestrator_quantization")]
    pub orchestrator_quantization: String,

    /// Whether startup may load a model without being asked.
    ///
    /// Enabled by default so the orchestrator is ready without a manual load.
    /// Existing configurations that explicitly set this to false stay disabled.
    #[serde(default = "default_true")]
    pub auto_load_on_startup: bool,
}

fn default_orchestrator_model_id() -> String {
    DEFAULT_ORCHESTRATOR_MODEL_ID.to_string()
}

fn default_orchestrator_provider_id() -> String {
    DEFAULT_ORCHESTRATOR_PROVIDER_ID.to_string()
}

fn default_orchestrator_quantization() -> String {
    DEFAULT_ORCHESTRATOR_QUANTIZATION.to_string()
}

fn default_true() -> bool {
    true
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            default_temperature: 0.7,
            use_gpu: true,
            gpu_layers: 35,
            orchestrator_provider_id: default_orchestrator_provider_id(),
            orchestrator_model_id: default_orchestrator_model_id(),
            orchestrator_quantization: default_orchestrator_quantization(),
            auto_load_on_startup: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_orchestrator_auto_loads_on_gpu_by_default() {
        let settings = AiSettings::default();
        assert_eq!(settings.orchestrator_provider_id, DEFAULT_ORCHESTRATOR_PROVIDER_ID);
        assert_eq!(settings.orchestrator_model_id, DEFAULT_ORCHESTRATOR_MODEL_ID);
        assert_eq!(settings.orchestrator_quantization, DEFAULT_ORCHESTRATOR_QUANTIZATION);
        assert!(settings.auto_load_on_startup);
        assert!(settings.use_gpu);
        assert!(settings.gpu_layers > 0);
    }

    #[test]
    fn older_config_without_startup_fields_inherits_the_new_defaults() {
        let settings: AiSettings = serde_json::from_str(
            r#"{
                "max_context_length": 4096,
                "default_temperature": 0.7,
                "use_gpu": true,
                "gpu_layers": 35
            }"#,
        )
        .expect("legacy AI settings should still deserialize");

        assert_eq!(settings.orchestrator_model_id, DEFAULT_ORCHESTRATOR_MODEL_ID);
        assert_eq!(settings.orchestrator_provider_id, DEFAULT_ORCHESTRATOR_PROVIDER_ID);
        assert_eq!(settings.orchestrator_quantization, DEFAULT_ORCHESTRATOR_QUANTIZATION);
        assert!(settings.auto_load_on_startup);
    }
}

impl Default for SarathiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            language: "en".to_string(),
            backend_url: "http://localhost:8000".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            model_directory: "models".to_string(),
            download_directory: "downloads".to_string(),
            cache_directory: "cache".to_string(),
            log_level: "info".to_string(),
            ai_settings: AiSettings::default(),
            hf_token: String::new(),
        }
    }
}

//! Default configuration definitions

use serde::{Deserialize, Serialize};

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
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            default_temperature: 0.7,
            use_gpu: true,
            gpu_layers: 35,
        }
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

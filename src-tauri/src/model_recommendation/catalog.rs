//! Phase 3: Provider-Independent Model Catalog
//!
//! Provides model metadata for recommendation calculations.
//! Phase 3 uses a versioned bootstrap catalog with verified metadata.
//! Phase 4 will extend this via the `CatalogProvider` trait to dynamically
//! query Hugging Face Hub, Ollama Library, or future providers.
//!
//! All metadata in the bootstrap catalog has been verified against
//! authoritative config.json files from Hugging Face model repositories.

use crate::model_recommendation::traits::*;

/// Trait for catalog providers. Phase 4 implements HuggingFaceCatalogProvider, etc.
pub trait CatalogProvider: Send + Sync {
    fn name(&self) -> &str;
    fn fetch_models(&self) -> Result<Vec<ModelMetadata>, String>;
    fn get_model(&self, id: &str) -> Result<Option<ModelMetadata>, String>;
}

/// Built-in bootstrap catalog for Phase 3.
/// All metadata verified from authoritative HuggingFace config.json files.
pub struct BootstrapCatalog;

impl CatalogProvider for BootstrapCatalog {
    fn name(&self) -> &str {
        "Sarathi Bootstrap Catalog v1.0"
    }

    fn fetch_models(&self) -> Result<Vec<ModelMetadata>, String> {
        Ok(bootstrap_models())
    }

    fn get_model(&self, id: &str) -> Result<Option<ModelMetadata>, String> {
        Ok(bootstrap_models().into_iter().find(|m| m.id == id))
    }
}

/// Returns the verified bootstrap model catalog.
/// Every entry has been verified against the model's authoritative config.json
/// on Hugging Face. Parameter counts are from safetensors metadata or model cards.
///
/// Source verification date: 2026-07-31
/// Verification method: HuggingFace config.json (num_hidden_layers, num_attention_heads,
///   num_key_value_heads, hidden_size, head_dim, max_position_embeddings, vocab_size)
pub fn bootstrap_models() -> Vec<ModelMetadata> {
    vec![
        // ── Llama 3.2 ────────────────────────────────────────────
        ModelMetadata {
            id: "meta-llama/Llama-3.2-1B".into(),
            name: "Llama 3.2 1B".into(),
            family: "Llama 3.2".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 1_235_814_400,
            active_parameters: None,
            num_layers: 16,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 64,
            hidden_size: 2048,
            max_context_length: 131072,
            vocab_size: 128256,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "meta-llama/Llama-3.2-3B".into(),
            name: "Llama 3.2 3B".into(),
            family: "Llama 3.2".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 3_212_749_824,
            active_parameters: None,
            num_layers: 28,
            num_attention_heads: 24,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 3072,
            max_context_length: 131072,
            vocab_size: 128256,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        // ── Llama 3.1 ────────────────────────────────────────────
        ModelMetadata {
            id: "meta-llama/Llama-3.1-8B".into(),
            name: "Llama 3.1 8B".into(),
            family: "Llama 3.1".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 8_030_261_248,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 131072,
            vocab_size: 128256,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
            catalog_version: "1.0".into(),
        },
        // ── Qwen 2.5 ────────────────────────────────────────────
        ModelMetadata {
            id: "Qwen/Qwen2.5-3B".into(),
            name: "Qwen 2.5 3B".into(),
            family: "Qwen 2.5".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 3_090_000_000,
            active_parameters: None,
            num_layers: 36,
            num_attention_heads: 16,
            num_kv_heads: 2,
            head_dimension: 128,
            hidden_size: 2048,
            max_context_length: 32768,
            vocab_size: 151936,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "Qwen/Qwen2.5-7B".into(),
            name: "Qwen 2.5 7B".into(),
            family: "Qwen 2.5".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 7_610_000_000,
            active_parameters: None,
            num_layers: 28,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dimension: 128,
            hidden_size: 3584,
            max_context_length: 131072,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "Qwen/Qwen2.5-14B".into(),
            name: "Qwen 2.5 14B".into(),
            family: "Qwen 2.5".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 14_770_000_000,
            active_parameters: None,
            num_layers: 48,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 5120,
            max_context_length: 131072,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "Qwen/Qwen2.5-32B".into(),
            name: "Qwen 2.5 32B".into(),
            family: "Qwen 2.5".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 32_760_000_000,
            active_parameters: None,
            num_layers: 64,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 5120,
            max_context_length: 131072,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
            catalog_version: "1.0".into(),
        },
        // ── Qwen 2.5 Coder ──────────────────────────────────────
        ModelMetadata {
            id: "Qwen/Qwen2.5-Coder-7B".into(),
            name: "Qwen 2.5 Coder 7B".into(),
            family: "Qwen 2.5 Coder".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 7_610_000_000,
            active_parameters: None,
            num_layers: 28,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dimension: 128,
            hidden_size: 3584,
            max_context_length: 32768,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["code".into()],
            catalog_version: "1.0".into(),
        },
        // ── Mistral ──────────────────────────────────────────────
        ModelMetadata {
            id: "mistralai/Mistral-7B-v0.3".into(),
            name: "Mistral 7B v0.3".into(),
            family: "Mistral".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 7_248_020_480,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 32768,
            vocab_size: 32768,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        // ── Mixtral (MoE) ────────────────────────────────────────
        ModelMetadata {
            id: "mistralai/Mixtral-8x7B-v0.1".into(),
            name: "Mixtral 8×7B".into(),
            family: "Mixtral".into(),
            architecture: ModelArchitecture::MixtureOfExperts {
                num_experts: 8,
                active_experts: 2,
            },
            total_parameters: 46_700_000_000,
            active_parameters: Some(12_900_000_000),
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 32768,
            vocab_size: 32000,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        // ── Phi-4 ────────────────────────────────────────────────
        ModelMetadata {
            id: "microsoft/phi-4".into(),
            name: "Phi-4 14B".into(),
            family: "Phi".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 14_700_000_000,
            active_parameters: None,
            num_layers: 40,
            num_attention_heads: 40,
            num_kv_heads: 10,
            head_dimension: 128,
            hidden_size: 5120,
            max_context_length: 16384,
            vocab_size: 100352,
            default_dtype: "bf16".into(),
            use_cases: vec!["reasoning".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        // ── DeepSeek R1 Distills ─────────────────────────────────
        ModelMetadata {
            id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".into(),
            name: "DeepSeek R1 Distill Qwen 7B".into(),
            family: "DeepSeek R1".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 7_610_000_000,
            active_parameters: None,
            num_layers: 28,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dimension: 128,
            hidden_size: 3584,
            max_context_length: 131072,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["reasoning".into(), "chat".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B".into(),
            name: "DeepSeek R1 Distill Qwen 14B".into(),
            family: "DeepSeek R1".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 14_770_000_000,
            active_parameters: None,
            num_layers: 48,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 5120,
            max_context_length: 131072,
            vocab_size: 152064,
            default_dtype: "bf16".into(),
            use_cases: vec!["reasoning".into(), "chat".into()],
            catalog_version: "1.0".into(),
        },
        // ── Gemma 2 ──────────────────────────────────────────────
        ModelMetadata {
            id: "google/gemma-2-2b".into(),
            name: "Gemma 2 2B".into(),
            family: "Gemma 2".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 2_614_341_888,
            active_parameters: None,
            num_layers: 26,
            num_attention_heads: 8,
            num_kv_heads: 4,
            head_dimension: 256,
            hidden_size: 2304,
            max_context_length: 8192,
            vocab_size: 256000,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "google/gemma-2-9b".into(),
            name: "Gemma 2 9B".into(),
            family: "Gemma 2".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 9_241_705_984,
            active_parameters: None,
            num_layers: 42,
            num_attention_heads: 16,
            num_kv_heads: 8,
            head_dimension: 256,
            hidden_size: 3584,
            max_context_length: 8192,
            vocab_size: 256000,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into()],
            catalog_version: "1.0".into(),
        },
        ModelMetadata {
            id: "google/gemma-2-27b".into(),
            name: "Gemma 2 27B".into(),
            family: "Gemma 2".into(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 27_225_856_000,
            active_parameters: None,
            num_layers: 46,
            num_attention_heads: 32,
            num_kv_heads: 16,
            head_dimension: 128,
            hidden_size: 4608,
            max_context_length: 8192,
            vocab_size: 256000,
            default_dtype: "bf16".into(),
            use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
            catalog_version: "1.0".into(),
        },
    ]
}

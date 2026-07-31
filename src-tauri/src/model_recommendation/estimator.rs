//! Phase 3: Memory & Fit Estimator
//!
//! Calculates exact memory requirements for a model at a given quantization
//! and context length using documented, deterministic formulas.
//!
//! ## Formulas
//!
//! ### Weight Memory
//! `weight_bytes = total_parameters × bits_per_weight / 8`
//!
//! For MoE models, `total_parameters` includes ALL experts (all weights
//! must be loaded into memory). Active parameters are used only for
//! speed estimation (deferred to future phase).
//!
//! ### KV Cache Memory (GQA-Aware)
//! `kv_cache_bytes = 2 × layers × kv_heads × head_dim × context × bytes_per_element`
//!
//! Where:
//! - 2: Key and Value tensors
//! - layers: Number of transformer layers
//! - kv_heads: Number of KV heads (fewer than query heads for GQA/MQA)
//! - head_dim: Per-head dimension
//! - context: Sequence length in tokens
//! - bytes_per_element: 2 for FP16 KV cache (standard in llama.cpp/Ollama/vLLM)
//!
//! ### Runtime/Compute Overhead
//! `overhead_bytes = (weight_bytes + kv_cache_bytes) × overhead_factor`
//!
//! The overhead factor (default 12%) is a configurable conservative heuristic
//! accounting for:
//! - CUDA/Metal/Vulkan context initialization buffers
//! - GGUF metadata and tensor index structures
//! - Activation tensors during forward pass
//! - Memory allocator fragmentation
//!
//! Future runtime measurements can calibrate/replace this value.
//!
//! ### Total Memory
//! `total = weight_bytes + kv_cache_bytes + overhead_bytes`
//!
//! ## Assumptions
//! - KV cache uses FP16 precision regardless of weight quantization
//! - Batch size is 1 (single-user local inference)
//! - MoE: all expert weights resident; routing overhead negligible

use crate::model_recommendation::traits::*;

/// Estimate weight memory in bytes for a model at a given quantization.
pub fn estimate_weight_memory(model: &ModelMetadata, quant: &QuantizationSpec) -> u64 {
    // For MoE models, ALL parameters (all experts) must be loaded
    let params = model.total_parameters;
    // weight_bytes = params × bits_per_weight / 8
    (params as f64 * quant.bits_per_weight / 8.0) as u64
}

/// Estimate KV cache memory in bytes for a given context length.
/// Uses GQA-aware formula: 2 × L × H_kv × D_head × T × E
pub fn estimate_kv_cache_memory(model: &ModelMetadata, context_length: u32, config: &EstimatorConfig) -> u64 {
    let layers = model.num_layers as u64;
    let kv_heads = model.num_kv_heads as u64;
    let head_dim = model.head_dimension as u64;
    let context = context_length as u64;
    let bytes_per_element = config.kv_cache_bytes_per_element as u64;
    let batch_size = config.batch_size as u64;

    // 2 × L × H_kv × D_head × T × B × E
    2 * layers * kv_heads * head_dim * context * batch_size * bytes_per_element
}

/// Estimate total memory requirement for a model configuration.
/// Returns (weight_bytes, kv_cache_bytes, overhead_bytes, total_bytes).
pub fn estimate_total_memory(
    model: &ModelMetadata,
    quant: &QuantizationSpec,
    context_length: u32,
    config: &EstimatorConfig,
) -> (u64, u64, u64, u64) {
    let weight_bytes = estimate_weight_memory(model, quant);
    let kv_cache_bytes = estimate_kv_cache_memory(model, context_length, config);
    let overhead_bytes = ((weight_bytes + kv_cache_bytes) as f64 * config.overhead_factor) as u64;
    let total = weight_bytes + kv_cache_bytes + overhead_bytes;
    (weight_bytes, kv_cache_bytes, overhead_bytes, total)
}

/// Standard quantization levels ordered from highest to lowest quality.
pub fn quantization_hierarchy() -> Vec<QuantizationSpec> {
    vec![
        QuantizationSpec { label: "Q8_0".to_string(),    bits_per_weight: 8.5,  quality_rank: 8 },
        QuantizationSpec { label: "Q6_K".to_string(),    bits_per_weight: 6.5,  quality_rank: 7 },
        QuantizationSpec { label: "Q5_K_M".to_string(),  bits_per_weight: 5.5,  quality_rank: 6 },
        QuantizationSpec { label: "Q4_K_M".to_string(),  bits_per_weight: 4.85, quality_rank: 5 },
        QuantizationSpec { label: "Q4_0".to_string(),    bits_per_weight: 4.5,  quality_rank: 4 },
        QuantizationSpec { label: "Q3_K_M".to_string(),  bits_per_weight: 3.5,  quality_rank: 3 },
        QuantizationSpec { label: "Q2_K".to_string(),    bits_per_weight: 2.5,  quality_rank: 2 },
    ]
}

// ─── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_llama31_8b() -> ModelMetadata {
        ModelMetadata {
            id: "meta-llama/Llama-3.1-8B".to_string(),
            name: "Llama 3.1 8B".to_string(),
            family: "Llama 3.1".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 8_030_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 131072,
            vocab_size: 128256,
            default_dtype: "bf16".to_string(),
            use_cases: vec!["chat".to_string(), "general".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    fn make_mixtral_8x7b() -> ModelMetadata {
        ModelMetadata {
            id: "mistralai/Mixtral-8x7B-v0.1".to_string(),
            name: "Mixtral 8×7B".to_string(),
            family: "Mixtral".to_string(),
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
            default_dtype: "bf16".to_string(),
            use_cases: vec!["chat".to_string(), "general".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    fn make_codelama_7b() -> ModelMetadata {
        ModelMetadata {
            id: "codellama/CodeLlama-7b-hf".to_string(),
            name: "CodeLlama 7B".to_string(),
            family: "CodeLlama".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 6_740_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32, // MHA — no GQA
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 16384,
            vocab_size: 32016,
            default_dtype: "bf16".to_string(),
            use_cases: vec!["code".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    #[test]
    fn test_weight_memory_7b_q4() {
        let model = make_llama31_8b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let weight_mem = estimate_weight_memory(&model, &q4km);
        // 8.03B × 4.85 / 8 ≈ 4.87 GB
        assert!(weight_mem > 4 * GB);
        assert!(weight_mem < 6 * GB);
    }

    #[test]
    fn test_weight_memory_moe_mixtral() {
        let model = make_mixtral_8x7b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let weight_mem = estimate_weight_memory(&model, &q4km);
        // 46.7B × 4.85 / 8 ≈ 28.3 GB — ALL experts loaded
        assert!(weight_mem > 25 * GB);
        assert!(weight_mem < 32 * GB);
    }

    #[test]
    fn test_kv_cache_gqa_8k() {
        let model = make_llama31_8b();
        let config = EstimatorConfig::default();
        let kv = estimate_kv_cache_memory(&model, 8192, &config);
        // 2 × 32 × 8 × 128 × 8192 × 2 = 1,073,741,824 = 1 GB
        assert_eq!(kv, GB);
    }

    #[test]
    fn test_kv_cache_mha_vs_gqa() {
        let config = EstimatorConfig::default();
        // CodeLlama uses MHA (32 KV heads = 32 query heads)
        let codelama = make_codelama_7b();
        let kv_mha = estimate_kv_cache_memory(&codelama, 8192, &config);
        // Llama 3.1 uses GQA (8 KV heads, 32 query heads)
        let llama = make_llama31_8b();
        let kv_gqa = estimate_kv_cache_memory(&llama, 8192, &config);
        // MHA should be 4× larger than GQA (32/8 = 4)
        assert_eq!(kv_mha, kv_gqa * 4);
    }

    #[test]
    fn test_total_memory_with_overhead() {
        let model = make_llama31_8b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let config = EstimatorConfig::default(); // 12% overhead
        let (w, kv, oh, total) = estimate_total_memory(&model, &q4km, 8192, &config);
        assert_eq!(total, w + kv + oh);
        // Overhead should be ~12% of (weights + kv)
        let expected_oh = ((w + kv) as f64 * 0.12) as u64;
        assert_eq!(oh, expected_oh);
    }

    #[test]
    fn test_quantization_hierarchy_order() {
        let hierarchy = quantization_hierarchy();
        for i in 1..hierarchy.len() {
            assert!(hierarchy[i].quality_rank < hierarchy[i - 1].quality_rank);
            assert!(hierarchy[i].bits_per_weight < hierarchy[i - 1].bits_per_weight);
        }
    }
}

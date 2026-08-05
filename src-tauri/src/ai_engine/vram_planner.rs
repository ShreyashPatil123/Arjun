//! GPU Offload Planning — deciding how many layers actually fit in VRAM.
//!
//! Replaces the heuristic previously inlined in `InferenceManager::build_load_config`,
//! which had three defects that all bias toward over-committing VRAM:
//!
//! 1. `vram_gb >= 6.0` forced full offload regardless of model size, so an 8 GB
//!    model on a 6 GB card requested every layer.
//! 2. Partial offload computed `ratio * 32.0`, hardcoding 32 layers. Real models
//!    range from 16 (Llama-3.2-1B) to 36 (Qwen3-4B) to 80 (70B).
//! 3. KV cache was ignored entirely. At 8K context it is often 1 GB+ — larger
//!    than the margin the old check reserved.
//!
//! On a 4 GB card the combination is decisive: a 2.5 GB model passes
//! `vram >= model_size + 1.0`, requests full offload, and then fails or silently
//! spills once the KV cache and compute buffers are allocated.
//!
//! ## Approximation
//!
//! Exact KV cache size needs `n_layer`, `n_head_kv`, and `head_dim` from GGUF
//! metadata, which is not parsed at load time. [`estimate_kv_bytes_per_token`]
//! substitutes a conservative size-banded estimate — deliberately biased high,
//! because under-offloading costs speed while over-offloading costs a crash.
//! `model_recommendation::estimator` has the exact GQA-aware formula for the
//! catalog path, where the metadata is available.

/// VRAM held back for the OS, desktop compositor, and other applications.
///
/// On Windows the desktop alone commonly holds 0.5–1.0 GB on a discrete GPU.
const OS_RESERVE_BYTES: u64 = 900 * 1024 * 1024;

/// Fraction of remaining VRAM reserved for llama.cpp compute buffers
/// (activations, scratch, allocator fragmentation).
const COMPUTE_OVERHEAD_FRACTION: f64 = 0.12;

/// Below this much usable VRAM, GPU offload is not worth attempting.
const MIN_USABLE_VRAM_BYTES: u64 = 512 * 1024 * 1024;

/// Layer count assumed when GGUF metadata is unavailable.
///
/// Only scales the *proportion* offloaded; llama.cpp clamps a too-large value
/// to the real layer count, so erring high is safe.
const ASSUMED_LAYERS_FALLBACK: u32 = 32;

/// Value meaning "offload everything" to llama.cpp.
pub const FULL_OFFLOAD: u32 = 999;

/// The offload decision, with the reasoning that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuOffloadPlan {
    /// Layers to request. 0 means CPU-only; [`FULL_OFFLOAD`] means all.
    pub gpu_layers: u32,
    /// Whether every layer is expected to fit.
    pub full_offload: bool,
    /// Human-readable explanation for the load log.
    pub reason: String,
}

impl GpuOffloadPlan {
    fn cpu_only(reason: impl Into<String>) -> Self {
        Self {
            gpu_layers: 0,
            full_offload: false,
            reason: reason.into(),
        }
    }
}

/// Conservative KV-cache cost per token, in bytes.
///
/// Banded by model size as a proxy for `layers × kv_heads × head_dim`.
/// Reference points (FP16 KV, GQA):
/// - Llama-3.2-1B: 16 layers × 8 KV heads × 64 dim  →  32 KB/token
/// - Qwen3-4B:     36 layers × 8 KV heads × 128 dim → 144 KB/token
/// - Llama-3-8B:   32 layers × 8 KV heads × 128 dim → 128 KB/token
pub fn estimate_kv_bytes_per_token(model_bytes: u64) -> u64 {
    const GB: u64 = 1024 * 1024 * 1024;
    match model_bytes {
        b if b < GB => 32 * 1024,             // ~1B params
        b if b < 3 * GB => 64 * 1024,         // ~1–3B params
        b if b < 8 * GB => 160 * 1024,        // ~3–8B params (covers Qwen3-4B)
        _ => 256 * 1024,                      // larger
    }
}

/// Plans GPU offload for a model given real VRAM, model size, and context.
///
/// `total_layers` should come from GGUF metadata when known; `None` falls back
/// to [`ASSUMED_LAYERS_FALLBACK`].
pub fn plan_gpu_offload(
    vram_total_bytes: u64,
    model_bytes: u64,
    context_length: u32,
    total_layers: Option<u32>,
) -> GpuOffloadPlan {
    if vram_total_bytes == 0 {
        return GpuOffloadPlan::cpu_only("No GPU VRAM detected");
    }

    // Usable VRAM after the OS and desktop take their share.
    let usable = vram_total_bytes.saturating_sub(OS_RESERVE_BYTES);
    if usable < MIN_USABLE_VRAM_BYTES {
        return GpuOffloadPlan::cpu_only(format!(
            "Only {:.2} GB usable VRAM after {:.2} GB OS reserve — below the {:.2} GB minimum",
            as_gb(usable),
            as_gb(OS_RESERVE_BYTES),
            as_gb(MIN_USABLE_VRAM_BYTES)
        ));
    }

    let kv_bytes = estimate_kv_bytes_per_token(model_bytes)
        .saturating_mul(context_length.max(1) as u64);

    // KV cache and compute buffers are charged before any weights.
    let after_kv = usable.saturating_sub(kv_bytes);
    let compute_reserve = (after_kv as f64 * COMPUTE_OVERHEAD_FRACTION) as u64;
    let weight_budget = after_kv.saturating_sub(compute_reserve);

    if weight_budget == 0 {
        return GpuOffloadPlan::cpu_only(format!(
            "KV cache for {} tokens (~{:.2} GB) exhausts {:.2} GB usable VRAM",
            context_length,
            as_gb(kv_bytes),
            as_gb(usable)
        ));
    }

    if weight_budget >= model_bytes {
        return GpuOffloadPlan {
            gpu_layers: FULL_OFFLOAD,
            full_offload: true,
            reason: format!(
                "Full offload: {:.2} GB weight budget covers the {:.2} GB model \
                 (VRAM {:.2} GB − {:.2} GB OS − {:.2} GB KV@{} − {:.2} GB compute)",
                as_gb(weight_budget),
                as_gb(model_bytes),
                as_gb(vram_total_bytes),
                as_gb(OS_RESERVE_BYTES),
                as_gb(kv_bytes),
                context_length,
                as_gb(compute_reserve)
            ),
        };
    }

    // Partial offload proportional to the fraction of weights that fit.
    let layers = total_layers.unwrap_or(ASSUMED_LAYERS_FALLBACK).max(1);
    let fraction = weight_budget as f64 / model_bytes as f64;
    let offload = (fraction * layers as f64).floor() as u32;

    if offload == 0 {
        return GpuOffloadPlan::cpu_only(format!(
            "Weight budget {:.2} GB covers under one layer of the {:.2} GB model",
            as_gb(weight_budget),
            as_gb(model_bytes)
        ));
    }

    GpuOffloadPlan {
        gpu_layers: offload,
        full_offload: false,
        reason: format!(
            "Partial offload: {}/{} layers ({:.0}% of the {:.2} GB model fits in a \
             {:.2} GB weight budget after {:.2} GB KV@{})",
            offload,
            layers,
            fraction * 100.0,
            as_gb(model_bytes),
            as_gb(weight_budget),
            as_gb(kv_bytes),
            context_length
        ),
    }
}

/// Smallest context worth loading. Below this a model is not useful for coding.
const MIN_VIABLE_CONTEXT: u32 = 2048;

/// Largest context length this machine can afford for a model.
///
/// A model's advertised maximum (e.g. Qwen2.5's 32768) says what the *weights*
/// support, not what a given GPU can hold. The KV cache grows linearly with
/// context, so on a small card a full-length context can cost more memory than
/// the model itself. Requesting it anyway is a reliable way to run out of VRAM
/// mid-generation.
///
/// Returns `desired` unchanged when there is no GPU to budget against — in that
/// case the limit is system RAM, which is handled elsewhere.
pub fn max_affordable_context(
    vram_total_bytes: u64,
    model_bytes: u64,
    desired_context: u32,
) -> u32 {
    if vram_total_bytes == 0 || desired_context == 0 {
        return desired_context;
    }

    let usable = vram_total_bytes.saturating_sub(OS_RESERVE_BYTES);
    let compute_reserve = (usable as f64 * COMPUTE_OVERHEAD_FRACTION) as u64;

    // Whatever is left after weights and compute buffers is available for KV.
    let kv_budget = usable
        .saturating_sub(model_bytes)
        .saturating_sub(compute_reserve);

    if kv_budget == 0 {
        // Weights alone fill the card; llama.cpp will spill layers to CPU and
        // the KV for those lives in system RAM. Keep a small usable context
        // rather than reporting zero.
        return MIN_VIABLE_CONTEXT.min(desired_context);
    }

    let per_token = estimate_kv_bytes_per_token(model_bytes).max(1);
    let affordable = (kv_budget / per_token).min(u32::MAX as u64) as u32;

    affordable.clamp(MIN_VIABLE_CONTEXT.min(desired_context), desired_context)
}

fn as_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn no_vram_means_cpu_only() {
        let plan = plan_gpu_offload(0, 2 * GB, 4096, None);
        assert_eq!(plan.gpu_layers, 0);
    }

    #[test]
    fn tiny_gpus_fall_back_to_cpu() {
        // 1 GB card: nothing usable after the OS reserve.
        let plan = plan_gpu_offload(GB, 2 * GB, 4096, None);
        assert_eq!(plan.gpu_layers, 0);
        assert!(plan.reason.contains("usable VRAM"));
    }

    #[test]
    fn regression_4gb_card_does_not_full_offload_a_25gb_model() {
        // The exact case the old heuristic got wrong: 4 GB RTX 3050 laptop.
        // `vram_gb >= model_size_gb + 1.0` → 4.0 >= 3.5 → requested all layers,
        // ignoring ~1.3 GB of KV cache at 8K context.
        let plan = plan_gpu_offload(4 * GB, 2560 * 1024 * 1024, 8192, Some(36));

        assert!(
            !plan.full_offload,
            "must not claim full offload on a 4 GB card: {}",
            plan.reason
        );
        assert!(plan.gpu_layers < FULL_OFFLOAD);
    }

    #[test]
    fn regression_large_vram_does_not_blindly_full_offload() {
        // Old rule: `vram_gb >= 6.0` → full offload regardless of model size.
        // A 20 GB model on an 8 GB card must not request every layer.
        let plan = plan_gpu_offload(8 * GB, 20 * GB, 8192, Some(80));

        assert!(!plan.full_offload, "20 GB model cannot fit 8 GB VRAM");
        assert!(plan.gpu_layers > 0 && plan.gpu_layers < 80);
    }

    #[test]
    fn a_roomy_card_offloads_everything() {
        let plan = plan_gpu_offload(24 * GB, 4 * GB, 8192, Some(36));

        assert!(plan.full_offload);
        assert_eq!(plan.gpu_layers, FULL_OFFLOAD);
    }

    #[test]
    fn context_length_reduces_offloadable_layers() {
        let short = plan_gpu_offload(8 * GB, 6 * GB, 2048, Some(32));
        let long = plan_gpu_offload(8 * GB, 6 * GB, 32768, Some(32));

        assert!(
            long.gpu_layers <= short.gpu_layers,
            "a larger KV cache must not permit more layers ({} vs {})",
            long.gpu_layers,
            short.gpu_layers
        );
    }

    #[test]
    fn layer_count_is_respected_rather_than_assumed_32() {
        // Same memory pressure, different architectures.
        let sixteen = plan_gpu_offload(6 * GB, 8 * GB, 4096, Some(16));
        let eighty = plan_gpu_offload(6 * GB, 8 * GB, 4096, Some(80));

        assert!(
            eighty.gpu_layers > sixteen.gpu_layers,
            "an 80-layer model should offload more layers than a 16-layer one at \
             the same fitting fraction ({} vs {})",
            eighty.gpu_layers,
            sixteen.gpu_layers
        );
        assert!(sixteen.gpu_layers <= 16, "cannot offload more layers than exist");
    }

    #[test]
    fn kv_estimate_grows_with_model_size() {
        assert!(
            estimate_kv_bytes_per_token(512 * 1024 * 1024)
                < estimate_kv_bytes_per_token(4 * GB)
        );
        assert!(estimate_kv_bytes_per_token(4 * GB) < estimate_kv_bytes_per_token(20 * GB));
    }

    #[test]
    fn a_4gb_card_cannot_afford_a_32k_context() {
        // The exact case the certified runtime profile got wrong: it hardcoded
        // context_length 32768 regardless of the machine.
        let affordable = max_affordable_context(4 * GB, 2560 * 1024 * 1024, 32768);

        assert!(
            affordable < 32768,
            "4 GB card must not be told it can hold a 32k context, got {affordable}"
        );
        assert!(affordable >= MIN_VIABLE_CONTEXT, "must stay usable, got {affordable}");
    }

    #[test]
    fn a_large_card_keeps_the_full_context() {
        let affordable = max_affordable_context(24 * GB, 4 * GB, 32768);
        assert_eq!(affordable, 32768, "a 24 GB card can afford the model's full context");
    }

    #[test]
    fn affordable_context_never_exceeds_what_was_asked_for() {
        // Even with enormous VRAM, we never invent context the model lacks.
        assert_eq!(max_affordable_context(80 * GB, GB, 8192), 8192);
    }

    #[test]
    fn no_gpu_leaves_the_request_untouched() {
        // CPU-only is bounded by system RAM, handled elsewhere.
        assert_eq!(max_affordable_context(0, 4 * GB, 16384), 16384);
    }

    #[test]
    fn context_shrinks_as_the_model_grows() {
        let small = max_affordable_context(8 * GB, GB, 32768);
        let large = max_affordable_context(8 * GB, 6 * GB, 32768);

        assert!(
            large <= small,
            "a bigger model leaves less room for KV cache ({large} vs {small})"
        );
    }

    #[test]
    fn plans_never_panic_across_extremes() {
        let cases = [
            (0u64, 0u64, 0u32),
            (u64::MAX, u64::MAX, u32::MAX),
            (4 * GB, 1, 1),
            (1, 4 * GB, 128_000),
        ];

        for (vram, model, ctx) in cases {
            let plan = plan_gpu_offload(vram, model, ctx, None);
            assert!(plan.gpu_layers == 0 || plan.gpu_layers > 0);
        }
    }
}

//! Inference Backend / Runtime Selector
//!
//! Reports how Sarathi will actually run a model.
//!
//! This module previously ranked Ollama above llama.cpp whenever CUDA was
//! present, so model cards advertised "Backend: Ollama". Sarathi has never used
//! Ollama — it embeds llama.cpp in-process through the `llama-cpp-2` crate and
//! has no code path that shells out to any external server. The listing was
//! aspirational, and told users to expect software the app does not touch.
//!
//! There is exactly one backend. What varies is the acceleration, and that
//! depends on two things that must both hold:
//!
//! 1. **The build** — `llama-cpp-sys-2` compiles llama.cpp with `GGML_CUDA=OFF`
//!    unless the `cuda` feature is enabled. A CPU-only binary ignores every GPU
//!    layer request silently.
//! 2. **The hardware** — the machine must actually have a usable GPU.
//!
//! Reporting CUDA when either is missing repeats the bug that made the runtime
//! claim GPU offload while running entirely on CPU.

use crate::model_recommendation::traits::*;

/// Acceleration compiled into this binary, independent of hardware.
///
/// `None` means llama.cpp was built CPU-only, so no GPU claim can be honest
/// regardless of what the machine has installed.
pub fn compiled_acceleration() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("CUDA")
    } else if cfg!(feature = "vulkan") {
        Some("Vulkan")
    } else {
        None
    }
}

/// Human-readable description of how a model will actually execute.
///
/// Examples: `llama.cpp · CUDA`, `llama.cpp · CPU`,
/// `llama.cpp · CPU (built without GPU support)`.
pub fn describe_execution(gpu: Option<&GpuMemoryBudget>) -> String {
    let hardware_capable = gpu
        .map(|g| g.cuda_available || g.rocm_available || g.vulkan_available)
        .unwrap_or(false);

    match (compiled_acceleration(), hardware_capable) {
        (Some(accel), true) => format!("llama.cpp · {accel}"),
        // A GPU is present but this build cannot use it. Say so, rather than
        // letting the user assume their card is doing the work.
        (None, true) => "llama.cpp · CPU (built without GPU support)".to_string(),
        (_, false) => "llama.cpp · CPU".to_string(),
    }
}

/// Backends Sarathi can actually run.
///
/// Always exactly one. The parameter is kept so callers need not change, and so
/// the signature still reads as hardware-dependent should that become true.
pub fn compatible_backends(_gpu: &GpuMemoryBudget) -> Vec<InferenceBackend> {
    vec![InferenceBackend::LlamaCppGguf]
}

/// Backends available in CPU-only mode — the same single engine.
pub fn cpu_only_backends() -> Vec<InferenceBackend> {
    vec![InferenceBackend::LlamaCppGguf]
}

/// Selects the backend to run with.
///
/// Always llama.cpp: it is the only engine Sarathi links against.
pub fn select_preferred_backend(_backends: &[InferenceBackend], _has_cuda: bool) -> InferenceBackend {
    InferenceBackend::LlamaCppGguf
}

/// Format a RunMode for human-readable display.
pub fn format_run_mode(mode: &RunMode) -> String {
    match mode {
        RunMode::PureGpu { gpu_index } => format!("Pure GPU (GPU #{})", gpu_index),
        RunMode::GpuWithCpuOffload { gpu_index, offload_fraction } =>
            format!("GPU #{} + CPU Offload ({:.0}%)", gpu_index, offload_fraction * 100.0),
        RunMode::MultiGpu { gpu_indices } =>
            format!("Multi-GPU ({:?})", gpu_indices),
        RunMode::PureCpu => "CPU Only".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_gpu() -> GpuMemoryBudget {
        GpuMemoryBudget {
            gpu_index: 0, gpu_model: "RTX 3050 Laptop".into(), gpu_type: GpuType::Dedicated,
            total_dedicated_vram: 4 * 1024 * 1024 * 1024, usable_dedicated_vram: 3 * 1024 * 1024 * 1024,
            total_shared_memory: 0, usable_shared_memory: 0,
            cuda_available: true, rocm_available: false, vulkan_available: true, directml_available: true,
            compute_capability: Some("8.6".into()),
        }
    }

    #[test]
    fn regression_ollama_is_never_offered() {
        // Model cards showed "Backend: Ollama" on CUDA machines. Sarathi has no
        // Ollama code path at all — the label sent users to install software the
        // app never touches.
        let backends = compatible_backends(&cuda_gpu());

        assert!(!backends.contains(&InferenceBackend::Ollama));
        assert_eq!(select_preferred_backend(&backends, true), InferenceBackend::LlamaCppGguf);
        assert!(!cpu_only_backends().contains(&InferenceBackend::Ollama));
    }

    #[test]
    fn only_the_engine_we_actually_link_is_offered() {
        // vLLM and DirectML are equally unreachable from this codebase.
        let backends = compatible_backends(&cuda_gpu());

        assert_eq!(backends, vec![InferenceBackend::LlamaCppGguf]);
    }

    #[test]
    fn execution_description_matches_how_this_binary_was_built() {
        let described = describe_execution(Some(&cuda_gpu()));

        match compiled_acceleration() {
            // GPU-enabled build on GPU hardware: name the acceleration.
            Some(accel) => assert_eq!(described, format!("llama.cpp · {accel}")),
            // CPU-only build: must not imply the GPU is being used, even though
            // the machine has one.
            None => {
                assert!(described.contains("CPU"), "got: {described}");
                assert!(
                    described.contains("built without GPU support"),
                    "a CPU-only build on GPU hardware must say why: {described}"
                );
            }
        }
    }

    #[test]
    fn a_machine_without_a_gpu_reports_plain_cpu() {
        let described = describe_execution(None);
        assert_eq!(described, "llama.cpp · CPU");
    }
}

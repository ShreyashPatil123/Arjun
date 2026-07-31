//! Phase 3: Inference Backend / Runtime Selector
//!
//! Determines compatible inference backends for each GPU and run mode.
//! Phase 3 does NOT install or launch backends — it only determines
//! which would be compatible.

use crate::model_recommendation::traits::*;

/// Determines compatible backends for a given GPU budget.
pub fn compatible_backends(gpu: &GpuMemoryBudget) -> Vec<InferenceBackend> {
    let mut backends = Vec::new();

    if gpu.cuda_available {
        backends.push(InferenceBackend::Ollama);
        backends.push(InferenceBackend::LlamaCppGguf);
        backends.push(InferenceBackend::VllmCuda);
    } else if gpu.rocm_available {
        backends.push(InferenceBackend::Ollama);
        backends.push(InferenceBackend::LlamaCppGguf);
    } else if gpu.vulkan_available {
        backends.push(InferenceBackend::LlamaCppGguf);
        backends.push(InferenceBackend::VulkanCompute);
    }

    if gpu.directml_available && !gpu.cuda_available {
        backends.push(InferenceBackend::DirectML);
    }

    // llama.cpp always supports CPU fallback
    if !backends.contains(&InferenceBackend::LlamaCppGguf) {
        backends.push(InferenceBackend::LlamaCppGguf);
    }

    backends
}

/// Determines compatible backends for CPU-only run mode.
pub fn cpu_only_backends() -> Vec<InferenceBackend> {
    vec![
        InferenceBackend::LlamaCppGguf,
        InferenceBackend::Ollama,
    ]
}

/// Selects the preferred backend from a list of compatible ones.
/// Priority: Ollama (if CUDA) > llama.cpp > vLLM > DirectML > Vulkan
pub fn select_preferred_backend(backends: &[InferenceBackend], has_cuda: bool) -> InferenceBackend {
    if has_cuda {
        if backends.contains(&InferenceBackend::Ollama) {
            return InferenceBackend::Ollama;
        }
        if backends.contains(&InferenceBackend::LlamaCppGguf) {
            return InferenceBackend::LlamaCppGguf;
        }
        if backends.contains(&InferenceBackend::VllmCuda) {
            return InferenceBackend::VllmCuda;
        }
    }
    if backends.contains(&InferenceBackend::LlamaCppGguf) {
        return InferenceBackend::LlamaCppGguf;
    }
    if backends.contains(&InferenceBackend::Ollama) {
        return InferenceBackend::Ollama;
    }
    if backends.contains(&InferenceBackend::DirectML) {
        return InferenceBackend::DirectML;
    }
    if backends.contains(&InferenceBackend::VulkanCompute) {
        return InferenceBackend::VulkanCompute;
    }
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

    #[test]
    fn test_runtime_cuda_preference() {
        let gpu = GpuMemoryBudget {
            gpu_index: 0, gpu_model: "RTX 5060".into(), gpu_type: GpuType::Dedicated,
            total_dedicated_vram: 8 * 1024 * 1024 * 1024, usable_dedicated_vram: 7 * 1024 * 1024 * 1024,
            total_shared_memory: 0, usable_shared_memory: 0,
            cuda_available: true, rocm_available: false, vulkan_available: true, directml_available: true,
            compute_capability: Some("12.0".into()),
        };
        let backends = compatible_backends(&gpu);
        assert!(backends.contains(&InferenceBackend::Ollama));
        assert!(backends.contains(&InferenceBackend::LlamaCppGguf));
        assert!(backends.contains(&InferenceBackend::VllmCuda));
        let preferred = select_preferred_backend(&backends, true);
        assert_eq!(preferred, InferenceBackend::Ollama);
    }

    #[test]
    fn test_runtime_vulkan_amd_igpu() {
        let gpu = GpuMemoryBudget {
            gpu_index: 0, gpu_model: "Radeon 780M".into(), gpu_type: GpuType::Integrated,
            total_dedicated_vram: 512 * 1024 * 1024, usable_dedicated_vram: 0,
            total_shared_memory: 12 * 1024 * 1024 * 1024, usable_shared_memory: 6 * 1024 * 1024 * 1024,
            cuda_available: false, rocm_available: false, vulkan_available: true, directml_available: true,
            compute_capability: None,
        };
        let backends = compatible_backends(&gpu);
        assert!(backends.contains(&InferenceBackend::LlamaCppGguf));
        assert!(backends.contains(&InferenceBackend::DirectML));
        let preferred = select_preferred_backend(&backends, false);
        assert_eq!(preferred, InferenceBackend::LlamaCppGguf);
    }
}

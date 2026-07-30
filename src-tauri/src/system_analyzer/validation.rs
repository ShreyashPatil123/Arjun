//! Validation engine evaluating hardware capability scores and readiness for local AI

use crate::system_analyzer::traits::{HardwareProfile, SystemValidationResult};

/// Validates a HardwareProfile against local AI runtime requirements
pub fn validate_profile(profile: &HardwareProfile) -> SystemValidationResult {
    let memory = profile.memory.current();
    let gpus = profile.gpus.current();
    let storage_drives = profile.storage.current();
    let software = profile.software.current();

    let mut score: u32 = 0;
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut recommendations = Vec::new();

    // 1. RAM Check (Max 30 pts)
    let ram_gb = memory.total_bytes / (1024 * 1024 * 1024);
    if ram_gb >= 32 {
        score += 30;
    } else if ram_gb >= 16 {
        score += 22;
        recommendations.push("Upgrading RAM to 32GB allows running larger quantized models smoothly.".to_string());
    } else if ram_gb >= 8 {
        score += 12;
        warnings.push("System RAM is 8GB. 16GB+ recommended for optimal local AI performance.".to_string());
        recommendations.push("Consider upgrading RAM to 16GB or 32GB.".to_string());
    } else {
        errors.push("System RAM is under 8GB minimum requirement for local model execution.".to_string());
    }

    // 2. GPU & VRAM Check (Max 35 pts)
    let max_vram_bytes = gpus.iter().map(|g| g.vram_total_bytes).max().unwrap_or(0);
    let max_vram_gb = max_vram_bytes / (1024 * 1024 * 1024);
    let has_dedicated_gpu = gpus.iter().any(|g| g.is_dedicated);

    if has_dedicated_gpu {
        if max_vram_gb >= 16 {
            score += 35;
        } else if max_vram_gb >= 12 {
            score += 30;
        } else if max_vram_gb >= 8 {
            score += 25;
            recommendations.push("8GB VRAM is great for 7B-8B parameter models with Q4/Q5 quantization.".to_string());
        } else if max_vram_gb >= 4 {
            score += 15;
            warnings.push("GPU VRAM is under 8GB. Larger models will offload to CPU RAM.".to_string());
        } else {
            score += 10;
            warnings.push("Dedicated GPU has low VRAM. CPU inference will be used as fallback.".to_string());
        }
    } else {
        score += 8;
        warnings.push("No dedicated GPU detected. Models will run on CPU, which may be slower.".to_string());
        recommendations.push("An NVIDIA RTX GPU (8GB+ VRAM) is recommended for fast inference.".to_string());
    }

    // 3. CUDA / Acceleration Check (Max 15 pts)
    let has_cuda = gpus.iter().any(|g| g.cuda_supported);
    let has_rocm = gpus.iter().any(|g| g.rocm_supported);
    if has_cuda {
        score += 15;
    } else if has_rocm {
        score += 12;
    } else if gpus.iter().any(|g| g.vulkan_supported) {
        score += 10;
    } else {
        score += 5;
    }

    // 4. Storage Space Check (Max 15 pts)
    let max_free_storage = storage_drives.iter().map(|s| s.free_bytes).max().unwrap_or(0);
    let max_free_gb = max_free_storage / (1024 * 1024 * 1024);

    if max_free_gb >= 50 {
        score += 15;
    } else if max_free_gb >= 20 {
        score += 10;
        recommendations.push("At least 50GB free disk space recommended for multiple GGUF/Safetensors models.".to_string());
    } else if max_free_gb >= 10 {
        score += 5;
        warnings.push("Disk space is below 20GB. Free up disk space before downloading large models.".to_string());
    } else {
        errors.push("Insufficient disk space (<10GB) available for local model storage.".to_string());
    }

    // 5. Software Environment Check (Max 5 pts)
    if software.ollama.installed || software.python.installed || software.git.installed {
        score += 5;
    } else {
        recommendations.push("Install Ollama or Python for seamless local model runtime execution.".to_string());
    }

    if score > 100 {
        score = 100;
    }

    let is_ready_for_ai = errors.is_empty() && ram_gb >= 8;

    SystemValidationResult {
        is_ready_for_ai,
        score,
        warnings,
        errors,
        recommendations,
    }
}

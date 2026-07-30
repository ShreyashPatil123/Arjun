//! Normalization module combining raw metrics into a HardwareProfile

use crate::system_analyzer::traits::{
    AICapabilityProfile, CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, OsInfo, OverrideValue,
    SoftwareEnvironment, StorageInfo, SystemPaths, SystemValidationResult, AIRuntimeInfo,
};
use crate::system_analyzer::validation::validate_profile;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

/// Combines raw collector metrics into a full HardwareProfile
pub fn normalize_profile(
    raw_cpu: CpuInfo,
    raw_gpus: Vec<GpuInfo>,
    raw_memory: MemoryInfo,
    raw_storage: Vec<StorageInfo>,
    raw_os: OsInfo,
    raw_software: SoftwareEnvironment,
    raw_ai_runtimes: Vec<AIRuntimeInfo>,
    raw_paths: SystemPaths,
) -> HardwareProfile {
    let ai_capabilities = compute_ai_capabilities(&raw_cpu, &raw_gpus, &raw_memory);

    let id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();

    let mut profile = HardwareProfile {
        id,
        profile_created_at: now.clone(),
        profile_updated_at: now,
        cpu: OverrideValue::new(raw_cpu),
        gpus: OverrideValue::new(raw_gpus),
        memory: OverrideValue::new(raw_memory),
        storage: OverrideValue::new(raw_storage),
        os: OverrideValue::new(raw_os),
        software: OverrideValue::new(raw_software),
        ai_runtimes: OverrideValue::new(raw_ai_runtimes),
        paths: raw_paths,
        ai_capabilities,
        validation: SystemValidationResult {
            is_ready_for_ai: false,
            score: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
            recommendations: Vec::new(),
        },
    };

    let validation = validate_profile(&profile);
    profile.validation = validation;

    profile
}

/// Evaluates hardware parameters to infer AI model running capabilities
pub fn compute_ai_capabilities(
    cpu: &CpuInfo,
    gpus: &[GpuInfo],
    memory: &MemoryInfo,
) -> AICapabilityProfile {
    let max_vram = gpus.iter().map(|g| g.vram_total_bytes).max().unwrap_or(0);
    let has_cuda = gpus.iter().any(|g| g.cuda_supported);
    let has_rocm = gpus.iter().any(|g| g.rocm_supported);

    let preferred_inference_backend = if has_cuda {
        Some("cuda".to_string())
    } else if has_rocm {
        Some("rocm".to_string())
    } else if gpus.iter().any(|g| g.vulkan_supported) {
        Some("vulkan".to_string())
    } else {
        Some("cpu".to_string())
    };

    // Calculate maximum recommended model size in bytes
    let max_recommended_model_size_bytes = if max_vram >= 16 * 1024 * 1024 * 1024 {
        Some(24 * 1024 * 1024 * 1024) // ~34B model quantized
    } else if max_vram >= 8 * 1024 * 1024 * 1024 {
        Some(10 * 1024 * 1024 * 1024) // ~13B model quantized
    } else if max_vram >= 4 * 1024 * 1024 * 1024 || memory.total_bytes >= 16 * 1024 * 1024 * 1024 {
        Some(6 * 1024 * 1024 * 1024) // ~7B model quantized
    } else {
        Some(3 * 1024 * 1024 * 1024) // ~3B model quantized
    };

    let mut recommended_quantizations = vec!["Q4_K_M".to_string(), "Q5_K_M".to_string()];
    if max_vram >= 12 * 1024 * 1024 * 1024 {
        recommended_quantizations.push("Q8_0".to_string());
        recommended_quantizations.push("FP16".to_string());
    }

    let recommended_context_length = if max_vram >= 12 * 1024 * 1024 * 1024 {
        Some(16384)
    } else if max_vram >= 6 * 1024 * 1024 * 1024 {
        Some(8192)
    } else {
        Some(4096)
    };

    let multi_model_capable = memory.total_bytes >= 24 * 1024 * 1024 * 1024 || max_vram >= 12 * 1024 * 1024 * 1024;
    let lora_ready = max_vram >= 6 * 1024 * 1024 * 1024 || memory.total_bytes >= 16 * 1024 * 1024 * 1024;
    let vision_ready = max_vram >= 8 * 1024 * 1024 * 1024 || memory.total_bytes >= 16 * 1024 * 1024 * 1024;
    let embedding_ready = true;

    let mut extra_capabilities = HashMap::new();
    extra_capabilities.insert(
        "avx2_supported".to_string(),
        serde_json::Value::Bool(cpu.simd_capabilities.contains(&"AVX2".to_string())),
    );
    extra_capabilities.insert(
        "total_gpus".to_string(),
        serde_json::Value::Number(serde_json::Number::from(gpus.len())),
    );

    AICapabilityProfile {
        max_recommended_model_size_bytes,
        recommended_quantizations,
        recommended_context_length,
        preferred_inference_backend,
        multi_model_capable,
        lora_ready,
        vision_ready,
        embedding_ready,
        extra_capabilities,
    }
}

//! Phase 3: Adaptive Resource Budget Calculator
//!
//! Calculates safe usable memory for inference from Phase 2 HardwareProfile.
//! Uses live OS telemetry (available_bytes, vram_free_bytes) and applies
//! configurable percentage-based safety margins.
//!
//! NEVER kills processes, clears memory, or forces resource reclamation.
//! Memory domains (dedicated VRAM, GPU shared memory, system RAM) are
//! kept strictly separate — never pooled.

use crate::model_recommendation::traits::*;
use crate::system_analyzer::traits::HardwareProfile;

/// Calculates a MemoryBudget from a HardwareProfile using adaptive safety margins.
pub fn calculate_budget(profile: &HardwareProfile, config: &BudgetConfig) -> MemoryBudget {
    let memory = profile.memory.current();
    let gpus = profile.gpus.current();

    // ── System RAM Budget ────────────────────────────────────────────────
    let total_ram = memory.total_bytes;
    let available_ram = memory.available_bytes;

    // Adaptive OS reservation: percentage of total, clamped to [min, max]
    let os_reservation = {
        let proportional = (total_ram as f64 * config.ram_os_reserve_fraction) as u64;
        proportional.max(config.ram_os_reserve_min).min(config.ram_os_reserve_max)
    };

    // Usable = min(what's actually free, theoretical max after reservations)
    // This ensures we never exceed either the live-free or the budget cap.
    let theoretical_max = total_ram.saturating_sub(os_reservation + config.ram_sarathi_reserve);
    let live_usable = available_ram.saturating_sub(config.ram_sarathi_reserve);
    let usable_for_inference = theoretical_max.min(live_usable);

    let system_ram = SystemRamBudget {
        total_bytes: total_ram,
        available_bytes: available_ram,
        usable_for_inference,
        ram_speed_mts: memory.speed_mts,
    };

    // ── Per-GPU Memory Budgets ───────────────────────────────────────────
    let gpu_budgets: Vec<GpuMemoryBudget> = gpus
        .iter()
        .enumerate()
        .map(|(idx, gpu)| {
            let gpu_type = if gpu.is_dedicated {
                GpuType::Dedicated
            } else {
                GpuType::Integrated
            };

            let total_dedicated_vram = gpu.dedicated_video_memory_bytes;
            let total_shared_memory = gpu.shared_system_memory_bytes;

            // Dedicated VRAM budget
            let usable_dedicated_vram = if total_dedicated_vram > 0 {
                let reserve = {
                    let proportional = (total_dedicated_vram as f64 * config.vram_reserve_fraction) as u64;
                    proportional.max(config.vram_reserve_min)
                };
                // Prefer live free VRAM telemetry if available (from NVML)
                let live_free = gpu.vram_free_bytes;
                if live_free > 0 && live_free < total_dedicated_vram {
                    // Use live free as-is (already accounts for current usage)
                    live_free
                } else {
                    total_dedicated_vram.saturating_sub(reserve)
                }
            } else {
                0
            };

            // Shared system memory budget
            // Note: shared memory competes with system RAM.
            // We apply a conservative fraction to avoid double-counting.
            // For dGPUs: shared memory CAN be used by some backends (llama.cpp
            // CPU offload uses system RAM which is the same physical resource).
            // We model the pool but the scorer decides per-backend whether to use it.
            let usable_shared_memory = if total_shared_memory > 0 {
                (total_shared_memory as f64 * config.shared_memory_usable_fraction) as u64
            } else {
                0
            };

            GpuMemoryBudget {
                gpu_index: idx,
                gpu_model: gpu.model.clone(),
                gpu_type,
                total_dedicated_vram,
                usable_dedicated_vram,
                total_shared_memory,
                usable_shared_memory,
                cuda_available: gpu.cuda_supported,
                rocm_available: gpu.rocm_supported,
                vulkan_available: gpu.vulkan_supported,
                directml_available: gpu.directx_supported,
                compute_capability: gpu.compute_capability.clone(),
            }
        })
        .collect();

    log::info!(
        "[RECOMMENDATION] Budget: RAM {:.1} GB usable / {:.1} GB total, {} GPU(s)",
        usable_for_inference as f64 / 1_073_741_824.0,
        total_ram as f64 / 1_073_741_824.0,
        gpu_budgets.len()
    );
    for gb in &gpu_budgets {
        log::info!(
            "[RECOMMENDATION]   GPU#{}: {} ({:?}) — VRAM {:.1} GB usable, Shared {:.1} GB usable",
            gb.gpu_index,
            gb.gpu_model,
            gb.gpu_type,
            gb.usable_dedicated_vram as f64 / 1_073_741_824.0,
            gb.usable_shared_memory as f64 / 1_073_741_824.0,
        );
    }

    MemoryBudget {
        gpu_budgets,
        system_ram,
    }
}

// ─── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_analyzer::traits::*;

    fn make_test_profile(
        total_ram: u64,
        available_ram: u64,
        gpus: Vec<GpuInfo>,
    ) -> HardwareProfile {
        HardwareProfile {
            id: "test".to_string(),
            profile_created_at: "2026-01-01T00:00:00Z".to_string(),
            profile_updated_at: "2026-01-01T00:00:00Z".to_string(),
            cpu: OverrideValue::new(CpuInfo {
                manufacturer: "Test".to_string(),
                model: "Test CPU".to_string(),
                architecture: "x86_64".to_string(),
                physical_cores: 8,
                logical_processors: 16,
                base_frequency_mhz: 3300,
                boost_frequency_mhz: 0,
                cache_l1_kb: None,
                cache_l2_kb: None,
                cache_l3_kb: None,
                virtualization_supported: true,
                simd_capabilities: vec!["AVX2".to_string()],
            }),
            gpus: OverrideValue::new(gpus),
            memory: OverrideValue::new(MemoryInfo {
                total_bytes: total_ram,
                available_bytes: available_ram,
                used_bytes: total_ram - available_ram,
                memory_type: "DDR5".to_string(),
                speed_mts: Some(5600),
                total_slots: Some(2),
                populated_slots: Some(2),
            }),
            storage: OverrideValue::new(vec![]),
            os: OverrideValue::new(OsInfo {
                name: "Windows".to_string(),
                edition: "11 Home".to_string(),
                version: "24H2".to_string(),
                build_number: "26200".to_string(),
                architecture: "x86_64".to_string(),
                locale: "en-US".to_string(),
            }),
            software: OverrideValue::new(SoftwareEnvironment {
                python: SoftwareDetectorInfo { name: "Python".to_string(), installed: false, version: None, path: None },
                rust: SoftwareDetectorInfo { name: "Rust".to_string(), installed: false, version: None, path: None },
                cargo: SoftwareDetectorInfo { name: "Cargo".to_string(), installed: false, version: None, path: None },
                git: SoftwareDetectorInfo { name: "Git".to_string(), installed: false, version: None, path: None },
                nodejs: SoftwareDetectorInfo { name: "Node.js".to_string(), installed: false, version: None, path: None },
                npm: SoftwareDetectorInfo { name: "npm".to_string(), installed: false, version: None, path: None },
                pnpm: SoftwareDetectorInfo { name: "pnpm".to_string(), installed: false, version: None, path: None },
                ollama: SoftwareDetectorInfo { name: "Ollama".to_string(), installed: false, version: None, path: None },
                cuda_toolkit: SoftwareDetectorInfo { name: "CUDA".to_string(), installed: false, version: None, path: None },
                vc_redistributable: SoftwareDetectorInfo { name: "VC++".to_string(), installed: false, version: None, path: None },
                additional: vec![],
            }),
            ai_runtimes: OverrideValue::new(vec![]),
            paths: SystemPaths {
                user_home: "C:\\Users\\test".to_string(),
                downloads: "C:\\Users\\test\\Downloads".to_string(),
                documents: "C:\\Users\\test\\Documents".to_string(),
                desktop: "C:\\Users\\test\\Desktop".to_string(),
                app_data: "C:\\Users\\test\\AppData".to_string(),
                cache_dir: "C:\\Users\\test\\.cache".to_string(),
                model_storage_dir: "C:\\Users\\test\\.sarathi\\models".to_string(),
            },
            ai_capabilities: AICapabilityProfile {
                max_recommended_model_size_bytes: None,
                recommended_quantizations: vec![],
                recommended_context_length: None,
                preferred_inference_backend: None,
                multi_model_capable: false,
                lora_ready: false,
                vision_ready: false,
                embedding_ready: false,
                extra_capabilities: std::collections::HashMap::new(),
            },
            validation: SystemValidationResult {
                is_ready_for_ai: false,
                score: 0,
                warnings: vec![],
                errors: vec![],
                recommendations: vec![],
            },
        }
    }

    fn make_gpu(model: &str, is_dedicated: bool, vram: u64, shared: u64, cuda: bool) -> GpuInfo {
        GpuInfo {
            vendor: if cuda { "NVIDIA".to_string() } else { "AMD".to_string() },
            model: model.to_string(),
            gpu_type: if is_dedicated { "Dedicated".to_string() } else { "Integrated".to_string() },
            is_dedicated,
            dedicated_video_memory_bytes: vram,
            dedicated_system_memory_bytes: 0,
            shared_system_memory_bytes: shared,
            total_available_graphics_memory_bytes: vram + shared,
            vram_total_bytes: vram,
            vram_free_bytes: 0,
            driver_version: None,
            vendor_id: None,
            device_id: None,
            compute_capability: None,
            cuda_supported: cuda,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: true,
            opencl_supported: true,
            detection_source: "Test".to_string(),
            confidence: "High".to_string(),
        }
    }

    const GB: u64 = 1_073_741_824;

    #[test]
    fn test_budget_low_ram_cpu_only() {
        let profile = make_test_profile(4 * GB, 2 * GB, vec![]);
        let budget = calculate_budget(&profile, &BudgetConfig::default());
        assert_eq!(budget.gpu_budgets.len(), 0);
        // OS reserve = max(4GB * 0.10, 1GB) = 1GB. Sarathi = 256MB.
        // Theoretical = 4GB - 1GB - 256MB = 2.75 GB
        // Live = 2GB - 256MB = 1.75 GB
        // Usable = min(2.75, 1.75) = 1.75 GB
        assert!(budget.system_ram.usable_for_inference > GB);
        assert!(budget.system_ram.usable_for_inference < 2 * GB);
    }

    #[test]
    fn test_budget_igpu_only() {
        let igpu = make_gpu("Intel UHD 770", false, 512 * 1024 * 1024, 8 * GB, false);
        let profile = make_test_profile(16 * GB, 10 * GB, vec![igpu]);
        let budget = calculate_budget(&profile, &BudgetConfig::default());
        assert_eq!(budget.gpu_budgets.len(), 1);
        let gb0 = &budget.gpu_budgets[0];
        assert_eq!(gb0.gpu_type, GpuType::Integrated);
        // iGPU shared usable = 8GB * 0.50 = 4 GB
        assert_eq!(gb0.usable_shared_memory, 4 * GB);
        // Small BIOS buffer: 512MB - max(512MB * 0.10, 256MB) = 512MB - 256MB = 256MB
        assert!(gb0.usable_dedicated_vram <= 512 * 1024 * 1024);
    }

    #[test]
    fn test_budget_nvidia_dgpu() {
        let dgpu = make_gpu("RTX 3060", true, 12 * GB, 16 * GB, true);
        let profile = make_test_profile(16 * GB, 10 * GB, vec![dgpu]);
        let budget = calculate_budget(&profile, &BudgetConfig::default());
        assert_eq!(budget.gpu_budgets.len(), 1);
        let gb0 = &budget.gpu_budgets[0];
        assert_eq!(gb0.gpu_type, GpuType::Dedicated);
        assert!(gb0.cuda_available);
        // VRAM reserve = max(12GB * 0.10, 256MB) = 1.2GB
        // Usable VRAM = 12GB - 1.2GB = 10.8GB
        assert!(gb0.usable_dedicated_vram > 10 * GB);
        assert!(gb0.usable_dedicated_vram < 12 * GB);
        // dGPU shared memory is modeled separately (not forced to zero)
        assert!(gb0.usable_shared_memory > 0);
    }

    #[test]
    fn test_budget_hybrid_igpu_dgpu() {
        let dgpu = make_gpu("RTX 5060", true, 8 * GB, 12 * GB, true);
        let igpu = make_gpu("Radeon 780M", false, 512 * 1024 * 1024, 12 * GB, false);
        let profile = make_test_profile(24 * GB, 14 * GB, vec![dgpu, igpu]);
        let budget = calculate_budget(&profile, &BudgetConfig::default());
        assert_eq!(budget.gpu_budgets.len(), 2);
        assert_eq!(budget.gpu_budgets[0].gpu_type, GpuType::Dedicated);
        assert_eq!(budget.gpu_budgets[1].gpu_type, GpuType::Integrated);
        // Each GPU has independent budgets
        assert!(budget.gpu_budgets[0].usable_dedicated_vram > 6 * GB);
        assert!(budget.gpu_budgets[1].usable_shared_memory > 4 * GB);
    }

    #[test]
    fn test_budget_multi_gpu() {
        let gpu0 = make_gpu("RTX 3090", true, 24 * GB, 32 * GB, true);
        let gpu1 = make_gpu("RTX 3090", true, 24 * GB, 32 * GB, true);
        let profile = make_test_profile(64 * GB, 50 * GB, vec![gpu0, gpu1]);
        let budget = calculate_budget(&profile, &BudgetConfig::default());
        assert_eq!(budget.gpu_budgets.len(), 2);
        // Each GPU gets its own independent budget
        assert!(budget.gpu_budgets[0].usable_dedicated_vram > 20 * GB);
        assert!(budget.gpu_budgets[1].usable_dedicated_vram > 20 * GB);
    }
}

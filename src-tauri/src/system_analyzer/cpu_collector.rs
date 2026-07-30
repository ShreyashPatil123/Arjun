//! CPU specs collector using sysinfo and system APIs

use crate::system_analyzer::traits::CpuInfo;
use sysinfo::{CpuRefreshKind, System};

/// Detects system CPU details
pub fn detect_cpu() -> CpuInfo {
    let mut sys = System::new();
    sys.refresh_cpu_specifics(CpuRefreshKind::everything());

    let cpus = sys.cpus();

    let model = if !cpus.is_empty() {
        let brand = cpus[0].brand().trim();
        if brand.is_empty() {
            "Unknown".to_string()
        } else {
            brand.to_string()
        }
    } else {
        "Unknown".to_string()
    };

    let manufacturer = if !cpus.is_empty() {
        let vendor = cpus[0].vendor_id().trim();
        if vendor.is_empty() {
            let model_lower = model.to_lowercase();
            if model_lower.contains("intel") {
                "Intel".to_string()
            } else if model_lower.contains("amd") {
                "AMD".to_string()
            } else if model_lower.contains("apple") {
                "Apple".to_string()
            } else {
                "Unknown".to_string()
            }
        } else {
            vendor.to_string()
        }
    } else {
        "Unknown".to_string()
    };

    let architecture = std::env::consts::ARCH.to_string();
    let logical_processors = sys.cpus().len() as u32;
    let physical_cores = sys.physical_core_count().unwrap_or(logical_processors as usize) as u32;

    let base_frequency_mhz = if !cpus.is_empty() {
        cpus[0].frequency()
    } else {
        0
    };

    let boost_frequency_mhz = cpus
        .iter()
        .map(|c| c.frequency())
        .max()
        .unwrap_or(base_frequency_mhz);

    let mut simd_capabilities = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            simd_capabilities.push("SSE".to_string());
        }
        if is_x86_feature_detected!("sse2") {
            simd_capabilities.push("SSE2".to_string());
        }
        if is_x86_feature_detected!("sse3") {
            simd_capabilities.push("SSE3".to_string());
        }
        if is_x86_feature_detected!("ssse3") {
            simd_capabilities.push("SSSE3".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            simd_capabilities.push("SSE4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            simd_capabilities.push("SSE4.2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            simd_capabilities.push("AVX".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            simd_capabilities.push("AVX2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            simd_capabilities.push("FMA".to_string());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        simd_capabilities.push("NEON".to_string());
    }

    let (cache_l1_kb, cache_l2_kb, cache_l3_kb, virtualization_supported) =
        query_cpu_cache_and_virt();

    CpuInfo {
        manufacturer,
        model,
        architecture,
        physical_cores: if physical_cores == 0 { 1 } else { physical_cores },
        logical_processors: if logical_processors == 0 { 1 } else { logical_processors },
        base_frequency_mhz,
        boost_frequency_mhz,
        cache_l1_kb,
        cache_l2_kb,
        cache_l3_kb,
        virtualization_supported,
        simd_capabilities,
    }
}

fn query_cpu_cache_and_virt() -> (Option<u32>, Option<u32>, Option<u32>, bool) {
    let mut virt = false;
    let l1 = None;
    let mut l2 = None;
    let mut l3 = None;

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("wmic")
            .args([
                "cpu",
                "get",
                "L2CacheSize,L3CacheSize,VirtualizationFirmwareEnabled",
                "/format:list",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let line = line.trim();
                if let Some((k, v)) = line.split_once('=') {
                    match k.trim() {
                        "L2CacheSize" => {
                            if let Ok(val) = v.trim().parse::<u32>() {
                                if val > 0 {
                                    l2 = Some(val);
                                }
                            }
                        }
                        "L3CacheSize" => {
                            if let Ok(val) = v.trim().parse::<u32>() {
                                if val > 0 {
                                    l3 = Some(val);
                                }
                            }
                        }
                        "VirtualizationFirmwareEnabled" => {
                            if v.trim().eq_ignore_ascii_case("TRUE") {
                                virt = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    (l1, l2, l3, virt)
}

//! CPU specs collector using sysinfo and PowerShell CIM queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::CpuInfo;
use serde::Deserialize;
use sysinfo::{CpuRefreshKind, System};

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimProcessorInfo {
    name: Option<String>,
    manufacturer: Option<String>,
    number_of_cores: Option<u32>,
    number_of_logical_processors: Option<u32>,
    max_clock_speed: Option<u32>,
    virtualization_firmware_enabled: Option<bool>,
}

/// Detects system CPU details
pub fn detect_cpu() -> CpuInfo {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 CPU Collector Started");

    let mut sys = System::new_all();
    sys.refresh_cpu_specifics(CpuRefreshKind::everything());

    let cpus = sys.cpus();

    let mut model = if !cpus.is_empty() {
        cpus[0].brand().trim().to_string()
    } else {
        String::new()
    };

    let mut manufacturer = if !cpus.is_empty() {
        cpus[0].vendor_id().trim().to_string()
    } else {
        String::new()
    };

    let mut logical_processors = sys.cpus().len() as u32;
    let mut physical_cores = sys.physical_core_count().unwrap_or(logical_processors as usize) as u32;
    let mut base_frequency_mhz = if !cpus.is_empty() { cpus[0].frequency() } else { 0 };
    let mut boost_frequency_mhz = cpus.iter().map(|c| c.frequency()).max().unwrap_or(base_frequency_mhz);
    let mut virtualization_supported = false;

    // Fallback/enrich via PowerShell Get-CimInstance on Windows
    #[cfg(target_os = "windows")]
    {
        if let Ok(cim_proc) = query_cim_processor() {
            log::info!("[SYSTEM ANALYZER DEBUG] ✓ CIM Processor query succeeded: {:?}", cim_proc);
            if model.is_empty() || model == "Unknown" {
                if let Some(n) = cim_proc.name {
                    model = n.trim().to_string();
                }
            }
            if manufacturer.is_empty() || manufacturer == "Unknown" {
                if let Some(m) = cim_proc.manufacturer {
                    manufacturer = m.trim().to_string();
                }
            }
            if physical_cores == 0 {
                if let Some(c) = cim_proc.number_of_cores {
                    physical_cores = c;
                }
            }
            if logical_processors == 0 {
                if let Some(lp) = cim_proc.number_of_logical_processors {
                    logical_processors = lp;
                }
            }
            if base_frequency_mhz == 0 {
                if let Some(spd) = cim_proc.max_clock_speed {
                    base_frequency_mhz = spd as u64;
                    if boost_frequency_mhz == 0 {
                        boost_frequency_mhz = spd as u64;
                    }
                }
            }
            if let Some(virt) = cim_proc.virtualization_firmware_enabled {
                virtualization_supported = virt;
            }
        } else {
            log::warn!("[SYSTEM ANALYZER DEBUG] ⚠️ CIM Processor query failed or empty");
        }
    }

    if model.is_empty() {
        model = "Unknown".to_string();
    }
    if manufacturer.is_empty() {
        let model_lower = model.to_lowercase();
        if model_lower.contains("intel") {
            manufacturer = "Intel".to_string();
        } else if model_lower.contains("amd") || model_lower.contains("ryzen") {
            manufacturer = "AMD".to_string();
        } else if model_lower.contains("apple") {
            manufacturer = "Apple".to_string();
        } else {
            manufacturer = "Unknown".to_string();
        }
    }

    let architecture = std::env::consts::ARCH.to_string();

    let mut simd_capabilities = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") { simd_capabilities.push("SSE".to_string()); }
        if is_x86_feature_detected!("sse2") { simd_capabilities.push("SSE2".to_string()); }
        if is_x86_feature_detected!("sse3") { simd_capabilities.push("SSE3".to_string()); }
        if is_x86_feature_detected!("ssse3") { simd_capabilities.push("SSSE3".to_string()); }
        if is_x86_feature_detected!("sse4.1") { simd_capabilities.push("SSE4.1".to_string()); }
        if is_x86_feature_detected!("sse4.2") { simd_capabilities.push("SSE4.2".to_string()); }
        if is_x86_feature_detected!("avx") { simd_capabilities.push("AVX".to_string()); }
        if is_x86_feature_detected!("avx2") { simd_capabilities.push("AVX2".to_string()); }
        if is_x86_feature_detected!("fma") { simd_capabilities.push("FMA".to_string()); }
    }
    #[cfg(target_arch = "aarch64")]
    {
        simd_capabilities.push("NEON".to_string());
    }

    let info = CpuInfo {
        manufacturer,
        model: model.clone(),
        architecture,
        physical_cores: if physical_cores == 0 { 1 } else { physical_cores },
        logical_processors: if logical_processors == 0 { 1 } else { logical_processors },
        base_frequency_mhz,
        boost_frequency_mhz,
        cache_l1_kb: None,
        cache_l2_kb: None,
        cache_l3_kb: None,
        virtualization_supported,
        simd_capabilities,
    };

    log::info!("[SYSTEM ANALYZER DEBUG] ✓ CPU Detection Finished: {}, Virtualization={}", model, virtualization_supported);
    info
}

#[cfg(target_os = "windows")]
fn query_cim_processor() -> Result<CimProcessorInfo, String> {
    let output = create_hidden_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, VirtualizationFirmwareEnabled | ConvertTo-Json",
        ])
        .output()
        .map_err(|e| format!("Failed to spawn powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_Processor returned non-zero status".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("powershell returned empty stdout".to_string());
    }

    if let Ok(info) = serde_json::from_str::<CimProcessorInfo>(&stdout) {
        Ok(info)
    } else if let Ok(list) = serde_json::from_str::<Vec<CimProcessorInfo>>(&stdout) {
        if let Some(first) = list.into_iter().next() {
            Ok(first)
        } else {
            Err("Empty list from CIM".to_string())
        }
    } else {
        Err(format!("JSON parse error from stdout: {}", stdout))
    }
}

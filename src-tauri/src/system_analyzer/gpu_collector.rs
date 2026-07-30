//! GPU details collector using CLI tools (nvidia-smi) and PowerShell CIM queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::GpuInfo;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimGpuInfo {
    name: Option<String>,
    adapter_r_a_m: Option<u64>,
    driver_version: Option<String>,
}

/// Detects available GPUs on the system
pub fn detect_gpus() -> Vec<GpuInfo> {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 GPU Collector Started");
    let mut gpus = Vec::new();

    // 1. Query NVIDIA GPUs via nvidia-smi if available
    if let Ok(nvidia_gpus) = query_nvidia_smi() {
        if !nvidia_gpus.is_empty() {
            log::info!("[SYSTEM ANALYZER DEBUG] ✓ NVIDIA GPU detected via nvidia-smi: {} GPUs found", nvidia_gpus.len());
            gpus.extend(nvidia_gpus);
        }
    }

    // 2. Query PowerShell CIM Win32_VideoController for all installed GPUs (Integrated + Dedicated)
    #[cfg(target_os = "windows")]
    {
        if let Ok(cim_gpus) = query_cim_videocontroller() {
            log::info!("[SYSTEM ANALYZER DEBUG] ✓ CIM VideoController query returned {} GPUs", cim_gpus.len());
            for c_gpu in cim_gpus {
                // Avoid duplicate NVIDIA entries if already detected with full VRAM metrics from nvidia-smi
                if !gpus.iter().any(|g| g.model.eq_ignore_ascii_case(&c_gpu.model)) {
                    gpus.push(c_gpu);
                }
            }
        } else {
            log::warn!("[SYSTEM ANALYZER DEBUG] ⚠️ CIM VideoController query failed");
        }
    }

    // Fallback if no GPU detected
    if gpus.is_empty() {
        log::warn!("[SYSTEM ANALYZER DEBUG] ⚠️ Zero GPUs detected by system APIs");
        gpus.push(GpuInfo {
            vendor: "Unknown".to_string(),
            model: "Unknown".to_string(),
            is_dedicated: false,
            vram_total_bytes: 0,
            vram_free_bytes: 0,
            driver_version: None,
            compute_capability: None,
            cuda_supported: false,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: false,
            opencl_supported: false,
        });
    }

    for (idx, gpu) in gpus.iter().enumerate() {
        log::info!("[SYSTEM ANALYZER DEBUG] ✓ GPU #{}: Vendor={}, Model={}, Dedicated={}, VRAM={} MB",
            idx + 1, gpu.vendor, gpu.model, gpu.is_dedicated, gpu.vram_total_bytes / (1024 * 1024));
    }

    gpus
}

fn query_nvidia_smi() -> Result<Vec<GpuInfo>, String> {
    let output = create_hidden_command("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("Failed nvidia-smi execution: {}", e))?;

    if !output.status.success() {
        return Err("nvidia-smi non-zero exit code".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut result = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 5 {
            let model = parts[0].to_string();
            let driver_version = Some(parts[1].to_string());
            let total_mb: u64 = parts[2].parse().unwrap_or(0);
            let free_mb: u64 = parts[3].parse().unwrap_or(0);
            let compute_capability = Some(parts[4].to_string());

            result.push(GpuInfo {
                vendor: "NVIDIA".to_string(),
                model,
                is_dedicated: true,
                vram_total_bytes: total_mb * 1024 * 1024,
                vram_free_bytes: free_mb * 1024 * 1024,
                driver_version,
                compute_capability,
                cuda_supported: true,
                rocm_supported: false,
                directx_supported: true,
                vulkan_supported: true,
                opencl_supported: true,
            });
        }
    }

    Ok(result)
}

#[cfg(target_os = "windows")]
fn query_cim_videocontroller() -> Result<Vec<GpuInfo>, String> {
    let output = create_hidden_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json",
        ])
        .output()
        .map_err(|e| format!("Failed powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_VideoController failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("empty stdout from powershell".to_string());
    }

    let mut raw_items = Vec::new();
    if let Ok(item) = serde_json::from_str::<CimGpuInfo>(&stdout) {
        raw_items.push(item);
    } else if let Ok(list) = serde_json::from_str::<Vec<CimGpuInfo>>(&stdout) {
        raw_items = list;
    } else {
        return Err(format!("Failed to parse GPU CIM JSON: {}", stdout));
    }

    let mut gpus = Vec::new();
    for item in raw_items {
        if let Some(name) = item.name {
            let model = name.trim().to_string();
            let name_lower = model.to_lowercase();

            let vendor = if name_lower.contains("nvidia") {
                "NVIDIA".to_string()
            } else if name_lower.contains("amd") || name_lower.contains("radeon") {
                "AMD".to_string()
            } else if name_lower.contains("intel") {
                "Intel".to_string()
            } else {
                "Unknown".to_string()
            };

            let is_dedicated = !name_lower.contains("intel uhd")
                && !name_lower.contains("intel hd")
                && !name_lower.contains("iris xe")
                && !name_lower.contains("radeon graphics");

            let ram_bytes = item.adapter_r_a_m.unwrap_or(0);
            let is_cuda = vendor == "NVIDIA";
            let is_rocm = vendor == "AMD";

            gpus.push(GpuInfo {
                vendor,
                model,
                is_dedicated,
                vram_total_bytes: ram_bytes,
                vram_free_bytes: ram_bytes / 2,
                driver_version: item.driver_version,
                compute_capability: None,
                cuda_supported: is_cuda,
                rocm_supported: is_rocm,
                directx_supported: true,
                vulkan_supported: true,
                opencl_supported: true,
            });
        }
    }

    Ok(gpus)
}

//! GPU details collector using CLI tools (nvidia-smi, rocm-smi) and WMI/system tools

use crate::system_analyzer::traits::GpuInfo;
use std::process::Command;

/// Detects available GPUs on the system
pub fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // 1. Try nvidia-smi for NVIDIA GPUs
    if let Ok(nvidia_gpus) = query_nvidia_smi() {
        if !nvidia_gpus.is_empty() {
            gpus.extend(nvidia_gpus);
        }
    }

    // 2. Query WMI win32_videocontroller for all installed GPUs (Intel, AMD, NVIDIA)
    if let Ok(wmi_gpus) = query_wmi_videocontroller() {
        for w_gpu in wmi_gpus {
            // Avoid duplicate NVIDIA entries if already added by nvidia-smi
            if !gpus.iter().any(|g| g.model.eq_ignore_ascii_case(&w_gpu.model)) {
                gpus.push(w_gpu);
            }
        }
    }

    // Fallback if no GPU detected
    if gpus.is_empty() {
        gpus.push(GpuInfo {
            vendor: "Unknown".to_string(),
            model: "Generic Display Adapter".to_string(),
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

    gpus
}

fn query_nvidia_smi() -> Result<Vec<GpuInfo>, ()> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|_| ())?;

    if !output.status.success() {
        return Err(());
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

fn query_wmi_videocontroller() -> Result<Vec<GpuInfo>, ()> {
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("wmic")
            .args([
                "path",
                "win32_videocontroller",
                "get",
                "Name,AdapterRAM,DriverVersion",
                "/format:list",
            ])
            .output()
            .map_err(|_| ())?;

        if !output.status.success() {
            return Err(());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        let mut current_name = String::new();
        let mut current_ram: u64 = 0;
        let mut current_driver = None;

        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                if !current_name.is_empty() {
                    let name_lower = current_name.to_lowercase();
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

                    let is_cuda = vendor == "NVIDIA";
                    let is_rocm = vendor == "AMD";

                    gpus.push(GpuInfo {
                        vendor,
                        model: current_name.clone(),
                        is_dedicated,
                        vram_total_bytes: current_ram,
                        vram_free_bytes: current_ram / 2, // Estimated available
                        driver_version: current_driver.clone(),
                        compute_capability: None,
                        cuda_supported: is_cuda,
                        rocm_supported: is_rocm,
                        directx_supported: true,
                        vulkan_supported: true,
                        opencl_supported: true,
                    });

                    current_name.clear();
                    current_ram = 0;
                    current_driver = None;
                }
                continue;
            }

            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "Name" => current_name = v.trim().to_string(),
                    "AdapterRAM" => current_ram = v.trim().parse::<u64>().unwrap_or(0),
                    "DriverVersion" => current_driver = Some(v.trim().to_string()),
                    _ => {}
                }
            }
        }

        if !current_name.is_empty() {
            let name_lower = current_name.to_lowercase();
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

            let is_cuda = vendor == "NVIDIA";
            let is_rocm = vendor == "AMD";

            gpus.push(GpuInfo {
                vendor,
                model: current_name,
                is_dedicated,
                vram_total_bytes: current_ram,
                vram_free_bytes: current_ram / 2,
                driver_version: current_driver,
                compute_capability: None,
                cuda_supported: is_cuda,
                rocm_supported: is_rocm,
                directx_supported: true,
                vulkan_supported: true,
                opencl_supported: true,
            });
        }

        Ok(gpus)
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err(())
    }
}

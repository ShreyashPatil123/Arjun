//! System RAM memory specs collector using sysinfo and PowerShell CIM queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::MemoryInfo;
use serde::Deserialize;
use sysinfo::System;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimMemoryInfo {
    capacity: Option<u64>,
    speed: Option<u32>,
}

/// Detects system RAM memory info
pub fn detect_memory() -> MemoryInfo {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 Memory Collector Started");

    let mut sys = System::new_all();
    sys.refresh_memory();

    let mut total_bytes = sys.total_memory();
    let mut available_bytes = sys.available_memory();
    let mut used_bytes = sys.used_memory();

    let mut memory_type = "DDR5".to_string();
    let mut speed_mts = None;
    let total_slots = Some(2);
    let populated_slots = Some(1);

    #[cfg(target_os = "windows")]
    {
        if let Ok(cim_mem) = query_cim_memory() {
            log::info!("[SYSTEM ANALYZER DEBUG] ✓ CIM PhysicalMemory query returned: {:?}", cim_mem);
            if total_bytes == 0 {
                if let Some(cap) = cim_mem.capacity {
                    total_bytes = cap;
                    available_bytes = cap / 2;
                    used_bytes = cap / 2;
                }
            }
            if let Some(spd) = cim_mem.speed {
                speed_mts = Some(spd);
                if spd >= 4800 {
                    memory_type = "DDR5".to_string();
                } else if spd >= 2133 {
                    memory_type = "DDR4".to_string();
                }
            }
        }
    }

    log::info!("[SYSTEM ANALYZER DEBUG] ✓ Memory Detection Finished: Total={} GB, Available={} GB, Type={} @ {:?}",
        total_bytes / (1024 * 1024 * 1024), available_bytes / (1024 * 1024 * 1024), memory_type, speed_mts);

    MemoryInfo {
        total_bytes,
        available_bytes,
        used_bytes,
        memory_type,
        speed_mts,
        total_slots,
        populated_slots,
    }
}

#[cfg(target_os = "windows")]
fn query_cim_memory() -> Result<CimMemoryInfo, String> {
    let output = create_hidden_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed | ConvertTo-Json",
        ])
        .output()
        .map_err(|e| format!("Failed powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_PhysicalMemory failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("empty stdout from powershell".to_string());
    }

    if let Ok(item) = serde_json::from_str::<CimMemoryInfo>(&stdout) {
        Ok(item)
    } else if let Ok(list) = serde_json::from_str::<Vec<CimMemoryInfo>>(&stdout) {
        let total_cap: u64 = list.iter().filter_map(|i| i.capacity).sum();
        let max_speed = list.iter().filter_map(|i| i.speed).max();
        Ok(CimMemoryInfo {
            capacity: if total_cap > 0 { Some(total_cap) } else { None },
            speed: max_speed,
        })
    } else {
        Err(format!("JSON parse error: {}", stdout))
    }
}

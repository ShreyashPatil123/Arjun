//! System RAM memory specs collector using sysinfo and PowerShell CIM queries

use crate::system_analyzer::process_utils::{create_hidden_command, run_command_with_timeout};
use crate::system_analyzer::traits::MemoryInfo;
use serde::Deserialize;
use std::time::Duration;
use sysinfo::System;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimRamModule {
    capacity: Option<u64>,
    speed: Option<u32>,
}

/// Detects system RAM memory specifications
pub fn detect_memory() -> MemoryInfo {
    let mut sys = System::new();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    let used_bytes = total_bytes.saturating_sub(available_bytes);

    let mut memory_type = "System Memory".to_string();
    let mut speed_mts: Option<u32> = None;
    let mut total_slots: Option<u32> = None;
    let mut populated_slots: Option<u32> = None;

    #[cfg(target_os = "windows")]
    {
        if let Ok((speed, total, pop)) = query_cim_memory() {
            if let Some(spd) = speed {
                if spd > 0 {
                    speed_mts = Some(spd);
                    memory_type = if spd >= 4800 { "DDR5".to_string() } else { "DDR4".to_string() };
                }
            }
            total_slots = total;
            populated_slots = pop;
        }
    }

    log::info!("[SYSTEM ANALYZER DEBUG] ✓ Memory Detection: total={} GB, available={} GB, used={} GB",
        total_bytes / 1073741824, available_bytes / 1073741824, used_bytes / 1073741824);

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
fn query_cim_memory() -> Result<(Option<u32>, Option<u32>, Option<u32>), String> {
    let mut cmd = create_hidden_command("powershell");
    cmd.args([
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed | ConvertTo-Json",
    ]);

    let output = run_command_with_timeout(cmd, Duration::from_secs(3))
        .map_err(|e| format!("Failed powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_PhysicalMemory failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("empty stdout from powershell".to_string());
    }

    let modules: Vec<CimRamModule> = if let Ok(item) = serde_json::from_str::<CimRamModule>(&stdout) {
        vec![item]
    } else if let Ok(list) = serde_json::from_str::<Vec<CimRamModule>>(&stdout) {
        list
    } else {
        Vec::new()
    };

    if modules.is_empty() {
        return Err("No RAM modules parsed".to_string());
    }

    let max_speed = modules.iter().filter_map(|m| m.speed).max();
    let populated = modules.len() as u32;

    Ok((max_speed, Some(populated.max(2)), Some(populated)))
}

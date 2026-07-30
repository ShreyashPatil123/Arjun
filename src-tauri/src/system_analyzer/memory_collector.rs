//! System RAM memory specs collector using sysinfo and WMI/OS queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::MemoryInfo;
use sysinfo::System;

/// Detects system RAM memory info
pub fn detect_memory() -> MemoryInfo {
    let mut sys = System::new();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    let used_bytes = sys.used_memory();

    let (memory_type, speed_mts, total_slots, populated_slots) = query_memory_hardware_details();

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

fn query_memory_hardware_details() -> (String, Option<u32>, Option<u32>, Option<u32>) {
    let mut memory_type = "DDR".to_string();
    let mut speed_mts = None;
    let mut total_slots = None;
    let mut populated_slots = None;

    #[cfg(target_os = "windows")]
    {
        // Query memory chip details via WMI
        if let Ok(output) = create_hidden_command("wmic")
            .args([
                "memorychip",
                "get",
                "Speed,MemoryType,SMBIOSMemoryType",
                "/format:list",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut count = 0;
            for line in stdout.lines() {
                let line = line.trim();
                if let Some((k, v)) = line.split_once('=') {
                    match k.trim() {
                        "Speed" => {
                            if let Ok(sp) = v.trim().parse::<u32>() {
                                if sp > 0 {
                                    speed_mts = Some(sp);
                                }
                            }
                        }
                        "SMBIOSMemoryType" => {
                            if let Ok(mt) = v.trim().parse::<u32>() {
                                match mt {
                                    20 => memory_type = "DDR".to_string(),
                                    21 => memory_type = "DDR2".to_string(),
                                    24 => memory_type = "DDR3".to_string(),
                                    26 => memory_type = "DDR4".to_string(),
                                    34 => memory_type = "DDR5".to_string(),
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if line.starts_with("Speed=") {
                    count += 1;
                }
            }
            if count > 0 {
                populated_slots = Some(count);
            }
        }

        // Query physical memory array for slot counts
        if let Ok(output) = create_hidden_command("wmic")
            .args([
                "path",
                "Win32_PhysicalMemoryArray",
                "get",
                "MemoryDevices",
                "/format:list",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some((k, v)) = line.trim().split_once('=') {
                    if k.trim() == "MemoryDevices" {
                        if let Ok(slots) = v.trim().parse::<u32>() {
                            if slots > 0 {
                                total_slots = Some(slots);
                            }
                        }
                    }
                }
            }
        }
    }

    (memory_type, speed_mts, total_slots, populated_slots)
}

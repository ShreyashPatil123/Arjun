//! System storage drives collector using sysinfo and PowerShell CIM queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::StorageInfo;
use serde::Deserialize;
use sysinfo::Disks;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimDiskInfo {
    device_i_d: Option<String>,
    volume_name: Option<String>,
    size: Option<u64>,
    free_space: Option<u64>,
    file_system: Option<String>,
}

/// Detects available storage drives
pub fn detect_storage() -> Vec<StorageInfo> {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 Storage Collector Started");
    let mut storage_list = Vec::new();

    let disks = Disks::new_with_refreshed_list();
    for disk in disks.iter() {
        let name = disk.name().to_string_lossy().to_string();
        let mount_point = disk.mount_point().to_string_lossy().to_string();
        let total_bytes = disk.total_space();
        let free_bytes = disk.available_space();
        let file_system = disk.file_system().to_string_lossy().to_string();
        let drive_type = format!("{:?}", disk.kind());
        let is_ai_storage_ready = free_bytes >= 20 * 1024 * 1024 * 1024; // 20GB+ free

        let drive_name = if name.trim().is_empty() {
            format!("Storage Drive ({})", mount_point)
        } else {
            name
        };

        storage_list.push(StorageInfo {
            drive_name,
            mount_point,
            drive_type,
            total_bytes,
            free_bytes,
            file_system,
            is_ai_storage_ready,
        });
    }

    // Fallback/enrichment via PowerShell CIM Win32_LogicalDisk
    #[cfg(target_os = "windows")]
    {
        if storage_list.is_empty() {
            if let Ok(cim_disks) = query_cim_storage() {
                log::info!("[SYSTEM ANALYZER DEBUG] ✓ CIM LogicalDisk query returned {} drives", cim_disks.len());
                for c_disk in cim_disks {
                    if let Some(dev_id) = c_disk.device_i_d {
                        let total_bytes = c_disk.size.unwrap_or(0);
                        let free_bytes = c_disk.free_space.unwrap_or(0);
                        let is_ai_storage_ready = free_bytes >= 20 * 1024 * 1024 * 1024;

                        storage_list.push(StorageInfo {
                            drive_name: format!("Local Disk ({})", dev_id),
                            mount_point: format!("{}\\", dev_id),
                            drive_type: "SSD/NVMe".to_string(),
                            total_bytes,
                            free_bytes,
                            file_system: c_disk.file_system.unwrap_or_else(|| "NTFS".to_string()),
                            is_ai_storage_ready,
                        });
                    }
                }
            }
        }
    }

    for (idx, drive) in storage_list.iter().enumerate() {
        log::info!("[SYSTEM ANALYZER DEBUG] ✓ Drive #{}: Name={}, Mount={}, Total={} GB, Free={} GB, FS={}",
            idx + 1, drive.drive_name, drive.mount_point, drive.total_bytes / (1024 * 1024 * 1024), drive.free_bytes / (1024 * 1024 * 1024), drive.file_system);
    }

    storage_list
}

#[cfg(target_os = "windows")]
fn query_cim_storage() -> Result<Vec<CimDiskInfo>, String> {
    let output = create_hidden_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID, VolumeName, Size, FreeSpace, FileSystem | ConvertTo-Json",
        ])
        .output()
        .map_err(|e| format!("Failed powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_LogicalDisk failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("empty stdout from powershell".to_string());
    }

    if let Ok(item) = serde_json::from_str::<CimDiskInfo>(&stdout) {
        Ok(vec![item])
    } else if let Ok(list) = serde_json::from_str::<Vec<CimDiskInfo>>(&stdout) {
        Ok(list)
    } else {
        Err(format!("Failed to parse Storage CIM JSON: {}", stdout))
    }
}

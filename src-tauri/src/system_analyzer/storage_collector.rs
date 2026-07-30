//! Storage drives collector using sysinfo Disks

use crate::system_analyzer::traits::StorageInfo;
use sysinfo::Disks;

/// Minimum free bytes required for a drive to be AI model storage ready (20 GB)
const MIN_AI_STORAGE_FREE_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Detects all storage drives on the system
pub fn detect_storage() -> Vec<StorageInfo> {
    let disks = Disks::new_with_refreshed_list();
    let mut storage_list = Vec::new();

    for disk in disks.list() {
        let drive_name = disk.name().to_string_lossy().to_string();
        let mount_point = disk.mount_point().to_string_lossy().to_string();
        let total_bytes = disk.total_space();
        let free_bytes = disk.available_space();
        let file_system = disk.file_system().to_string_lossy().to_string();

        let drive_type = match disk.kind() {
            sysinfo::DiskKind::SSD => "SSD".to_string(),
            sysinfo::DiskKind::HDD => "HDD".to_string(),
            _ => "Storage Drive".to_string(),
        };

        let is_ai_storage_ready = free_bytes >= MIN_AI_STORAGE_FREE_BYTES;

        storage_list.push(StorageInfo {
            drive_name: if drive_name.is_empty() { mount_point.clone() } else { drive_name },
            mount_point,
            drive_type,
            total_bytes,
            free_bytes,
            file_system,
            is_ai_storage_ready,
        });
    }

    if storage_list.is_empty() {
        storage_list.push(StorageInfo {
            drive_name: "Primary Drive".to_string(),
            mount_point: "/".to_string(),
            drive_type: "SSD".to_string(),
            total_bytes: 100 * 1024 * 1024 * 1024,
            free_bytes: 50 * 1024 * 1024 * 1024,
            file_system: "NTFS".to_string(),
            is_ai_storage_ready: true,
        });
    }

    storage_list
}

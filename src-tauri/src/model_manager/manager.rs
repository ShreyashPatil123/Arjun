//! Model Manager Engine
//!
//! Scans installed models directory, provides model deletion functionality,
//! and computes storage usage metrics.

use std::fs;
use std::path::{Path, PathBuf};
use anyhow::Result;
use sysinfo::{Disks, System};

use crate::download_manager::traits::{InstalledModel, StorageSummary};

pub struct ModelManager;

impl ModelManager {
    /// Recursively scans <app_data_dir>/models/ for completed .gguf models
    pub fn list_installed_models(app_data_dir: &Path) -> Vec<InstalledModel> {
        let mut installed = Vec::new();
        let models_dir = app_data_dir.join("models");

        if !models_dir.exists() {
            return installed;
        }

        Self::scan_dir_for_gguf(&models_dir, &mut installed);
        installed
    }

    fn scan_dir_for_gguf(dir: &Path, results: &mut Vec<InstalledModel>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::scan_dir_for_gguf(&path, results);
                } else if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "gguf" {
                            if let Ok(meta) = fs::metadata(&path) {
                                let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                                
                                // Extract metadata from path hierarchy: models/<provider>/<clean_model_id>/<quantization>/filename.gguf
                                let components: Vec<String> = path
                                    .iter()
                                    .map(|c| c.to_string_lossy().to_string())
                                    .collect();

                                let len = components.len();
                                let (provider_id, model_id, quantization) = if len >= 4 {
                                    (
                                        components[len - 4].clone(),
                                        components[len - 3].replace('_', "/"),
                                        components[len - 2].clone(),
                                    )
                                } else {
                                    ("huggingface".to_string(), file_name.clone(), "GGUF".to_string())
                                };

                                let model_name = model_id.split('/').last().unwrap_or(&model_id).to_string();

                                results.push(InstalledModel {
                                    id: format!("{}_{}", model_id.replace('/', "_"), quantization),
                                    model_id,
                                    model_name,
                                    provider_id,
                                    quantization,
                                    format: "GGUF".to_string(),
                                    backend: "llama.cpp (GGUF)".to_string(),
                                    file_name,
                                    file_path: path.to_string_lossy().to_string(),
                                    size_bytes: meta.len(),
                                    installed_at: chrono::Utc::now().to_rfc3339(),
                                    is_ready: meta.len() > 0,
                                    checksum: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Deletes an installed model directory and files from disk
    pub fn delete_installed_model(app_data_dir: &Path, provider_id: &str, model_id: &str, quantization: &str) -> Result<()> {
        let clean_model_id = model_id.replace('/', "_");
        let target_dir = app_data_dir
            .join("models")
            .join(provider_id)
            .join(clean_model_id)
            .join(quantization);

        if target_dir.exists() {
            fs::remove_dir_all(&target_dir)?;
            log::info!("[MODEL_MANAGER] ✓ Deleted model directory: {:?}", target_dir);
        }

        Ok(())
    }

    /// Computes summary of storage usage
    pub fn get_storage_summary(app_data_dir: &Path) -> StorageSummary {
        let installed = Self::list_installed_models(app_data_dir);
        let total_models_bytes: u64 = installed.iter().map(|m| m.size_bytes).sum();

        let disks = Disks::new_with_refreshed_list();

        let mut available_disk_space_bytes = 0;
        let mut total_disk_space_bytes = 0;

        let models_dir = app_data_dir.join("models");
        let path_str = models_dir.to_string_lossy();
        let drive_prefix = if path_str.len() >= 3 && &path_str[1..3] == ":\\" {
            &path_str[0..3]
        } else {
            "C:\\"
        };

        for disk in &disks {
            let mount = disk.mount_point().to_string_lossy();
            if mount.eq_ignore_ascii_case(drive_prefix) || path_str.starts_with(mount.as_ref()) {
                available_disk_space_bytes = disk.available_space();
                total_disk_space_bytes = disk.total_space();
                break;
            }
        }

        StorageSummary {
            models_directory: models_dir.to_string_lossy().to_string(),
            total_installed_models: installed.len(),
            total_models_bytes,
            available_disk_space_bytes,
            total_disk_space_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_installed_models_empty() {
        let temp_dir = std::env::temp_dir().join("sarathi_test_models_empty");
        let installed = ModelManager::list_installed_models(&temp_dir);
        assert_eq!(installed.len(), 0);
    }
}

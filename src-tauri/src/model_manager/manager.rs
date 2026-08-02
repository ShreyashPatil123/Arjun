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
    /// Recursively scans <app_data_dir>/models/ for model packages
    pub fn list_installed_models(app_data_dir: &Path) -> Vec<InstalledModel> {
        let mut installed = Vec::new();
        let models_dir = app_data_dir.join("models");

        if !models_dir.exists() {
            return installed;
        }

        // Iterate over provider directories: models/<provider>/<sanitized_model_id>/
        if let Ok(providers) = fs::read_dir(&models_dir) {
            for provider_entry in providers.flatten() {
                let provider_path = provider_entry.path();
                if provider_path.is_dir() {
                    let provider_id = provider_path.file_name().unwrap_or_default().to_string_lossy().to_string();

                    if let Ok(packages) = fs::read_dir(&provider_path) {
                        for pkg_entry in packages.flatten() {
                            let pkg_path = pkg_entry.path();
                            if pkg_path.is_dir() {
                                let folder_name = pkg_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                                let inferred_model_id = folder_name.replace('_', "/");

                                if let Ok(manifest) = crate::adapter_manager::AdapterRegistry::ensure_valid_manifest(
                                    &pkg_path,
                                    &provider_id,
                                    &inferred_model_id,
                                ) {
                                    let full_gguf_path = pkg_path.join(&manifest.base_model.file_path);
                                    if full_gguf_path.exists() && full_gguf_path.is_file() {
                                        let file_name = full_gguf_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                                        let model_id = manifest.base_model.model_id.clone();
                                        let model_name = manifest.base_model.model_name.clone();
                                        let quantization = manifest.base_model.quantization.clone();
                                        let size_bytes = manifest.base_model.size_bytes;

                                        installed.push(InstalledModel {
                                            id: format!("{}_{}", model_id.replace('/', "_"), quantization),
                                            model_id,
                                            model_name,
                                            provider_id: manifest.provider_id,
                                            quantization,
                                            format: "GGUF".to_string(),
                                            backend: "llama.cpp (GGUF)".to_string(),
                                            file_name,
                                            file_path: full_gguf_path.to_string_lossy().to_string(),
                                            size_bytes,
                                            installed_at: chrono::Utc::now().to_rfc3339(),
                                            is_ready: size_bytes > 0,
                                            checksum: None,
                                            adapters: Some(manifest.adapters),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        installed
    }

    /// Deletes an installed model directory and files from disk
    pub fn delete_installed_model(app_data_dir: &Path, provider_id: &str, model_id: &str, quantization: &str) -> Result<()> {
        let clean_model_id = model_id.replace('/', "_");
        let package_dir = app_data_dir
            .join("models")
            .join(provider_id)
            .join(&clean_model_id);

        let legacy_quant_dir = package_dir.join(quantization);

        if package_dir.exists() {
            fs::remove_dir_all(&package_dir)?;
            log::info!("[MODEL_MANAGER] ✓ Deleted model package directory: {:?}", package_dir);
        } else if legacy_quant_dir.exists() {
            fs::remove_dir_all(&legacy_quant_dir)?;
            log::info!("[MODEL_MANAGER] ✓ Deleted model directory: {:?}", legacy_quant_dir);
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

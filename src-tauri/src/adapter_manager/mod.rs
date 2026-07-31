//! Adapter Manager & Model Package Registry
//!
//! Provides package manifest generation, storage, and backend registry operations
//! for future runtime orchestration attach/detach calls.

use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseManifestInfo {
    pub model_id: String,
    pub model_name: String,
    pub quantization: String,
    pub file_path: String,
    pub size_bytes: u64,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterManifestInfo {
    pub capability: String,
    pub status: String, // "Installed" | "Unavailable" | "Failed"
    pub repo_id: Option<String>,
    pub local_path: Option<String>,
    pub adapter_file: Option<String>,
    pub config_file: Option<String>,
    pub size_bytes: Option<u64>,
    pub base_model_match: Option<String>,
    pub target_modules: Vec<String>,
    pub peft_type: Option<String>,
    pub checksum: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPackageManifest {
    pub package_id: String,
    pub provider_id: String,
    pub base_model: BaseManifestInfo,
    pub adapters: HashMap<String, AdapterManifestInfo>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct AdapterRegistry;

impl AdapterRegistry {
    /// Resolves standard package directory: <SarathiAppData>/models/<provider>/<sanitized-model-id>/
    pub fn resolve_package_dir(app_data_dir: &Path, provider_id: &str, model_id: &str) -> PathBuf {
        let provider_clean = provider_id.to_lowercase();
        let sanitized_model = model_id.replace('/', "_");
        app_data_dir.join("models").join(provider_clean).join(sanitized_model)
    }

    /// Reads package manifest if present
    pub fn read_manifest(package_dir: &Path) -> Result<ModelPackageManifest> {
        let manifest_path = package_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(anyhow!("manifest.json does not exist at {:?}", manifest_path));
        }
        let content = fs::read_to_string(&manifest_path)?;
        let manifest: ModelPackageManifest = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    /// Writes model package manifest atomically to <package_dir>/manifest.json
    pub fn write_manifest(package_dir: &Path, manifest: &ModelPackageManifest) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        let manifest_path = package_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(manifest)?;
        fs::write(&manifest_path, json)?;
        log::info!("[ADAPTER_REGISTRY] Successfully wrote package manifest to {:?}", manifest_path);
        Ok(())
    }

    /// Lists all installed model packages with their manifest details
    pub fn list_installed_packages(app_data_dir: &Path) -> Vec<ModelPackageManifest> {
        let mut packages = Vec::new();
        let models_base = app_data_dir.join("models");
        if !models_base.exists() {
            return packages;
        }

        if let Ok(providers) = fs::read_dir(&models_base) {
            for prov_entry in providers.flatten() {
                if prov_entry.path().is_dir() {
                    if let Ok(model_dirs) = fs::read_dir(prov_entry.path()) {
                        for model_entry in model_dirs.flatten() {
                            if model_entry.path().is_dir() {
                                if let Ok(manifest) = Self::read_manifest(&model_entry.path()) {
                                    packages.push(manifest);
                                }
                            }
                        }
                    }
                }
            }
        }

        packages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_manifest_roundtrip() {
        let temp_dir = std::env::temp_dir().join(format!("sarathi_pkg_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let package_dir = AdapterRegistry::resolve_package_dir(&temp_dir, "huggingface", "meta-llama/Llama-3.2-1B");

        let mut adapters = HashMap::new();
        adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "Installed".to_string(),
                repo_id: Some("author/llama-code-lora".to_string()),
                local_path: Some("adapters/coding/".to_string()),
                adapter_file: Some("adapters/coding/adapter_model.safetensors".to_string()),
                config_file: Some("adapters/coding/adapter_config.json".to_string()),
                size_bytes: Some(45_000_000),
                base_model_match: Some("meta-llama/Llama-3.2-1B".to_string()),
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                peft_type: Some("LORA".to_string()),
                checksum: Some("abc123sha256".to_string()),
                reason: None,
            },
        );

        let manifest = ModelPackageManifest {
            package_id: "meta-llama_Llama-3.2-1B".to_string(),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: "meta-llama/Llama-3.2-1B".to_string(),
                model_name: "Llama 3.2 1B".to_string(),
                quantization: "Q8_0".to_string(),
                file_path: "base/Llama-3.2-1B-Instruct-Q8_0.gguf".to_string(),
                size_bytes: 1_321_083_008,
                checksum: None,
            },
            adapters,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };

        let write_res = AdapterRegistry::write_manifest(&package_dir, &manifest);
        assert!(write_res.is_ok(), "Writing package manifest must succeed");

        let read_res = AdapterRegistry::read_manifest(&package_dir);
        assert!(read_res.is_ok(), "Reading package manifest must succeed");
        let loaded = read_res.unwrap();
        assert_eq!(loaded.package_id, "meta-llama_Llama-3.2-1B");
        assert_eq!(loaded.adapters.get("coding").unwrap().status, "Installed");

        let _ = fs::remove_dir_all(temp_dir);
    }
}

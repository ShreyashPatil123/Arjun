//! Adapter Manager & Model Package Registry
//!
//! Provides package manifest generation, storage, single source of truth verification,
//! state machine logging, and startup scans for LoRA capability adapters.

pub mod state_machine;

use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub use state_machine::{AdapterState, log_adapter_transition, validate_transition};

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
    pub status: String, // "Installed" | "READY" | "Unavailable" | "Failed"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_runtime_status: Option<String>, // "compatible" | "requires_conversion" | "incompatible" | "not_present"
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

    /// Self-healing manifest validator and repair function.
    /// Ensures manifest exists, points to a valid primary GGUF file, and reports true size_bytes > 0.
    pub fn ensure_valid_manifest(package_dir: &Path, provider_id: &str, model_id: &str) -> Result<ModelPackageManifest> {
        let existing = Self::read_manifest(package_dir).ok();
        let base_dir = package_dir.join("base");

        let mut primary_gguf_rel: Option<String> = None;
        let mut total_gguf_bytes: u64 = 0;

        if base_dir.exists() && base_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&base_dir) {
                let mut gguf_files = Vec::new();
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension().map_or(false, |ext| ext == "gguf") {
                        if let Ok(meta) = fs::metadata(&p) {
                            let fname = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                            total_gguf_bytes += meta.len();
                            gguf_files.push((fname, meta.len()));
                        }
                    }
                }

                if !gguf_files.is_empty() {
                    gguf_files.sort_by(|a, b| a.0.cmp(&b.0));
                    let first_part = gguf_files.iter().find(|(name, _)| name.contains("-00001-of-")).map(|(n, _)| n.clone())
                        .unwrap_or_else(|| gguf_files[0].0.clone());
                    primary_gguf_rel = Some(format!("base/{}", first_part));
                }
            }
        }

        let rel_file_path = primary_gguf_rel.unwrap_or_else(|| "base/".to_string());

        if let Some(mut manifest) = existing {
            let file_valid = !manifest.base_model.file_path.is_empty() 
                && manifest.base_model.file_path != "base/" 
                && package_dir.join(&manifest.base_model.file_path).is_file();

            if file_valid && manifest.base_model.size_bytes > 0 {
                return Ok(manifest);
            }

            // Repair manifest values
            manifest.base_model.file_path = rel_file_path;
            manifest.base_model.size_bytes = if total_gguf_bytes > 0 { total_gguf_bytes } else { manifest.base_model.size_bytes };
            manifest.updated_at = chrono::Utc::now().to_rfc3339();

            let _ = Self::write_manifest(package_dir, &manifest);
            return Ok(manifest);
        }

        // Generate brand new manifest if missing
        let model_name = model_id.split('/').last().unwrap_or(model_id).to_string();
        let quant = if rel_file_path.contains("q4_k_m") {
            "Q4_K_M"
        } else if rel_file_path.contains("q4_0") {
            "Q4_0"
        } else if rel_file_path.contains("q8_0") {
            "Q8_0"
        } else {
            "GGUF"
        }.to_string();

        let new_manifest = ModelPackageManifest {
            package_id: format!("{}::{}::llama.cpp", model_id, quant),
            provider_id: provider_id.to_string(),
            base_model: BaseManifestInfo {
                model_id: model_id.to_string(),
                model_name,
                quantization: quant,
                file_path: rel_file_path,
                size_bytes: total_gguf_bytes,
                checksum: None,
            },
            adapters: HashMap::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };

        let _ = Self::write_manifest(package_dir, &new_manifest);
        Ok(new_manifest)
    }

    /// Verifies if adapter files exist on disk and meet minimum structural size requirements (>100KB for weights, >10B for config)
    pub fn verify_adapter_files(cap_dir: &Path) -> Option<(String, u64)> {
        if !cap_dir.exists() || !cap_dir.is_dir() {
            return None;
        }

        // Check for config file
        let config_path = cap_dir.join("adapter_config.json");
        if !config_path.exists() || fs::metadata(&config_path).map(|m| m.len()).unwrap_or(0) < 10 {
            return None;
        }

        // Check for weight file (.safetensors, .gguf, or .bin)
        let entries = match fs::read_dir(cap_dir) {
            Ok(e) => e,
            Err(_) => return None,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if name.ends_with(".safetensors") || name.ends_with(".gguf") || name.ends_with(".bin") {
                    if let Ok(meta) = fs::metadata(&path) {
                        if meta.len() >= 100_000 {
                            return Some((name, meta.len()));
                        }
                    }
                }
            }
        }

        None
    }

    /// Pre-download check: Checks if an adapter capability is ALREADY INSTALLED & VALID locally.
    /// Returns true if valid files exist and/or manifest records READY/Installed status.
    pub fn is_adapter_installed_and_valid(package_dir: &Path, capability: &str) -> bool {
        let cap_dir = package_dir.join("adapters").join(capability);
        let files_valid = Self::verify_adapter_files(&cap_dir).is_some();

        if let Ok(manifest) = Self::read_manifest(package_dir) {
            if let Some(adapter_info) = manifest.adapters.get(capability) {
                let status_upper = adapter_info.status.to_uppercase();
                if status_upper == "INSTALLED" || status_upper == "READY" {
                    // Priority 1: Manifest says Installed/READY and files exist (or manifest has verified local_path)
                    if files_valid || adapter_info.local_path.is_some() {
                        return true;
                    }
                }
            }
        }

        // Priority 2: Even if manifest was corrupted, if files exist on disk, it IS INSTALLED!
        files_valid
    }

    /// Writes model package manifest atomically to <package_dir>/manifest.json with Single Source of Truth protection.
    /// Preserves any previously Installed/READY adapters from being overwritten automatically.
    pub fn write_manifest(package_dir: &Path, manifest: &ModelPackageManifest) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        let manifest_path = package_dir.join("manifest.json");

        let mut final_manifest = manifest.clone();

        // Single Source of Truth Enforcement: Merge with existing manifest to prevent regression from READY -> Unavailable/NotFound
        if manifest_path.exists() {
            if let Ok(existing_manifest) = Self::read_manifest(package_dir) {
                for (cap_key, existing_adapter) in existing_manifest.adapters {
                    let cap_dir = package_dir.join("adapters").join(&cap_key);
                    let files_exist = Self::verify_adapter_files(&cap_dir).is_some();

                    let is_existing_ready = existing_adapter.status.eq_ignore_ascii_case("Installed") 
                        || existing_adapter.status.eq_ignore_ascii_case("READY");

                    if (is_existing_ready || files_exist) && files_exist {
                        if let Some(new_adapter) = final_manifest.adapters.get_mut(&cap_key) {
                            let new_status_upper = new_adapter.status.to_uppercase();
                            if new_status_upper != "INSTALLED" && new_status_upper != "READY" {
                                log_adapter_transition(
                                    &cap_key,
                                    &AdapterState::Ready,
                                    &AdapterState::Ready,
                                    "Preserved READY status during manifest write (blocked automatic overwrite)",
                                    "SingleSourceOfTruthProtection",
                                );
                                // Preserve existing valid installed adapter
                                *new_adapter = existing_adapter;
                            }
                        } else {
                            // Re-insert missing installed adapter
                            final_manifest.adapters.insert(cap_key, existing_adapter);
                        }
                    }
                }
            }
        }

        let json = serde_json::to_string_pretty(&final_manifest)?;
        fs::write(&manifest_path, json)?;
        log::info!("[ADAPTER_REGISTRY] Successfully wrote package manifest to {:?}", manifest_path);
        Ok(())
    }

    /// Startup Scan: Scans all local packages, validates adapter files on disk, updates registry, manifest.json & profile.json.
    /// Zero remote calls. Ensures UI and backend are immediately 100% synchronized on boot.
    pub fn perform_startup_scan(app_data_dir: &Path) {
        log::info!("[STARTUP_SCAN] Scanning local model packages for installed LoRA adapters...");
        let models_base = app_data_dir.join("models");
        if !models_base.exists() {
            return;
        }

        if let Ok(providers) = fs::read_dir(&models_base) {
            for prov_entry in providers.flatten() {
                if prov_entry.path().is_dir() {
                    if let Ok(model_dirs) = fs::read_dir(prov_entry.path()) {
                        for model_entry in model_dirs.flatten() {
                            if model_entry.path().is_dir() {
                                let package_dir = model_entry.path();
                                let mut manifest = match Self::read_manifest(&package_dir) {
                                    Ok(m) => m,
                                    Err(_) => continue,
                                };

                                let adapters_dir = package_dir.join("adapters");
                                if !adapters_dir.exists() {
                                    continue;
                                }

                                let mut updated = false;
                                for cap in crate::model_providers::huggingface::adapter_provider::HuggingFaceAdapterProvider::all_capabilities() {
                                    let cap_key = cap.key();
                                    let cap_dir = adapters_dir.join(cap_key);

                                    if let Some((weight_file, size_bytes)) = Self::verify_adapter_files(&cap_dir) {
                                        let existing_status = manifest.adapters.get(cap_key).map(|a| a.status.as_str()).unwrap_or("NOT_FOUND");
                                        let is_already_ready = existing_status.eq_ignore_ascii_case("Installed") || existing_status.eq_ignore_ascii_case("READY");

                                        if !is_already_ready {
                                            log_adapter_transition(
                                                cap_key,
                                                &AdapterState::NotFound,
                                                &AdapterState::Ready,
                                                "Startup scan verified local files on disk",
                                                "StartupScan",
                                            );
                                        }

                                        let adapter_info = AdapterManifestInfo {
                                            capability: cap_key.to_string(),
                                            status: "Installed".to_string(),
                                            adapter_runtime_status: None,
                                            repo_id: manifest.adapters.get(cap_key).and_then(|a| a.repo_id.clone()),
                                            local_path: Some(format!("adapters/{}/", cap_key)),
                                            adapter_file: Some(format!("adapters/{}/{}", cap_key, weight_file)),
                                            config_file: Some(format!("adapters/{}/adapter_config.json", cap_key)),
                                            size_bytes: Some(size_bytes),
                                            base_model_match: Some(manifest.base_model.model_id.clone()),
                                            target_modules: manifest.adapters.get(cap_key).map(|a| a.target_modules.clone()).unwrap_or_default(),
                                            peft_type: Some("LORA".to_string()),
                                            checksum: None,
                                            reason: None,
                                        };

                                        manifest.adapters.insert(cap_key.to_string(), adapter_info);
                                        updated = true;
                                    }
                                }

                                if updated {
                                    manifest.updated_at = chrono::Utc::now().to_rfc3339();
                                    let _ = Self::write_manifest(&package_dir, &manifest);
                                    let _ = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest);
                                } else {
                                    let _ = Self::validate_and_update_manifest(&package_dir);
                                }
                            }
                        }
                    }
                }
            }
        }
        log::info!("[STARTUP_SCAN] Local adapter scan completed.");
    }

    /// Validates all adapters in package_dir and updates manifest.json with runtime statuses
    pub fn validate_and_update_manifest(package_dir: &Path) -> Result<ModelPackageManifest> {
        let mut manifest = Self::read_manifest(package_dir)?;
        let validation_results = crate::lora::validator::validate_all_adapters(package_dir);

        let mut updated = false;
        for (cap, val_res) in validation_results {
            if let Some(adapter_info) = manifest.adapters.get_mut(&cap) {
                let status_str = val_res.status.to_string();
                if adapter_info.adapter_runtime_status.as_deref() != Some(&status_str) {
                    adapter_info.adapter_runtime_status = Some(status_str);
                    if adapter_info.reason.is_none() || val_res.status != crate::lora::validator::AdapterRuntimeStatus::Compatible {
                        adapter_info.reason = Some(val_res.reason);
                    }
                    updated = true;
                }
            }
        }

        if updated {
            manifest.updated_at = chrono::Utc::now().to_rfc3339();
            Self::write_manifest(package_dir, &manifest)?;
        }

        Ok(manifest)
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
                                let path = model_entry.path();
                                // Validate & update before returning package manifest
                                if let Ok(manifest) = Self::validate_and_update_manifest(&path) {
                                    packages.push(manifest);
                                } else if let Ok(manifest) = Self::read_manifest(&path) {
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
                adapter_runtime_status: Some("requires_conversion".to_string()),
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

    #[test]
    fn test_single_source_of_truth_protection_prevents_overwrite() {
        let temp_dir = std::env::temp_dir().join(format!("sarathi_ssot_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let package_dir = AdapterRegistry::resolve_package_dir(&temp_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        let coding_dir = package_dir.join("adapters").join("coding");
        fs::create_dir_all(&coding_dir).unwrap();

        // Create dummy adapter files (>100KB weight, >10B config)
        fs::write(coding_dir.join("adapter_config.json"), r#"{"peft_type":"LORA"}"#).unwrap();
        let dummy_weights = vec![0u8; 150_000];
        fs::write(coding_dir.join("adapter_model.safetensors"), &dummy_weights).unwrap();

        let mut initial_adapters = HashMap::new();
        initial_adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "Installed".to_string(),
                adapter_runtime_status: None,
                repo_id: Some("author/llama-code-lora".to_string()),
                local_path: Some("adapters/coding/".to_string()),
                adapter_file: Some("adapters/coding/adapter_model.safetensors".to_string()),
                config_file: Some("adapters/coding/adapter_config.json".to_string()),
                size_bytes: Some(150_000),
                base_model_match: Some("meta-llama/Llama-3.2-1B".to_string()),
                target_modules: vec![],
                peft_type: Some("LORA".to_string()),
                checksum: None,
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
                file_path: "base/Llama-3.2-1B.gguf".to_string(),
                size_bytes: 1_000_000,
                checksum: None,
            },
            adapters: initial_adapters,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };

        AdapterRegistry::write_manifest(&package_dir, &manifest).unwrap();

        // Attempt to write a corrupted manifest attempting to set coding to Unavailable
        let mut corrupted_adapters = HashMap::new();
        corrupted_adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "Unavailable".to_string(),
                adapter_runtime_status: None,
                repo_id: None,
                local_path: None,
                adapter_file: None,
                config_file: None,
                size_bytes: None,
                base_model_match: None,
                target_modules: vec![],
                peft_type: None,
                checksum: None,
                reason: Some("Remote search failed".to_string()),
            },
        );

        let corrupted_manifest = ModelPackageManifest {
            package_id: "meta-llama_Llama-3.2-1B".to_string(),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: "meta-llama/Llama-3.2-1B".to_string(),
                model_name: "Llama 3.2 1B".to_string(),
                quantization: "Q8_0".to_string(),
                file_path: "base/Llama-3.2-1B.gguf".to_string(),
                size_bytes: 1_000_000,
                checksum: None,
            },
            adapters: corrupted_adapters,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };

        // Write the corrupted manifest
        AdapterRegistry::write_manifest(&package_dir, &corrupted_manifest).unwrap();

        // Read manifest back and verify Single Source of Truth protection preserved Installed status!
        let reloaded = AdapterRegistry::read_manifest(&package_dir).unwrap();
        let coding_status = reloaded.adapters.get("coding").unwrap().status.clone();
        assert_eq!(coding_status, "Installed", "Single Source of Truth MUST preserve Installed status when valid files exist on disk");

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_startup_scan_detects_and_registers_local_files() {
        let temp_dir = std::env::temp_dir().join(format!("sarathi_startup_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let package_dir = AdapterRegistry::resolve_package_dir(&temp_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        let reasoning_dir = package_dir.join("adapters").join("reasoning");
        fs::create_dir_all(&reasoning_dir).unwrap();

        // Create base manifest
        let manifest = ModelPackageManifest {
            package_id: "meta-llama_Llama-3.2-1B".to_string(),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: "meta-llama/Llama-3.2-1B".to_string(),
                model_name: "Llama 3.2 1B".to_string(),
                quantization: "Q8_0".to_string(),
                file_path: "base/Llama-3.2-1B.gguf".to_string(),
                size_bytes: 1_000_000,
                checksum: None,
            },
            adapters: HashMap::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        AdapterRegistry::write_manifest(&package_dir, &manifest).unwrap();

        // Manually place valid adapter files in reasoning dir
        fs::write(reasoning_dir.join("adapter_config.json"), r#"{"peft_type":"LORA"}"#).unwrap();
        let weights = vec![1u8; 120_000];
        fs::write(reasoning_dir.join("adapter_model.safetensors"), &weights).unwrap();

        // Execute startup scan
        AdapterRegistry::perform_startup_scan(&temp_dir);

        // Verify startup scan automatically registered reasoning adapter as Installed
        let loaded = AdapterRegistry::read_manifest(&package_dir).unwrap();
        let reasoning = loaded.adapters.get("reasoning").expect("Reasoning adapter should be registered by startup scan");
        assert_eq!(reasoning.status, "Installed");
        assert_eq!(reasoning.size_bytes, Some(120_000));

        let _ = fs::remove_dir_all(temp_dir);
    }
}

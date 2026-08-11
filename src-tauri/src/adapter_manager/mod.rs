//! Adapter Manager & Model Package Registry
//!
//! Provides package manifest generation, storage, single source of truth verification,
//! state machine logging, and startup scans for LoRA capability adapters.

pub mod gguf;
pub mod state_machine;
pub mod store;

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

/// `Default` exists so construction sites can use `..Default::default()` and a
/// later field addition does not have to touch all of them. It is never a valid
/// record on its own — `capability` and `status` are always supplied.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    // Everything below is `serde(default)` so a manifest written before these
    // existed still deserializes. An installed model is expensive to re-acquire;
    // a schema addition must never orphan one.
    /// Strength the adapter is bound at. `None` means
    /// [`crate::capability::DEFAULT_LORA_SCALE`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    /// LoRA rank, from the adapter's own configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    /// LoRA alpha, after any rsLoRA compensation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Base architecture the adapter was converted against, e.g. `qwen2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// `user` or `auto-discovery`. Keeps the capability sweep from overwriting a
    /// slot the user assigned by hand.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// `stated`, `suggested`, or `manual` — see [`crate::capability::assign`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment_confidence: Option<String>,
}

/// Written by the user installing an adapter themselves.
pub const SOURCE_USER: &str = "user";
/// Written by the automatic capability sweep during a model download.
pub const SOURCE_AUTO_DISCOVERY: &str = "auto-discovery";

/// Below this, a "weight file" is a stub, an error page, or a truncated download
/// rather than a real adapter.
const MIN_ADAPTER_WEIGHT_BYTES: u64 = 100_000;
/// An `adapter_config.json` smaller than this cannot hold a usable object.
const MIN_ADAPTER_CONFIG_BYTES: u64 = 10;

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

    /// Finds the loadable weight file in an adapter directory, with its size.
    ///
    /// A GGUF adapter needs no `adapter_config.json`. It declares its own
    /// `adapter.type`, which [`gguf::verify_is_lora_adapter`] checks directly —
    /// that is the real authority. Requiring the sidecar JSON as well rejected
    /// every adapter installed from a ready-made GGUF, because that path has no
    /// config to save. PEFT checkpoints still need one: without it there is no
    /// way to know how the adapter was trained.
    ///
    /// GGUF wins when a directory holds both, which is what a conversion looks
    /// like the moment before its source safetensors are removed.
    pub fn verify_adapter_files(cap_dir: &Path) -> Option<(String, u64)> {
        if !cap_dir.exists() || !cap_dir.is_dir() {
            return None;
        }

        let config_path = cap_dir.join("adapter_config.json");
        let has_config =
            fs::metadata(&config_path).map(|m| m.len() >= MIN_ADAPTER_CONFIG_BYTES).unwrap_or(false);

        let entries = match fs::read_dir(cap_dir) {
            Ok(e) => e,
            Err(_) => return None,
        };

        let mut peft_weights: Option<(String, u64)> = None;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let size = match fs::metadata(&path) {
                Ok(m) if m.len() >= MIN_ADAPTER_WEIGHT_BYTES => m.len(),
                _ => continue,
            };

            let lower = name.to_lowercase();
            if lower.ends_with(".gguf") {
                return Some((name, size));
            }
            if has_config && (lower.ends_with(".safetensors") || lower.ends_with(".bin")) {
                peft_weights.get_or_insert((name, size));
            }
        }

        peft_weights
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

    /// Writes the manifest, obeying whatever the caller asks for.
    ///
    /// The Single Source of Truth protection in [`Self::write_manifest`] exists to
    /// stop *automatic* passes — startup scans, discovery sweeps — from
    /// downgrading an adapter that is installed and working. It is not meant to
    /// overrule the person using the app.
    ///
    /// It would, though: the protection triggers whenever
    /// `adapters/<capability>/` holds files, which is exactly the shape
    /// auto-discovery writes. Reassigning such an adapter through the protected
    /// path was silently reverted, leaving the UI showing a change that had not
    /// happened.
    pub fn write_manifest_user_initiated(
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        Self::persist(package_dir, manifest)
    }

    fn persist(package_dir: &Path, manifest: &ModelPackageManifest) -> Result<()> {
        let manifest_path = package_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(manifest)?;
        fs::write(&manifest_path, json)?;
        log::info!("[ADAPTER_REGISTRY] Successfully wrote package manifest to {:?}", manifest_path);
        Ok(())
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

        Self::persist(package_dir, &final_manifest)
    }

    /// Startup Scan: Scans all local packages, validates adapter files on disk, updates registry, manifest.json & profile.json.
    /// Zero remote calls. Ensures UI and backend are immediately 100% synchronized on boot.
    pub fn perform_startup_scan(app_data_dir: &Path) {
        log::info!("[STARTUP_SCAN] Scanning local model packages for installed LoRA adapters...");
        let models_base = app_data_dir.join("models");
        if !models_base.exists() {
            return;
        }

        for package_dir in Self::installed_package_dirs(&models_base) {
            Self::scan_package_adapters(&package_dir);
        }

        log::info!("[STARTUP_SCAN] Local adapter scan completed.");
    }

    /// Every `<models>/<provider>/<package>/` directory on disk.
    fn installed_package_dirs(models_base: &Path) -> Vec<PathBuf> {
        let providers = match fs::read_dir(models_base) {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };

        providers
            .flatten()
            .map(|provider| provider.path())
            .filter(|path| path.is_dir())
            .filter_map(|provider| fs::read_dir(provider).ok())
            .flat_map(|packages| packages.flatten().map(|p| p.path()))
            .filter(|path| path.is_dir())
            .collect()
    }

    /// Registers every adapter directory in one package against the manifest.
    fn scan_package_adapters(package_dir: &Path) {
        let mut manifest = match Self::read_manifest(package_dir) {
            Ok(m) => m,
            Err(_) => return,
        };

        let adapters_dir = package_dir.join("adapters");
        if !adapters_dir.exists() {
            return;
        }

        // Adapters live in one of two shapes: `adapters/<capability>/`, written
        // by the automatic sweep during a model download, and
        // `adapters/<sanitised-repo-id>/`, written when the user installs one
        // themselves. Both are walked.
        //
        // Scanning only the first shape is what left every hand-installed
        // adapter unregistered — and so permanently unbindable, since
        // `try_bind_adapter` can only reach what the manifest names.
        let subdirs = match fs::read_dir(&adapters_dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        let mut updated = false;
        for entry in subdirs.flatten() {
            if Self::register_adapter_dir(&entry.path(), &mut manifest) {
                updated = true;
            }
        }

        if updated {
            manifest.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = Self::write_manifest(package_dir, &manifest);
            let _ = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(
                package_dir,
                &manifest,
            );
        } else {
            let _ = Self::validate_and_update_manifest(package_dir);
        }
    }

    /// Adds or refreshes the manifest record for one adapter directory.
    ///
    /// Returns whether the manifest changed.
    fn register_adapter_dir(dir: &Path, manifest: &mut ModelPackageManifest) -> bool {
        if !dir.is_dir() {
            return false;
        }

        let dir_name = match dir.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => return false,
        };

        let (weight_file, size_bytes) = match Self::verify_adapter_files(dir) {
            Some(found) => found,
            None => return false,
        };

        // The record already pointing into *this* directory, if there is one.
        //
        // Everything below hangs off the distinction between "refresh what this
        // adapter already holds" and "claim a slot for it": reading the target
        // slot blindly would copy another adapter's rank onto this one and
        // overwrite whichever adapter the user had put there.
        let prefix = format!("adapters/{}/", dir_name);
        let own = manifest
            .adapters
            .iter()
            .find(|(_, a)| a.adapter_file.as_deref().map(|f| f.starts_with(&prefix)).unwrap_or(false))
            .map(|(key, record)| (key.clone(), record.clone()));

        let (cap_key, inferred_confidence) = match &own {
            Some((key, _)) => (key.clone(), None),
            None => match Self::claim_free_capability(&dir_name, dir, manifest) {
                Some(found) => found,
                None => return false,
            },
        };

        let previous = own.as_ref().map(|(_, record)| record);
        let was_ready = previous
            .map(|a| {
                a.status.eq_ignore_ascii_case("Installed") || a.status.eq_ignore_ascii_case("READY")
            })
            .unwrap_or(false);

        if !was_ready {
            log_adapter_transition(
                &cap_key,
                &AdapterState::NotFound,
                &AdapterState::Ready,
                "Startup scan verified local files on disk",
                "StartupScan",
            );
        }

        let is_gguf = weight_file.to_lowercase().ends_with(".gguf");

        let adapter_info = AdapterManifestInfo {
            capability: cap_key.clone(),
            status: "Installed".to_string(),
            // A GGUF on disk is loadable as it stands; anything else still needs
            // converting, and saying so is what stops the resolver from handing
            // llama.cpp a file it will reject.
            adapter_runtime_status: Some(
                if is_gguf { "compatible" } else { "requires_conversion" }.to_string(),
            ),
            repo_id: previous
                .and_then(|a| a.repo_id.clone())
                .or_else(|| Self::read_source_repo(dir)),
            local_path: Some(format!("adapters/{}/", dir_name)),
            adapter_file: Some(format!("adapters/{}/{}", dir_name, weight_file)),
            config_file: Some(format!("adapters/{}/adapter_config.json", dir_name)),
            size_bytes: Some(size_bytes),
            base_model_match: Some(manifest.base_model.model_id.clone()),
            target_modules: previous.map(|a| a.target_modules.clone()).unwrap_or_default(),
            peft_type: Some("LORA".to_string()),
            checksum: None,
            reason: None,
            // Everything a disk scan cannot rediscover is carried forward. The
            // rank came from a config file the conversion deleted, and the slot
            // may have been chosen by hand — re-deriving either would quietly
            // undo the user's decision.
            scale: previous.and_then(|a| a.scale),
            rank: previous.and_then(|a| a.rank),
            alpha: previous.and_then(|a| a.alpha),
            architecture: previous.and_then(|a| a.architecture.clone()),
            source: previous.and_then(|a| a.source.clone()),
            assignment_confidence: previous
                .and_then(|a| a.assignment_confidence.clone())
                .or(inferred_confidence),
        };

        manifest.adapters.insert(cap_key, adapter_info);
        true
    }

    /// Picks a capability for an adapter that does not already hold one.
    ///
    /// Only a *free* slot is claimed. If another adapter already fills the
    /// capability this one would land in, that adapter keeps it: a scan runs on
    /// every launch, and one that could evict a binding would undo a user's
    /// choice silently and repeatedly. The newcomer stays installed and
    /// assignable by hand.
    fn claim_free_capability(
        dir_name: &str,
        dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Option<(String, Option<String>)> {
        // A directory named after a capability is self-describing — that is the
        // shape the discovery sweep writes. Anything else is a repository name,
        // and offline there are no tags to consult, so the name is all there is:
        // a hint, recorded as one.
        let (key, confidence) = if crate::capability::assign::is_known_capability(dir_name) {
            (dir_name.to_string(), None)
        } else {
            let inferred = Self::read_source_repo(dir)
                .and_then(|repo_id| crate::capability::assign::infer(&repo_id, &[]));

            match inferred {
                Some(a) => (a.capability, Some(a.confidence.as_str().to_string())),
                // Nothing says what this adapter is for. It stays installed and
                // visible in Storage awaiting a choice, rather than being filed
                // under a guess that would silently never activate.
                None => {
                    log::info!(
                        "[STARTUP_SCAN] '{}' is installed but has no capability assigned yet",
                        dir.display()
                    );
                    return None;
                }
            }
        };

        let taken = manifest
            .adapters
            .get(&key)
            .map(|a| a.adapter_file.is_some())
            .unwrap_or(false);

        if taken {
            log::info!(
                "[STARTUP_SCAN] '{}' suits '{}', which another adapter already fills — left unassigned",
                dir.display(),
                key
            );
            return None;
        }

        Some((key, confidence))
    }

    /// The repository an adapter was installed from, as recorded by `source.txt`.
    fn read_source_repo(dir: &Path) -> Option<String> {
        let raw = fs::read_to_string(dir.join("source.txt")).ok()?;
        let trimmed = raw.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
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

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sarathi_am_{name}_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base_manifest() -> ModelPackageManifest {
        ModelPackageManifest {
            package_id: "Qwen_Qwen3-4B".to_string(),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: "Qwen/Qwen3-4B".to_string(),
                model_name: "Qwen3 4B".to_string(),
                quantization: "Q4_K_M".to_string(),
                file_path: "base/model.gguf".to_string(),
                size_bytes: 1,
                checksum: None,
            },
            adapters: HashMap::new(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    /// Writes an adapter directory holding a weight file of `weight_name`.
    fn install_adapter(package: &Path, dir_name: &str, weight_name: &str, source: Option<&str>) {
        let dir = package.join("adapters").join(dir_name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(weight_name), vec![0u8; 150_000]).unwrap();
        if let Some(repo) = source {
            fs::write(dir.join("source.txt"), repo).unwrap();
        }
    }

    #[test]
    fn a_gguf_adapter_needs_no_sidecar_config() {
        // The ready-GGUF install path never writes adapter_config.json. Demanding
        // one rejected every adapter installed that way, which then could not be
        // registered and so could never bind.
        let package = scratch("gguf_no_config");
        install_adapter(&package, "someone_coding-lora", "adapter.gguf", None);

        let found = AdapterRegistry::verify_adapter_files(
            &package.join("adapters").join("someone_coding-lora"),
        );

        assert_eq!(found.map(|(name, _)| name), Some("adapter.gguf".to_string()));
    }

    #[test]
    fn peft_weights_still_require_a_config() {
        // Without it there is no way to know how the adapter was trained, so it
        // cannot be converted either.
        let package = scratch("peft_needs_config");
        install_adapter(&package, "coding", "adapter_model.safetensors", None);

        assert!(AdapterRegistry::verify_adapter_files(&package.join("adapters").join("coding"))
            .is_none());

        fs::write(
            package.join("adapters").join("coding").join("adapter_config.json"),
            r#"{"peft_type":"LORA"}"#,
        )
        .unwrap();

        assert!(AdapterRegistry::verify_adapter_files(&package.join("adapters").join("coding"))
            .is_some());
    }

    #[test]
    fn gguf_wins_when_a_directory_holds_both() {
        // What a conversion looks like in the instant before its source
        // safetensors are removed. Picking the safetensors would register an
        // adapter llama.cpp cannot load.
        let package = scratch("both_formats");
        install_adapter(&package, "coding", "adapter_model.safetensors", None);
        let dir = package.join("adapters").join("coding");
        fs::write(dir.join("adapter_config.json"), r#"{"peft_type":"LORA"}"#).unwrap();
        fs::write(dir.join("adapter.gguf"), vec![0u8; 150_000]).unwrap();

        let found = AdapterRegistry::verify_adapter_files(&dir);

        assert_eq!(found.map(|(name, _)| name), Some("adapter.gguf".to_string()));
    }

    #[test]
    fn the_startup_scan_registers_a_repo_named_directory() {
        // The bug this exists for: a hand-installed adapter lives under its
        // repository name, the scan only ever looked at `adapters/<capability>/`,
        // so it was never written to the manifest and could never be bound.
        let package = scratch("scan_repo_named");
        install_adapter(
            &package,
            "someone_sql-coder-lora",
            "adapter.gguf",
            Some("someone/sql-coder-lora"),
        );

        let mut manifest = base_manifest();
        AdapterRegistry::register_adapter_dir(
            &package.join("adapters").join("someone_sql-coder-lora"),
            &mut manifest,
        );

        let record = manifest.adapters.get("coding").expect("should have claimed the coding slot");
        assert_eq!(
            record.adapter_file.as_deref(),
            Some("adapters/someone_sql-coder-lora/adapter.gguf")
        );
        assert_eq!(record.adapter_runtime_status.as_deref(), Some("compatible"));
        assert_eq!(record.repo_id.as_deref(), Some("someone/sql-coder-lora"));
        // Inferred from the name alone, so it must not claim the author said so.
        assert_eq!(record.assignment_confidence.as_deref(), Some("suggested"));
    }

    #[test]
    fn an_unidentifiable_adapter_is_left_unassigned() {
        let package = scratch("scan_unassignable");
        install_adapter(&package, "someone_mystery-v2", "adapter.gguf", Some("someone/mystery-v2"));

        let mut manifest = base_manifest();
        let changed = AdapterRegistry::register_adapter_dir(
            &package.join("adapters").join("someone_mystery-v2"),
            &mut manifest,
        );

        assert!(!changed, "nothing should be claimed");
        assert!(manifest.adapters.is_empty(), "a guessed slot would never activate");
    }

    #[test]
    fn a_scan_never_overwrites_a_manual_assignment() {
        // The user filed a coding-named adapter under research. Re-inferring on
        // every launch would quietly undo that, forever.
        let package = scratch("scan_respects_manual");
        install_adapter(
            &package,
            "someone_sql-coder-lora",
            "adapter.gguf",
            Some("someone/sql-coder-lora"),
        );

        let mut manifest = base_manifest();
        manifest.adapters.insert(
            "research".to_string(),
            AdapterManifestInfo {
                capability: "research".to_string(),
                status: "installed".to_string(),
                adapter_file: Some(
                    "adapters/someone_sql-coder-lora/adapter.gguf".to_string(),
                ),
                assignment_confidence: Some("manual".to_string()),
                rank: Some(16),
                ..Default::default()
            },
        );

        AdapterRegistry::register_adapter_dir(
            &package.join("adapters").join("someone_sql-coder-lora"),
            &mut manifest,
        );

        assert!(manifest.adapters.contains_key("research"));
        assert!(!manifest.adapters.contains_key("coding"), "the name must not win over the user");
        let record = manifest.adapters.get("research").unwrap();
        assert_eq!(record.assignment_confidence.as_deref(), Some("manual"));
        assert_eq!(record.rank, Some(16), "shape data a disk scan cannot rediscover must survive");
    }

    #[test]
    fn a_scan_never_evicts_the_adapter_already_filling_a_slot() {
        // Two adapters both suit `coding`: one hand-installed and assigned, one
        // sitting in the capability-named directory the sweep uses. The scan runs
        // on every launch, so a scan that could take the slot would undo the
        // user's choice again and again.
        let package = scratch("scan_no_evict");
        install_adapter(&package, "coding", "adapter.gguf", None);
        install_adapter(
            &package,
            "someone_sql-coder-lora",
            "adapter.gguf",
            Some("someone/sql-coder-lora"),
        );

        let mut manifest = base_manifest();
        manifest.adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "installed".to_string(),
                adapter_file: Some(
                    "adapters/someone_sql-coder-lora/adapter.gguf".to_string(),
                ),
                assignment_confidence: Some("manual".to_string()),
                rank: Some(8),
                ..Default::default()
            },
        );

        let changed = AdapterRegistry::register_adapter_dir(
            &package.join("adapters").join("coding"),
            &mut manifest,
        );

        assert!(!changed, "the occupied slot must be left alone");
        let record = &manifest.adapters["coding"];
        assert_eq!(
            record.adapter_file.as_deref(),
            Some("adapters/someone_sql-coder-lora/adapter.gguf"),
            "the user's adapter must keep the slot"
        );
        assert_eq!(record.rank, Some(8), "and must not inherit the other adapter's shape");
    }

    #[test]
    fn refreshing_an_adapter_keeps_its_own_data_not_the_slots() {
        // Re-scanning an adapter that already holds a slot must reuse *its*
        // record. Reading the slot blindly is the same bug from the other side.
        let package = scratch("scan_refresh_own");
        install_adapter(
            &package,
            "someone_sql-coder-lora",
            "adapter.gguf",
            Some("someone/sql-coder-lora"),
        );

        let mut manifest = base_manifest();
        manifest.adapters.insert(
            "research".to_string(),
            AdapterManifestInfo {
                capability: "research".to_string(),
                status: "installed".to_string(),
                adapter_file: Some(
                    "adapters/someone_sql-coder-lora/adapter.gguf".to_string(),
                ),
                rank: Some(64),
                scale: Some(0.5),
                ..Default::default()
            },
        );

        AdapterRegistry::register_adapter_dir(
            &package.join("adapters").join("someone_sql-coder-lora"),
            &mut manifest,
        );

        let record = &manifest.adapters["research"];
        assert_eq!(record.rank, Some(64));
        assert_eq!(record.scale, Some(0.5));
        assert!(!manifest.adapters.contains_key("coding"));
    }

    #[test]
    fn a_manifest_written_before_the_new_fields_existed_still_loads() {
        // An installed model is expensive to re-acquire; a schema addition must
        // never orphan one.
        let json = r#"{
            "packageId": "pkg",
            "providerId": "huggingface",
            "baseModel": {
                "modelId": "Qwen/Qwen3-4B",
                "modelName": "Qwen3 4B",
                "quantization": "Q4_K_M",
                "filePath": "base/model.gguf",
                "sizeBytes": 1
            },
            "adapters": {
                "coding": {
                    "capability": "coding",
                    "status": "Installed",
                    "repoId": "someone/coding-lora",
                    "localPath": "adapters/coding/",
                    "adapterFile": "adapters/coding/adapter.gguf",
                    "configFile": null,
                    "sizeBytes": 4096,
                    "baseModelMatch": null,
                    "targetModules": [],
                    "peftType": "LORA",
                    "checksum": null,
                    "reason": null
                }
            },
            "createdAt": "",
            "updatedAt": ""
        }"#;

        let loaded: ModelPackageManifest = serde_json::from_str(json).expect("must deserialize");
        let coding = loaded.adapters.get("coding").unwrap();

        assert_eq!(coding.status, "Installed");
        assert_eq!(coding.scale, None);
        assert_eq!(coding.rank, None);
        assert_eq!(coding.assignment_confidence, None);
    }

    #[test]
    fn the_new_fields_round_trip_through_disk() {
        let package = scratch("new_fields_roundtrip");
        let mut manifest = base_manifest();
        manifest.adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "installed".to_string(),
                adapter_file: Some("adapters/x/adapter.gguf".to_string()),
                scale: Some(0.75),
                rank: Some(32),
                alpha: Some(64.0),
                architecture: Some("qwen3".to_string()),
                source: Some(SOURCE_USER.to_string()),
                assignment_confidence: Some("manual".to_string()),
                ..Default::default()
            },
        );

        AdapterRegistry::write_manifest_user_initiated(&package, &manifest).unwrap();
        let loaded = AdapterRegistry::read_manifest(&package).unwrap();
        let coding = loaded.adapters.get("coding").unwrap();

        assert_eq!(coding.scale, Some(0.75));
        assert_eq!(coding.rank, Some(32));
        assert_eq!(coding.alpha, Some(64.0));
        assert_eq!(coding.architecture.as_deref(), Some("qwen3"));
        assert_eq!(coding.source.as_deref(), Some(SOURCE_USER));
        assert_eq!(coding.assignment_confidence.as_deref(), Some("manual"));
    }

    #[test]
    fn a_user_initiated_write_can_release_a_slot_the_protected_path_would_restore() {
        // The Single Source of Truth guard fires whenever `adapters/<capability>/`
        // holds files, which is exactly the shape auto-discovery writes. Routing
        // a reassignment through the protected path silently reverted it.
        let package = scratch("user_initiated_release");
        install_adapter(&package, "coding", "adapter.gguf", None);

        let mut manifest = base_manifest();
        manifest.adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "Installed".to_string(),
                adapter_file: Some("adapters/coding/adapter.gguf".to_string()),
                ..Default::default()
            },
        );
        AdapterRegistry::write_manifest(&package, &manifest).unwrap();

        // Release it.
        manifest.adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "Unavailable".to_string(),
                ..Default::default()
            },
        );

        AdapterRegistry::write_manifest(&package, &manifest).unwrap();
        assert_eq!(
            AdapterRegistry::read_manifest(&package).unwrap().adapters["coding"].status,
            "Installed",
            "the protected path is expected to refuse this"
        );

        AdapterRegistry::write_manifest_user_initiated(&package, &manifest).unwrap();
        assert_eq!(
            AdapterRegistry::read_manifest(&package).unwrap().adapters["coding"].status,
            "Unavailable",
            "an explicit user action must not be overruled"
        );
    }

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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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

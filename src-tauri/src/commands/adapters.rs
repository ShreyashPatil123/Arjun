//! IPC for downloading and managing LoRA adapters.
//!
//! Adapters install beside the model they belong to, so removing a model takes
//! its adapters with it rather than leaving orphans.

use std::path::PathBuf;

use tauri::{AppHandle, Manager};

use crate::adapter_manager::store::{self, InstalledAdapter};
use crate::adapter_manager::AdapterRegistry;

/// Largest file accepted as a LoRA adapter.
///
/// Adapters are deltas, not weights: rank-64 on a 7B model lands near 200 MB,
/// and even generous ones stay well under a gigabyte. 2 GB leaves plenty of
/// room while still catching a merged model published under an adapter tag.
const MAX_ADAPTER_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Adapters on disk for one model, with their total footprint.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledAdapters {
    pub adapters: Vec<InstalledAdapter>,
    pub total_bytes: u64,
}

/// Finds a repository file by exact base name, ignoring any folder prefix.
fn find_named(filenames: &[String], wanted: &str) -> Option<String> {
    filenames
        .iter()
        .find(|f| f.rsplit('/').next().unwrap_or(f).eq_ignore_ascii_case(wanted))
        .cloned()
}

/// The PEFT weight file, if this repository ships one.
///
/// `.safetensors` is preferred over the older pickle `.bin`: pickle files
/// execute arbitrary code when loaded by torch, and while Sarathi never loads
/// them that way, declining to install them keeps that hazard off the disk.
fn find_peft_weights(filenames: &[String]) -> Option<String> {
    find_named(filenames, "adapter_model.safetensors")
}

/// Downloads one file from an adapter repository into memory.
///
/// Cleans up `target_dir` on any failure so a half-installed adapter never
/// appears in the list. The size ceiling is applied before the body is read,
/// because buffering a merged model would cost gigabytes of memory before
/// anything could reject it.
async fn fetch_adapter_file(
    client: &reqwest::Client,
    token: Option<&str>,
    repo_id: &str,
    filename: &str,
    target_dir: &std::path::Path,
) -> Result<Vec<u8>, String> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true");
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let fail = |dir: &std::path::Path, message: String| -> String {
        let _ = std::fs::remove_dir_all(dir);
        message
    };

    let resp = req
        .send()
        .await
        .map_err(|e| fail(target_dir, format!("download failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(fail(
            target_dir,
            format!("download of {filename} failed with {}", resp.status()),
        ));
    }

    // A LoRA adapter is a small delta — tens to a few hundred megabytes. A file
    // far past that is a merged model wearing an adapter label.
    if let Some(len) = resp.content_length() {
        if len > MAX_ADAPTER_BYTES {
            return Err(fail(
                target_dir,
                format!(
                    "This file is {:.1} GB. LoRA adapters are small add-ons, so this is \
                     almost certainly a complete model rather than an adapter.",
                    len as f64 / 1_073_741_824.0
                ),
            ));
        }
    }

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| fail(target_dir, format!("download was interrupted: {e}")))
}

fn package_dir_for(app: &AppHandle, provider_id: &str, model_id: &str) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve the app data folder: {e}"))?;
    Ok(AdapterRegistry::resolve_package_dir(&dir, provider_id, model_id))
}

#[tauri::command]
pub async fn list_installed_adapters(
    app: AppHandle,
    provider_id: String,
    model_id: String,
) -> Result<InstalledAdapters, String> {
    let package = package_dir_for(&app, &provider_id, &model_id)?;
    let adapters = store::list_installed(&package, &model_id);
    let total_bytes = adapters.iter().map(|a| a.size_bytes).sum();
    Ok(InstalledAdapters { adapters, total_bytes })
}

/// Downloads a LoRA adapter for a model.
///
/// The repository's file list is checked **before** any bytes are fetched: a
/// PEFT safetensors adapter cannot be loaded by llama.cpp, so it is refused
/// with an explanation rather than downloaded and left to fail at load time.
#[tauri::command]
pub async fn download_adapter(
    app: AppHandle,
    provider_id: String,
    model_id: String,
    adapter_repo_id: String,
) -> Result<InstalledAdapter, String> {
    let package = package_dir_for(&app, &provider_id, &model_id)?;
    let token = crate::config::hf_token::get();

    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .build()
        .map_err(|e| format!("could not create an HTTP client: {e}"))?;

    // 1. Read the file list and decide whether this is installable at all.
    let mut info_req = client.get(format!(
        "https://huggingface.co/api/models/{adapter_repo_id}"
    ));
    if let Some(t) = &token {
        info_req = info_req.bearer_auth(t);
    }

    let info: serde_json::Value = info_req
        .send()
        .await
        .map_err(|e| format!("could not reach HuggingFace: {e}"))?
        .json()
        .await
        .map_err(|e| format!("unexpected response from HuggingFace: {e}"))?;

    let filenames: Vec<String> = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| f.get("rfilename").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // The author's tags decide which capability this adapter fills, and they are
    // only available here — the manifest write below happens after the download,
    // by which point re-fetching them would mean a second round trip.
    let tags: Vec<String> = info
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    // A repository either ships a GGUF that loads as-is, or PEFT safetensors
    // that Sarathi converts locally. Only when it is neither is there nothing
    // to install.
    let ready_gguf = match store::check_installable(&filenames) {
        Ok(name) => Some(name),
        Err(refusal) => {
            if find_peft_weights(&filenames).is_some() {
                None
            } else {
                return Err(refusal.to_string());
            }
        }
    };

    // An adapter has to live with the model it attaches to, not with the
    // repository the user happened to be browsing.
    //
    // Those differ routinely: adapters are published against the original model
    // (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) while the GGUF weights come from
    // whoever quantised it (`bartowski/...-GGUF`). Installing under the browsed
    // repository put the adapter in a package with no base model, so the startup
    // scan never registered it and inference could never find it — a converted
    // adapter that looked installed and did nothing.
    //
    // Falling back to the browsed package keeps the case where someone installs
    // an adapter before its model working exactly as before.
    let base_gguf = crate::lora::convert::arch::resolve_base_gguf(&package).ok();

    let install_package = base_gguf
        .as_ref()
        .and_then(|gguf| gguf.parent()?.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| package.clone());

    if install_package != package {
        log::info!(
            "[ADAPTERS] Installing into '{}', which holds the base model for '{model_id}'",
            install_package.display()
        );
    }

    // 2. Fetch it.
    let dir_name = store::adapter_dir_name(&adapter_repo_id);
    let target_dir = store::adapters_root(&install_package).join(&dir_name);
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("could not create the adapter folder: {e}"))?;

    let file_path = target_dir.join("adapter.gguf");
    let downloaded_bytes;
    // Kept so the manifest can record the adapter's shape. The conversion
    // deletes the config it came from, so this is the last point at which the
    // rank and alpha are knowable without re-downloading.
    let mut conversion: Option<crate::lora::ConversionSummary> = None;

    match ready_gguf {
        Some(gguf_file) => {
            let bytes = fetch_adapter_file(
                &client,
                token.as_deref(),
                &adapter_repo_id,
                &gguf_file,
                &target_dir,
            )
            .await?;

            std::fs::write(&file_path, &bytes)
                .map_err(|e| format!("could not save the adapter: {e}"))?;
            downloaded_bytes = bytes.len();
        }

        // PEFT path: fetch the weights and their configuration, then convert in
        // place. The conversion is what makes the adapter loadable, so a failure
        // here has to clean up — leaving safetensors behind would show as an
        // installed adapter that nothing can use.
        None => {
            let weights = find_peft_weights(&filenames)
                .expect("PEFT weights were present when the source was chosen");

            let weight_bytes = fetch_adapter_file(
                &client,
                token.as_deref(),
                &adapter_repo_id,
                &weights,
                &target_dir,
            )
            .await?;

            std::fs::write(target_dir.join("adapter_model.safetensors"), &weight_bytes)
                .map_err(|e| format!("could not save the adapter weights: {e}"))?;

            let config = find_named(&filenames, "adapter_config.json").ok_or_else(|| {
                let _ = std::fs::remove_dir_all(&target_dir);
                "This adapter has no adapter_config.json, so Sarathi cannot tell how it \
                 was trained."
                    .to_string()
            })?;

            let config_bytes = fetch_adapter_file(
                &client,
                token.as_deref(),
                &adapter_repo_id,
                &config,
                &target_dir,
            )
            .await?;

            std::fs::write(target_dir.join("adapter_config.json"), &config_bytes)
                .map_err(|e| format!("could not save the adapter configuration: {e}"))?;

            // The base model already on disk supplies the architecture, so this
            // needs no network and cannot disagree with what the adapter will
            // actually attach to.
            let base_gguf = base_gguf.ok_or_else(|| {
                let _ = std::fs::remove_dir_all(&target_dir);
                format!(
                    "No base model for '{model_id}' is installed. Download the model first — \
                     an adapter is converted against the model it attaches to."
                )
            })?;

            let summary = crate::lora::convert_adapter(&target_dir, &base_gguf).map_err(|e| {
                let _ = std::fs::remove_dir_all(&target_dir);
                format!("{e:#}")
            })?;

            // The safetensors were the input, and keeping them doubles the
            // adapter's footprint for a file nothing reads again. Removing them
            // only after the GGUF exists means a failed conversion still leaves
            // the download intact for the rollback above to clear.
            let source = target_dir.join("adapter_model.safetensors");
            if let Err(e) = std::fs::remove_file(&source) {
                log::warn!("[ADAPTERS] Could not remove {}: {e}", source.display());
            }

            downloaded_bytes = summary.output_bytes as usize;
            conversion = Some(summary);
        }
    }

    // Verify what actually arrived. Checking the magic bytes only proves it is
    // *a* GGUF; a whole model passes that. Reading what the file declares itself
    // to be is what separates an adapter from a model. A converted adapter goes
    // through the same gate as a downloaded one — nothing is trusted because of
    // where it came from.
    if let Err(e) = crate::adapter_manager::gguf::verify_is_lora_adapter(&file_path) {
        let _ = std::fs::remove_dir_all(&target_dir);
        return Err(e.to_string());
    }

    // Record where it came from so the UI can link back to the source, and so a
    // later startup scan can re-infer the capability if the manifest is lost.
    let _ = std::fs::write(target_dir.join("source.txt"), &adapter_repo_id);

    // Register it against a capability.
    //
    // This is the step whose absence made every installed adapter inert: the
    // capability resolver can only bind what `manifest.adapters[<capability>]`
    // names, so an adapter that was downloaded, converted and verified but never
    // recorded here would show as installed and never once be used.
    let assignment = crate::capability::assign::infer(&adapter_repo_id, &tags);
    let capability = assignment.as_ref().map(|a| a.capability.clone());

    if let Some(assigned) = &assignment {
        if let Err(e) = register_adapter(
            &install_package,
            &provider_id,
            &model_id,
            &adapter_repo_id,
            &dir_name,
            assigned,
            downloaded_bytes as u64,
            conversion.as_ref(),
        ) {
            // The files are on disk and valid; only the wiring failed. Losing the
            // adapter over it would be worse than leaving it unassigned, and the
            // startup scan will try again on the next launch.
            log::warn!(
                "[ADAPTERS] Installed '{adapter_repo_id}' but could not register it: {e:#}"
            );
        }
    } else {
        log::info!(
            "[ADAPTERS] '{adapter_repo_id}' gives no sign of what it is for — installed \
             unassigned, awaiting a choice"
        );
    }

    log::info!(
        "[ADAPTERS] Installed '{adapter_repo_id}' for '{model_id}' ({downloaded_bytes} bytes, \
         capability {})",
        capability.as_deref().unwrap_or("unassigned"),
    );

    Ok(InstalledAdapter {
        id: dir_name,
        name: adapter_repo_id
            .split('/')
            .next_back()
            .unwrap_or(&adapter_repo_id)
            .replace(['-', '_'], " "),
        repo_id: adapter_repo_id,
        base_model_id: model_id,
        file_path: file_path.to_string_lossy().to_string(),
        size_bytes: downloaded_bytes as u64,
        assignment_confidence: assignment.map(|a| a.confidence.as_str().to_string()),
        capability,
    })
}

/// Writes an installed adapter into its package manifest under a capability.
///
/// Takes the package that holds the *base model*, which is not always the one
/// the user was browsing — see the note above `install_package`.
#[allow(clippy::too_many_arguments)]
fn register_adapter(
    install_package: &std::path::Path,
    provider_id: &str,
    model_id: &str,
    adapter_repo_id: &str,
    dir_name: &str,
    assignment: &crate::capability::CapabilityAssignment,
    size_bytes: u64,
    conversion: Option<&crate::lora::ConversionSummary>,
) -> anyhow::Result<()> {
    // A package downloaded before manifests existed, or one whose manifest was
    // lost, still needs somewhere to put this. `ensure_valid_manifest` builds one
    // from what is on disk.
    let mut manifest = match AdapterRegistry::read_manifest(install_package) {
        Ok(m) => m,
        Err(_) => AdapterRegistry::ensure_valid_manifest(install_package, provider_id, model_id)?,
    };

    let record = crate::adapter_manager::AdapterManifestInfo {
        capability: assignment.capability.clone(),
        status: "installed".to_string(),
        // It has just been through `verify_is_lora_adapter`, so it is loadable as
        // it stands — which is exactly what this status is asked about later.
        adapter_runtime_status: Some("compatible".to_string()),
        repo_id: Some(adapter_repo_id.to_string()),
        local_path: Some(format!("adapters/{}/", dir_name)),
        adapter_file: Some(format!(
            "adapters/{}/{}",
            dir_name,
            crate::lora::convert::CONVERTED_FILENAME
        )),
        config_file: None,
        size_bytes: Some(size_bytes),
        base_model_match: Some(manifest.base_model.model_id.clone()),
        target_modules: conversion.map(|c| c.target_modules.clone()).unwrap_or_default(),
        peft_type: Some("LORA".to_string()),
        checksum: None,
        reason: None,
        // Left unset so the resolver applies its own default. Writing 1.0 here
        // would freeze today's default into every record ever created.
        scale: None,
        rank: conversion.and_then(|c| c.rank),
        alpha: conversion.map(|c| c.alpha),
        architecture: conversion.map(|c| c.architecture.clone()),
        source: Some(crate::adapter_manager::SOURCE_USER.to_string()),
        assignment_confidence: Some(assignment.confidence.as_str().to_string()),
    };

    manifest.adapters.insert(assignment.capability.clone(), record);
    manifest.updated_at = chrono::Utc::now().to_rfc3339();
    AdapterRegistry::write_manifest(install_package, &manifest)
}

#[tauri::command]
pub async fn remove_adapter(
    app: AppHandle,
    provider_id: String,
    model_id: String,
    adapter_id: String,
) -> Result<(), String> {
    let package = package_dir_for(&app, &provider_id, &model_id)?;
    store::remove(&package, &adapter_id).map_err(|e| e.to_string())?;

    // Drop the manifest record too. Leaving it behind would point a capability at
    // a file that no longer exists, which the resolver survives but reports as a
    // missing adapter every turn rather than cleanly falling back.
    if let Err(e) =
        clear_assignment(&package, &adapter_id, "The adapter that filled this slot was removed.")
    {
        log::warn!("[ADAPTERS] Removed '{adapter_id}' but its manifest entry lingers: {e:#}");
    }

    Ok(())
}

/// Releases whatever capability an adapter currently holds.
///
/// Returns the record that was cleared, so a caller reassigning the same adapter
/// can carry its shape data — rank, alpha, architecture — across to the new slot
/// instead of losing it.
fn clear_assignment(
    package: &std::path::Path,
    adapter_id: &str,
    note: &str,
) -> anyhow::Result<Option<crate::adapter_manager::AdapterManifestInfo>> {
    let mut manifest = match AdapterRegistry::read_manifest(package) {
        Ok(m) => m,
        // No manifest means nothing to clear.
        Err(_) => return Ok(None),
    };

    let prefix = format!("adapters/{}/", adapter_id);
    let held = manifest
        .adapters
        .iter()
        .find(|(_, a)| a.adapter_file.as_deref().map(|f| f.starts_with(&prefix)).unwrap_or(false))
        .map(|(key, record)| (key.clone(), record.clone()));

    let Some((key, record)) = held else {
        return Ok(None);
    };

    // The slot is replaced rather than deleted. `write_manifest` re-inserts any
    // key it finds missing whose files still exist, so a bare removal would be
    // merged straight back in.
    manifest.adapters.insert(
        key.clone(),
        crate::adapter_manager::AdapterManifestInfo {
            capability: key,
            status: "Unavailable".to_string(),
            reason: Some(note.to_string()),
            ..Default::default()
        },
    );
    manifest.updated_at = chrono::Utc::now().to_rfc3339();
    // User-initiated: the protected path would restore the record we are
    // deliberately clearing whenever the adapter sits in a capability-named
    // directory.
    AdapterRegistry::write_manifest_user_initiated(package, &manifest)?;

    Ok(Some(record))
}

/// Points a capability at an installed adapter, or unassigns one.
///
/// `capability` of `None` means "stop using this adapter" — the files stay on
/// disk. Assigning a capability that another adapter already holds displaces
/// that one; it becomes unassigned rather than being deleted, because only one
/// adapter can be bound per capability and the user has just said which.
#[tauri::command]
pub async fn set_adapter_capability(
    app: AppHandle,
    provider_id: String,
    model_id: String,
    adapter_id: String,
    capability: Option<String>,
) -> Result<(), String> {
    let package = package_dir_for(&app, &provider_id, &model_id)?;

    if let Some(key) = &capability {
        if !crate::capability::assign::is_known_capability(key) {
            return Err(format!("'{key}' is not a capability Sarathi can route to."));
        }
    }

    // Whatever this adapter held before, it does not hold any more. Doing this
    // first keeps an adapter from occupying two slots after a reassignment, and
    // hands back the old record so its shape data survives the move.
    let previous = clear_assignment(
        &package,
        &adapter_id,
        "This adapter was assigned to a different capability.",
    )
    .map_err(|e| e.to_string())?;

    let Some(key) = capability else {
        log::info!("[ADAPTERS] '{adapter_id}' unassigned for '{model_id}'");
        return Ok(());
    };

    let adapter_dir = store::adapters_root(&package).join(&adapter_id);
    let (weight_file, size_bytes) = AdapterRegistry::verify_adapter_files(&adapter_dir)
        .ok_or_else(|| format!("No usable adapter found in '{adapter_id}'."))?;

    let mut manifest = AdapterRegistry::read_manifest(&package)
        .map_err(|e| format!("could not read the model's manifest: {e}"))?;

    let previous = previous.as_ref();
    let record = crate::adapter_manager::AdapterManifestInfo {
        capability: key.clone(),
        status: "installed".to_string(),
        adapter_runtime_status: Some(
            if weight_file.to_lowercase().ends_with(".gguf") {
                "compatible"
            } else {
                "requires_conversion"
            }
            .to_string(),
        ),
        repo_id: std::fs::read_to_string(adapter_dir.join("source.txt"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        local_path: Some(format!("adapters/{}/", adapter_id)),
        adapter_file: Some(format!("adapters/{}/{}", adapter_id, weight_file)),
        config_file: None,
        size_bytes: Some(size_bytes),
        base_model_match: Some(manifest.base_model.model_id.clone()),
        target_modules: previous.map(|a| a.target_modules.clone()).unwrap_or_default(),
        peft_type: Some("LORA".to_string()),
        checksum: None,
        reason: None,
        // Shape data survives a reassignment: it describes the adapter, not the
        // slot, and re-deriving it would mean reading a config the conversion
        // already deleted.
        scale: previous.and_then(|a| a.scale),
        rank: previous.and_then(|a| a.rank),
        alpha: previous.and_then(|a| a.alpha),
        architecture: previous.and_then(|a| a.architecture.clone()),
        source: Some(crate::adapter_manager::SOURCE_USER.to_string()),
        assignment_confidence: Some(
            crate::capability::AssignmentConfidence::Manual.as_str().to_string(),
        ),
    };

    manifest.adapters.insert(key.clone(), record);
    manifest.updated_at = chrono::Utc::now().to_rfc3339();
    AdapterRegistry::write_manifest_user_initiated(&package, &manifest)
        .map_err(|e| e.to_string())?;

    log::info!("[ADAPTERS] '{adapter_id}' assigned to '{key}' for '{model_id}'");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_manager::{AdapterManifestInfo, BaseManifestInfo, ModelPackageManifest};
    use crate::capability::assign::{AssignmentConfidence, CapabilityAssignment};
    use std::collections::HashMap;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sarathi_cmd_adapters_{name}_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A package with a manifest and one installed GGUF adapter directory.
    fn package_with_adapter(name: &str, dir_name: &str) -> PathBuf {
        let package = scratch(name);
        let adapters = package.join("adapters").join(dir_name);
        fs::create_dir_all(&adapters).unwrap();
        fs::write(adapters.join("adapter.gguf"), vec![0u8; 150_000]).unwrap();
        fs::write(adapters.join("source.txt"), "someone/sql-coder-lora").unwrap();

        let manifest = ModelPackageManifest {
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
        };
        AdapterRegistry::write_manifest_user_initiated(&package, &manifest).unwrap();
        package
    }

    #[test]
    fn registering_an_adapter_makes_it_reachable_from_a_capability() {
        // The gap this closes: before, an adapter was downloaded, converted and
        // verified, and then nothing wrote it into the manifest — so the resolver,
        // which can only bind what the manifest names, never saw it.
        let package = package_with_adapter("register", "someone_sql-coder-lora");
        let assignment = CapabilityAssignment {
            capability: "coding".to_string(),
            confidence: AssignmentConfidence::Stated,
        };
        let summary = crate::lora::ConversionSummary {
            output: package.join("adapters/someone_sql-coder-lora/adapter.gguf"),
            architecture: "qwen3".to_string(),
            tensors_written: 128,
            non_lora_skipped: 0,
            alpha: 32.0,
            output_bytes: 150_000,
            rank: Some(16),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        };

        register_adapter(
            &package,
            "huggingface",
            "Qwen/Qwen3-4B",
            "someone/sql-coder-lora",
            "someone_sql-coder-lora",
            &assignment,
            150_000,
            Some(&summary),
        )
        .unwrap();

        let record = AdapterRegistry::read_manifest(&package).unwrap().adapters.remove("coding");
        let record = record.expect("the coding slot must now be filled");

        assert_eq!(
            record.adapter_file.as_deref(),
            Some("adapters/someone_sql-coder-lora/adapter.gguf")
        );
        assert_eq!(record.adapter_runtime_status.as_deref(), Some("compatible"));
        assert_eq!(record.assignment_confidence.as_deref(), Some("stated"));
        assert_eq!(record.source.as_deref(), Some(crate::adapter_manager::SOURCE_USER));
        // Shape data the conversion knew and the config file no longer holds.
        assert_eq!(record.rank, Some(16));
        assert_eq!(record.alpha, Some(32.0));
        assert_eq!(record.architecture.as_deref(), Some("qwen3"));
        assert_eq!(record.target_modules, vec!["q_proj".to_string(), "v_proj".to_string()]);
        // Left unset so the resolver supplies its own default rather than
        // freezing today's value into the record.
        assert_eq!(record.scale, None);
    }

    #[test]
    fn clearing_an_assignment_returns_the_record_so_its_shape_survives_a_move() {
        let package = package_with_adapter("clear", "someone_sql-coder-lora");
        let mut manifest = AdapterRegistry::read_manifest(&package).unwrap();
        manifest.adapters.insert(
            "coding".to_string(),
            AdapterManifestInfo {
                capability: "coding".to_string(),
                status: "installed".to_string(),
                adapter_file: Some(
                    "adapters/someone_sql-coder-lora/adapter.gguf".to_string(),
                ),
                rank: Some(16),
                architecture: Some("qwen3".to_string()),
                ..Default::default()
            },
        );
        AdapterRegistry::write_manifest_user_initiated(&package, &manifest).unwrap();

        let cleared = clear_assignment(&package, "someone_sql-coder-lora", "moved").unwrap();

        let cleared = cleared.expect("the record it held must come back");
        assert_eq!(cleared.rank, Some(16));
        assert_eq!(cleared.architecture.as_deref(), Some("qwen3"));

        let after = AdapterRegistry::read_manifest(&package).unwrap();
        assert_eq!(after.adapters["coding"].status, "Unavailable");
        assert_eq!(after.adapters["coding"].adapter_file, None);
    }

    #[test]
    fn clearing_is_a_no_op_for_an_adapter_that_holds_nothing() {
        let package = package_with_adapter("clear_noop", "someone_sql-coder-lora");

        let cleared = clear_assignment(&package, "someone_sql-coder-lora", "moved").unwrap();

        assert!(cleared.is_none());
    }

    #[test]
    fn clearing_only_touches_the_slot_the_named_adapter_holds() {
        // Two adapters, two slots. Releasing one must leave the other alone.
        let package = package_with_adapter("clear_scoped", "coder-lora");
        let other = package.join("adapters").join("math-lora");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("adapter.gguf"), vec![0u8; 150_000]).unwrap();

        let mut manifest = AdapterRegistry::read_manifest(&package).unwrap();
        for (slot, dir) in [("coding", "coder-lora"), ("mathematics", "math-lora")] {
            manifest.adapters.insert(
                slot.to_string(),
                AdapterManifestInfo {
                    capability: slot.to_string(),
                    status: "installed".to_string(),
                    adapter_file: Some(format!("adapters/{dir}/adapter.gguf")),
                    ..Default::default()
                },
            );
        }
        AdapterRegistry::write_manifest_user_initiated(&package, &manifest).unwrap();

        clear_assignment(&package, "coder-lora", "moved").unwrap();

        let after = AdapterRegistry::read_manifest(&package).unwrap();
        assert_eq!(after.adapters["coding"].status, "Unavailable");
        assert_eq!(after.adapters["mathematics"].status, "installed");
    }
}

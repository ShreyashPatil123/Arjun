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
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty());

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

    let gguf_file = store::check_installable(&filenames).map_err(|e| e.to_string())?;

    // 2. Fetch it.
    let dir_name = store::adapter_dir_name(&adapter_repo_id);
    let target_dir = store::adapters_root(&package).join(&dir_name);
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("could not create the adapter folder: {e}"))?;

    let url =
        format!("https://huggingface.co/{adapter_repo_id}/resolve/main/{gguf_file}?download=true");
    let mut req = client.get(&url);
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }

    let resp = req.send().await.map_err(|e| format!("download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download failed with {}", resp.status()));
    }

    // A LoRA adapter is a small delta — tens to a few hundred megabytes. A file
    // far past that is a merged model wearing an adapter label, and buffering it
    // would take gigabytes of memory before anything could reject it.
    if let Some(len) = resp.content_length() {
        if len > MAX_ADAPTER_BYTES {
            let _ = std::fs::remove_dir_all(&target_dir);
            return Err(format!(
                "This file is {:.1} GB. LoRA adapters are small add-ons, so this is \
                 almost certainly a complete model rather than an adapter.",
                len as f64 / 1_073_741_824.0
            ));
        }
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("download was interrupted: {e}"))?;

    let file_path = target_dir.join("adapter.gguf");
    std::fs::write(&file_path, &bytes).map_err(|e| format!("could not save the adapter: {e}"))?;

    // Verify what actually arrived. Checking the magic bytes only proves it is
    // *a* GGUF; a whole model passes that. Reading what the file declares itself
    // to be is what separates an adapter from a model.
    if let Err(e) = crate::adapter_manager::gguf::verify_is_lora_adapter(&file_path) {
        let _ = std::fs::remove_dir_all(&target_dir);
        return Err(e.to_string());
    }

    // Record where it came from so the UI can link back to the source.
    let _ = std::fs::write(target_dir.join("source.txt"), &adapter_repo_id);

    log::info!(
        "[ADAPTERS] Installed '{adapter_repo_id}' for '{model_id}' ({} bytes)",
        bytes.len()
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
        size_bytes: bytes.len() as u64,
    })
}

#[tauri::command]
pub async fn remove_adapter(
    app: AppHandle,
    provider_id: String,
    model_id: String,
    adapter_id: String,
) -> Result<(), String> {
    let package = package_dir_for(&app, &provider_id, &model_id)?;
    store::remove(&package, &adapter_id).map_err(|e| e.to_string())
}

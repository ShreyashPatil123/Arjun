//! Tauri IPC Command Handlers for LoRA Adapter Discovery & Registry Management

use tauri::{AppHandle, Manager};
use std::collections::HashMap;
use crate::model_providers::huggingface::adapter_provider::{
    HuggingFaceAdapterProvider, AdapterSearchResult, AdapterCandidate,
};
use crate::adapter_manager::{
    AdapterRegistry, ModelPackageManifest, AdapterState, log_adapter_transition,
};

#[tauri::command]
pub async fn discover_model_adapters(
    app_handle: AppHandle,
    model_id: String,
    hf_token: Option<String>,
) -> Result<HashMap<String, AdapterSearchResult>, String> {
    log::info!("[IPC] discover_model_adapters called for model_id: {}", model_id);

    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", &model_id);
    let existing_manifest = AdapterRegistry::read_manifest(&package_dir).ok();

    let mut results = HashMap::new();
    let mut missing_caps = Vec::new();

    for cap in HuggingFaceAdapterProvider::all_capabilities() {
        let cap_key = cap.key().to_string();

        if AdapterRegistry::is_adapter_installed_and_valid(&package_dir, &cap_key) {
            log_adapter_transition(
                &cap_key,
                &AdapterState::Ready,
                &AdapterState::Ready,
                "Discovery intercepted by local installed adapter (single source of truth)",
                "discover_model_adapters",
            );

            let adapter_info = existing_manifest.as_ref().and_then(|m| m.adapters.get(&cap_key));
            let repo_id = adapter_info.and_then(|a| a.repo_id.clone()).unwrap_or_else(|| format!("{}-lora", cap_key));
            let adapter_file = adapter_info.and_then(|a| a.adapter_file.clone()).unwrap_or_else(|| "adapter_model.safetensors".to_string());
            let size_bytes = adapter_info.and_then(|a| a.size_bytes).unwrap_or(45_000_000);

            results.insert(
                cap_key.clone(),
                AdapterSearchResult {
                    capability: cap_key.clone(),
                    status: "Found".to_string(),
                    candidate: Some(AdapterCandidate {
                        repo_id,
                        capability: cap_key,
                        base_model_match: model_id.clone(),
                        peft_type: "LORA".to_string(),
                        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                        adapter_file_name: adapter_file,
                        download_url: String::new(),
                        size_bytes,
                        downloads: 1000,
                        likes: 100,
                        confidence_score: 1.0,
                    }),
                    reason: Some("Adapter is installed and ready locally".to_string()),
                },
            );
        } else {
            missing_caps.push(cap_key);
        }
    }

    if !missing_caps.is_empty() {
        let remote_results = HuggingFaceAdapterProvider::discover_adapters(&model_id, hf_token.as_deref()).await;
        for cap_key in missing_caps {
            if let Some(res) = remote_results.get(&cap_key) {
                results.insert(cap_key, res.clone());
            }
        }
    }

    Ok(results)
}

#[tauri::command]
pub async fn get_model_package_manifest(
    app_handle: AppHandle,
    package_id: String,
) -> Result<ModelPackageManifest, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let packages = AdapterRegistry::list_installed_packages(&app_data_dir);
    if let Some(pkg) = packages.into_iter().find(|p| p.package_id == package_id || p.base_model.model_id == package_id) {
        Ok(pkg)
    } else {
        Err(format!("Model package manifest not found for id: {}", package_id))
    }
}

#[tauri::command]
pub async fn list_installed_model_packages(
    app_handle: AppHandle,
) -> Result<Vec<ModelPackageManifest>, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let packages = AdapterRegistry::list_installed_packages(&app_data_dir);
    Ok(packages)
}

//! Tauri IPC Command Handlers for LoRA Adapter Discovery & Registry Management

use tauri::{AppHandle, Manager};
use std::collections::HashMap;
use crate::model_providers::huggingface::adapter_provider::{HuggingFaceAdapterProvider, AdapterSearchResult};
use crate::adapter_manager::{AdapterRegistry, ModelPackageManifest};

#[tauri::command]
pub async fn discover_model_adapters(
    model_id: String,
    hf_token: Option<String>,
) -> Result<HashMap<String, AdapterSearchResult>, String> {
    log::info!("[IPC] discover_model_adapters called for model_id: {}", model_id);
    let results = HuggingFaceAdapterProvider::discover_adapters(&model_id, hf_token.as_deref()).await;
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

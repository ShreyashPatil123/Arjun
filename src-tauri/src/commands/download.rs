//! Phase 4 Tauri Commands for Model Downloads & Storage Management

use tauri::{AppHandle, Manager, State};
use std::sync::Arc;
use anyhow::Result;

use crate::download_manager::traits::{DownloadTask, InstalledModel, StorageSummary};
use crate::download_manager::DownloadManager;
use crate::model_manager::ModelManager;

#[tauri::command]
pub async fn start_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    model_id: String,
    model_name: String,
    provider_id: String,
    quantization: String,
    format: String,
    backend: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    download_mgr
        .start_download(
            app_handle.clone(),
            app_data_dir,
            model_id,
            model_name,
            provider_id,
            quantization,
            format,
            backend,
            hf_token,
        )
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn pause_model_download(
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
) -> Result<(), String> {
    download_mgr.pause_download(&task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn resume_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    download_mgr
        .resume_download(app_handle.clone(), app_data_dir, &task_id, hf_token)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn cancel_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
) -> Result<(), String> {
    download_mgr.cancel_download(&app_handle, &task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_active_downloads(
    download_mgr: State<'_, Arc<DownloadManager>>,
) -> Result<Vec<DownloadTask>, String> {
    Ok(download_mgr.list_tasks())
}

#[tauri::command]
pub fn get_installed_models(
    app_handle: AppHandle,
) -> Result<Vec<InstalledModel>, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::list_installed_models(&app_data_dir))
}

#[tauri::command]
pub fn delete_installed_model(
    app_handle: AppHandle,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<(), String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    ModelManager::delete_installed_model(&app_data_dir, &provider_id, &model_id, &quantization)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_storage_summary(
    app_handle: AppHandle,
) -> Result<StorageSummary, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::get_storage_summary(&app_data_dir))
}

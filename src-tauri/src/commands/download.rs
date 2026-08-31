//! Phase 4 Tauri Commands for Model Downloads & Storage Management

use tauri::{AppHandle, Manager, State};
use std::sync::Arc;
use anyhow::Result;

use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::download_manager::traits::{DownloadTask, InstalledModel, StorageSummary};
use crate::download_manager::DownloadManager;
use crate::identity::Permission;
use crate::model_manager::ModelManager;

#[tauri::command]
pub async fn start_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    session: State<'_, CurrentSession>,
    model_id: String,
    model_name: String,
    provider_id: String,
    quantization: String,
    format: String,
    backend: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    // A model download is the first half of installing a model. The
    // matrix puts it under `ImportModel`. `User` and below must not
    // start one.
    require_permission(&session, Permission::ImportModel)?;

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
    session: State<'_, CurrentSession>,
    task_id: String,
) -> Result<(), String> {
    require_permission(&session, Permission::ImportModel)?;
    download_mgr.pause_download(&task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn resume_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    session: State<'_, CurrentSession>,
    task_id: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    require_permission(&session, Permission::ImportModel)?;

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
    session: State<'_, CurrentSession>,
    task_id: String,
) -> Result<(), String> {
    require_permission(&session, Permission::ImportModel)?;
    download_mgr.cancel_download(&app_handle, &task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_active_downloads(
    download_mgr: State<'_, Arc<DownloadManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<DownloadTask>, String> {
    // The list of in-flight downloads is not sensitive; any signed-in
    // user can see it. The matrix does not gate read-only inspection.
    require_session(&session)?;
    Ok(download_mgr.list_tasks())
}

#[tauri::command]
pub fn get_installed_models(
    app_handle: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<Vec<InstalledModel>, String> {
    require_session(&session)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::list_installed_models(&app_data_dir))
}

#[tauri::command]
pub fn delete_installed_model(
    app_handle: AppHandle,
    session: State<'_, CurrentSession>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<(), String> {
    // Deleting a model is the inverse of installing one. Same gate.
    require_permission(&session, Permission::ImportModel)?;

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
    session: State<'_, CurrentSession>,
) -> Result<StorageSummary, String> {
    require_session(&session)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(ModelManager::get_storage_summary(&app_data_dir))
}

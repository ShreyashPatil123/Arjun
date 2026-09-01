//! Model Intelligence Tauri IPC Commands

use std::sync::Arc;

use tauri::{Manager, State};

use crate::model_package::ModelPackageRegistry;
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::model_intelligence::{
    telemetry::{ModelAggregate, TelemetrySink},
    InferenceParameters, ModelIntelligenceManager, ModelProfile,
};

/// Reads the in-memory aggregate for the Model Health page. Bounded:
/// one row per model id, with the most recent calls collapsed. The
/// audit log is the source of truth for full history; this is the
/// "what is right now" view.
#[tauri::command]
pub async fn model_health_snapshot(
    state: tauri::State<'_, Arc<TelemetrySink>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ModelAggregate>, String> {
    // Read-only model health summary. The matrix does not gate it
    // beyond sign-in.
    require_session(&session)?;
    Ok(state.snapshot())
}

#[tauri::command]
pub async fn get_model_profile(
    app_handle: tauri::AppHandle,
    session: State<'_, CurrentSession>,
    provider_id: String,
    model_id: String,
) -> Result<ModelProfile, String> {
    // Read-only profile inspection. Any signed-in user may see it.
    require_session(&session)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir =
        ModelPackageRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest =
        ModelPackageRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest)
        .map_err(|e| format!("Failed to load model profile: {}", e))
}

#[tauri::command]
pub async fn update_model_profile(
    app_handle: tauri::AppHandle,
    session: State<'_, CurrentSession>,
    provider_id: String,
    model_id: String,
    params: InferenceParameters,
) -> Result<ModelProfile, String> {
    // Writing a model profile changes how the model answers the next
    // prompt. That is part of the model-configuration story, so the
    // matrix puts it under `ImportModel`.
    require_permission(&session, Permission::ImportModel)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir =
        ModelPackageRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest =
        ModelPackageRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    let mut profile = ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest)
        .map_err(|e| e.to_string())?;

    profile.active_user_params = Some(params);
    profile.updated_at = chrono::Utc::now().to_rfc3339();

    ModelIntelligenceManager::write_profile(&package_dir, &profile)
        .map_err(|e| format!("Failed to write profile: {}", e))?;

    Ok(profile)
}

#[tauri::command]
pub async fn refresh_model_profile(
    app_handle: tauri::AppHandle,
    session: State<'_, CurrentSession>,
    provider_id: String,
    model_id: String,
) -> Result<ModelProfile, String> {
    // Re-deriving a model profile re-reads the on-disk manifest and
    // recomputes the model's capabilities. The matrix puts any write
    // that changes model configuration under `ImportModel`.
    require_permission(&session, Permission::ImportModel)?;
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir =
        ModelPackageRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest =
        ModelPackageRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    ModelIntelligenceManager::refresh_profile(&package_dir, &manifest)
        .map_err(|e| format!("Failed to refresh model profile: {}", e))
}

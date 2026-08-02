//! Model Intelligence Tauri IPC Commands

use tauri::Manager;

use crate::adapter_manager::AdapterRegistry;
use crate::model_intelligence::{
    AdapterRouteResult, AdapterRouter, ModelIntelligenceManager, ModelProfile, InferenceParameters,
};

#[tauri::command]
pub async fn get_model_profile(
    app_handle: tauri::AppHandle,
    provider_id: String,
    model_id: String,
) -> Result<ModelProfile, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest = AdapterRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest)
        .map_err(|e| format!("Failed to load model profile: {}", e))
}

#[tauri::command]
pub async fn update_model_profile(
    app_handle: tauri::AppHandle,
    provider_id: String,
    model_id: String,
    params: InferenceParameters,
) -> Result<ModelProfile, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest = AdapterRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

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
    provider_id: String,
    model_id: String,
) -> Result<ModelProfile, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest = AdapterRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    ModelIntelligenceManager::refresh_profile(&package_dir, &manifest)
        .map_err(|e| format!("Failed to refresh model profile: {}", e))
}

#[tauri::command]
pub async fn route_prompt_capability(
    app_handle: tauri::AppHandle,
    provider_id: String,
    model_id: String,
    prompt: String,
    user_override: Option<String>,
) -> Result<AdapterRouteResult, String> {
    let start_time = std::time::Instant::now();
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app_data_dir: {}", e))?;

    let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, &provider_id, &model_id);
    let manifest = AdapterRegistry::read_manifest(&package_dir).map_err(|e| e.to_string())?;

    let route = AdapterRouter::select_adapter_for_prompt(
        &package_dir,
        &manifest,
        &prompt,
        user_override.as_deref(),
    );

    log::info!(
        "[INTENT_ROUTER] Prompt Intent Analysis complete in {}ms | Intent: {:?} | Target Capability: '{}' | Selected Adapter: {:?}",
        start_time.elapsed().as_millis(), route.intent, route.target_capability, route.selected_adapter_name
    );

    Ok(route)
}

//! System IPC commands for Sarathi

use serde_json::json;
use tauri::AppHandle;
use tauri::Manager;

use crate::core::app_state::{get_app_state, AppStateData};
use crate::system_analyzer::{get_system_analyzer_manager, HardwareProfile, SystemValidationResult};

/// Returns basic application info
#[tauri::command]
pub async fn get_app_info(app: AppHandle) -> Result<serde_json::Value, String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "data_dir": data_dir,
    }))
}

/// Returns current application state
#[tauri::command]
pub async fn get_app_state_info() -> Result<AppStateData, String> {
    Ok(get_app_state().get())
}

/// Records activity to log (placeholder for future implementation)
#[tauri::command]
pub async fn log_activity(
    action: String,
    category: String,
    details: Option<String>,
) -> Result<(), String> {
    log::info!("Activity: {} [{}] - {:?}", action, category, details);
    Ok(())
}

/// Retrieves cached hardware profile if available
#[tauri::command]
pub async fn get_hardware_profile() -> Result<Option<HardwareProfile>, String> {
    Ok(get_system_analyzer_manager().get_profile())
}

/// Triggers full system analysis and updates cached profile
#[tauri::command]
pub async fn analyze_system() -> Result<HardwareProfile, String> {
    get_system_analyzer_manager()
        .analyze_system()
        .map_err(|e| e.to_string())
}

/// Applies a manual override to a hardware/software profile field
#[tauri::command]
pub async fn override_hardware_value(
    field_path: String,
    value: serde_json::Value,
) -> Result<HardwareProfile, String> {
    get_system_analyzer_manager()
        .override_value(&field_path, value)
        .map_err(|e| e.to_string())
}

/// Reverts a hardware/software field override back to detected value
#[tauri::command]
pub async fn revert_hardware_override(field_path: String) -> Result<HardwareProfile, String> {
    get_system_analyzer_manager()
        .revert_override(&field_path)
        .map_err(|e| e.to_string())
}

/// Evaluates current system hardware readiness for AI model execution
#[tauri::command]
pub async fn validate_system() -> Result<SystemValidationResult, String> {
    let manager = get_system_analyzer_manager();
    if let Some(profile) = manager.get_profile() {
        Ok(profile.validation)
    } else {
        let profile = manager.analyze_system().map_err(|e| e.to_string())?;
        Ok(profile.validation)
    }
}

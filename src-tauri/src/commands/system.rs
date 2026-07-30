//! System commands

use serde_json::json;
use tauri::AppHandle;
use tauri::Manager;

use crate::core::app_state::{get_app_state, AppStateData};

/// Returns basic application info
#[tauri::command]
pub async fn get_app_info(app: AppHandle) -> Result<serde_json::Value, String> {
    let data_dir = app.path().app_data_dir().unwrap_or_default().to_string_lossy().to_string();
    
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
pub async fn log_activity(action: String, category: String, details: Option<String>) -> Result<(), String> {
    log::info!("Activity: {} [{}] - {:?}", action, category, details);
    // In the future this will write to the database
    Ok(())
}

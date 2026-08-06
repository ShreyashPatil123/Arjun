//! Configuration commands

use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Manager;
use std::path::PathBuf;

use crate::config::{ConfigManager, SarathiConfig};
use crate::core::event_bus::{get_event_bus, SarathiEvent};

/// Gets the entire configuration as JSON string
#[tauri::command]
pub async fn get_config(app: AppHandle) -> Result<String, String> {
    let path = ConfigManager::get_config_path(&app);
    let config = ConfigManager::load(&path).map_err(|e| e.to_string())?;
    serde_json::to_string(&config).map_err(|e| e.to_string())
}

/// Sets the entire configuration from JSON string
#[tauri::command]
pub async fn set_config(app: AppHandle, config: String) -> Result<(), String> {
    let new_config: SarathiConfig = serde_json::from_str(&config).map_err(|e| e.to_string())?;
    let path = ConfigManager::get_config_path(&app);
    
    ConfigManager::save(&new_config, &path).map_err(|e| e.to_string())?;
    
    // Publish config changed event
    let event_bus = get_event_bus();
    if let Ok(value) = serde_json::to_value(&new_config) {
        event_bus.publish(SarathiEvent::ConfigChanged, Some(value));
    }
    
    Ok(())
}

/// The HuggingFace token's status, without returning the token itself.
///
/// A saved secret is not echoed back to the UI — the field shows whether one is
/// set and where it came from, which is all that is needed to decide whether to
/// replace it.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfTokenStatus {
    /// True when Hub requests will be authenticated.
    pub configured: bool,
    /// `"settings"`, `"environment"`, or `"none"`.
    pub source: String,
}

#[tauri::command]
pub fn get_hf_token_status() -> HfTokenStatus {
    HfTokenStatus {
        configured: crate::config::hf_token::is_present(),
        source: crate::config::hf_token::source().to_string(),
    }
}

/// Saves the HuggingFace token and applies it to the running process.
///
/// An empty string clears the saved token, after which any `HF_TOKEN` in the
/// environment takes over again.
#[tauri::command]
pub async fn set_hf_token(app: AppHandle, token: String) -> Result<HfTokenStatus, String> {
    let path = ConfigManager::get_config_path(&app);
    let mut config = ConfigManager::load(&path).map_err(|e| e.to_string())?;

    config.hf_token = token.trim().to_string();
    ConfigManager::save(&config, &path).map_err(|e| e.to_string())?;

    // Applied in-process as well as saved, so the next browse uses it without
    // a restart.
    crate::config::hf_token::set(Some(config.hf_token.clone()));

    // The cached sweep was fetched under the previous credentials and is a
    // smaller slice of the library than a token allows; dropping it means the
    // next browse re-fetches rather than showing the anonymous result.
    crate::commands::catalog::invalidate_browse_cache();

    log::info!(
        "[CONFIG] HuggingFace token {}",
        if config.hf_token.is_empty() { "cleared" } else { "saved" }
    );

    Ok(get_hf_token_status())
}

/// Gets a specific config value by key (simplified implementation)
#[tauri::command]
pub async fn get_config_value(app: AppHandle, key: String) -> Result<Option<String>, String> {
    let config_json = get_config(app).await?;
    let config_val: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| e.to_string())?;
    
    if let Some(val) = config_val.get(&key) {
        Ok(Some(val.to_string()))
    } else {
        Ok(None)
    }
}

/// Sets a specific config value (simplified implementation)
#[tauri::command]
pub async fn set_config_value(app: AppHandle, key: String, value: String) -> Result<(), String> {
    let config_json = get_config(app.clone()).await?;
    let mut config_val: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| e.to_string())?;
    
    // Parse value if it's JSON, otherwise treat as string
    let parsed_val = serde_json::from_str(&value).unwrap_or(serde_json::Value::String(value));
    
    if let Some(obj) = config_val.as_object_mut() {
        obj.insert(key, parsed_val);
    }
    
    let updated_config_str = serde_json::to_string(&config_val).map_err(|e| e.to_string())?;
    set_config(app, updated_config_str).await
}

/// Returns the default config map
#[tauri::command]
pub async fn get_default_config() -> Result<String, String> {
    let config = SarathiConfig::default();
    serde_json::to_string(&config).map_err(|e| e.to_string())
}

/// Resets configuration to defaults
#[tauri::command]
pub async fn reset_config(app: AppHandle) -> Result<(), String> {
    let config = SarathiConfig::default();
    let config_str = serde_json::to_string(&config).map_err(|e| e.to_string())?;
    set_config(app, config_str).await
}

/// Returns app paths
#[tauri::command]
pub async fn get_app_paths(app: AppHandle) -> Result<HashMap<String, String>, String> {
    let mut paths = HashMap::new();
    
    if let Ok(data_dir) = app.path().app_data_dir() {
        paths.insert("data_dir".to_string(), data_dir.to_string_lossy().to_string());
    }
    
    if let Ok(log_dir) = app.path().app_log_dir() {
        paths.insert("log_dir".to_string(), log_dir.to_string_lossy().to_string());
    }
    
    let config_path = ConfigManager::get_config_path(&app);
    paths.insert("config_path".to_string(), config_path.to_string_lossy().to_string());
    
    Ok(paths)
}

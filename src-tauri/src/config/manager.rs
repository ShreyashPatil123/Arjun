//! Configuration manager

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Manager;

use super::defaults::SarathiConfig;

/// Manages application configuration
pub struct ConfigManager;

impl ConfigManager {
    /// Loads configuration from a JSON file, or returns defaults if not exists
    pub fn load(path: &Path) -> Result<SarathiConfig> {
        if !path.exists() {
            return Ok(SarathiConfig::default());
        }

        let data = fs::read_to_string(path)?;
        let config: SarathiConfig = serde_json::from_str(&data)?;
        
        Ok(config)
    }

    /// Saves configuration to a JSON file
    pub fn save(config: &SarathiConfig, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let data = serde_json::to_string_pretty(config)?;
        fs::write(path, data)?;
        
        Ok(())
    }

    /// Gets the standard configuration file path for the app
    pub fn get_config_path(app: &tauri::AppHandle) -> PathBuf {
        let app_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
        app_dir.join("config.json")
    }
}

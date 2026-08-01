//! Persistent Model Session Manager
//!
//! Remembers the last loaded model, active profile, and user overrides across app restarts.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastActiveSession {
    pub provider_id: String,
    pub model_id: String,
    pub quantization: String,
    pub loaded_at: String,
    pub auto_restore_enabled: bool,
}

pub struct SessionManager;

impl SessionManager {
    fn get_session_file_path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join("session.json")
    }

    /// Saves the current active model session state
    pub fn save_session(
        app_data_dir: &Path,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
    ) -> Result<()> {
        let session = LastActiveSession {
            provider_id: provider_id.to_string(),
            model_id: model_id.to_string(),
            quantization: quantization.to_string(),
            loaded_at: chrono::Utc::now().to_rfc3339(),
            auto_restore_enabled: true,
        };

        fs::create_dir_all(app_data_dir)?;
        let file_path = Self::get_session_file_path(app_data_dir);
        let data = serde_json::to_string_pretty(&session)?;
        fs::write(&file_path, data)?;
        log::info!("[SESSION] Saved active session for model '{}' at {:?}", model_id, file_path);
        Ok(())
    }

    /// Loads the last active session if present
    pub fn load_session(app_data_dir: &Path) -> Result<Option<LastActiveSession>> {
        let file_path = Self::get_session_file_path(app_data_dir);
        if !file_path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&file_path)?;
        let session: LastActiveSession = serde_json::from_str(&data)?;
        Ok(Some(session))
    }

    /// Clears active session on explicit model unload
    pub fn clear_session(app_data_dir: &Path) -> Result<()> {
        let file_path = Self::get_session_file_path(app_data_dir);
        if file_path.exists() {
            let _ = fs::remove_file(&file_path);
            log::info!("[SESSION] Cleared active model session file {:?}", file_path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_clear_session() {
        let path_buf = std::env::temp_dir().join("sarathi_test_session");
        let path = path_buf.as_path();

        let _ = SessionManager::clear_session(path);
        assert!(SessionManager::load_session(path).unwrap().is_none());

        SessionManager::save_session(path, "huggingface", "meta-llama/Llama-3.2-1B", "Q8_0").unwrap();

        let loaded = SessionManager::load_session(path).unwrap().unwrap();
        assert_eq!(loaded.provider_id, "huggingface");
        assert_eq!(loaded.model_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(loaded.quantization, "Q8_0");
        assert!(loaded.auto_restore_enabled);

        SessionManager::clear_session(path).unwrap();
        assert!(SessionManager::load_session(path).unwrap().is_none());
    }
}

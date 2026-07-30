//! System standard paths collector

use crate::system_analyzer::traits::SystemPaths;
use std::path::PathBuf;

/// Detects standard directories for the application and current user
pub fn detect_paths() -> SystemPaths {
    let home_dir = get_user_home_dir();
    let downloads = home_dir.join("Downloads").to_string_lossy().to_string();
    let documents = home_dir.join("Documents").to_string_lossy().to_string();
    let desktop = home_dir.join("Desktop").to_string_lossy().to_string();

    let app_data = get_app_data_dir(&home_dir);
    let cache_dir = get_cache_dir(&home_dir);
    let model_storage_dir = PathBuf::from(&app_data)
        .join("models")
        .to_string_lossy()
        .to_string();

    SystemPaths {
        user_home: home_dir.to_string_lossy().to_string(),
        downloads,
        documents,
        desktop,
        app_data,
        cache_dir,
        model_storage_dir,
    }
}

fn get_user_home_dir() -> PathBuf {
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home);
    }
    PathBuf::from("C:\\Users\\Default")
}

fn get_app_data_dir(home_dir: &PathBuf) -> String {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata)
            .join("Sarathi")
            .to_string_lossy()
            .to_string();
    }
    home_dir
        .join(".sarathi")
        .join("data")
        .to_string_lossy()
        .to_string()
}

fn get_cache_dir(home_dir: &PathBuf) -> String {
    if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(localappdata)
            .join("Sarathi")
            .join("cache")
            .to_string_lossy()
            .to_string();
    }
    home_dir
        .join(".sarathi")
        .join("cache")
        .to_string_lossy()
        .to_string()
}

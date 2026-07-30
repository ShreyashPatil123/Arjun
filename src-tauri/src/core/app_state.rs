//! Global application state manager

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};

/// Represents the current status of the application
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AppStatus {
    Initializing,
    Ready,
    Downloading,
    Installing,
    LoadingModel,
    LoadingLoRA,
    Chatting,
    Error,
}

/// The state of the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStateData {
    pub status: AppStatus,
    pub version: String,
    pub is_first_run: bool,
    pub uptime_start: DateTime<Utc>,
}

/// Thread-safe wrapper for application state
#[derive(Clone)]
pub struct AppState {
    data: Arc<Mutex<AppStateData>>,
}

impl AppState {
    /// Creates a new AppState instance
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(AppStateData {
                status: AppStatus::Initializing,
                version: env!("CARGO_PKG_VERSION").to_string(),
                is_first_run: true, // This should be determined by config/db check
                uptime_start: Utc::now(),
            })),
        }
    }

    /// Gets a copy of the current state
    pub fn get(&self) -> AppStateData {
        self.data.lock().unwrap().clone()
    }

    /// Updates the application status
    pub fn set_status(&self, status: AppStatus) {
        let mut data = self.data.lock().unwrap();
        data.status = status;
    }

    /// Updates first run flag
    pub fn set_first_run(&self, is_first_run: bool) {
        let mut data = self.data.lock().unwrap();
        data.is_first_run = is_first_run;
    }
}

static APP_STATE: OnceLock<AppState> = OnceLock::new();

/// Gets the global app state
pub fn get_app_state() -> &'static AppState {
    APP_STATE.get_or_init(|| AppState::new())
}

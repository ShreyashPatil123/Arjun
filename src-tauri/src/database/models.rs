//! Database models

use serde::{Deserialize, Serialize};

/// Represents a configuration setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Setting {
    pub key: String,
    pub value: String,
    pub value_type: String,
    pub updated_at: String,
}

/// Represents an activity log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityLogEntry {
    pub id: i64,
    pub action: String,
    pub category: String,
    pub details: Option<String>,
    pub created_at: String,
}

/// Represents a model in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub provider: String,
    pub name: String,
    pub size_bytes: i64,
    pub format: String,
    pub quantization: String,
    pub status: String,
    pub local_path: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Represents a download record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRecord {
    pub id: String,
    pub model_id: String,
    pub url: String,
    pub file_path: String,
    pub total_bytes: i64,
    pub downloaded_bytes: i64,
    pub status: String,
    pub checksum: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Represents an installed LoRA adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledLoRA {
    pub id: String,
    pub name: String,
    pub base_model_id: String,
    pub file_path: String,
    pub size_bytes: i64,
    pub adapter_type: String,
    pub metadata: Option<String>,
    pub is_active: i32, // bool as i32 for SQLite
    pub created_at: String,
    pub updated_at: String,
}

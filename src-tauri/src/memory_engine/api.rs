//! Tauri IPC Commands for Phase 6 Memory Engine
//! Exposes Memory Engine stats, profile, project switching, search, edit, and export to frontend.

use std::sync::Arc;
use tauri::State;
use serde_json::Value;

use crate::commands::governance::{require_session, CurrentSession};
use crate::memory_engine::manager::MemoryManager;
use crate::memory_engine::persistence::{ProjectRecord, UserProfileRecord};
use crate::memory_engine::traits::{ProviderHealthStatus, ScoredCandidate};

#[tauri::command]
pub async fn get_memory_health_status(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
) -> Result<ProviderHealthStatus, String> {
    // Memory is per-user data, not per-role. The matrix does not name
    // it; the per-user scoping lives inside the manager. Sign-in is
    // enough to attribute every read.
    require_session(&session)?;
    memory_mgr.check_health().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_user_profile_memory(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<UserProfileRecord>, String> {
    require_session(&session)?;
    memory_mgr.get_user_profile().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn update_user_profile_fact(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
    key: String,
    value: String,
    category: String,
) -> Result<(), String> {
    require_session(&session)?;
    memory_mgr.set_user_profile_fact(&key, &value, &category).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_memory_projects(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ProjectRecord>, String> {
    require_session(&session)?;
    memory_mgr.list_projects().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn create_memory_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
    name: String,
    description: Option<String>,
) -> Result<ProjectRecord, String> {
    require_session(&session)?;
    memory_mgr.create_project(&name, description.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn switch_active_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
    project_id: String,
) -> Result<String, String> {
    require_session(&session)?;
    memory_mgr.set_active_project_id(&project_id);
    Ok(project_id)
}

#[tauri::command]
pub fn get_active_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
) -> Result<String, String> {
    require_session(&session)?;
    Ok(memory_mgr.get_active_project_id())
}

#[tauri::command]
pub async fn search_memory_nodes(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
    query: String,
    project_id: Option<String>,
) -> Result<Vec<ScoredCandidate>, String> {
    require_session(&session)?;
    memory_mgr.search_memories(&query, project_id.as_deref()).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_memory_node_by_id(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
    id: String,
) -> Result<(), String> {
    require_session(&session)?;
    memory_mgr.delete_memory(&id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_memory_diagnostics(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    session: State<'_, CurrentSession>,
) -> Result<serde_json::Value, String> {
    require_session(&session)?;
    let health = memory_mgr.check_health().await.unwrap_or_else(|_| crate::memory_engine::traits::ProviderHealthStatus {
        status: "degraded".to_string(),
        registered_providers: vec![],
        capabilities: serde_json::json!({}),
    });

    let (nodes_cnt, profile_cnt, projects_cnt) = memory_mgr.get_counts().unwrap_or((0, 0, 0));
    let active_proj = memory_mgr.get_active_project_id();

    Ok(serde_json::json!({
        "memory_provider": memory_mgr.provider_id(),
        "sidecar_status": if health.status == "healthy" { "online" } else { "offline" },
        "database_status": "connected",
        "memory_counts": {
            "memory_nodes": nodes_cnt,
            "user_profile": profile_cnt,
            "projects": projects_cnt
        },
        "active_project": active_proj,
        "health_status": health,
        "last_error": null
    }))
}

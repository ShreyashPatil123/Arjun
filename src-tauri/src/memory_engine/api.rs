//! Tauri IPC Commands for Phase 6 Memory Engine
//! Exposes Memory Engine stats, profile, project switching, search, edit, and export to frontend.

use std::sync::Arc;
use tauri::State;
use serde_json::Value;

use crate::memory_engine::manager::MemoryManager;
use crate::memory_engine::persistence::{ProjectRecord, UserProfileRecord};
use crate::memory_engine::traits::{ProviderHealthStatus, ScoredCandidate};

#[tauri::command]
pub async fn get_memory_health_status(
    memory_mgr: State<'_, Arc<MemoryManager>>,
) -> Result<ProviderHealthStatus, String> {
    memory_mgr.check_health().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_user_profile_memory(
    memory_mgr: State<'_, Arc<MemoryManager>>,
) -> Result<Vec<UserProfileRecord>, String> {
    memory_mgr.get_user_profile().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn update_user_profile_fact(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    key: String,
    value: String,
    category: String,
) -> Result<(), String> {
    memory_mgr.set_user_profile_fact(&key, &value, &category).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_memory_projects(
    memory_mgr: State<'_, Arc<MemoryManager>>,
) -> Result<Vec<ProjectRecord>, String> {
    memory_mgr.list_projects().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn create_memory_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    name: String,
    description: Option<String>,
) -> Result<ProjectRecord, String> {
    memory_mgr.create_project(&name, description.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn switch_active_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    project_id: String,
) -> Result<String, String> {
    memory_mgr.set_active_project_id(&project_id);
    Ok(project_id)
}

#[tauri::command]
pub fn get_active_project(
    memory_mgr: State<'_, Arc<MemoryManager>>,
) -> Result<String, String> {
    Ok(memory_mgr.get_active_project_id())
}

#[tauri::command]
pub async fn search_memory_nodes(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    query: String,
    project_id: Option<String>,
) -> Result<Vec<ScoredCandidate>, String> {
    memory_mgr.search_memories(&query, project_id.as_deref()).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_memory_node_by_id(
    memory_mgr: State<'_, Arc<MemoryManager>>,
    id: String,
) -> Result<(), String> {
    memory_mgr.delete_memory(&id).map_err(|e| e.to_string())
}

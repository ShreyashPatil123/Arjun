//! Phase 5 Tauri IPC Commands for Local Inference Runtime
//!
//! Provides commands to load/unload models, query inference status,
//! send chat prompts, and stop ongoing generation.

use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

use crate::ai_engine::traits::*;
use crate::ai_engine::manager::InferenceManager;

#[tauri::command]
pub async fn load_installed_model(
    app_handle: AppHandle,
    inference_mgr: State<'_, Arc<InferenceManager>>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<LoadedModelInfo, String> {
    log::info!(
        "[STAGE 2 IPC] load_installed_model command entered: provider_id='{}', model_id='{}', quantization='{}'",
        provider_id, model_id, quantization
    );

    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| {
            let err = format!("[STAGE 2 IPC ERROR] Failed to resolve AppData directory: {:?}", e);
            log::error!("{}", err);
            err
        })?;

    log::info!("[STAGE 2 IPC] AppData directory resolved: {:?}", app_data_dir);

    let pack_mgr = app_handle.state::<std::sync::Arc<crate::model_recommendation::pack_manager::PackManager>>();
    let validation = crate::model_recommendation::runtime_validator::RuntimeValidator::validate_before_load(
        &pack_mgr,
        &model_id,
        false, // Developer override default false
    ).map_err(|e| format!("[PRE-LOAD VALIDATION FAILED] Safe Abort: {}", e))?;

    if let Some(warn) = &validation.warning {
        log::warn!("[PRE-LOAD VALIDATION WARNING] {}", warn);
    }

    let mgr = inference_mgr.inner().clone();
    let res = tokio::task::spawn_blocking(move || {
        mgr.load_installed_model(
            &app_handle,
            &app_data_dir,
            &provider_id,
            &model_id,
            &quantization,
        )
    })
    .await
    .map_err(|e| {
        let err = format!("[STAGE 2 IPC ERROR] Task join error: {:?}", e);
        log::error!("{}", err);
        err
    })?
    .map_err(|e| {
        let err = format!("[STAGE 2 IPC ERROR] Manager load failed: {:#}", e);
        log::error!("{}", err);
        err
    });

    match &res {
        Ok(info) => log::info!("[STAGE 2 IPC] load_installed_model succeeded: {:?}", info),
        Err(e) => log::error!("[STAGE 2 IPC FAILED] Returning error: {}", e),
    }

    res
}

#[tauri::command]
pub async fn unload_active_model(
    app_handle: AppHandle,
    inference_mgr: State<'_, Arc<InferenceManager>>,
) -> Result<(), String> {
    inference_mgr
        .unload_active_model(&app_handle)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_inference_status(
    inference_mgr: State<'_, Arc<InferenceManager>>,
) -> Result<InferenceStatusPayload, String> {
    let status = inference_mgr.get_status();
    let loaded_model = inference_mgr.get_loaded_model_info();

    Ok(InferenceStatusPayload {
        status: status.to_string(),
        step: None,
        model: loaded_model,
        error: None,
    })
}

#[tauri::command]
pub async fn send_chat_message(
    app_handle: AppHandle,
    inference_mgr: State<'_, Arc<InferenceManager>>,
    memory_mgr: State<'_, Arc<crate::memory_engine::MemoryManager>>,
    messages: Vec<ChatMessage>,
    params: Option<GenerationParams>,
) -> Result<(), String> {
    let params = params.unwrap_or_default();
    let mgr = inference_mgr.inner().clone();
    let mem = memory_mgr.inner().clone();

    // Extract facts from last user turn & prepare memory-injected messages
    let mut final_messages = messages.clone();
    if let Some(last_msg) = messages.last() {
        if last_msg.role == "user" {
            let _ = mem.process_user_turn(&last_msg.content, None).await;
            if let Ok(injected) = mem.prepare_injected_messages(&messages, &last_msg.content).await {
                final_messages = injected;
            }
        }
    }

    tokio::task::spawn_blocking(move || {
        mgr.send_chat_message(&app_handle, final_messages, params)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn stop_chat_generation(
    inference_mgr: State<'_, Arc<InferenceManager>>,
) -> Result<(), String> {
    inference_mgr.stop_generation();
    Ok(())
}

#[tauri::command]
pub async fn restore_last_session(
    app_handle: AppHandle,
    inference_mgr: State<'_, Arc<InferenceManager>>,
) -> Result<Option<LoadedModelInfo>, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app_data_dir: {}", e))?;

    if let Ok(Some(session)) = crate::ai_engine::session::SessionManager::load_session(&app_data_dir) {
        if session.auto_restore_enabled {
            log::info!("[SESSION] Auto-restoring last active model session: {:?}", session);
            let mgr = inference_mgr.inner().clone();
            let app_handle_clone = app_handle.clone();
            let app_data_dir_clone = app_data_dir.clone();

            let res = tokio::task::spawn_blocking(move || {
                mgr.load_installed_model(
                    &app_handle_clone,
                    &app_data_dir_clone,
                    &session.provider_id,
                    &session.model_id,
                    &session.quantization,
                )
            })
            .await;

            if let Ok(Ok(info)) = res {
                log::info!("[SESSION] Auto-restored model successfully: {:?}", info);
                return Ok(Some(info));
            } else {
                log::warn!("[SESSION WARN] Auto-restore failed for session, falling back to manual load.");
            }
        }
    }

    Ok(None)
}

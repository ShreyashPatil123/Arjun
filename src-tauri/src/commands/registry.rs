//! Commands for the model registry and the router.

use std::sync::Arc;

use tauri::{AppHandle, Manager, State};

use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::config::ConfigManager;
use crate::core::event_bus::{get_event_bus, SarathiEvent};
use crate::identity::Permission;
use crate::policy::Classification;
use crate::ai_engine::activation::{ActivationOutcome, InferenceLoader, ModelActivator};
use crate::audit::{AuditKind, AuditService};
use crate::registry::router::{ModelRouter, RoutingDecision};
use crate::registry::{ModelEntry, ModelRegistry};
use crate::system_analyzer::gpu_collector;
use crate::ai_engine::startup::StartupModelTarget;
use crate::download_manager::traits::InstalledModel;

/// The activator, shared across commands.
pub type SharedActivator = Arc<ModelActivator<InferenceLoader>>;

/// A routed and loaded model, ready to run.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedModel {
    pub routing: RoutingDecision,
    pub activation: ActivationOutcome,
}

/// The exact installed model variant selected to run the orchestrator.
#[tauri::command]
pub async fn get_orchestrator_model(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<StartupModelTarget, String> {
    require_session(&session)?;
    let config = ConfigManager::load(&ConfigManager::get_config_path(&app))
        .map_err(|e| e.to_string())?;
    Ok(StartupModelTarget {
        provider_id: config.ai_settings.orchestrator_provider_id,
        model_id: config.ai_settings.orchestrator_model_id,
        quantization: config.ai_settings.orchestrator_quantization,
    })
}

/// Selects any ready installed model as the orchestrator. Administrator only.
/// The exact provider/model/quantization coordinates are persisted so startup
/// never guesses between two variants of the same model.
#[tauri::command]
pub async fn set_orchestrator_model(
    app: AppHandle,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<StartupModelTarget, String> {
    let signed_in = require_permission(&session, Permission::ModifyPolicy)?;
    let requested = StartupModelTarget {
        provider_id: provider_id.trim().to_string(),
        model_id: model_id.trim().to_string(),
        quantization: quantization.trim().to_string(),
    };
    if requested.provider_id.is_empty()
        || requested.model_id.is_empty()
        || requested.quantization.is_empty()
    {
        return Err("Provider, model and quantization are all required.".to_string());
    }

    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let installed = crate::model_manager::ModelManager::list_installed_models(&app_data);
    let selected = resolve_installed_orchestrator(&installed, &requested)?;

    let config_path = ConfigManager::get_config_path(&app);
    let mut config = ConfigManager::load(&config_path).map_err(|e| e.to_string())?;
    config.ai_settings.orchestrator_provider_id = selected.provider_id.clone();
    config.ai_settings.orchestrator_model_id = selected.model_id.clone();
    config.ai_settings.orchestrator_quantization = selected.quantization.clone();
    config.ai_settings.auto_load_on_startup = true;
    config.ai_settings.use_gpu = true;
    ConfigManager::save(&config, &config_path).map_err(|e| e.to_string())?;

    get_event_bus().publish(
        SarathiEvent::ConfigChanged,
        Some(serde_json::json!({ "orchestrator": &selected })),
    );
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!(
            "Set orchestrator to {} ({})",
            selected.model_id, selected.quantization
        ),
        Some(serde_json::json!({
            "providerId": &selected.provider_id,
            "modelId": &selected.model_id,
            "quantization": &selected.quantization,
        })),
    );

    Ok(selected)
}

fn resolve_installed_orchestrator(
    installed: &[InstalledModel],
    requested: &StartupModelTarget,
) -> Result<StartupModelTarget, String> {
    installed
        .iter()
        .find(|model| {
            requested.matches_installed(model) && model.is_ready && model.size_bytes > 0
        })
        .map(StartupModelTarget::from_installed)
        .ok_or_else(|| {
            format!(
                "{} ({}) is not a ready installed model.",
                requested.model_id, requested.quantization
            )
        })
}

/// Every registered model, including disabled ones.
#[tauri::command]
pub async fn list_registered_models(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ModelEntry>, String> {
    // Read-only inspection. Any signed-in user may see the registry; the
    // matrix does not gate read paths for the model list itself.
    require_session(&session)?;
    Ok(registry.all().to_vec())
}

/// Where the manifest lives, so an administrator can find the file to edit.
///
/// Registering a model is editing this file and restarting — there is no import
/// wizard to go through, and no code change. Showing the path makes that
/// concrete rather than a claim in the documentation.
#[tauri::command]
pub async fn model_manifest_path(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
) -> Result<String, String> {
    require_session(&session)?;
    Ok(registry.manifest_path().display().to_string())
}

/// Shows which model would handle a prompt, without running anything.
///
/// This is what makes automatic selection visible instead of implicit: the same
/// routing the orchestrator will use, reported before the task starts, with the
/// reasons that produced it.
#[tauri::command]
pub async fn preview_routing(
    registry: State<'_, Arc<ModelRegistry>>,
    session: State<'_, CurrentSession>,
    prompt: String,
    classification: Option<Classification>,
) -> Result<RoutingDecision, String> {
    // Routing reveals which models exist and what they are cleared for, so it
    // needs a signed-in user like anything else.
    require_session(&session)?;

    // Read from the live hardware rather than a stored figure: the right model
    // on a workstation is the wrong one on a laptop, and adapting to the machine
    // it is actually on is the whole point of this router.
    //
    // The largest GPU wins on a multi-GPU box, matching what the inference
    // engine will use. No GPU at all reports zero, and the planner turns that
    // into a CPU-only plan rather than an error.
    let vram = gpu_collector::detect_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    ModelRouter::route(
        &registry,
        &prompt,
        classification,
        vram,
        None,
        false,
        &[],
        &[],
    )
    .map_err(|failure| failure.reason)
}

/// Picks the right model for a prompt and loads it, with no human step.
///
/// This is the automatic selection the problem statement asks to be
/// demonstrated: a coding request and a summarisation request each end with a
/// different model resident, and the trace records the routing reasons and what
/// was evicted to get there.
///
/// Refuses while another task holds the model, rather than swapping underneath
/// it. The refusal names the holder so the wait is explicable.
#[tauri::command]
pub async fn prepare_model_for(
    registry: State<'_, Arc<ModelRegistry>>,
    activator: State<'_, SharedActivator>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    prompt: String,
    classification: Option<Classification>,
) -> Result<PreparedModel, String> {
    let signed_in = require_permission(&session, Permission::ImportModel)?;

    let vram = gpu_collector::detect_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    let routing = ModelRouter::route(
        &registry,
        &prompt,
        classification,
        vram,
        None,
        false,
        &[],
        &[],
    )
    .map_err(|failure| failure.reason)?;

    let activation = activator
        .ensure_ready(&registry, &routing.model_id, &signed_in.user.id)
        .map_err(|e| e.message())?;

    // Recorded whether or not a swap happened: "which model answered this" is
    // exactly the question an auditor asks afterwards, and it cannot be
    // reconstructed later from the prompt alone.
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!(
            "Routed to {} ({}){}",
            routing.model_name,
            routing.role.label(),
            match &activation.evicted {
                Some(evicted) => format!(", releasing {evicted}"),
                None if activation.already_resident => ", already loaded".to_string(),
                None => ", loaded".to_string(),
            }
        ),
        Some(serde_json::json!({
            "modelId": routing.model_id,
            "role": routing.role,
            "intent": routing.intent,
            "confidence": routing.confidence,
            "usedFallback": routing.used_fallback,
            "reasons": routing.reasons,
            "evicted": activation.evicted,
            "alreadyResident": activation.already_resident,
            "tookMs": activation.took_ms,
        })),
    );

    Ok(PreparedModel { routing, activation })
}

/// Which model is loaded right now, and who is using it.
#[tauri::command]
pub async fn model_residency(
    activator: State<'_, SharedActivator>,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "heldBy": activator.current_holder(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed(quantization: &str, is_ready: bool) -> InstalledModel {
        InstalledModel {
            id: format!("custom-{quantization}"),
            model_id: "org/custom-orchestrator".to_string(),
            model_name: "Custom Orchestrator".to_string(),
            provider_id: "huggingface".to_string(),
            quantization: quantization.to_string(),
            format: "GGUF".to_string(),
            backend: "llama.cpp (GGUF)".to_string(),
            file_name: "model.gguf".to_string(),
            file_path: "/models/model.gguf".to_string(),
            size_bytes: 1_000,
            installed_at: String::new(),
            is_ready,
            checksum: None,
        }
    }

    #[test]
    fn administrator_selection_resolves_the_exact_ready_variant() {
        let requested = StartupModelTarget::from_installed(&installed("Q6_K", true));
        let selected = resolve_installed_orchestrator(
            &[installed("Q4_K_M", true), installed("Q6_K", true)],
            &requested,
        )
        .expect("the exact requested variant should be selectable");

        assert_eq!(selected, requested);
    }

    #[test]
    fn incomplete_models_cannot_become_the_orchestrator() {
        let model = installed("Q6_K", false);
        let requested = StartupModelTarget::from_installed(&model);
        assert!(resolve_installed_orchestrator(&[model], &requested).is_err());
    }
}

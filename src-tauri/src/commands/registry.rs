//! Commands for the model registry and the router.

use std::sync::Arc;

use tauri::State;

use crate::commands::governance::{require_session, CurrentSession};
use crate::policy::Classification;
use crate::ai_engine::activation::{ActivationOutcome, InferenceLoader, ModelActivator};
use crate::audit::{AuditKind, AuditService};
use crate::registry::router::{ModelRouter, RoutingDecision};
use crate::registry::{ModelEntry, ModelRegistry};
use crate::system_analyzer::gpu_collector;

/// The activator, shared across commands.
pub type SharedActivator = Arc<ModelActivator<InferenceLoader>>;

/// A routed and loaded model, ready to run.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedModel {
    pub routing: RoutingDecision,
    pub activation: ActivationOutcome,
}

/// Every registered model, including disabled ones.
#[tauri::command]
pub async fn list_registered_models(
    registry: State<'_, Arc<ModelRegistry>>,
) -> Result<Vec<ModelEntry>, String> {
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
) -> Result<String, String> {
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
    let signed_in = require_session(&session)?;

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

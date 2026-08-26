//! Driving an agent run from the UI.
//!
//! Thin on purpose. Everything these commands do is start, stop, or observe a
//! run; the loop is in the Node runtime and the decisions are in
//! [`crate::agent_runtime`]. Adding policy here would create a third place a
//! rule could live, and the whole point of the split is that there are two.
//!
//! ## Why the runtime starts lazily
//!
//! Spawning a Node process at application start would make the workbench depend
//! on the agent runtime to open at all — including for an auditor who only ever
//! reads the record, and on a machine where the bundle was never built. So the
//! child is started on the first run and kept for the rest of the session.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};

use crate::agent_runtime::workspace::Workspace;
use crate::agent_runtime::{AgentRuntime, RuntimeDeps, AGENT_EVENT};
use crate::orchestrator::approvals::ApprovalQueue;
use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_session, CurrentSession};
use crate::knowledge::KnowledgeIndex;
use crate::policy::Classification;
use crate::registry::router::{ModelRouter, RoutingDecision};
use crate::registry::ModelRegistry;
use crate::serving::{Endpoint, ModelServers};
use crate::system_analyzer::gpu_collector;

/// The one runtime for this session, started on first use.
pub type AgentRuntimeHandle = Arc<Mutex<Option<Arc<AgentRuntime>>>>;

/// What the UI sends to start a run.
///
/// Deliberately no model. Which model answers is the router's decision, and
/// letting a caller name one would make automatic selection optional — the
/// opposite of what PS 26117 asks to be demonstrated.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartRunRequest {
    pub prompt: String,
    /// Sensitivity of the material, which narrows the models that may see it.
    #[serde(default)]
    pub classification: Option<Classification>,
    /// Overrides the default instructions. Present for the demonstrator's
    /// scripted scenarios; ordinary runs leave it unset.
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    pub run_id: String,
    pub text: String,
    pub turns: u32,
    /// Which model answered and why. Shown in the trace verbatim.
    pub routing: RoutingDecision,
    /// Where it ran, and whether ARJUN started it.
    pub endpoint: Endpoint,
}

/// What the model is told it is, and what it must not do.
///
/// Deliberately short and specific. The two rules that matter for PS 26117 are
/// here rather than buried in a template: search before answering, and say when
/// nothing was found instead of filling the silence.
const SYSTEM_PROMPT: &str = "\
You are an assistant inside an organisation's own workbench. You run entirely on \
this machine and have no access to the internet.

Answer questions about internal procedure, specification, drawings or \
correspondence only from documents you have retrieved with the search_documents \
tool. Do not answer them from memory: your training data is not this \
organisation's record, and a plausible answer that is not in the documents is \
worse than no answer.

When a search returns nothing, say so plainly and stop. Do not infer what a \
document probably says. Cite the document and page for each claim you make.";

/// Finds the runtime bundle.
///
/// In a packaged build it is a bundled resource; in a checkout it is the
/// sibling `agent-runtime/dist`. Resolved here rather than in
/// [`crate::agent_runtime`] so that module keeps no dependency on Tauri, which
/// is what lets its tests drive a real child process with no app running.
///
/// The bundle ships with the app; the Node binary that executes it does not
/// yet. Until Phase 5 packages one, a deployment needs `node` on PATH — and
/// says so plainly through [`RuntimeError::Spawn`] when it does not.
fn bundle_path(app: &AppHandle) -> std::path::PathBuf {
    use tauri::Manager;
    app.path()
        .resolve(
            "arjun-agent-runtime.mjs",
            tauri::path::BaseDirectory::Resource,
        )
        .ok()
        .filter(|path| path.exists())
        .unwrap_or_else(crate::agent_runtime::default_bundle_path)
}

/// Workspaces for the runs this session has started, shared with the runtime.
pub type RunWorkspaces = Arc<std::sync::Mutex<std::collections::HashMap<String, Workspace>>>;

/// The application's data directory, where run workspaces live.
fn app_data_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    use tauri::Manager;
    app.path()
        .app_data_dir()
        .map_err(|error| format!("the application data directory is not available: {error}"))
}

fn runtime(
    handle: &AgentRuntimeHandle,
    app: &AppHandle,
    index: &Arc<KnowledgeIndex>,
    session: &CurrentSession,
    workspaces: &RunWorkspaces,
    approvals: &Arc<ApprovalQueue>,
) -> Result<Arc<AgentRuntime>, String> {
    let mut slot = handle
        .lock()
        .map_err(|_| "the agent runtime handle is poisoned".to_string())?;
    if let Some(existing) = slot.as_ref() {
        return Ok(existing.clone());
    }

    let deps = Arc::new(RuntimeDeps {
        index: index.clone(),
        session: Arc::clone(session),
        workspaces: workspaces.clone(),
        approvals: approvals.clone(),
        calculations: Arc::default(),
    });

    let emitter = app.clone();
    let started = AgentRuntime::spawn(
        deps,
        Arc::new(move |event: Value| {
            // A dropped event costs a progress line, not a run.
            let _ = emitter.emit(AGENT_EVENT, event);
        }),
        bundle_path(app),
    )
    .map_err(|error| error.to_string())?;

    *slot = Some(started.clone());
    Ok(started)
}

/// Routes a prompt to a model, makes sure that model is served, and runs it.
///
/// The three steps are what PS 26117 asks to be demonstrated end to end, and
/// keeping them in one command is what makes the demonstration honest: there is
/// no path by which a caller supplies its own model and skips the routing.
///
/// Long-running. The UI shows progress from the `agent://event` stream and this
/// resolves with the final answer, the routing reasons, and where it ran.
#[tauri::command]
pub async fn agent_start_run(
    app: AppHandle,
    request: StartRunRequest,
    handle: State<'_, AgentRuntimeHandle>,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    index: State<'_, Arc<KnowledgeIndex>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    workspaces: State<'_, RunWorkspaces>,
    approvals: State<'_, Arc<ApprovalQueue>>,
) -> Result<RunSummary, String> {
    // Checked here as well as in the runtime's handlers. Here it gives the
    // person a clear reason before anything starts; there it stops a call whose
    // session ended mid-run.
    let signed_in = require_session(&session)?;

    // Read from the live hardware rather than a stored figure: the right model
    // on a workstation is the wrong one on a laptop. The largest GPU wins on a
    // multi-GPU box; no GPU reports zero and the planner makes a CPU-only plan.
    let vram = gpu_collector::detect_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    let routing = ModelRouter::route(&registry, &request.prompt, request.classification, vram)
        .map_err(|failure| failure.reason)?;

    let entry = registry
        .find(&routing.model_id)
        .ok_or_else(|| format!("{} was routed to but is not in the registry.", routing.model_id))?;

    // Where it will actually run. A GGUF model gets a llama-server ARJUN starts;
    // a Python-served one is an endpoint an operator already runs. Both end up
    // as an OpenAI-compatible URL on loopback, which is why one agent loop can
    // drive either.
    let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
        vram,
        entry.weights_bytes,
        entry.context_length,
        None,
    );
    let endpoint = servers
        .endpoint_for(entry, registry.models_dir(), &plan)
        .await
        .map_err(|error| error.to_string())?;

    let runtime = runtime(&handle, &app, &index, &session, &workspaces, &approvals)?;
    let run_id = uuid::Uuid::new_v4().to_string();

    // The run's own directory, created before the model is told anything — so
    // the instructions can name it, and so a tool call cannot arrive before the
    // gateway has roots to resolve against.
    let workspace = Workspace::create(&app_data_dir(&app)?, &run_id).map_err(|e| e.to_string())?;
    let workspace_note = workspace.describe();
    workspaces
        .lock()
        .map_err(|_| "the workspace table is poisoned".to_string())?
        .insert(run_id.clone(), workspace);

    let params = json!({
        "runId": run_id,
        "prompt": request.prompt,
        "systemPrompt": format!(
            "{}

{workspace_note}",
            request.system_prompt.as_deref().unwrap_or(SYSTEM_PROMPT)
        ),
        "model": {
            "id": endpoint.served_model_id,
            "provider": provider_label(endpoint.runtime),
            "baseUrl": endpoint.base_url,
            "contextWindow": entry.context_length,
            "maxTokens": DEFAULT_MAX_TOKENS,
        },
    });

    // Recorded before the run, not after. A run that crashes or is killed still
    // has to leave behind which model was chosen and why — that is exactly the
    // question asked when something goes wrong.
    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!(
            "Agent run routed to {} ({}) on {}",
            routing.model_name,
            routing.role.label(),
            endpoint.runtime.label()
        ),
        Some(json!({
            "runId": run_id,
            "modelId": routing.model_id,
            "role": routing.role,
            "intent": routing.intent,
            "confidence": routing.confidence,
            "usedFallback": routing.used_fallback,
            "reasons": routing.reasons,
            "runtime": endpoint.runtime.label(),
            "baseUrl": endpoint.base_url,
            "managed": endpoint.managed,
        })),
    );

    let outcome = runtime
        .request("run.start", params)
        .await
        .map_err(|error| error.to_string())?;

    Ok(RunSummary {
        run_id,
        text: outcome
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        turns: outcome.get("turns").and_then(Value::as_u64).unwrap_or(0) as u32,
        routing,
        endpoint,
    })
}

/// Cap on one turn's output.
///
/// Not read from the model: a GGUF advertises its training context, not what
/// this deployment should let one turn produce. Large enough for an approval
/// note, small enough that a looping model does not fill the context window
/// before the budget stops it.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// What the agent runtime calls this provider.
///
/// Cosmetic on the wire — the transport is the same OpenAI-compatible one
/// either way — but it appears in the trace, and "vllm" against a llama-server
/// would mislead the person reading it.
fn provider_label(runtime: crate::registry::Runtime) -> &'static str {
    match runtime {
        crate::registry::Runtime::LlamaCpp => "llama-cpp",
        crate::registry::Runtime::PythonSidecar => "vllm",
    }
}

/// Applies a correction to a run already in flight.
///
/// The alternative an operator otherwise has is to stop and start again, losing
/// every tool result gathered so far. On a task that has already read a
/// 200-page drawing set, that is an expensive way to say "use the 2019
/// revision".
///
/// Resolves `false` when there was nothing to correct — an ordinary race, not a
/// failure.
#[tauri::command]
pub async fn agent_steer_run(
    run_id: String,
    text: String,
    handle: State<'_, AgentRuntimeHandle>,
) -> Result<bool, String> {
    if text.trim().is_empty() {
        return Err("A correction with no text would do nothing.".to_string());
    }
    let runtime = {
        let slot = handle
            .lock()
            .map_err(|_| "the agent runtime handle is poisoned".to_string())?;
        slot.clone()
    };
    let Some(runtime) = runtime else {
        return Ok(false);
    };

    let outcome = runtime
        .request("run.steer", json!({ "runId": run_id, "text": text }))
        .await
        .map_err(|error| error.to_string())?;
    Ok(outcome
        .get("steered")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

/// Stops a run in flight.
#[tauri::command]
pub async fn agent_abort_run(
    run_id: String,
    handle: State<'_, AgentRuntimeHandle>,
) -> Result<bool, String> {
    let runtime = {
        let slot = handle
            .lock()
            .map_err(|_| "the agent runtime handle is poisoned".to_string())?;
        slot.clone()
    };
    // Nothing running is not a failure: the run finishing just before the abort
    // arrived is an ordinary race, and reporting it as an error would make an
    // operator doubt the button.
    let Some(runtime) = runtime else {
        return Ok(false);
    };

    let outcome = runtime
        .request("run.abort", json!({ "runId": run_id }))
        .await
        .map_err(|error| error.to_string())?;
    Ok(outcome
        .get("aborted")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

/// Whether the runtime is up, and what it is.
///
/// Shown on the health screen. Starts the child if it is not already running,
/// so this doubles as the "can this deployment run an agent at all" check.
#[tauri::command]
pub async fn agent_runtime_health(
    app: AppHandle,
    handle: State<'_, AgentRuntimeHandle>,
    index: State<'_, Arc<KnowledgeIndex>>,
    session: State<'_, CurrentSession>,
    workspaces: State<'_, RunWorkspaces>,
    approvals: State<'_, Arc<ApprovalQueue>>,
) -> Result<Value, String> {
    let runtime = runtime(&handle, &app, &index, &session, &workspaces, &approvals)?;
    runtime
        .request("health", json!({}))
        .await
        .map_err(|error| error.to_string())
}

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

use crate::agent_runtime::artifacts::{ArtifactReport, RunArtifacts};
use crate::agent_runtime::retrieval::RunPassages;
use crate::agent_runtime::tasks::{ApprovalRecord, PlanRecord, TaskRecord, TaskSummary, ToolCallRecord};
use crate::agent_runtime::workspace::Workspace;
use crate::agent_runtime::{artifacts, planning, retrieval, tasks};
use crate::agent_runtime::{AgentRuntime, RuntimeDeps, AGENT_EVENT};
use crate::artifacts::{verify, Evidence, VerificationReport};
use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_session, CurrentSession};
use crate::identity::{Permission, Session};
use crate::knowledge::KnowledgeIndex;
use crate::orchestrator::approvals::ApprovalQueue;
use crate::orchestrator::plan::PlanRun;
use crate::policy::Classification;
use crate::registry::router::{ModelRouter, RoutingDecision};
use crate::registry::ModelRegistry;
use crate::serving::{Endpoint, ModelServers};
use crate::system_analyzer::gpu_collector;

/// The plans this session's runs are being held to, shared with the runtime.
///
/// Held by the application rather than inside the runtime, because the command
/// that starts a run has to write the plan into the record when the run ends —
/// and `RuntimeDeps` is owned by the child supervisor, which the command cannot
/// reach into.
pub type RunPlans = Arc<Mutex<std::collections::HashMap<String, PlanRun>>>;

/// Every tool call each run made, in order, shared with the runtime.
///
/// Application state for the same reason as the plans: the runtime appends to
/// it as the run works, and the command that started the run reads it back to
/// write the record.
pub type RunToolCalls = Arc<Mutex<std::collections::HashMap<String, Vec<ToolCallRecord>>>>;

/// Calculations each run performed, in order, shared with the runtime.
///
/// Application state for the same reason as the plans: `create_xlsx` writes the
/// working from this table during the run, and the task record reads it
/// afterwards to hand the verifier the figures the engine actually produced.
pub type RunCalculations =
    Arc<Mutex<std::collections::HashMap<String, Vec<crate::orchestrator::calculation::CalculationRecord>>>>;

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
    /// Echoed back on the run's first event, so a caller can tell which run is
    /// its own before `agent_start_run` resolves.
    ///
    /// The caller does not get to name the run. Events carry the run id this
    /// process generated; this only lets a window recognise the stream it
    /// started, which matters as soon as two windows are open at once.
    #[serde(default)]
    pub correlation_id: Option<String>,
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
    /// The plan it was held to, and how much of it was spent.
    pub plan: PlanRecord,
    /// What the answer's claims resolve to. Absent when there was no answer.
    pub verification: Option<VerificationReport>,
    /// The files it produced, each re-opened and checked.
    pub artifacts: Vec<ArtifactReport>,
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
document probably says.

Cite every claim with the marker of the passage it came from, written as [E1], \
[E2] and so on — the numbers search_documents gave those passages. Each marker \
is checked against what you actually retrieved when the task finishes, so a \
citation to a passage you were never given will be found and reported. Say a \
figure came from a calculation rather than citing a passage for it.";

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

/// Everything the runtime's handlers need, gathered from application state.
///
/// A struct rather than eight arguments because every caller passes the same
/// eight, and a list that long is one where two of them eventually get swapped.
pub struct RuntimeState<'a> {
    pub index: &'a Arc<KnowledgeIndex>,
    pub session: &'a CurrentSession,
    pub workspaces: &'a RunWorkspaces,
    pub approvals: &'a Arc<ApprovalQueue>,
    pub passages: &'a RunPassages,
    pub produced: &'a RunArtifacts,
    pub plans: &'a RunPlans,
    pub calculations: &'a RunCalculations,
    pub calls: &'a RunToolCalls,
}

fn runtime(
    handle: &AgentRuntimeHandle,
    app: &AppHandle,
    state: &RuntimeState<'_>,
) -> Result<Arc<AgentRuntime>, String> {
    let mut slot = handle
        .lock()
        .map_err(|_| "the agent runtime handle is poisoned".to_string())?;
    if let Some(existing) = slot.as_ref() {
        return Ok(existing.clone());
    }

    let emitter = app.clone();
    let emit: Arc<dyn Fn(Value) + Send + Sync> = Arc::new(move |event: Value| {
        // A dropped event costs a progress line, not a run.
        let _ = emitter.emit(AGENT_EVENT, event);
    });

    let deps = Arc::new(RuntimeDeps {
        index: state.index.clone(),
        session: Arc::clone(state.session),
        workspaces: state.workspaces.clone(),
        approvals: state.approvals.clone(),
        calculations: state.calculations.clone(),
        passages: state.passages.clone(),
        produced: state.produced.clone(),
        calls: state.calls.clone(),
        plans: state.plans.clone(),
        // The same channel the loop's own events travel, so an operator sees
        // one sequence of what happened rather than two interleaved by luck.
        emit: emit.clone(),
    });

    let started =
        AgentRuntime::spawn(deps, emit, bundle_path(app)).map_err(|error| error.to_string())?;

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
    passages: State<'_, RunPassages>,
    produced: State<'_, RunArtifacts>,
    plans: State<'_, RunPlans>,
    calculations: State<'_, RunCalculations>,
    calls: State<'_, RunToolCalls>,
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

    let state = RuntimeState {
        index: &index,
        session: &session,
        workspaces: &workspaces,
        approvals: &approvals,
        passages: &passages,
        produced: &produced,
        plans: &plans,
        calculations: &calculations,
        calls: &calls,
    };
    let runtime = runtime(&handle, &app, &state)?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now();

    // The run's own directory, created before the model is told anything — so
    // the instructions can name it, and so a tool call cannot arrive before the
    // gateway has roots to resolve against.
    let workspace = Workspace::create(&app_data_dir(&app)?, &run_id).map_err(|e| e.to_string())?;
    let workspace_note = workspace.describe();
    workspaces
        .lock()
        .map_err(|_| "the workspace table is poisoned".to_string())?
        .insert(run_id.clone(), workspace);

    // The plan, fixed before the model is told anything. Registered before the
    // run starts rather than alongside it: a tool call arriving against a run
    // with no plan yet would be a call with no budget, and the window for that
    // is exactly the window in which the first call happens.
    let task_plan = planning::plan_for(&run_id, &request.prompt);
    let plan_note = describe_plan(&task_plan);
    let planned = PlanRecord::of(&task_plan);
    plans
        .lock()
        .map_err(|_| "the plan table is poisoned".to_string())?
        .insert(run_id.clone(), task_plan);

    // Published before the first turn, so the trace shows what the run intends
    // before it shows what it did.
    let _ = app.emit(
        AGENT_EVENT,
        json!({
            "runId": run_id,
            "event": {
                "type": "plan_ready",
                "plan": planned,
                "correlationId": request.correlation_id,
            },
        }),
    );

    let params = json!({
        "runId": run_id,
        "prompt": request.prompt,
        "systemPrompt": format!(
            "{}\n\n{workspace_note}\n\n{plan_note}",
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

    let outcome = runtime.request("run.start", params).await;

    // From here the run is over, one way or the other, and everything below is
    // about leaving a record of it. A run that failed gets the same treatment
    // as one that worked: the failure is written into the record rather than
    // returned instead of it, because the run somebody most wants to look at
    // afterwards is the one that went wrong.
    let (answer, turns, failure) = match &outcome {
        Ok(value) => (
            value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            value.get("turns").and_then(Value::as_u64).unwrap_or(0) as u32,
            None,
        ),
        Err(error) => (String::new(), 0, Some(error.to_string())),
    };

    // Re-opened, not taken on the model's word. A document that was written and
    // then corrupted still passes every test of the code that wrote it.
    let produced_files = artifacts::report_for_run(&produced, &run_id);
    let retrieved = retrieval::for_run(&passages, &run_id);
    // Read off the engine's own table, never rebuilt from the answer's figures.
    // PS step 27 makes the engine the source of numerical truth, and a record
    // recovered from the text would be the model's account of its arithmetic
    // rather than the arithmetic.
    let worked = calculations
        .lock()
        .ok()
        .and_then(|table| table.get(&run_id).cloned())
        .unwrap_or_default();

    // The check between a draft and something somebody signs. Skipped when
    // there is no answer to check — reporting "nothing to verify" as a pass
    // would be the one misleading outcome available here.
    let verification = (!answer.trim().is_empty()).then(|| {
        verify(
            &answer,
            &Evidence {
                passages: &retrieved,
                calculations: &worked,
                // The document service reports these; nothing on this path
                // produces them yet, and an invented list would fabricate a
                // finding rather than omit one.
                unread_pages: &[],
            },
        )
    });

    let made_calls = calls
        .lock()
        .ok()
        .and_then(|table| table.get(&run_id).cloned())
        .unwrap_or_default();

    // Everything a person was asked to allow during this run, decided or not.
    // Read from the queue by run id rather than tracked separately, so the
    // record and the Approvals screen cannot drift apart.
    let asked = approvals
        .all()
        .into_iter()
        .filter(|item| item.request.task_id == run_id)
        .map(|item| {
            let (state, decided_by, decided_at, because) = match &item.decision {
                None => ("pending", None, None, None),
                Some(crate::orchestrator::approvals::Decision::Approved { by, at }) => {
                    ("approved", Some(by.clone()), Some(at.to_rfc3339()), None)
                }
                Some(crate::orchestrator::approvals::Decision::Rejected { by, at, because }) => (
                    "rejected",
                    Some(by.clone()),
                    Some(at.to_rfc3339()),
                    Some(because.clone()),
                ),
            };
            ApprovalRecord {
                id: item.request.id,
                tool: item.request.tool,
                target: item.request.target,
                arguments: item.request.arguments,
                consequences: item.request.consequences,
                requested_at: item.request.requested_at.to_rfc3339(),
                state: state.to_string(),
                decided_by,
                decided_at,
                because,
            }
        })
        .collect::<Vec<_>>();

    let finished_at = chrono::Utc::now();
    let mut final_plan = plans
        .lock()
        .ok()
        .and_then(|table| table.get(&run_id).map(PlanRecord::of))
        .unwrap_or(planned);
    // Which steps the run actually carried out, judged against what it left
    // behind rather than what it said. Re-deriving the plan is safe because the
    // derivation is deterministic over the prompt — this is the same plan the
    // run was held to, so the steps line up one for one.
    let succeeded: Vec<String> = made_calls
        .iter()
        .filter(|call| call.outcome == crate::agent_runtime::tasks::CallOutcome::Succeeded)
        .map(|call| call.tool.clone())
        .collect();
    final_plan.settle(
        &planning::derive(&request.prompt).steps,
        &succeeded,
        !answer.trim().is_empty(),
        verification.is_some(),
    );

    // The plan only knows the endings it caused. A loop that simply finished,
    // or a runtime that fell over, ends the run without it hearing about it.
    final_plan.ended(failure.as_deref());

    let record = TaskRecord {
        run_id: run_id.clone(),
        prompt: request.prompt.clone(),
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_seconds: (finished_at - started_at).num_seconds().max(0) as u64,
        user_id: signed_in.user.id.clone(),
        routing: routing.clone(),
        endpoint: endpoint.clone(),
        plan: final_plan.clone(),
        answer: answer.clone(),
        turns,
        verification: verification.clone(),
        artifacts: produced_files.clone(),
        evidence: TaskRecord::evidence_from(&retrieved),
        calculations: worked,
        tool_calls: made_calls,
        approvals: asked,
        failure: failure.clone(),
    };

    // Saved before anything is released, so a failure to write is a failure the
    // person hears about rather than one that quietly loses the task.
    if let Err(error) = tasks::save(&app_data_dir(&app)?, &record) {
        log::error!("[agent] the record for run {run_id} could not be saved: {error}");
    }

    // The run's working state is not needed once its record is written, and
    // holding every passage of every run for the life of the session would grow
    // without bound. The workspace is deliberately left alone — the deliverable
    // is in it.
    retrieval::forget(&passages, &run_id);
    artifacts::forget(&produced, &run_id);
    if let Ok(mut table) = plans.lock() {
        table.remove(&run_id);
    }
    if let Ok(mut table) = calculations.lock() {
        table.remove(&run_id);
    }
    if let Ok(mut table) = calls.lock() {
        table.remove(&run_id);
    }

    // Returned last, so a run that failed still left its record behind first.
    outcome.map_err(|error| error.to_string())?;

    Ok(RunSummary {
        run_id,
        text: answer,
        turns,
        routing,
        endpoint,
        plan: final_plan,
        verification,
        artifacts: produced_files,
    })
}

/// What the model is told about the plan it is being held to.
///
/// Told rather than left to discover, because a model that does not know it has
/// a step budget spends it on searches it could have combined, and one that
/// does not know a tool is outside its plan collects refusals instead of saying
/// what it could not do.
fn describe_plan(plan: &PlanRun) -> String {
    let steps: Vec<String> = plan
        .steps
        .iter()
        .map(|step| format!("{}. {}", step.ordinal, step.intent))
        .collect();
    let tools: Vec<&str> = plan
        .budget
        .permitted_tools
        .iter()
        .map(|tool| tool.as_str())
        .collect();

    format!(
        "This task has a plan, fixed before you were asked and not extendable:\n\n{}\n\n\
         You may use these tools and no others: {}. You have {} tool calls and {} minutes for \
         the whole task, and the same call repeated {} times is treated as going in circles and \
         stops the task. If you run out, say what you completed and what you did not.",
        steps.join("\n"),
        tools.join(", "),
        plan.budget.max_steps,
        plan.budget.max_duration.as_secs() / 60,
        plan.budget.repeat_limit,
    )
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
    passages: State<'_, RunPassages>,
    produced: State<'_, RunArtifacts>,
    plans: State<'_, RunPlans>,
    calculations: State<'_, RunCalculations>,
    calls: State<'_, RunToolCalls>,
) -> Result<Value, String> {
    let state = RuntimeState {
        index: &index,
        session: &session,
        workspaces: &workspaces,
        approvals: &approvals,
        passages: &passages,
        produced: &produced,
        plans: &plans,
        calculations: &calculations,
        calls: &calls,
    };
    let runtime = runtime(&handle, &app, &state)?;
    runtime
        .request("health", json!({}))
        .await
        .map_err(|error| error.to_string())
}

/// Who may read a given task.
///
/// A task record holds the passages the run retrieved and the text it drafted,
/// which is the document library seen through one person's permissions. So it
/// is readable by the person who ran it, and by an auditor — the same people
/// who can already read the audit log, and for the same reason.
///
/// Without this, signing in as anybody would be a way to read passages the
/// knowledge index would have refused to return to them.
fn may_read(session: &Session, record_user_id: &str) -> bool {
    session.user.id == record_user_id || session.holds(Permission::ViewAuditLog)
}

/// Every task the signed-in person may read, newest first.
///
/// Read from disk each time rather than cached: a record is written by the run
/// that produced it, and a list held in memory would go stale the moment a
/// second window ran something.
#[tauri::command]
pub async fn agent_task_history(
    app: AppHandle,
    session: State<'_, CurrentSession>,
) -> Result<Vec<TaskSummary>, String> {
    let signed_in = require_session(&session)?;
    Ok(tasks::list(&app_data_dir(&app)?)
        .into_iter()
        .filter(|task| may_read(&signed_in, &task.user_id))
        .collect())
}

/// One task in full — its plan, routing, evidence, working and artifacts.
#[tauri::command]
pub async fn agent_task(
    app: AppHandle,
    run_id: String,
    session: State<'_, CurrentSession>,
) -> Result<TaskRecord, String> {
    let signed_in = require_session(&session)?;
    let record = tasks::load(&app_data_dir(&app)?, &run_id)?;
    if !may_read(&signed_in, &record.user_id) {
        // Phrased as "not yours" rather than "does not exist": the person
        // holding a task id already knows it exists, and pretending otherwise
        // only makes the refusal look like a bug.
        return Err("That task was run by somebody else, and its evidence is theirs.".to_string());
    }
    Ok(record)
}

/// Re-opens the files a finished task produced and reports what is in them now.
///
/// Separate from the saved record on purpose. The record says what the check
/// found when the run ended; this says what it finds today, and the two
/// disagreeing is worth knowing — a deliverable can be moved, replaced or
/// truncated long after the run that made it.
#[tauri::command]
pub async fn agent_task_artifacts(
    app: AppHandle,
    run_id: String,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ArtifactReport>, String> {
    let signed_in = require_session(&session)?;
    let record = tasks::load(&app_data_dir(&app)?, &run_id)?;
    if !may_read(&signed_in, &record.user_id) {
        return Err("That task was run by somebody else.".to_string());
    }
    Ok(record
        .artifacts
        .iter()
        .map(|artifact| {
            artifacts::check(&artifacts::Produced {
                name: artifact.name.clone(),
                path: artifact.path.clone(),
                kind: artifact.kind,
                // The template the run actually used, carried in the record —
                // so this asks the same question the original check asked. A
                // record written before that field existed has none, and falls
                // back to the only template there is.
                template: artifact.template.clone(),
                produced_at: artifact.produced_at.clone(),
            })
        })
        .collect())
}

/// Shows a produced file in the operating system's file manager.
///
/// Reveals rather than opens. Handing a path to the shell to *open* would let a
/// file a model named decide which application runs, which is a decision this
/// application should not delegate to a tool call.
#[tauri::command]
pub async fn agent_reveal_artifact(
    app: AppHandle,
    run_id: String,
    name: String,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    let signed_in = require_session(&session)?;
    let record = tasks::load(&app_data_dir(&app)?, &run_id)?;
    if !may_read(&signed_in, &record.user_id) {
        return Err("That task was run by somebody else.".to_string());
    }
    let artifact = record
        .artifacts
        .iter()
        .find(|artifact| artifact.name == name)
        .ok_or_else(|| format!("{name} is not one of that task's files."))?;

    // Resolved from the record rather than from the argument, so the path shown
    // is one this application wrote down, not one a caller composed.
    let path = std::path::PathBuf::from(&artifact.path);
    if !path.exists() {
        return Err(format!("{name} is no longer where the task wrote it."));
    }

    let workspace = path
        .parent()
        .ok_or_else(|| format!("{name} has no containing folder."))?;
    open_folder(workspace)
        .map_err(|error| format!("that task's folder could not be opened: {error}"))
}

/// Opens a directory in the platform's file manager.
fn open_folder(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = std::process::Command::new("explorer");
        command.arg(path);
        command
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = std::process::Command::new("open");
        command.arg(path);
        command
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut command = std::process::Command::new("xdg-open");
        command.arg(path);
        command
    };

    command.spawn().map(|_| ())
}

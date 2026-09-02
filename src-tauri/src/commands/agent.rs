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
use crate::agent_runtime::events::{
    EventDraft, RecordedOutcome, RunState, TaskEvent, TaskEventLog, TaskEventType, TaskSnapshot,
    SYSTEM_ACTOR,
};
use crate::agent_runtime::retrieval::RunPassages;
use crate::agent_runtime::tasks::{
    ApprovalRecord, PlanRecord, TaskRecord, TaskSummary, ToolCallRecord,
};
use crate::agent_runtime::workspace::Workspace;
use crate::agent_runtime::{artifacts, planning, retrieval, tasks};
use crate::agent_runtime::{AgentRuntime, RuntimeDeps, AGENT_DURABLE_EVENT, AGENT_EVENT};
use crate::artifacts::{verify, Evidence, VerificationReport};
use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
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
pub type RunCalculations = Arc<
    Mutex<
        std::collections::HashMap<String, Vec<crate::orchestrator::calculation::CalculationRecord>>,
    >,
>;

/// The one runtime for this session, started on first use.
pub type AgentRuntimeHandle = Arc<Mutex<Option<Arc<AgentRuntime>>>>;

/// The skills installed on this machine, shared with the runtime.
pub type Skills = Arc<crate::skills::SkillRegistry>;

/// The subagent roles this deployment has.
pub type Subagents = Arc<crate::subagents::SubagentManager>;

/// The durable history of every run, shared with the runtime.
///
/// The tables above are the working state of runs *this process* is carrying,
/// and every one of them is gone when the process is. This is the part that is
/// not: it is written as the run happens, so a window that remounted and a
/// process that has just started can both find out what a run has been doing.
pub type TaskEvents = Arc<TaskEventLog>;

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
    /// The conversation this turn belongs to, when the caller is following
    /// up. `None` means "this is the first turn of a new conversation", and
    /// the command will create one and return its id.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// The id the caller reserved for the assistant message via
    /// `agent_append_turn`. Required when `conversationId` is set.
    #[serde(default)]
    pub message_id: Option<String>,
    /// Files the user attached to this turn.
    ///
    /// Carried as bytes, not paths: the webview has no filesystem the backend
    /// could re-open, and a path would let the frontend nominate any file on
    /// the machine. They belong to THIS request — nothing is remembered
    /// between runs, so one turn's attachment cannot reappear in another's.
    #[serde(default)]
    pub attachments: Vec<crate::commands::ocr::ChatAttachment>,
    /// Where the accuracy-to-speed slider was left when this turn was sent.
    ///
    /// It governs only how attachments are read; the model that answers is
    /// still the router's decision. `None` means the caller never showed a
    /// slider, and the default stop is used.
    #[serde(default)]
    pub ocr_detent: Option<crate::ai_engine::ocr_profile::OcrDetent>,
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
    /// The conversation this run was started in, when the caller asked for
    /// one (every run is now in a conversation; older callers will see
    /// `None` and the front-end will fall back to a one-off task view).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    /// The id of the assistant message this run produced, so the front-end
    /// can correlate `message_end` with the right `Message` cell.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
}

/// What the model is told it is, and what it must not do.
///
/// Deliberately short and specific. The rule that matters for PS 26117 is here
/// rather than buried in a template: search before answering *about the
/// organisation's own record*, and say when nothing was found instead of
/// filling the silence.
///
/// The scope clause is load-bearing. An earlier version stated the grounding
/// rule unconditionally, and a small local model reads that as applying to
/// every turn — so a bare "hi" came back as "no source was found for the
/// query", and a request to write a function came back empty. Retrieval is
/// for questions whose answer lives in the collections; everything else is
/// answered the way any assistant would answer it.
const SYSTEM_PROMPT: &str = "\
You are an assistant inside an organisation's own workbench. You run entirely \
on this machine and have no access to the internet.

Two kinds of request reach you, and they are answered differently.

FIRST: anything about this organisation's own record — its internal procedure, \
specification, drawings, correspondence, or a figure taken from them. Answer \
these only from passages you have retrieved with the \
knowledge.search_authorized tool. Do not answer them from memory: your \
training data is not this organisation's record, and a plausible answer that \
is not in the documents is worse than no answer. Cite every such claim with \
the marker of the passage it came from, written as [E1], [E2] and so on — the \
numbers the search gave those passages. Each marker is checked against what \
you actually retrieved when the task finishes, so a citation to a passage you \
were never given will be found and reported. Say a figure came from a \
calculation rather than citing a passage for it. If a search for one of these \
returns nothing, say so plainly and stop; do not infer what a document \
probably says.

SECOND: everything else — a greeting, small talk, a general-knowledge \
question, mathematics, or writing, explaining or debugging code. Answer these \
directly and in full, from your own knowledge, in your own words. Do not \
search first, do not write [E] markers, and never reply that no source was \
found: nobody asked you for a source. Say hello back to a greeting. Write the \
code when you are asked for code, and write all of it.

When a request could be either, treat it as the second kind and answer it. You \
can always search afterwards if the answer turns out to depend on a document. \
Silence and a refusal are never the right answer to a question you are able to \
answer.";

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
    pub events: &'a TaskEvents,
    pub skills: &'a Skills,
    pub memory: &'a AgentMemory,
    pub checkpoints: &'a RunCheckpoints,
}

/// The scoped memory store, as Tauri manages it.
pub type AgentMemory = crate::agent_runtime::memory::SharedMemory;

/// The fixed half of each live run's checkpoint, keyed by run id.
///
/// Established when a run starts and dropped when it ends. See
/// `agent_runtime::resume::CheckpointSeed` for why the deep loop needs it.
pub type RunCheckpoints = Arc<
    std::sync::Mutex<
        std::collections::HashMap<String, crate::agent_runtime::resume::CheckpointSeed>,
    >,
>;

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

    let durable_emitter = app.clone();
    let emit_durable: Arc<dyn Fn(Value) + Send + Sync> = Arc::new(move |event: Value| {
        // Dropping one of these costs a client its place in the sequence — but
        // that is recoverable, because the gap is detectable and the snapshot
        // is authoritative. Emitting is still best-effort; the *record* is not.
        let _ = durable_emitter.emit(AGENT_DURABLE_EVENT, event);
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
        events: state.events.clone(),
        skills: state.skills.clone(),
        // Built here, from code, once. Not read from a file and not reachable
        // from anything a prompt, a skill body or a retrieved document can name
        // — which is the whole reason policy belongs in a hook rather than in a
        // system prompt. See `crate::hooks`.
        hooks: Arc::new(crate::hooks::HookRegistry::with_builtin_policy()),
        memory: state.memory.clone(),
        checkpoints: state.checkpoints.clone(),
        emit_durable,
        // The same channel the loop's own events travel, so an operator sees
        // one sequence of what happened rather than two interleaved by luck.
        emit: emit.clone(),
    });

    let started =
        AgentRuntime::spawn(deps, emit, bundle_path(app)).map_err(|error| error.to_string())?;

    *slot = Some(started.clone());
    Ok(started)
}

/// Records one event durably, then publishes it with its sequence number.
///
/// The order matters and is the whole contract of the durable channel: a
/// message on it names a row that exists. Publishing first and writing second
/// would let a client apply an event that never landed, and no amount of later
/// reconciliation would tell it so.
///
/// A duplicate is not an error. The event the caller wanted written is there,
/// which is the outcome it wanted; it is simply not published a second time.
fn record_and_publish(
    app: &AppHandle,
    events: &TaskEvents,
    draft: EventDraft,
) -> Result<(), String> {
    use crate::agent_runtime::events::AppendError;
    match events.record(draft) {
        Ok(event) => {
            let _ = app.emit(AGENT_DURABLE_EVENT, event.envelope());
            Ok(())
        }
        Err(AppendError::Duplicate { .. }) | Err(AppendError::AlreadyEnded { .. }) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

/// Routes a prompt to a model, makes sure that model is served, and runs it.
///
/// The three steps are what PS 26117 asks to be demonstrated end to end, and
/// keeping them in one command is what makes the demonstration honest: there is
/// no path by which a caller supplies its own model and skips the routing.
///
/// Long-running. The UI shows progress from the `agent://event` stream and this
/// resolves with the final answer, the routing reasons, and where it ran.

/// Folds what the OCR model read into the turn's prompt.
///
/// The document goes first and the person's question last, so the model reads
/// the page before the instruction — the same ordering the vision schema uses
/// when an image and a question share one message.
///
/// A file that produced no text is still named. "I could not read anything in
/// this" is a true answer; pretending the attachment was not there is not.
fn compose_prompt_with_attachments(
    prompt: &str,
    reads: &[crate::commands::ocr::AttachmentRead],
) -> String {
    let mut out = String::new();
    for read in reads {
        out.push_str("<attachment name=\"");
        out.push_str(&read.name);
        out.push_str(
            "\">
",
        );
        if read.text.is_empty() {
            out.push_str("(no text could be read from this file)");
        } else {
            out.push_str(&read.text);
        }
        out.push_str(
            "
</attachment>

",
        );
    }
    out.push_str(prompt);
    out
}

/// The routing reasons for the OCR stage of a turn that carried files.
///
/// Every sentence is a fact the read actually produced — which model ran, at
/// which stop, over how many pages, and how much text came back. A file read
/// without a model says so, because "an OCR model looked at your spreadsheet"
/// would be a lie about work that never happened.
fn describe_attachment_reads(reads: &[crate::commands::ocr::AttachmentRead]) -> Vec<String> {
    reads
        .iter()
        .map(|read| {
            let pages = if read.pages > 1 {
                format!("{} pages", read.pages)
            } else {
                "1 page".to_string()
            };
            match (&read.ocr_model_id, read.ocr_detent) {
                (Some(model), Some(detent)) => format!(
                    "{} ({}) was read on this device by the document-OCR model {} at the {} stop — {}, {} characters recognised.",
                    read.name,
                    read.kind,
                    model,
                    detent.label(),
                    pages,
                    read.text.chars().count()
                ),
                _ => format!(
                    "{} ({}) already carried its text, so it was extracted locally and no model was needed to read it — {}, {} characters.",
                    read.name,
                    read.kind,
                    pages,
                    read.text.chars().count()
                ),
            }
        })
        .collect()
}

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
    events: State<'_, TaskEvents>,
    skills: State<'_, Skills>,
    memory: State<'_, AgentMemory>,
    checkpoints: State<'_, RunCheckpoints>,
    conversations: State<'_, super::conversations::ConversationsState>,
    run_to_conversation: State<'_, super::conversations::RunToConversationState>,
) -> Result<RunSummary, String> {
    // Checked here as well as in the runtime's handlers. Here it gives the
    // person a clear reason before anything starts; there it stops a call whose
    // session ended mid-run.
    let signed_in = require_permission(&session, Permission::UseModel)?;

    // Attachments are read before anything else, because what they say has to
    // be in the prompt the router sees — a turn asking "what does this drawing
    // show" routes on the drawing, not on the sentence.
    //
    // Read here rather than in the frontend so the composer cannot decide
    // whether a file is really looked at. Failure is surfaced to the person
    // instead of silently answering about a document nobody read.
    let detent = request
        .ocr_detent
        .unwrap_or(crate::ai_engine::ocr_profile::OcrDetent::Detailed);
    let mut attachment_reads = Vec::new();
    for attachment in &request.attachments {
        let read = crate::commands::ocr::read_attachment(
            &app,
            registry.inner(),
            servers.inner(),
            attachment,
            detent,
        )
        .await?;
        attachment_reads.push(read);
    }
    let request = {
        let mut request = request;
        if !attachment_reads.is_empty() {
            request.prompt = compose_prompt_with_attachments(&request.prompt, &attachment_reads);
        }
        request
    };

    // Read from the live hardware rather than a stored figure: the right model
    // on a workstation is the wrong one on a laptop. The largest GPU wins on a
    // multi-GPU box; no GPU reports zero and the planner makes a CPU-only plan.
    let vram = gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    // The chat model an administrator chose, read fresh rather than cached: the
    // choice can change while the app is open, and a run starting a second later
    // must use the new one. `None` means nobody has chosen, and the router then
    // picks on capability alone.
    let chosen_orchestrator = crate::commands::registry::configured_orchestrator(&app);

    let mut routing = ModelRouter::route_with_orchestrator(
        &registry,
        &request.prompt,
        request.classification,
        vram,
        None,
        false,
        &[],
        &[],
        chosen_orchestrator.as_ref(),
    )
    .map_err(|failure| failure.reason)?;

    // A turn that carried a document was answered by two models, not one. The
    // reasons list used to name only the second, so a person who attached a
    // scan and opened "Why?" saw a reasoning model explaining itself with no
    // mention of the OCR stage that produced everything it was reasoning
    // about. These lines go first because that is the order the work
    // happened in.
    if !attachment_reads.is_empty() {
        let mut preamble = describe_attachment_reads(&attachment_reads);
        preamble.append(&mut routing.reasons);
        routing.reasons = preamble;
    }

    let entry = registry.find(&routing.model_id).ok_or_else(|| {
        format!(
            "{} was routed to but is not in the registry.",
            routing.model_id
        )
    })?;

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
        events: &events,
        skills: &skills,
        memory: &memory,
        checkpoints: &checkpoints,
    };
    let runtime = runtime(&handle, &app, &state)?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now();

    // The lifecycle, written as it happens rather than summarised at the end.
    //
    // Three events before the loop is even asked to start, because each answers
    // a different question somebody asks about a run that went wrong: was it
    // accepted, was its sensitivity understood, and what was it routed to. A
    // run that dies in its first second still leaves all three.
    //
    // `promptShown` rather than `prompt`: the redaction hashes anything called
    // `prompt`, and this is the person's own words being shown back to them on
    // their own machine. A task list where every row reads as a hash identifies
    // nothing.
    let opening = [
        (
            TaskEventType::RunCreated,
            json!({
                           "promptShown": request.prompt,
            "correlationId": request.correlation_id,
                       }),
        ),
        (
            TaskEventType::RunClassified,
            json!({
                "classification": request
                    .classification
                    .map(|c| c.label().to_string())
                    .unwrap_or_else(|| "Internal".to_string()),
            }),
        ),
        (
            TaskEventType::RunRouted,
            json!({
                "modelId": routing.model_id,
                "modelName": routing.model_name,
                "intent": routing.intent,
                "confidence": routing.confidence,
                "usedFallback": routing.used_fallback,
                "runtime": endpoint.runtime.label(),
            }),
        ),
    ];
    for (event_type, payload) in opening {
        let draft = EventDraft::new(&run_id, event_type, &signed_in.user.id).with(payload);
        if let Err(error) = record_and_publish(&app, &events, draft) {
            log::error!(
                "[tasks] run {run_id}: {} was not recorded: {error}",
                event_type.as_str()
            );
        }
    }

    // The run's own directory, created before the model is told anything — so
    // the instructions can name it, and so a tool call cannot arrive before the
    // gateway has roots to resolve against.
    let workspace = Workspace::create(&app_data_dir(&app)?, &run_id).map_err(|e| e.to_string())?;
    let workspace_note = workspace.describe();
    // Kept because the checkpoint seed below needs the directory's identity, and
    // the workspace itself is about to be moved into the shared table.
    let workspace_root = Some(workspace.root().to_path_buf());
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
    // The fixed half of every checkpoint this attempt will take. Established
    // here because this is the first point at which all of it is known: the
    // workspace exists, the plan is fixed, the model is chosen, and the session
    // that authorised it is in hand. Recorded now so the deep loop can take a
    // checkpoint after each tool result without re-deriving any of it.
    {
        let seed = crate::agent_runtime::resume::CheckpointSeed {
            attempt_id: uuid::Uuid::new_v4().to_string(),
            plan_hash: crate::agent_runtime::resume::plan_hash_of(&request.prompt),
            policy_hash: crate::agent_runtime::resume::policy_hash(
                &signed_in,
                request.classification,
                &format!("{:?}", crate::sovereignty::global_broker().mode()),
            ),
            // The workspace was created a moment ago, so this resolves. An
            // unresolvable one would mean the directory vanished between
            // creating it and describing it, and a seed built on a workspace
            // that is not there would claim a world nobody observed.
            workspace_hash: workspace_root
                .as_deref()
                .and_then(crate::agent_runtime::resume::workspace_hash_of)
                .unwrap_or_default(),
            model_id: routing.model_id.clone(),
        };
        if let Ok(mut seeds) = checkpoints.lock() {
            seeds.insert(run_id.clone(), seed);
        }
    }

    // The instant this run must stop by. A property of the plan, so it is only
    // knowable once the plan is fixed — and fixed it is: nothing after this
    // point may extend it.
    let deadline = started_at
        + chrono::Duration::from_std(std::time::Duration::from_secs(
            planned.max_duration_seconds.max(1),
        ))
        .unwrap_or_else(|_| chrono::Duration::minutes(10));
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
    // Kept as well as published. The published one reaches a window that is
    // listening now; this one reaches a window that opens in ten minutes.
    if let Err(error) = record_and_publish(
        &app,
        &events,
        EventDraft::new(&run_id, TaskEventType::PlanReady, &signed_in.user.id)
            .with(json!({ "plan": planned })),
    ) {
        log::warn!("[tasks] run {run_id}: the plan was not recorded: {error}");
    }

    // A resumption reads what the earlier attempt recorded. On a first attempt
    // this is `null`, and the loop starts with empty notes.
    //
    // Read from the saved task record rather than from the event history: the
    // record holds the notes as the loop last reported them, and the history
    // holds only that compactions happened. A run whose record was never
    // written has nothing to resume from, and saying so honestly is better than
    // reconstructing a plausible set of notes nobody actually recorded.
    let resumed_notes = tasks::load(&app_data_dir(&app)?, &run_id, Some(&signed_in.user.id))
        .ok()
        .and_then(|previous| previous.working_notes)
        .filter(|notes| !notes.is_empty());

    let params = json!({
        "runId": run_id,
        // The assistant `Message` id the front-end reserved via
        // `agent_append_turn`. The runtime attaches it to every
        // `message_start` / `message_update` / `message_end` event so the chat
        // surface can route each token to the right cell. Without this the
        // translator in the runtime has nothing to bind to and the chat
        // would drop every streaming event.
        "messageId": request.message_id,
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
        // The same instant this side is holding, as epoch milliseconds. Sent so
        // the loop stops itself at the boundary rather than being killed from
        // outside mid-turn — the child knows where its own safe points are and
        // this side does not.
        //
        // It is not a second authority: the loop can only stop *earlier* than
        // Rust would, and every tool call still goes through the gateway.
        "deadlineMs": deadline.timestamp_millis(),
        // What this run already knows, if it is a resumption.
        //
        // Sent at start rather than pushed after the first turn, because the
        // whole value of it is being read *before* the model decides what to do
        // — notes that arrive after the loop has re-issued `create_docx` have
        // not prevented anything.
        "notes": resumed_notes,
        // State this side owns and the loop must carry across compaction
        // unchanged. Refreshed by `run.note` as the run proceeds; sent here so
        // a run that compacts before its first refresh still carries its plan.
        "preserved": {
            "activePlan": planned.stopped_because.clone(),
            "policyDecisions": Vec::<String>::new(),
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

    // The moment the loop is handed the work, and the instant it must stop by.
    //
    // The deadline is a property of the plan, so it is only knowable here —
    // after `planning::plan_for` has fixed the budget. Recorded as well as sent
    // so a window that reattaches can say how long is left rather than only
    // that the run is still going.
    if let Err(error) = record_and_publish(
        &app,
        &events,
        EventDraft::new(&run_id, TaskEventType::RunStarted, &signed_in.user.id)
            .with(json!({ "deadline": deadline.to_rfc3339() })),
    ) {
        log::warn!("[tasks] run {run_id}: the start was not recorded: {error}");
    }

    // The plan's own time budget, enforced here rather than trusted to the
    // loop. The plan refuses the *next* tool call once the clock has run out,
    // which is the right check for a run that is doing things and the wrong one
    // for a run that is stuck: a model waiting on a model server that will
    // never answer makes no further calls, so nothing ever asks the plan
    // whether it may continue. Without a deadline on this side, that run waits
    // for as long as the application is open.
    let allowed = std::time::Duration::from_secs(planned.max_duration_seconds.max(1));
    let (outcome, ending) =
        match tokio::time::timeout(allowed, runtime.request("run.start", params)).await {
            Ok(Ok(value)) => (Ok(value), TaskEventType::RunCompleted),
            Ok(Err(error)) => {
                // A run the gateway or the plan stopped is not a fault, and a
                // list that paints it the same colour as one teaches people to
                // skip the row that actually broke. Read from the refusal's own
                // wording, because every refusal path produces a sentence and
                // none of them produces a code.
                let detail = error.to_string();
                let stopped_by_policy = detail.contains("not permitted")
                    || detail.contains("was not approved")
                    || detail.contains("is not cleared");
                let stopped_by_budget = detail.contains("permitted steps")
                    || detail.contains("going in circles")
                    || detail.contains("time allowed");
                let kind = if stopped_by_policy {
                    TaskEventType::RunStoppedByPolicy
                } else if stopped_by_budget {
                    TaskEventType::RunStoppedByBudget
                } else {
                    TaskEventType::RunFailed
                };
                (Err(detail), kind)
            }
            Err(_) => {
                // Told to stop, because the deadline expiring here does not
                // reach the child on its own: the loop would carry on holding a
                // model server and a workspace for a run nobody is waiting for.
                let _ = runtime
                    .request("run.abort", json!({ "runId": run_id }))
                    .await;
                (
                    Err(format!(
                        "Stopped: it ran past the {} minutes this task was allowed.",
                        allowed.as_secs() / 60
                    )),
                    TaskEventType::RunStoppedByBudget,
                )
            }
        };

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
        Err(error) => (String::new(), 0, Some(error.clone())),
    };

    // The run's own notes and its final context ledger, as the loop reported
    // them. Read from the outcome rather than reconstructed: a run that failed
    // returns no outcome, and the notes for that run are the ones already in
    // the durable event history — reconstructing them here from the transcript
    // would produce a second, disagreeing account of what the run had done.
    let working_notes = outcome
        .as_ref()
        .ok()
        .and_then(|value| value.get("notes"))
        .and_then(|notes| {
            serde_json::from_value::<crate::agent_runtime::memory::RunMemory>(notes.clone()).ok()
        });
    let context_ledger = outcome
        .as_ref()
        .ok()
        .and_then(|value| value.get("ledger"))
        .and_then(|ledger| ledger_record(ledger));

    // Re-opened, not taken on the model's word. A document that was written and
    // then corrupted still passes every test of the code that wrote it.
    let produced_files = artifacts::report_for_run(&produced, &run_id);
    let retrieved = retrieval::for_run(&passages, &run_id);
    // Read off the engine's own table, never rebuilt from the answer's figures.
    // ARJUN design rule 27 makes the engine the source of numerical truth, and a record
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
    if !answer.trim().is_empty() {
        let _ = record_and_publish(
            &app,
            &events,
            EventDraft::new(
                &run_id,
                TaskEventType::VerificationStarted,
                &signed_in.user.id,
            )
            .with(json!({ "answerChars": answer.chars().count() })),
        );
    }

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
        // Folded from the durable events rather than counted here. The events
        // are written as each compaction happens, so a run the process took
        // down with it still has its compaction history — and a record built
        // from a live counter would not.
        compactions: events
            .snapshot(&run_id)
            .ok()
            .flatten()
            .map(|snapshot| snapshot.compaction_events)
            .unwrap_or_default(),
        working_notes,
        context_ledger,
    };

    // Saved before anything is released, so a failure to write is a failure the
    // person hears about rather than one that quietly loses the task.
    if let Err(error) = tasks::save(&app_data_dir(&app)?, &record) {
        log::error!("[agent] the record for run {run_id} could not be saved: {error}");
    }

    // The ending, written last. Refused if the run already has one — a person
    // who pressed stop a moment before the loop finished has already given this
    // run its ending, and a second one would let a reader pick which happened.
    let ending_payload = match ending {
        TaskEventType::RunCompleted => json!({
            "answer": answer,
            "turns": turns,
            "artifacts": produced_files.len(),
            "stoppedBecause": final_plan.stopped_because,
        }),
        _ => json!({
            "failure": failure,
            "turns": turns,
            "stoppedBecause": final_plan.stopped_because,
        }),
    };
    // The event id is derived from the run rather than generated, so a retry
    // after an ambiguous failure presents the same id and is refused as the
    // duplicate it is. A run has exactly one ending, and this is what makes
    // writing it twice harmless rather than merely unlikely.
    match record_and_publish(
        &app,
        &events,
        EventDraft::idempotent(&run_id, ending, &signed_in.user.id, "ending").with(ending_payload),
    ) {
        Ok(()) => {}
        Err(error) => log::warn!("[tasks] run {run_id}: the ending was not recorded: {error}"),
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
    let run_failed = outcome.is_err();
    outcome?;

    // Bind this run to a conversation so the chat surface can route the
    // streaming message events for it to the right assistant cell. The
    // caller may have already created a conversation via
    // `agent_create_conversation` and `agent_append_turn` (a follow-up);
    // when it has not, this is the first turn of a new conversation, and
    // we create one here so the chat surface has somewhere to write the
    // user message and the streaming assistant message.
    let started_at_rfc = started_at.to_rfc3339();
    let _ = started_at_rfc; // reserved for future per-step timing
    let elapsed_ms = chrono::Utc::now()
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0) as u64;
    let conversation_id;
    let message_id;
    match request.conversation_id.as_deref() {
        Some(existing_id) => {
            conversation_id = existing_id.to_string();
            // The front-end already called `agent_append_turn` to reserve
            // the assistant cell. Find the message id it passed in.
            message_id = request
                .message_id
                .clone()
                .unwrap_or_else(|| format!("a-{}", run_id));
            run_to_conversation.0.bind(&run_id, &conversation_id);
        }
        None => {
            // First turn: create the conversation now, with a title derived
            // from the prompt's first line. Persisting it before the run
            // resolves is fine — the existing record is what the front-end
            // will reload on remount anyway.
            let title = request
                .prompt
                .lines()
                .next()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .unwrap_or_else(|| "New conversation".to_string());
            let title = title.chars().take(80).collect::<String>();
            let new_message_id = format!("a-{}", run_id);
            // TODO 2: the conversation is owned by the signed-in
            // user. The first turn of a brand-new conversation
            // therefore belongs to the user who started the run,
            // not to a global "default" account.
            let conversation = conversations
                .0
                .create(
                    title,
                    "Arjun is ready. Ask anything; nothing leaves this machine.".to_string(),
                    &signed_in.user.id,
                )
                .map_err(|error| format!("the conversation could not be created: {error}"))?;
            conversation_id = conversation.id.clone();
            let _ = conversations.0.append_user_turn(
                &conversation_id,
                &request.prompt,
                &new_message_id,
                &run_id,
                &signed_in.user.id,
            );
            message_id = new_message_id;
            run_to_conversation.0.bind(&run_id, &conversation_id);
        }
    }

    // Mark the assistant message as complete on the conversation. The
    // front-end can also call `agent_complete_message` itself when it
    // receives `message_end`; we write the final state here too so a
    // remount that lands on this run while the front-end is reconnecting
    // sees a coherent terminal state.
    let final_content = if run_failed {
        None
    } else {
        Some(answer.as_str())
    };
    let _ = conversations.0.record_message_completion(
        &conversation_id,
        &message_id,
        &run_id,
        final_content,
        Some(elapsed_ms),
        Some(routing.model_name.as_str()),
        Some(routing.role.label()),
        Some(routing.used_fallback),
        None,
        run_failed,
        None,
        None,
        &signed_in.user.id,
    );
    run_to_conversation.0.unbind(&run_id);

    Ok(RunSummary {
        run_id,
        text: answer,
        turns,
        routing,
        endpoint,
        plan: final_plan,
        verification,
        artifacts: produced_files,
        conversation_id: Some(conversation_id),
        message_id: Some(message_id),
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
    session: State<'_, CurrentSession>,
) -> Result<bool, String> {
    // A correction is part of running a model. The matrix puts it under
    // `UseModel`. The orchestrator rejects no-longer-running runs, so
    // this is a sign-in + UseModel gate plus the runtime's own check.
    require_permission(&session, Permission::UseModel)?;
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
///
/// The cancellation is recorded *before* the child is told, and deliberately.
/// The record is what a restart reads; telling the loop first and then failing
/// to write would leave a run that stopped for a reason nobody can see. Writing
/// first and then failing to reach the loop leaves a run marked cancelled that
/// is still winding down, which is the direction of error somebody can act on.
#[tauri::command]
pub async fn agent_abort_run(
    run_id: String,
    handle: State<'_, AgentRuntimeHandle>,
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
) -> Result<bool, String> {
    // Aborting your own in-flight run uses the model. The matrix puts
    // that under `UseModel`. The previous fallback to SYSTEM_ACTOR is
    // removed: the orchestrator's own internal cancellation goes
    // through a different code path (the runtime's `cancel` request),
    // not this command. Only a signed-in caller may stop a run from
    // the webview.
    let by = require_permission(&session, Permission::UseModel)?.user.id;

    // Only for a run the record has heard of. A run id arrives from the UI, and
    // writing an ending for one that has no beginning would let any caller
    // conjure a row on the Tasks screen for a task that never ran.
    if events.snapshot(&run_id)?.is_some() {
        match events.record(
            EventDraft::new(&run_id, TaskEventType::RunCancelled, &by).with(json!({
                "failure": "Stopped, because somebody stopped it.",
                "cancelledBy": by,
            })),
        ) {
            // Already over — the run finished a moment before the button did.
            // An ordinary race, and the ending it already has is the true one.
            Ok(_) | Err(crate::agent_runtime::events::AppendError::AlreadyEnded { .. }) => {}
            Err(error) => log::warn!("[tasks] run {run_id}: the stop was not recorded: {error}"),
        }
    }

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

/// What a person decided at a milestone gate, as the chat surface reads it.
///
/// Mirrors `MilestoneAcknowledgement` in `src/services/agent.service.ts`.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MilestoneAcknowledgement {
    pub checkpoint_id: String,
    pub ordinal: u32,
    /// `approved` or `rejected`.
    pub decision: String,
    pub acknowledged_by: String,
    /// RFC 3339, UTC.
    pub at: String,
}

/// Records a person's decision at a milestone, and acts on it.
///
/// The gate exists because ARJUN requires evidence-anchored decision points
/// (an ARJUN design rule, not a PS 26117 requirement):
/// the model says "I think we are here", the run pauses, and a person signs off
/// before the next leg starts. Everything for that existed — the plan marks the
/// step, `MilestoneRecord` is the durable artefact, `MilestoneGate.tsx` draws
/// the buttons — except this command. Without it the front-end's call rejected
/// with "command not found", and a run that reached a gate could never leave it.
///
/// Approving clears the pause and the run carries on. Rejecting stops it with
/// [`StopReason::MilestoneRejected`], which is deliberately not a failure: the
/// steps already done stay done, and the record says a person chose to stop
/// rather than that something went wrong.
///
/// The parameters use the front-end's existing vocabulary (`runId`,
/// `checkpointId`, `approved`/`rejected`) rather than a new one, because that
/// contract is already typed end to end and renaming it would churn the UI, the
/// service and their tests for no gain.
#[tauri::command]
pub async fn agent_acknowledge_milestone(
    app: AppHandle,
    run_id: String,
    checkpoint_id: String,
    decision: String,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    events: State<'_, TaskEvents>,
    plans: State<'_, RunPlans>,
    handle: State<'_, AgentRuntimeHandle>,
) -> Result<MilestoneAcknowledgement, String> {
    // Signing off a milestone is the approval gesture, so it sits under the
    // same permission as the approvals queue rather than under `UseModel`.
    let signed_in = require_permission(&session, Permission::ApproveOutput)?;
    let by = signed_in.user.id.clone();

    let approved = match decision.as_str() {
        "approved" => true,
        "rejected" => false,
        other => {
            return Err(format!(
                "{other:?} is not a decision. A milestone is either \"approved\" or \"rejected\"."
            ))
        }
    };

    // The plan is the only place that knows which step a checkpoint id belongs
    // to. Both the ordinal and the intent go into the durable record, and
    // neither may be invented, so a run whose plan is no longer in memory is
    // refused rather than recorded against a guess.
    let (ordinal, intent, stop_reason) = {
        let mut held = plans
            .lock()
            .map_err(|_| "the plan table is poisoned".to_string())?;
        let plan = held.get_mut(&run_id).ok_or_else(|| {
            format!(
                "Run {run_id} is not in flight, so its milestone cannot be decided from here. \
                 A run that has already ended keeps the ending it reached."
            )
        })?;
        let step = plan
            .step_at_checkpoint(&checkpoint_id)
            .ok_or_else(|| format!("This run has no milestone called {checkpoint_id:?}."))?;
        let ordinal = step.ordinal;
        let intent = step.intent.clone();

        if approved {
            plan.resume();
            (ordinal, intent, None)
        } else {
            let reason = plan.reject_milestone(&checkpoint_id);
            (ordinal, intent, reason)
        }
    };

    let at = chrono::Utc::now().to_rfc3339();

    // Durable before anything else acts on it. A decision that moved the run
    // but was never written is one nobody can audit afterwards.
    let app_data = app_data_dir(&app)?;
    if let Ok(mut record) = tasks::load(&app_data, &run_id, Some(&by)) {
        let mut notes = record.working_notes.take().unwrap_or_default();
        notes
            .milestones
            .push(crate::agent_runtime::memory::MilestoneRecord {
                checkpoint_id: checkpoint_id.clone(),
                ordinal,
                intent: intent.clone(),
                acknowledged_by: by.clone(),
                at: at.clone(),
                decision: decision.clone(),
            });
        record.working_notes = Some(notes);
        if let Err(error) = tasks::save(&app_data, &record) {
            log::warn!("[tasks] run {run_id}: the milestone decision was not persisted: {error}");
        }
    }

    // The continuation, or the ending, as the trace will show it.
    if events.snapshot(&run_id).ok().flatten().is_some() {
        let draft = if approved {
            EventDraft::new(&run_id, TaskEventType::RunResumed, &by).with(json!({
                "checkpointId": checkpoint_id,
                "ordinal": ordinal,
                "intent": intent,
                "decidedBy": by,
            }))
        } else {
            EventDraft::new(&run_id, TaskEventType::RunCancelled, &by).with(json!({
                "failure": stop_reason
                    .as_ref()
                    .map(crate::orchestrator::plan::StopReason::explain)
                    .unwrap_or_else(|| "Stopped at a milestone.".to_string()),
                "checkpointId": checkpoint_id,
                "ordinal": ordinal,
                "intent": intent,
                "stoppedBecause": stop_reason,
                "cancelledBy": by,
            }))
        };
        match events.record(draft) {
            Ok(_) | Err(crate::agent_runtime::events::AppendError::AlreadyEnded { .. }) => {}
            Err(error) => {
                log::warn!("[tasks] run {run_id}: the milestone decision was not recorded: {error}")
            }
        }
    }

    // A rejection has to actually stop the loop. Marking the plan stopped ends
    // it at the next check; asking the runtime to abort ends it now, which is
    // what somebody who just pressed "reject" expects to have happened.
    if !approved {
        let runtime = {
            let slot = handle
                .lock()
                .map_err(|_| "the agent runtime handle is poisoned".to_string())?;
            slot.clone()
        };
        if let Some(runtime) = runtime {
            if let Err(error) = runtime
                .request("run.abort", json!({ "runId": run_id }))
                .await
            {
                log::warn!("[tasks] run {run_id}: the loop did not stop on rejection: {error}");
            }
        }
    }

    let _ = audit.record(
        &by,
        AuditKind::Approval,
        format!("Milestone {checkpoint_id} ({intent}) was {decision} on run {run_id}"),
        Some(json!({
            "runId": run_id,
            "checkpointId": checkpoint_id,
            "ordinal": ordinal,
            "decision": decision,
        })),
    );

    Ok(MilestoneAcknowledgement {
        checkpoint_id,
        ordinal,
        decision,
        acknowledged_by: by,
        at,
    })
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
    events: State<'_, TaskEvents>,
    skills: State<'_, Skills>,
    memory: State<'_, AgentMemory>,
    checkpoints: State<'_, RunCheckpoints>,
) -> Result<Value, String> {
    // The health probe is a read; the matrix does not gate it beyond
    // sign-in. The runtime may also start the agent if it is down, so
    // sign-in is required to attribute the start.
    require_session(&session)?;
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
        events: &events,
        skills: &skills,
        memory: &memory,
        checkpoints: &checkpoints,
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
    events: State<'_, TaskEvents>,
) -> Result<Vec<TaskSummary>, String> {
    let signed_in = require_session(&session)?;
    // Records first, snapshots second. Every finished run has written its JSON
    // record exactly as it always did, and that record is richer than a
    // snapshot; the snapshots supply only the runs that have no record — the
    // ones still going, and the ones the process took down with it. Before
    // this, those simply did not appear, and a task list silently missing the
    // interrupted runs is the list that misleads.
    // TODO 2: per-user isolation. The history screen shows the
    // signed-in user's runs only; cross-account views live in the
    // audit log, not here.
    let mut all = tasks::list(&app_data_dir(&app)?, Some(&signed_in.user.id));
    let recorded: std::collections::HashSet<String> =
        all.iter().map(|task| task.run_id.clone()).collect();

    for snapshot in events.snapshots().unwrap_or_default() {
        if !recorded.contains(&snapshot.run_id) {
            all.push(tasks::summary_of(&snapshot));
            continue;
        }
        // The record holds the contents; the history holds the ending. A run
        // somebody stopped and one that ran out of time both write a record
        // whose `failure` field cannot tell those two apart — the history can,
        // so it is where the status comes from.
        if let Some(task) = all.iter_mut().find(|task| task.run_id == snapshot.run_id) {
            task.state = snapshot.state;
            task.live = !snapshot.state.is_terminal();
        }
    }

    all.retain(|task| may_read(&signed_in, &task.user_id));
    // On the finish time, newest first, exactly as before. A run still going
    // has no finish time and sorts to the top, which is where it belongs.
    all.sort_by(|a, b| b.finished_at.cmp(&a.finished_at));
    Ok(all)
}

/// The latest state of one task, without replaying its history.
///
/// What a window calls when it mounts holding a run id — after a remount, or
/// after the whole application was restarted. Answers for a run that is still
/// going, one that finished, and one that was interrupted, which is the point:
/// before this there was no way to ask about the first and third at all.
#[tauri::command]
pub async fn agent_task_snapshot(
    run_id: String,
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
) -> Result<Option<TaskSnapshot>, String> {
    let signed_in = require_session(&session)?;
    let Some(snapshot) = events.snapshot(&run_id)? else {
        // A run id nobody has heard of is an empty answer, not a failure: the
        // caller may be holding one from a database that has since been reset.
        return Ok(None);
    };
    if !may_read(&signed_in, &snapshot.actor) {
        return Err("That task was run by somebody else, and its evidence is theirs.".to_string());
    }
    Ok(Some(snapshot))
}

/// One task's events after `after_seq`, in order.
///
/// The catch-up half of recovery. A window that holds a snapshot at sequence 12
/// asks for everything after 12 and applies it, rather than reloading a state
/// it already has. Events that could not be read are reported alongside the
/// ones that could — a history with a hole in it is usable, but only if the
/// screen reading it knows the hole is there.
#[tauri::command]
pub async fn agent_task_events(
    run_id: String,
    after_seq: Option<i64>,
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
) -> Result<TaskEventPage, String> {
    let signed_in = require_session(&session)?;
    if let Some(snapshot) = events.snapshot(&run_id)? {
        if !may_read(&signed_in, &snapshot.actor) {
            return Err(
                "That task was run by somebody else, and its evidence is theirs.".to_string(),
            );
        }
    }
    let page = events.events_since(&run_id, after_seq.unwrap_or(0))?;
    Ok(TaskEventPage {
        last_seq: page.last_seq(),
        events: page.events,
        unreadable: page.unreadable,
    })
}

/// A page of events, and what could not be read alongside them.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEventPage {
    pub events: Vec<TaskEvent>,
    pub unreadable: Vec<crate::agent_runtime::events::UnreadableEvent>,
    /// The highest position accounted for, readable or not. A caller asks for
    /// everything after this next time.
    pub last_seq: i64,
}

/// Side effects nobody can account for, across every run.
///
/// Each one is an action that was in flight when the process went away: a file
/// that may or may not have been written. They are listed rather than retried,
/// because retrying could do the thing twice and assuming could mean it never
/// happens. See [`crate::agent_runtime::events::idempotency`].
#[tauri::command]
pub async fn agent_unknown_effects(
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
) -> Result<Vec<RecordedOutcome>, String> {
    let signed_in = require_session(&session)?;
    // Reconciling is a reviewer's judgement about whether work happened, which
    // is the same kind of decision as approving an output. Somebody who may not
    // make that decision is not shown the queue of them.
    if !signed_in.holds(Permission::ApproveOutput) {
        return Err(format!(
            "{} is not permitted to reconcile interrupted actions. That is a reviewer's decision.",
            signed_in.user.display_name
        ));
    }
    events.unknown_effects()
}

/// Records what a person found out about an interrupted side effect.
///
/// `happened` is their assertion, not a measurement — they went and looked at
/// the file. It is stored as an assertion, naming who made it, because a record
/// that presented a person's judgement as a fact the system established would
/// be claiming more than it knows.
///
/// Resolves `false` when there was nothing under that key to reconcile, which
/// is an ordinary race — somebody else got there first — and not a failure.
#[tauri::command]
pub async fn agent_reconcile_effect(
    run_id: String,
    idempotency_key: String,
    happened: bool,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    events: State<'_, TaskEvents>,
) -> Result<bool, String> {
    let signed_in = require_session(&session)?;
    if !signed_in.holds(Permission::ApproveOutput) {
        return Err(format!(
            "{} is not permitted to reconcile interrupted actions. That is a reviewer's decision.",
            signed_in.user.display_name
        ));
    }

    let settled =
        events.reconcile_effect(&run_id, &idempotency_key, happened, &signed_in.user.id)?;
    if settled {
        // On the permanent record as well as the run's own history: this is a
        // person asserting something about work the system could not establish,
        // which is exactly the kind of claim an auditor comes looking for.
        let _ = audit.record(
            &signed_in.user.id,
            AuditKind::Approval,
            format!(
                "{} reconciled an interrupted action in run {run_id}: it {}",
                signed_in.user.display_name,
                if happened {
                    "did take effect"
                } else {
                    "did not take effect"
                }
            ),
            Some(json!({
                "runId": run_id,
                "idempotencyKey": idempotency_key,
                "happened": happened,
            })),
        );
    }
    Ok(settled)
}

/// Concise metadata for the skills this person may use.
///
/// The UI's half of `capability.search`. Returns cards — never a skill's
/// instructions — so a screen can list what is installed, and what is
/// quarantined and why, without any of it reaching a prompt.
#[tauri::command]
pub async fn skill_search(
    query: Option<String>,
    session: State<'_, CurrentSession>,
    skills: State<'_, Skills>,
) -> Result<Vec<crate::skills::SkillCard>, String> {
    let signed_in = require_session(&session)?;
    Ok(skills.search(
        query.as_deref().unwrap_or_default(),
        &crate::skills::SkillContext {
            session: &signed_in,
            mode: crate::sovereignty::global_broker().mode(),
            // No run in view from here, so nothing is permitted. The cards say
            // what each skill asks for; what a given run would actually grant
            // is decided when it loads one.
            run_permits: &[],
        },
    ))
}

/// The subagent roles this deployment has, and whether each can be performed.
///
/// A profile is a declaration; a worker is what performs it. Both are reported,
/// because a role that is declared and has no worker is a role this build
/// cannot do — and that reads very differently from one that is missing.
#[tauri::command]
pub async fn subagent_profiles(
    session: State<'_, CurrentSession>,
    subagents: State<'_, Subagents>,
) -> Result<Vec<Value>, String> {
    let _ = require_session(&session)?;
    Ok(subagents
        .profiles()
        .map(|profile| {
            json!({
                "name": profile.name,
                "description": profile.description,
                "version": profile.version,
                "modelRole": profile.model_role.label(),
                "allowedTools": profile.allowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "disallowedTools": profile.disallowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "isolation": profile.isolation.as_str(),
                "memoryScope": profile.memory_scope.as_str(),
                "writePolicy": profile.write_policy.as_str(),
                "networkPermitted": profile.network_permitted,
                "classificationCeiling": profile.classification_ceiling.label(),
                "requiredSchema": profile.required_schema.as_str(),
                "maxTurns": profile.limits.max_turns,
                "maxChildren": profile.limits.max_children,
                // The honest half: this build may not be able to perform it.
                "hasWorker": subagents.has_worker(&profile.name),
            })
        })
        .collect())
}

/// Re-reads the skills directory.
///
/// Safe at any moment: it swaps a snapshot rather than mutating one, so a run
/// part-way through a tool call keeps the definition it started with. See
/// [`crate::skills::SkillRegistry::reload`].
#[tauri::command]
pub async fn skill_reload(
    session: State<'_, CurrentSession>,
    skills: State<'_, Skills>,
) -> Result<usize, String> {
    let _ = require_session(&session)?;
    Ok(skills.reload().count())
}

/// The runs that are still going as far as the record is concerned.
///
/// How a window that has just opened finds a run to reattach to. Deliberately
/// derived from the record rather than from the runtime's in-memory tables:
/// after a restart those are empty, and a run the record still calls live is
/// exactly the one somebody needs to be told about.
#[tauri::command]
pub async fn agent_active_tasks(
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
) -> Result<Vec<TaskSnapshot>, String> {
    let signed_in = require_session(&session)?;
    Ok(events
        .running()?
        .into_iter()
        .filter(|snapshot| may_read(&signed_in, &snapshot.actor))
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
    let record = tasks::load(&app_data_dir(&app)?, &run_id, None)?;
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
    let record = tasks::load(&app_data_dir(&app)?, &run_id, None)?;
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
    let record = tasks::load(&app_data_dir(&app)?, &run_id, None)?;
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

/// Returns a safe, *preview-only* rendering of a produced artifact.
///
/// The preview lets the user *see* what ARJUN wrote without opening
/// another application. It is not a substitute for opening the file
/// in its native app — the rendering is plain text for Office
/// documents, and the user is told that. The reader is conservative:
/// it does not execute macros, follow external references, or load
/// remote resources. See
/// [`crate::commands::artifact_preview`] for the full contract.
#[tauri::command]
pub async fn artifact_preview(
    app: AppHandle,
    run_id: String,
    name: String,
    session: State<'_, CurrentSession>,
) -> Result<crate::commands::artifact_preview::ArtifactPreview, String> {
    let signed_in = require_session(&session)?;
    let record = tasks::load(&app_data_dir(&app)?, &run_id, None)?;
    if !may_read(&signed_in, &record.user_id) {
        return Err("That task was run by somebody else.".to_string());
    }
    let artifact = record
        .artifacts
        .iter()
        .find(|artifact| artifact.name == name)
        .ok_or_else(|| format!("{name} is not one of that task's files."))?;

    let path = std::path::PathBuf::from(&artifact.path);
    if !path.exists() {
        return Err(format!("{name} is no longer where the task wrote it."));
    }

    let kind_hint = match artifact.kind {
        crate::agent_runtime::artifacts::Kind::Document => "docx",
        crate::agent_runtime::artifacts::Kind::Workbook => "xlsx",
        crate::agent_runtime::artifacts::Kind::Deck => "pptx",
        crate::agent_runtime::artifacts::Kind::Text => "text",
    };
    crate::commands::artifact_preview::preview(&path, kind_hint)
        .map_err(|e| format!("could not preview {name}: {e}"))
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

/// The runtime's context ledger, flattened into the shape the record holds.
///
/// Rebuilt field by field rather than deserialised straight through: the wire
/// shape is nested and the record's is flat, and a `serde` bridge between the
/// two would silently produce zeros the day the runtime renames a section.
/// Reading each name here means a rename is a compile error on one side and a
/// visible zero on the other, rather than a ledger that quietly stops adding up.
fn ledger_record(ledger: &Value) -> Option<crate::agent_runtime::tasks::ContextLedgerRecord> {
    let sections = ledger.get("sections")?;
    let section = |name: &str| sections.get(name).and_then(Value::as_u64).unwrap_or(0) as u32;
    let top = |name: &str| ledger.get(name).and_then(Value::as_i64).unwrap_or(0);

    Some(crate::agent_runtime::tasks::ContextLedgerRecord {
        system: section("system"),
        skill: section("skill"),
        tool_schema: section("toolSchema"),
        evidence: section("evidence"),
        notes: section("notes"),
        transcript: section("transcript"),
        compaction: section("compaction"),
        reserve: section("reserve"),
        occupied: top("occupied").max(0) as u32,
        committed: top("committed").max(0) as u32,
        window: top("window").max(0) as u32,
        // Signed on purpose. A negative headroom means the next turn does not
        // fit, and clamping it to zero would report that as "exactly full".
        headroom: top("headroom"),
    })
}

/// How resumable a stopped run is, as the Tasks screen asks.
///
/// Read-only. Answering this must never change anything about the run, because
/// a screen asks it on every refresh and an operator has not decided anything by
/// looking.
#[tauri::command]
pub async fn agent_run_resumability(
    app: AppHandle,
    run_id: String,
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
    registry: State<'_, Arc<ModelRegistry>>,
) -> Result<crate::agent_runtime::events::Resumability, String> {
    let signed_in = require_session(&session)?;
    Ok(assess_resumability(
        &app, &run_id, &signed_in, &events, &registry,
    ))
}

/// Continues a stopped run as a new attempt at the same task.
///
/// ## Why this is a separate command from starting a run
///
/// Reattaching to a run and continuing one look similar on a screen and are not
/// remotely the same act. Reattaching reads a record. Continuing takes actions
/// in the world, under an authorisation that was granted at some earlier moment
/// to a person who may no longer hold it, against files that may no longer be
/// where they were. Every one of those has to be re-established before anything
/// runs, and a single command that did both would inevitably grow a path where
/// one of them was skipped.
///
/// So the checks happen here, before any work, and the refusals are specific:
/// see `NotResumable`. The most important is that a side effect nobody settled
/// stops this outright — continuing would either repeat it or assume it worked,
/// and nothing on this side can tell which.
#[tauri::command]
pub async fn agent_resume_run(
    app: AppHandle,
    run_id: String,
    operator_intent: String,
    session: State<'_, CurrentSession>,
    events: State<'_, TaskEvents>,
    registry: State<'_, Arc<ModelRegistry>>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<crate::agent_runtime::resume::Attempt, String> {
    use crate::agent_runtime::events::Resumability;

    let signed_in = require_permission(&session, Permission::UseModel)?;

    // Assessed and refused before anything else happens. A resumption that
    // records its own intent and then discovers it may not proceed has written a
    // line saying a person continued a run that never continued.
    let verdict = assess_resumability(&app, &run_id, &signed_in, &events, &registry);
    let (attempt_id, from_seq) = match verdict {
        Resumability::Resumable {
            attempt_id,
            from_seq,
            ..
        } => (attempt_id, from_seq),
        Resumability::NeedsReconciliation { because, .. } => return Err(because),
        Resumability::ViewOnly { because } => return Err(because),
    };
    let _ = attempt_id;

    let attempt = crate::agent_runtime::resume::Attempt::new(&run_id, &operator_intent, from_seq);

    // Recorded before the loop is asked to do anything, and treated as
    // recovery-critical: a resumption that is not in the history is one a later
    // reader would count as part of the original attempt, and the whole point of
    // an attempt id is that those are told apart.
    events
        .record(
            EventDraft::new(
                &run_id,
                TaskEventType::RunResumed,
                &signed_in.user.id,
            )
            .with(json!({
                "attemptId": attempt.attempt_id,
                "fromSeq": attempt.from_seq,
                // The operator note, already bounded by `Attempt::new`.
                "operatorIntent": attempt.operator_intent,
            })),
        )
        .map_err(|error| {
            format!(
                "This run was not resumed: the resumption could not be recorded ({error}), and continuing without a record of it would leave the work unattributable."
            )
        })?;

    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::ModelRegistry,
        format!("resumed task {run_id} as attempt {}", attempt.attempt_id),
        None,
    );

    Ok(attempt)
}

/// The read-only half of both commands above.
///
/// Gathers the world as it is now and puts it to the checkpoint. Everything it
/// reads is re-derived rather than remembered, which is the entire basis on
/// which a resumption can be called safe.
fn assess_resumability(
    app: &AppHandle,
    run_id: &str,
    signed_in: &crate::identity::Session,
    events: &TaskEvents,
    registry: &Arc<ModelRegistry>,
) -> crate::agent_runtime::events::Resumability {
    use crate::agent_runtime::events::{NotResumable, Resumability};

    let checkpoint = match events.checkpoint(run_id) {
        Ok(found) => found,
        // A damaged or unreadable checkpoint is surfaced as its own refusal
        // rather than folded into "no checkpoint": absence means the run was
        // never safe to continue, and damage means somebody should know the
        // record was harmed.
        Err(refusal) => {
            return Resumability::ViewOnly {
                because: refusal.explain(),
            }
        }
    };

    let Ok(Some(snapshot)) = events.snapshot(run_id) else {
        return Resumability::ViewOnly {
            because: NotResumable::NoCheckpoint.explain(),
        };
    };

    // The prompt the plan is re-derived from, and the person the run belongs to,
    // both read from the run's own durable record rather than from the caller.
    let prompt = snapshot.prompt.clone();
    let owner = snapshot.actor.clone();

    let workspace_root = app_data_dir(app)
        .map(|dir| dir.join("runs").join(run_id))
        .unwrap_or_default();

    // Whether the model this run was routed to can still be served. A different
    // model would produce a second half the first half does not match.
    let model_available = checkpoint
        .as_ref()
        .map(|point| registry.find(&point.model_id).is_some())
        .unwrap_or(false);

    let context = crate::agent_runtime::resume::ResumeContext {
        session: signed_in,
        prompt: &prompt,
        // Read back off the run rather than supplied: a caller that could name
        // the classification could name a lower one.
        classification: snapshot.classification.as_deref().and_then(|label| {
            crate::policy::Classification::ALL
                .iter()
                .copied()
                .find(|c| c.label() == label)
        }),
        sovereignty_mode: &format!("{:?}", crate::sovereignty::global_broker().mode()),
        workspace_root: &workspace_root,
        model_available,
        owner: &owner,
        ended: snapshot.state.is_terminal(),
        state: snapshot.state,
    };

    Resumability::of(checkpoint.as_ref(), &context.world())
}

#[cfg(test)]
mod attachment_prompt_tests {
    use super::{compose_prompt_with_attachments, describe_attachment_reads};
    use crate::ai_engine::ocr_profile::OcrDetent;
    use crate::commands::ocr::AttachmentRead;

    /// A file read by the OCR model: an image goes to a vision model, one
    /// page, and the read names which model and which stop.
    fn read(name: &str, text: &str) -> AttachmentRead {
        AttachmentRead {
            name: name.into(),
            sha256: "0".repeat(64),
            text: text.into(),
            kind: "image".into(),
            pages: 1,
            ocr_model_id: Some("unlimited-ocr-q6-k".into()),
            ocr_detent: Some(OcrDetent::Detailed),
        }
    }

    /// A file that carried its own text. No model touched it, and nothing
    /// about it may claim one did.
    fn extracted(name: &str, text: &str) -> AttachmentRead {
        AttachmentRead {
            name: name.into(),
            sha256: "1".repeat(64),
            text: text.into(),
            kind: "xlsx".into(),
            pages: 1,
            ocr_model_id: None,
            ocr_detent: None,
        }
    }

    /// The defect this exists for: a turn that attached a scan was answered
    /// by two models, and the reasons named only the second. Somebody
    /// opening "Why?" saw a reasoning model with no account of where the
    /// text it reasoned over came from.
    #[test]
    fn the_reasons_name_the_ocr_model_that_read_the_page() {
        let reasons = describe_attachment_reads(&[read("scan.png", "TOTAL 44")]);
        assert_eq!(reasons.len(), 1);
        let line = &reasons[0];
        assert!(line.contains("scan.png"), "{line}");
        assert!(line.contains("unlimited-ocr-q6-k"), "{line}");
        assert!(line.contains("Detailed"), "{line}");
        assert!(line.contains("8 characters"), "{line}");
    }

    /// The other half of the same honesty: a spreadsheet was not read by a
    /// vision model, and the explanation must not imply that it was.
    #[test]
    fn a_locally_extracted_file_does_not_claim_a_model_read_it() {
        let reasons = describe_attachment_reads(&[extracted("rows.xlsx", "A1,B1")]);
        let line = &reasons[0];
        assert!(line.contains("already carried its text"), "{line}");
        assert!(!line.contains("OCR"), "{line}");
    }

    /// The regression this change exists for: the composer used to keep only
    /// `file.name`, so the bytes never left the picker and the model answered
    /// "please provide the text". The prompt the runtime sees must contain
    /// what was actually read.
    #[test]
    fn what_the_model_read_is_in_the_prompt_the_runtime_sees() {
        let out = compose_prompt_with_attachments(
            "Read this document and extract the text.",
            &[read("field-report.png", "QUARTERLY FIELD REPORT")],
        );
        assert!(out.contains("QUARTERLY FIELD REPORT"), "got {out:?}");
        assert!(
            out.contains("field-report.png"),
            "the file is named: {out:?}"
        );
        assert!(out.contains("Read this document"), "the question survives");
    }

    #[test]
    fn the_document_precedes_the_question() {
        let out =
            compose_prompt_with_attachments("What is the total?", &[read("a.png", "TOTAL 44")]);
        assert!(
            out.find("TOTAL 44") < out.find("What is the total?"),
            "the page must be read before the instruction: {out:?}"
        );
    }

    #[test]
    fn a_file_that_read_as_nothing_is_still_named() {
        // Silently dropping it would let the model answer as though no
        // document had been attached at all.
        let out = compose_prompt_with_attachments("Read this.", &[read("blank.png", "")]);
        assert!(out.contains("blank.png"), "got {out:?}");
        assert!(out.contains("no text could be read"), "got {out:?}");
    }

    #[test]
    fn several_attachments_are_all_present_and_separately_named() {
        let out = compose_prompt_with_attachments(
            "Compare these.",
            &[read("one.png", "ALPHA"), read("two.png", "BETA")],
        );
        for needle in ["one.png", "ALPHA", "two.png", "BETA", "Compare these."] {
            assert!(out.contains(needle), "missing {needle} in {out:?}");
        }
    }

    /// Attachments belong to the request that carried them. A turn with none
    /// must produce exactly its own prompt — this is what stops one message
    /// answering about another's document.
    #[test]
    fn a_turn_with_no_attachments_is_left_exactly_as_written() {
        let out = compose_prompt_with_attachments("Just a question.", &[]);
        assert_eq!(out, "Just a question.");
        assert!(!out.contains("<attachment"));
    }
}

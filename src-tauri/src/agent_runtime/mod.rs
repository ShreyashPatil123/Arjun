//! Supervising the agent runtime, and answering it.
//!
//! The agent loop lives in a Node child process (`agent-runtime/`), built from
//! OpenClaw's `agent-core`. This module owns that process and serves the two
//! questions it asks: *may this tool call happen*, and *please perform it*.
//!
//! ## Why the loop is over there and the decisions are here
//!
//! The loop needs streaming, compaction, steering and abort recovery, which
//! OpenClaw already has and this project would otherwise have to grow. The
//! decisions need the user's permissions, the workspace boundary, the
//! sovereignty invariant and the audit record, which live here and should not
//! be copied into a second process to be re-derived.
//!
//! So the split is not by convenience but by authority: **the runtime may
//! request; only this side decides.** Nothing in the child process can widen
//! what a run is permitted to do, because it does not hold the information that
//! would let it.
//!
//! ## The two questions
//!
//! - `tool.authorize` puts a call through [`ToolGateway`] and, on an allow,
//!   returns a single-use grant bound to that exact call (see [`grants`]).
//! - `tool.execute` redeems the grant, *re-derives the verdict independently*,
//!   and only then runs the tool through [`LocalToolRunner`] — the same runner
//!   the retired Rust executor used, unchanged.
//!
//! Checking twice is deliberate. The grant covers a compromised runtime; the
//! re-check covers a bug in the grant. Neither alone is worth the claim being
//! made.

pub mod approval;
pub mod artifacts;
pub mod grants;
pub mod planning;
pub mod protocol;
pub mod retrieval;
pub mod tasks;
pub mod workspace;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

use crate::identity::Session;
use crate::knowledge::KnowledgeIndex;
use crate::orchestrator::approvals::ApprovalQueue;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::gateway::{GatewayVerdict, TaskContext, ToolGateway};
use crate::orchestrator::plan::Continuation;
use crate::orchestrator::runner::LocalToolRunner;
use crate::orchestrator::executor::ToolRunner;
use crate::orchestrator::tools::{ToolCall, ToolName};
use crate::policy::ApprovalState;
use grants::GrantLedger;
use protocol::{code, Frame, Outgoing, WireError};

/// Event name the UI listens on. One channel for every run; the payload carries
/// the run id so a listener can filter.
pub const AGENT_EVENT: &str = "agent://event";

/// Everything a handler needs that is not on the wire.
///
/// Held rather than reached for so the handlers stay testable without a Tauri
/// app: the tests at the bottom build one of these directly.
pub struct RuntimeDeps {
    pub index: Arc<KnowledgeIndex>,
    pub session: Arc<std::sync::RwLock<Option<Session>>>,
    /// Where a run's files live, keyed by run id.
    ///
    /// Per run rather than per process: a shared scratch directory would let one
    /// task read what an unrelated task left behind, and the audit record would
    /// show a legitimate read of a permitted path. See [`workspace`].
    pub workspaces: Arc<Mutex<HashMap<String, workspace::Workspace>>>,
    /// Where proposed actions go to be seen by a person.
    pub approvals: Arc<ApprovalQueue>,
    /// Calculations a run has performed, in order, keyed by run id.
    ///
    /// Accumulated rather than recomputed because `create_xlsx` writes the
    /// run's whole working — PS 26117 asks for *"calculations with steps
    /// shown"*, and a workbook rebuilt from the model's recollection of what it
    /// computed would be exactly the thing the calculation engine exists to
    /// avoid.
    pub calculations: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>>,
    /// Passages a run has retrieved, in the order its citation markers refer to.
    ///
    /// Kept for the same reason as the calculations: the verifier resolves each
    /// `[En]` in the final answer against what was actually retrieved, and it
    /// cannot do that against passages nobody kept. See [`retrieval`].
    pub passages: retrieval::RunPassages,
    /// Files a run has produced, so each can be re-opened and checked when the
    /// run ends rather than taken on the model's word. See [`artifacts`].
    pub produced: artifacts::RunArtifacts,
    /// Every tool call a run has made, in order.
    ///
    /// Kept here rather than reconstructed from the event stream, which is
    /// best-effort: a dropped event costs a progress line, and it should not
    /// also cost a line in the permanent record of what the run did.
    pub calls: Arc<Mutex<HashMap<String, Vec<tasks::ToolCallRecord>>>>,
    /// The plan each run is being held to, keyed by run id.
    ///
    /// The budget inside is fixed by [`planning`] before the model is told
    /// anything, and nothing on the runtime's side of the wire can reach it.
    pub plans: Arc<Mutex<HashMap<String, crate::orchestrator::plan::PlanRun>>>,
    /// Where run events go.
    ///
    /// The loop publishes its own events over the wire; these are the ones this
    /// side decides — a step spent, a plan exhausted. They travel the same
    /// channel because an operator watching a run should see one sequence of
    /// what happened, not two interleaved by luck.
    ///
    /// Injected rather than reached for, so this module keeps no dependency on
    /// Tauri. That is the same reason [`AgentRuntime::spawn`] takes an emitter,
    /// and it is what lets the tests drive all of this with no app running.
    pub emit: Arc<dyn Fn(Value) + Send + Sync>,
}

impl RuntimeDeps {
    /// The directories a given run may touch. None until the run has one, which
    /// makes every path-taking tool refuse rather than reach somewhere shared.
    fn roots_for(&self, run_id: &str) -> Vec<PathBuf> {
        self.workspaces
            .lock()
            .ok()
            .and_then(|table| table.get(run_id).map(workspace::Workspace::roots))
            .unwrap_or_default()
    }

    /// The run's workspace, for naming a produced file relative to it.
    fn root_for(&self, run_id: &str) -> Option<PathBuf> {
        self.roots_for(run_id).into_iter().next()
    }

    /// Publishes one event about a run, in the shape the UI already listens for.
    fn publish(&self, run_id: &str, event: Value) {
        (self.emit)(json!({ "runId": run_id, "event": event }));
    }
}

impl RuntimeDeps {
    fn session(&self) -> Result<Session, WireError> {
        self.session
            .read()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| {
                WireError::new(
                    code::REFUSED,
                    "No one is signed in, so no tool call can be attributed to a person. Sign in and start the task again.",
                )
            })
    }

    /// Whether confidential work is permitted right now.
    ///
    /// Read from the broker at the moment of the call rather than captured when
    /// the run started: switching the workbench into provisioning mode mid-run
    /// must stop the next tool call, not just the next run.
    fn confidential_work_permitted(&self) -> bool {
        crate::sovereignty::global_broker()
            .guard_confidential("agent tool call")
            .is_ok()
    }
}

/// A live agent runtime process.
pub struct AgentRuntime {
    outbound: mpsc::UnboundedSender<String>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>>,
    next_id: AtomicU64,
    child: Mutex<Option<Child>>,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("the agent runtime bundle is missing at {0}. Run `npm run build` in agent-runtime/.")]
    BundleMissing(PathBuf),
    #[error("the agent runtime could not be started: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("the agent runtime went away before answering")]
    Closed,
    #[error("{}", .0.message)]
    Remote(WireError),
}

/// Where the bundle lives when the caller has nothing better.
///
/// `ARJUN_AGENT_RUNTIME` wins, which is what the tests use. Otherwise the
/// development layout, so a checkout runs with no configuration. A packaged
/// build resolves its own resource directory and passes that to [`spawn`] --
/// this module does not depend on Tauri, which is what lets the tests drive a
/// real child process with no application running.
pub fn default_bundle_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("ARJUN_AGENT_RUNTIME") {
        return PathBuf::from(explicit);
    }
    // `CARGO_MANIFEST_DIR` is src-tauri/; the runtime is its sibling.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("agent-runtime/dist/arjun-agent-runtime.mjs"))
        .unwrap_or_default()
}

impl AgentRuntime {
    /// Starts the child and the three tasks that keep it fed.
    ///
    /// `emit` receives every `run.event` the runtime publishes. Injected rather
    /// than taking an `AppHandle` so this module does not depend on Tauri, which
    /// is what lets the tests drive a real child process with no app running.
    pub fn spawn(
        deps: Arc<RuntimeDeps>,
        emit: Arc<dyn Fn(Value) + Send + Sync>,
        bundle: PathBuf,
    ) -> Result<Arc<Self>, RuntimeError> {
        if !bundle.exists() {
            return Err(RuntimeError::BundleMissing(bundle));
        }

        let mut child = Command::new("node")
            .arg(&bundle)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // The runtime reaches only the loopback inference endpoint the
            // router chose. It has no use for inherited proxy configuration, and
            // an inherited `HTTP_PROXY` would be a way out of the machine that
            // nothing in this codebase put there.
            .env_remove("HTTP_PROXY")
            .env_remove("HTTPS_PROXY")
            .env_remove("http_proxy")
            .env_remove("https_proxy")
            .env_remove("ALL_PROXY")
            .env_remove("NPM_CONFIG_PROXY")
            .kill_on_drop(true)
            .spawn()
            .map_err(RuntimeError::Spawn)?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let (outbound, mut outbox) = mpsc::unbounded_channel::<String>();
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let runtime = Arc::new(Self {
            outbound: outbound.clone(),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            child: Mutex::new(Some(child)),
        });

        // Writer: one task owns stdin, so writes cannot interleave mid-line.
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = outbox.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        // Diagnostics. The runtime writes every log line here precisely so that
        // stdout stays parseable; forwarding them keeps that decision from
        // costing us the ability to debug.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::info!("[agent-runtime] {line}");
            }
        });

        // Reader: the only place inbound frames are interpreted.
        let reader_deps = deps.clone();
        let reader_outbound = outbound.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                let frame = match Frame::parse(&line) {
                    Ok(frame) => frame,
                    Err(error) => {
                        // Fatal for the channel: past a frame we cannot read,
                        // neither end knows what the other said.
                        log::error!("[agent-runtime] unparseable frame, closing channel: {error}");
                        break;
                    }
                };
                dispatch(frame, &reader_deps, &reader_outbound, &pending, &emit).await;
            }
            // Stream ended. Fail every caller still waiting rather than leaving
            // them to hang on a process that is gone.
            let waiting: Vec<_> = pending.lock().map(|mut p| p.drain().collect()).unwrap_or_default();
            for (_, sender) in waiting {
                let _ = sender.send(Err(WireError::new(
                    code::INTERNAL,
                    "the agent runtime stopped",
                )));
            }
        });

        Ok(runtime)
    }

    /// Sends a request and waits for its reply.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, RuntimeError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .map_err(|_| RuntimeError::Closed)?
            .insert(id.clone(), tx);

        let frame = Outgoing::Request {
            id: id.clone(),
            method: method.to_string(),
            params,
        };
        if self.outbound.send(frame.encode()).is_err() {
            self.pending.lock().ok().and_then(|mut p| p.remove(&id));
            return Err(RuntimeError::Closed);
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => Err(RuntimeError::Remote(error)),
            Err(_) => Err(RuntimeError::Closed),
        }
    }

    /// Stops the child. Idempotent, so shutdown paths can call it freely.
    pub async fn shutdown(&self) {
        let child = self.child.lock().ok().and_then(|mut slot| slot.take());
        if let Some(mut child) = child {
            let _ = child.kill().await;
        }
    }
}

/// Routes one inbound frame.
async fn dispatch(
    frame: Frame,
    deps: &Arc<RuntimeDeps>,
    outbound: &mpsc::UnboundedSender<String>,
    pending: &Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value, WireError>>>>>,
    emit: &Arc<dyn Fn(Value) + Send + Sync>,
) {
    match frame {
        Frame::Request { id, method, params } => {
            let reply = match handle(&method, params, deps).await {
                Ok(result) => Outgoing::Result { id, result },
                Err(error) => Outgoing::Error { id, error },
            };
            let _ = outbound.send(reply.encode());
        }
        Frame::Result { id, result } => {
            if let Some(sender) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                let _ = sender.send(Ok(result));
            }
        }
        Frame::Error { id, error } => {
            if let Some(sender) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                let _ = sender.send(Err(error));
            }
        }
        Frame::Notification { method, params } => {
            if method == "run.event" {
                emit(params);
            } else {
                log::debug!("[agent-runtime] unhandled notification {method}");
            }
        }
    }
}

/// The methods this side serves.
async fn handle(
    method: &str,
    params: Value,
    deps: &Arc<RuntimeDeps>,
) -> Result<Value, WireError> {
    match method {
        "tool.authorize" => authorize(params, deps).await,
        "tool.execute" => execute(params, deps),
        other => Err(WireError::new(
            code::UNKNOWN_METHOD,
            format!("no handler for {other}"),
        )),
    }
}

/// Fields both tool methods need off the wire.
pub struct CallParams {
    pub run_id: String,
    pub tool_call_id: String,
    pub tool: String,
    pub args: Value,
    /// The model driving this run, stamped onto anything it produces so a
    /// reader of the document knows what wrote it. Absent when the runtime did
    /// not say, which is recorded as "unrecorded" rather than guessed.
    pub model: Option<String>,
}

fn read_call(params: &Value) -> Result<CallParams, WireError> {
    let field = |name: &str| -> Result<String, WireError> {
        params
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                WireError::new(code::BAD_PARAMS, format!("missing string field {name:?}"))
            })
    };
    Ok(CallParams {
        run_id: field("runId")?,
        tool_call_id: field("toolCallId")?,
        tool: field("tool")?,
        args: params.get("args").cloned().unwrap_or(Value::Null),
        model: params
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// Grants outstanding across every run this process is serving.
///
/// Process-wide because a grant is meaningless outside the ledger that issued
/// it, and one ledger keeps "issued here, redeemed here" true no matter how many
/// runtimes are running.
fn ledger() -> &'static Mutex<GrantLedger> {
    static LEDGER: std::sync::OnceLock<Mutex<GrantLedger>> = std::sync::OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(GrantLedger::new()))
}

/// Decides one call, issuing a grant if the answer is yes.
///
/// When the gateway says a person must look first, this raises the request and
/// **waits** rather than refusing. From the loop's side that is simply a slow
/// authorisation; from the operator's, it appears on the approvals screen and
/// the run continues when they decide. Neither side has to model the other's
/// idea of waiting.
async fn authorize(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let call = read_call(&params)?;

    // The plan is consulted before the gateway. A task that is out of time
    // should not be asking about permissions, and "you have run out of steps"
    // is a more useful thing to tell a model than "that path is fine, but
    // nothing further will happen".
    if let Some(reason) = plan_refusal(&call, deps) {
        return Ok(json!({ "outcome": "refuse", "reason": reason }));
    }

    let verdict = decide(&call, deps, ApprovalState::NotRequested)?;

    let (tool, resolved_path) = match verdict {
        GatewayVerdict::Allow {
            tool,
            resolved_path,
        } => (tool, resolved_path),
        GatewayVerdict::Refuse { reason } => {
            return Ok(json!({ "outcome": "refuse", "reason": reason }))
        }
        GatewayVerdict::NeedsApproval {
            tool,
            summary,
            resolved_path,
        } => {
            let session = deps.session()?;
            let target = resolved_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| call.tool.clone());
            let outcome = approval::await_decision(
                &deps.approvals,
                &session,
                &call.run_id,
                tool,
                summary,
                target,
                render_arguments(&call.args),
            )
            .await;

            match outcome {
                approval::ApprovalOutcome::Approved { .. } => (tool, resolved_path),
                other => {
                    return Ok(json!({ "outcome": "refuse", "reason": other.refusal() }))
                }
            }
        }
    };

    let grant = ledger()
        .lock()
        .map_err(|_| WireError::new(code::INTERNAL, "grant ledger is poisoned"))?
        .issue(&call.run_id, &call.tool_call_id, &call.tool, &call.args);

    Ok(json!({
        "outcome": "allow",
        "tool": tool.as_str(),
        "grant": grant,
        "resolvedPath": resolved_path,
    }))
}

/// Puts a call to the run's plan, and says why not when the answer is no.
///
/// Two shapes of refusal, deliberately different:
///
/// - **A tool outside the plan** is refused without stopping the run. The model
///   reads the refusal and can do the rest of the work, or say plainly what it
///   could not do. Stopping there would turn one wrong guess by [`planning`]
///   into a lost run, which is far too high a price for a keyword miss.
/// - **Out of steps, out of time, or going in circles** stops the run, and
///   every later call is refused with the same sentence. Those are the
///   conditions PS Part C asks to be stopped at, and a limit a model could keep
///   retrying against would not be a limit.
///
/// A run with no plan is allowed through. That is not a hole: the only caller
/// that starts a run registers a plan first, and refusing every tool call for a
/// run this table has never heard of would break the runtime's own health check
/// rather than enforce anything.
fn plan_refusal(call: &CallParams, deps: &Arc<RuntimeDeps>) -> Option<String> {
    let stopped = {
        // A poisoned table is a panic that happened while the budget was being
        // read, and carrying on would mean running with no budget at all. That
        // is the one case here that fails closed: an unbounded run is worse
        // than a stopped one.
        let Ok(mut plans) = deps.plans.lock() else {
            return Some(
                "This task's plan cannot be read, so there is no budget to hold the work to and \
                 nothing further will be run. Start the task again."
                    .to_string(),
            );
        };
        // No plan at all is different, and is allowed: the runtime's own health
        // probe belongs to no run, and refusing it would break the check rather
        // than enforce anything.
        let plan = plans.get_mut(&call.run_id)?;

        // Checked before `may_call`, which halts the whole plan on an
        // unpermitted tool. Here that is one refused call and no more.
        let permitted = ToolName::from_str(&call.tool)
            .map(|tool| plan.budget.permits(tool))
            .unwrap_or(false);
        if !permitted {
            let allowed: Vec<&str> = plan
                .budget
                .permitted_tools
                .iter()
                .map(|tool| tool.as_str())
                .collect();
            return Some(format!(
                "{} is not one of the tools this task was planned to use. The plan allows: {}. \
                 Do what you can with those, and say plainly what you could not do.",
                call.tool,
                allowed.join(", ")
            ));
        }

        match plan.may_call(&ToolCall::new(call.tool.clone(), call.args.clone())) {
            Continuation::Proceed => return None,
            Continuation::Stop(reason) => reason,
        }
    };

    // Published so the trace says why the run went quiet. Emitted outside the
    // lock: the handler is arbitrary code, and holding the plan table across it
    // would let a slow listener block every other run's authorisation.
    deps.publish(
        &call.run_id,
        json!({
            "type": "plan_stopped",
            "reason": stopped.explain(),
            "tool": call.tool,
        }),
    );
    Some(stopped.explain())
}

/// Renders arguments the way an approver will read them.
///
/// Values are truncated: a write's `content` can be a whole document, and an
/// approval screen that makes somebody scroll past 30 KB to find the path is one
/// where they stop reading and start clicking yes.
fn render_arguments(args: &Value) -> Vec<String> {
    const MAX_VALUE_CHARS: usize = 200;
    let Some(object) = args.as_object() else {
        return Vec::new();
    };
    object
        .iter()
        .map(|(key, value)| {
            let rendered = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            if rendered.chars().count() > MAX_VALUE_CHARS {
                let head: String = rendered.chars().take(MAX_VALUE_CHARS).collect();
                format!("{key} = {head}… ({} characters)", rendered.chars().count())
            } else {
                format!("{key} = {rendered}")
            }
        })
        .collect()
}

/// Puts a call through the gateway. Shared by both methods so the two answers
/// cannot diverge.
fn decide(
    call: &CallParams,
    deps: &Arc<RuntimeDeps>,
    approval: ApprovalState,
) -> Result<GatewayVerdict, WireError> {
    let session = deps.session()?;
    let roots = deps.roots_for(&call.run_id);
    let tool_call = anchor_path(ToolCall::new(call.tool.clone(), call.args.clone()), &roots);
    let context = TaskContext {
        session: &session,
        workspace_roots: &roots,
        confidential_work_permitted: deps.confidential_work_permitted(),
        // What the caller already holds. `authorize` raises the request and
        // waits; `execute` passes `Granted` because it only runs after that
        // wait returned yes, and re-deciding as `NotRequested` would ask the
        // same person the same question a second time.
        approval,
    };
    Ok(ToolGateway::decide(&tool_call, &context))
}


/// Resolves a relative `path` argument against the run's workspace.
///
/// The gateway's containment check compares a path against the permitted roots,
/// so a bare `"note.txt"` fails it — it is not under any root, it is under
/// nothing. That would make every relative path a refusal, which matters
/// because relative is exactly what the model is told to use: an absolute path
/// is a temp directory with a UUID in it, and a 7B model asked to reproduce one
/// verbatim across a dozen calls will not.
///
/// So the anchoring happens here, before the gateway sees the call, and the
/// gateway's check is unchanged. Traversal is still refused: `../../etc/passwd`
/// joined onto the root still normalises to somewhere outside it, and
/// `resolve_within` still says no. This makes relative paths *expressible*, not
/// permitted — the containment decision stays exactly where it was.
fn anchor_path(call: ToolCall, roots: &[PathBuf]) -> ToolCall {
    let Some(root) = roots.first() else {
        // No workspace, so nothing to anchor against. The gateway refuses every
        // path-taking tool in that state, which is the correct outcome.
        return call;
    };
    // Copied out before `arguments` is moved: `text` borrows the call, and the
    // rewrite below takes ownership of what it borrows from.
    let Some(raw) = call.text("path").map(str::to_string) else {
        return call;
    };
    // Only a *purely* relative path is anchored. Anything carrying a root is
    // passed through to be judged as written, because `Path::join` replaces
    // rather than appends when the argument has one: on Windows
    // `C:\runs\<id>`.join("/etc/passwd") is `C:/etc/passwd` — outside the
    // workspace, silently. The gateway refuses that, but anchoring should not
    // be manufacturing paths that depend on a later check to be safe.
    let candidate = Path::new(&raw);
    if candidate.is_absolute() || candidate.has_root() {
        return call;
    }

    let mut arguments = call.arguments;
    if let Some(object) = arguments.as_object_mut() {
        object.insert(
            "path".to_string(),
            Value::String(root.join(&raw).display().to_string()),
        );
    }
    ToolCall {
        tool: call.tool,
        arguments,
    }
}

/// Redeems the grant, re-derives the verdict, then runs the tool.
fn execute(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let call = read_call(&params)?;
    let grant = params
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| WireError::new(code::REFUSED, "no authorisation grant was presented"))?;

    ledger()
        .lock()
        .map_err(|_| WireError::new(code::INTERNAL, "grant ledger is poisoned"))?
        .redeem(
            grant,
            &call.run_id,
            &call.tool_call_id,
            &call.tool,
            &call.args,
        )
        .map_err(|error| WireError::new(code::REFUSED, error.to_string()))?;

    // Independent of the grant. A grant proves the gateway said yes once; this
    // asks it again, because the state it decides against — the signed-in user,
    // the sovereignty mode — can have changed since.
    //
    // `Granted` because the grant is itself the evidence a person already said
    // yes: re-deciding as `NotRequested` would put the same request in front of
    // the same approver a second time, for an action they have just approved.
    let verdict = decide(&call, deps, ApprovalState::Granted)?;
    let (tool, resolved_path) = match verdict {
        GatewayVerdict::Allow {
            tool,
            resolved_path,
        } => (tool, resolved_path),
        GatewayVerdict::NeedsApproval { summary, .. } => {
            return Err(WireError::new(code::REFUSED, summary))
        }
        GatewayVerdict::Refuse { reason } => return Err(WireError::new(code::REFUSED, reason)),
    };

    let session = deps.session()?;
    // Anchored the same way the gateway saw it, so the tool acts on exactly the
    // path that was judged rather than on the raw argument.
    let tool_call = anchor_path(
        ToolCall::new(call.tool.clone(), call.args.clone()),
        &deps.roots_for(&call.run_id),
    );

    // Four tools are handled here rather than in `LocalToolRunner` because each
    // needs the run's accumulated state — its calculations, its evidence, the
    // files it has produced — and the runner is built fresh per call, so it
    // cannot hold any of it.
    let outcome = match tool {
        ToolName::CreateDocx => {
            artifacts::create_docx(&call, resolved_path.as_deref(), &session, &tool_call)
        }
        ToolName::CreateXlsx => {
            artifacts::create_xlsx(resolved_path.as_deref(), &deps.calculations, &call.run_id)
        }
        // Recorded as the run's evidence on the way past, and numbered once
        // across the whole run so a citation means one passage. See
        // [`retrieval`].
        ToolName::SearchDocuments => LocalToolRunner::new(deps.index.as_ref(), &session)
            .search_hits(&tool_call)
            .map(|(query, hits)| retrieval::record(&deps.passages, &call.run_id, &query, &hits)),
        ToolName::ValidateArtifact => {
            validate(deps, &call.run_id, resolved_path.as_deref(), &session, &tool_call)
        }
        _ => {
            let runner = LocalToolRunner::new(deps.index.as_ref(), &session);
            let result = runner.run(tool, &tool_call, resolved_path.as_deref());
            // A successful calculation is kept, so the workbook can show the
            // working rather than the model's memory of it.
            if tool == ToolName::RunCalculation && result.is_ok() {
                if let Ok(record) =
                    crate::orchestrator::calculation::evaluate(tool_call.text("expression").unwrap_or_default())
                {
                    if let Ok(mut table) = deps.calculations.lock() {
                        table.entry(call.run_id.clone()).or_default().push(record);
                    }
                }
            }
            result
        }
    };

    if outcome.is_ok() {
        remember_if_produced(deps, &call.run_id, tool, resolved_path.as_deref(), &tool_call);
    }
    record_call(deps, &call.run_id, tool.as_str(), &outcome);

    // Counted whatever the tool returned. A failed call cost the same wall
    // clock and the same context window as a successful one, and a budget that
    // only counts successes is one a model going in circles never reaches.
    record_step(deps, &call.run_id, tool);

    match outcome {
        Ok(text) => Ok(json!({ "text": text, "details": { "tool": tool.as_str() } })),
        // A tool that fails says why, in words the model can act on. Returned as
        // an error frame so the runtime turns it into an error tool result
        // rather than passing it off as an answer.
        Err(reason) => Err(WireError::new(code::TOOL_FAILED, reason)),
    }
}

/// Re-opens a file this run produced and says what is actually in it.
///
/// The runner's own check asks whether the path exists and is not empty, which
/// is all it *can* ask: it is built fresh per call and does not know the file
/// was rendered from the `approval_note` template. This does, because the run
/// remembered it — so `validate_artifact` on a document opens the package and
/// checks the sections are really there, which is what PS step 30 asks for and
/// what the tool's own description promises the model.
fn validate(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    resolved_path: Option<&Path>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let known = resolved_path.and_then(|path| {
        artifacts::for_run(&deps.produced, run_id)
            .into_iter()
            .find(|produced| Path::new(&produced.path) == path)
    });

    let Some(produced) = known else {
        // Not something this run produced, so there is no template to check it
        // against and no claim to make beyond what is on disk. The runner's
        // existence-and-size check is then the honest answer.
        let runner = LocalToolRunner::new(deps.index.as_ref(), session);
        return runner.run(ToolName::ValidateArtifact, tool_call, resolved_path);
    };

    let report = artifacts::check(&produced);
    if report.sound {
        Ok(format!("{}: {}", report.name, report.detail))
    } else {
        Err(format!(
            "{} did not pass its check: {}. Correct it and produce it again.",
            report.name,
            report.problems.join("; ")
        ))
    }
}

/// Records a file the call has just produced, so it can be re-opened later.
fn remember_if_produced(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    tool: ToolName,
    resolved_path: Option<&Path>,
    tool_call: &ToolCall,
) {
    let kind = match tool {
        ToolName::CreateDocx => artifacts::Kind::Document,
        ToolName::CreateXlsx => artifacts::Kind::Workbook,
        ToolName::WriteScopedFile => artifacts::Kind::Text,
        _ => return,
    };
    let Some(path) = resolved_path else { return };

    let template = if tool == ToolName::CreateDocx {
        tool_call.text("template").map(str::to_string)
    } else {
        None
    };
    let root = deps.root_for(run_id);
    artifacts::remember(
        &deps.produced,
        run_id,
        artifacts::produced_from(path, root.as_deref(), kind, template),
    );
}

/// Keeps what a tool call did, for the run's record.
///
/// A refusal is recorded as its own outcome rather than as a failure. The two
/// look the same to a naive reader and mean opposite things: a failure is the
/// tool going wrong, a refusal is the policy working, and a Tasks screen that
/// paints every refusal red teaches people to skip the ones that matter.
fn record_call(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    tool: &str,
    outcome: &Result<String, String>,
) {
    let record = match outcome {
        Ok(text) => tasks::ToolCallRecord::new(tool, tasks::CallOutcome::Succeeded, text),
        Err(reason) => {
            // The gateway and the plan both refuse in this wording; a tool that
            // simply went wrong does not. Read from the reason rather than
            // threaded through, because every refusal path already produces a
            // sentence and none of them produces a code.
            let refused = reason.contains("not permitted")
                || reason.contains("planned to use")
                || reason.contains("permitted steps")
                || reason.contains("was not approved")
                || reason.contains("going in circles");
            let kind = if refused {
                tasks::CallOutcome::Refused
            } else {
                tasks::CallOutcome::Failed
            };
            tasks::ToolCallRecord::new(tool, kind, reason)
        }
    };

    if let Ok(mut table) = deps.calls.lock() {
        table.entry(run_id.to_string()).or_default().push(record);
    }
}

/// Marks a step spent and publishes how far through the plan the run is.
fn record_step(deps: &Arc<RuntimeDeps>, run_id: &str, tool: ToolName) {
    // Built inside the lock, published outside it: the handler is arbitrary
    // code, and holding the plan table across a slow listener would stall every
    // other run's authorisation.
    let progress = {
        let Ok(mut plans) = deps.plans.lock() else {
            return;
        };
        let Some(plan) = plans.get_mut(run_id) else {
            return;
        };
        // `record_call`, not `record_step`: one planned step can take several
        // tool calls, and ticking a step off per call would report a document
        // as produced and checked after four searches.
        plan.record_call();
        json!({
            "type": "plan_step",
            "tool": tool.as_str(),
            "stepsTaken": plan.steps_taken(),
            "maxSteps": plan.budget.max_steps,
            "stepsPlanned": plan.steps.len(),
        })
    };
    deps.publish(run_id, progress);
}

/// Names the tool catalogue exposes. Used by the absence test in `tests/`.
pub fn catalogue() -> Vec<&'static str> {
    ToolName::ALL.iter().map(|tool| tool.as_str()).collect()
}


#[cfg(test)]
mod tests;

//! What one durable task event is, and what may be written into it.
//!
//! A run used to leave behind exactly one thing: a JSON file written after it
//! ended. That is enough to review a finished task and nothing at all for the
//! two cases that matter most — a run still going when the window remounts, and
//! a run that was going when the process died. Both look identical from the
//! outside (no file), and both used to be unrecoverable.
//!
//! So the record is now written *as the run happens*, one ordered event at a
//! time. This module is the vocabulary: the event types, the envelope every
//! event carries, and — the part with teeth — the redaction that decides what
//! is allowed into a payload in the first place.
//!
//! ## Why redaction is here and not at the call sites
//!
//! PS step 14 says confidential contents must not be copied into a record more
//! people can read than could read the original. A rule enforced at the call
//! sites is a rule that holds until somebody adds a call site. So every payload
//! goes through [`redact`] on its way into an [`EventDraft`], and the fields
//! that carry document text come out as a hash and a length. The event says
//! *that* a passage was retrieved and which one it was; it does not say what
//! the passage said.
//!
//! The same rule covers the model's own words. Nothing here carries a message,
//! a reasoning trace or a partial completion — the operational trace is plan,
//! decision, tool, evidence, approval, artifact and verifier status, and that
//! is all an event may hold.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// The shape events are written in.
///
/// Bumped when a payload's meaning changes, never when a field is added. Stored
/// on every row so a reader built for an older shape can say "this event is
/// newer than I am" instead of silently misreading it.
///
/// - **1** — the first durable history: run start, plan, tools, endings.
/// - **2** — explicit run states. Adds the lifecycle events the state machine
///   in [`super::machine`] needs, and the tool-effect events that make an
///   interrupted side effect recoverable.
pub const SCHEMA_VERSION: u32 = 2;

/// What happened. Coarse on purpose — the detail belongs in the payload, and a
/// type per detail would make every reader a match arm longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskEventType {
    // -- Lifecycle --------------------------------------------------------
    /// The run was accepted. Always the first event of a run.
    RunCreated,
    /// The sensitivity of the material is known, so routing can be narrowed.
    RunClassified,
    /// A model was chosen, with the reasons that led there.
    RunRouted,
    /// The plan it will be held to, fixed before the model was told anything.
    PlanReady,
    /// The loop began.
    RunStarted,

    // -- Progress ---------------------------------------------------------
    /// A step spent.
    PlanStep,
    /// The plan will permit nothing further. The *ending* event that follows
    /// says which kind of ending it was; this only says the plan is done.
    PlanStopped,
    /// One model turn finished.
    TurnEnded,
    /// Older history was replaced by a summary so the run could continue.
    ///
    /// Kept durably rather than only shown live, because it is a caveat on
    /// everything the run says afterwards: those answers rest on a summary of
    /// the earlier turns rather than on the turns themselves. A recovered trace
    /// that quietly dropped this would overstate its own grounding.
    ContextCompacted,

    // -- Tools ------------------------------------------------------------
    /// The gateway allowed a call and issued a grant.
    ToolAuthorized,
    /// The gateway, the plan or a person said no before the tool ran.
    ToolRefused,
    ToolSucceeded,
    ToolFailed,
    /// A side-effecting call arrived twice under one idempotency key. The
    /// recorded outcome was returned and the tool was *not* run again.
    ToolReplayed,
    /// A side-effecting call is about to happen. Written **before** the effect,
    /// so that a process which dies mid-call leaves evidence it was trying.
    ToolEffectPending,
    /// A side effect whose outcome nobody knows: it was in flight when the
    /// process went away. Needs a person before its key can be reused.
    ToolEffectUnknown,
    /// A person said what actually happened to an unknown side effect.
    ToolEffectReconciled,
    ArtifactProduced,

    // -- Subagents --------------------------------------------------------
    /// A bounded worker began, with the manifest of what it was permitted.
    SubagentStarted,
    /// It ended. The status says which way, and a failure, a timeout and a
    /// cancellation are each named rather than folded into one ending.
    SubagentStopped,

    // -- Approval ---------------------------------------------------------
    ApprovalRequested,
    ApprovalDecided,

    // -- Verification -----------------------------------------------------
    /// The answer is being checked against the evidence the run actually holds.
    VerificationStarted,

    // -- Endings ----------------------------------------------------------
    /// The loop finished and an answer was produced.
    RunCompleted,
    /// The run ended badly. `failure` in the payload is the sentence shown.
    RunFailed,
    /// A person stopped it.
    RunCancelled,
    /// It reached the limit the plan set — steps, time, or going in circles.
    RunStoppedByBudget,
    /// It needed to do something it is not permitted to do.
    RunStoppedByPolicy,
    /// It was still going when the process went away, and the next start found
    /// it. Written by recovery, never by the run itself — a run cannot record
    /// its own sudden death.
    RunDegraded,

    // -- Read-only legacy -------------------------------------------------
    /// Schema 1's spelling for what is now [`Self::RunStoppedByBudget`].
    /// Readable so an older database still folds; never written.
    RunTimedOut,
    /// Schema 1's spelling for what is now [`Self::RunDegraded`].
    RunInterrupted,
}

impl TaskEventType {
    /// Stable database spelling. Written out rather than derived from the
    /// variant name so renaming a variant cannot rewrite history.
    pub const fn as_str(self) -> &'static str {
        match self {
            TaskEventType::RunCreated => "run_created",
            TaskEventType::RunClassified => "run_classified",
            TaskEventType::RunRouted => "run_routed",
            TaskEventType::PlanReady => "plan_ready",
            TaskEventType::RunStarted => "run_started",
            TaskEventType::PlanStep => "plan_step",
            TaskEventType::PlanStopped => "plan_stopped",
            TaskEventType::TurnEnded => "turn_ended",
            TaskEventType::ContextCompacted => "context_compacted",
            TaskEventType::ToolAuthorized => "tool_authorized",
            TaskEventType::ToolRefused => "tool_refused",
            TaskEventType::ToolSucceeded => "tool_succeeded",
            TaskEventType::ToolFailed => "tool_failed",
            TaskEventType::ToolReplayed => "tool_replayed",
            TaskEventType::ToolEffectPending => "tool_effect_pending",
            TaskEventType::ToolEffectUnknown => "tool_effect_unknown",
            TaskEventType::ToolEffectReconciled => "tool_effect_reconciled",
            TaskEventType::ArtifactProduced => "artifact_produced",
            TaskEventType::SubagentStarted => "subagent_started",
            TaskEventType::SubagentStopped => "subagent_stopped",
            TaskEventType::ApprovalRequested => "approval_requested",
            TaskEventType::ApprovalDecided => "approval_decided",
            TaskEventType::VerificationStarted => "verification_started",
            TaskEventType::RunCompleted => "run_completed",
            TaskEventType::RunFailed => "run_failed",
            TaskEventType::RunCancelled => "run_cancelled",
            TaskEventType::RunStoppedByBudget => "run_stopped_by_budget",
            TaskEventType::RunStoppedByPolicy => "run_stopped_by_policy",
            TaskEventType::RunDegraded => "run_degraded",
            TaskEventType::RunTimedOut => "run_timed_out",
            TaskEventType::RunInterrupted => "run_interrupted",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Some(match raw {
            "run_created" => TaskEventType::RunCreated,
            "run_classified" => TaskEventType::RunClassified,
            "run_routed" => TaskEventType::RunRouted,
            "plan_ready" => TaskEventType::PlanReady,
            "run_started" => TaskEventType::RunStarted,
            "plan_step" => TaskEventType::PlanStep,
            "plan_stopped" => TaskEventType::PlanStopped,
            "turn_ended" => TaskEventType::TurnEnded,
            "context_compacted" => TaskEventType::ContextCompacted,
            "tool_authorized" => TaskEventType::ToolAuthorized,
            "tool_refused" => TaskEventType::ToolRefused,
            "tool_succeeded" => TaskEventType::ToolSucceeded,
            "tool_failed" => TaskEventType::ToolFailed,
            "tool_replayed" => TaskEventType::ToolReplayed,
            "tool_effect_pending" => TaskEventType::ToolEffectPending,
            "tool_effect_unknown" => TaskEventType::ToolEffectUnknown,
            "tool_effect_reconciled" => TaskEventType::ToolEffectReconciled,
            "artifact_produced" => TaskEventType::ArtifactProduced,
            "subagent_started" => TaskEventType::SubagentStarted,
            "subagent_stopped" => TaskEventType::SubagentStopped,
            "approval_requested" => TaskEventType::ApprovalRequested,
            "approval_decided" => TaskEventType::ApprovalDecided,
            "verification_started" => TaskEventType::VerificationStarted,
            "run_completed" => TaskEventType::RunCompleted,
            "run_failed" => TaskEventType::RunFailed,
            "run_cancelled" => TaskEventType::RunCancelled,
            "run_stopped_by_budget" => TaskEventType::RunStoppedByBudget,
            "run_stopped_by_policy" => TaskEventType::RunStoppedByPolicy,
            "run_degraded" => TaskEventType::RunDegraded,
            "run_timed_out" => TaskEventType::RunTimedOut,
            "run_interrupted" => TaskEventType::RunInterrupted,
            _ => return None,
        })
    }

    /// Whether this event ends the run. A run whose history has no terminal
    /// event is a run that is still going — which is how recovery finds the
    /// ones the process took down with it.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskEventType::RunCompleted
                | TaskEventType::RunFailed
                | TaskEventType::RunCancelled
                | TaskEventType::RunStoppedByBudget
                | TaskEventType::RunStoppedByPolicy
                | TaskEventType::RunDegraded
                | TaskEventType::RunTimedOut
                | TaskEventType::RunInterrupted
        )
    }
}

/// One event as it is stored and as it is read back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEvent {
    pub run_id: String,
    /// Unique across every run. Supplied by the writer so a retry after an
    /// ambiguous failure presents the same id and is rejected as the duplicate
    /// it is, rather than landing twice. See [`EventDraft::idempotent`].
    pub event_id: String,
    /// Monotonic within the run, starting at 1, no gaps. Assigned by the store
    /// inside the same transaction as the insert.
    pub seq: i64,
    pub event_type: TaskEventType,
    /// RFC 3339, UTC.
    pub at: String,
    /// Who caused it — a user id, or `system` for the application itself.
    pub actor: String,
    pub schema_version: u32,
    /// Redacted. See [`redact`].
    pub payload: Value,
    /// SHA-256 over the canonical form of `payload`. Lets a reader tell a
    /// payload that was rewritten underneath it from one that was not.
    pub payload_hash: String,
}

impl TaskEvent {
    /// How a durable event travels to a window.
    ///
    /// Carries the sequence number, which is the whole point: a client that
    /// receives seq 14 having last applied seq 12 knows it missed one, and can
    /// ask for a snapshot rather than drawing a trace with a hole in it. The
    /// best-effort loop stream has no such number and cannot offer that.
    ///
    /// The payload is the redacted one, unchanged — there is no second
    /// redaction pass here, because there is nothing that could have been added
    /// since [`EventDraft::with`] ran.
    pub fn envelope(&self) -> Value {
        json!({
            "runId": self.run_id,
            "seq": self.seq,
            "eventId": self.event_id,
            "eventType": self.event_type,
            "at": self.at,
            "actor": self.actor,
            "schemaVersion": self.schema_version,
            "payload": self.payload,
        })
    }
}

/// An event that was on disk and could not be read.
///
/// Kept and reported rather than dropped silently: a history with a hole in it
/// is still usable, but only if the screen reading it knows the hole is there.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UnreadableEvent {
    pub seq: i64,
    pub event_id: String,
    /// What is wrong with it, in words.
    pub problem: String,
}

/// An event on its way in.
#[derive(Debug, Clone)]
pub struct EventDraft {
    pub run_id: String,
    pub event_id: String,
    pub event_type: TaskEventType,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub payload: Value,
}

/// The actor recorded when the application itself caused something.
pub const SYSTEM_ACTOR: &str = "system";

impl EventDraft {
    /// A new event with a fresh id and the current time.
    pub fn new(
        run_id: impl Into<String>,
        event_type: TaskEventType,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type,
            at: Utc::now(),
            actor: actor.into(),
            payload: Value::Object(Map::new()),
        }
    }

    /// A new event whose id is derived from what it is about.
    ///
    /// This is what makes a class of event idempotent rather than merely
    /// deduplicable: two writers describing the same thing — the same approval
    /// decision, the same run finishing, the same tool call settling — compute
    /// the same id without coordinating, and the second one is refused as the
    /// duplicate it is.
    ///
    /// `discriminator` must identify the thing uniquely within the run: an
    /// approval id, a tool-call id, or a constant for a once-per-run event.
    pub fn idempotent(
        run_id: impl Into<String>,
        event_type: TaskEventType,
        actor: impl Into<String>,
        discriminator: &str,
    ) -> Self {
        let run_id = run_id.into();
        let event_id = digest(&format!(
            "{run_id}\u{1f}{}\u{1f}{discriminator}",
            event_type.as_str()
        ));
        Self {
            event_id,
            ..Self::new(run_id, event_type, actor)
        }
    }

    /// Attaches a payload, redacted on the way in.
    ///
    /// There is deliberately no way to attach one that is not redacted. A
    /// caller who genuinely needs the contents kept has the task record and the
    /// workspace for that; the event stream is not the place, and making it
    /// impossible here is cheaper than reviewing every future call site.
    pub fn with(mut self, payload: Value) -> Self {
        self.payload = redact(payload);
        self
    }

    /// Uses a caller-chosen event id.
    pub fn with_event_id(mut self, event_id: impl Into<String>) -> Self {
        self.event_id = event_id.into();
        self
    }

    /// Sets the moment it happened, for events reconstructed after the fact.
    pub fn at(mut self, at: DateTime<Utc>) -> Self {
        self.at = at;
        self
    }
}

/// Payload keys whose values may carry the contents of somebody's documents,
/// the model's own words, or a credential.
///
/// Redacted wherever they appear, at any depth. The list is deliberately
/// generous — a field wrongly hashed costs a line of debugging, a field wrongly
/// kept costs a disclosure — and it is matched case-insensitively because the
/// two sides of this wire disagree about camelCase.
const CONFIDENTIAL_KEYS: &[&str] = &[
    "answer",
    "apikey",
    "args",
    "arguments",
    "body",
    "completion",
    "content",
    "credential",
    "detail",
    "excerpt",
    "expression",
    "key",
    "message",
    "passage",
    "password",
    "prompt",
    "query",
    "reasoning",
    "result",
    "secret",
    "source",
    "text",
    "thinking",
    "token",
];

/// Replaces every confidential-looking value with a hash of itself.
///
/// Structure is preserved so the shape of an event stays readable: a redacted
/// string becomes `{"sha256": "…", "chars": 812}`, which still tells a reader
/// that something was there and how much of it, and still lets two events be
/// compared for being about the same content.
pub fn redact(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    if is_confidential(&key) {
                        (key, digest_of(&value))
                    } else {
                        (key, redact(value))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(redact).collect()),
        other => other,
    }
}

/// Whether a payload key names something that must not be stored in the clear.
pub fn is_confidential(key: &str) -> bool {
    CONFIDENTIAL_KEYS
        .iter()
        .any(|blocked| key.eq_ignore_ascii_case(blocked))
}

/// The stand-in a redacted value is replaced by.
fn digest_of(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        Value::String(text) => json!({
            "sha256": digest(text),
            "chars": text.chars().count(),
        }),
        other => {
            let canonical = canonical(other);
            json!({
                "sha256": digest(&canonical),
                "chars": canonical.chars().count(),
            })
        }
    }
}

/// SHA-256 of some text, hex.
pub fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// One JSON value, written the same way every time.
///
/// Object keys are sorted explicitly rather than relying on `serde_json`'s map
/// being ordered: that depends on a Cargo feature, and a hash whose stability
/// depends on a feature flag somebody else can turn on is not a stable hash.
pub fn canonical(value: &Value) -> String {
    match value {
        Value::Object(fields) => {
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|key| format!("{}:{}", Value::String(key.clone()), canonical(&fields[key])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// The seal over a payload.
pub fn payload_hash(payload: &Value) -> String {
    digest(&canonical(payload))
}

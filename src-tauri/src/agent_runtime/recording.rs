//! How the runtime writes what it is doing into the durable record.
//!
//! The glue between [`super`], which decides and performs tool calls, and
//! [`events`], which stores an ordered history of them. Kept apart from both:
//! the decision path should not grow a second concern, and the event store
//! should not learn what a `RuntimeDeps` is.
//!
//! ## Why this duplicates the in-memory tables
//!
//! `RuntimeDeps` already accumulates a run's tool calls, passages and
//! artifacts, and the task record is built from those when the run ends. Every
//! one of those tables dies with the process. So does the account of the run
//! they were going to produce.
//!
//! What is written here is the same information going somewhere that survives.
//! The duplication is the point: one copy is convenient and fast and is lost on
//! a crash, and the other is the one a window that reopens can read.

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use super::events;
use super::{CallParams, RuntimeDeps};
use crate::orchestrator::tools::ToolName;

impl RuntimeDeps {
    /// Who to attribute a record to right now.
    ///
    /// Falls back to `system` rather than refusing: an event that could not be
    /// attributed is still worth having, and the tool call it describes has
    /// already been through a gateway that *does* refuse when nobody is signed
    /// in.
    pub(super) fn actor(&self) -> String {
        self.session
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|session| session.user.id.clone()))
            .unwrap_or_else(|| events::SYSTEM_ACTOR.to_string())
    }

    /// Writes one event into the run's durable history.
    ///
    /// Best-effort by design, and the asymmetry with `RuntimeDeps::publish` is deliberate:
    /// a dropped UI event costs a progress line, a dropped durable event costs
    /// a line in the history a restart reads. So this one is logged when it
    /// fails, and it still does not stop the run — a task that refuses to
    /// proceed because its history could not be written would trade a
    /// recoverable gap for a certain loss.
    pub(super) fn remember(&self, run_id: &str, event_type: events::TaskEventType, payload: Value) {
        let draft = events::EventDraft::new(run_id, event_type, self.actor()).with(payload);
        match self.events.record(draft) {
            // Published only once it is on disk, and carrying the sequence
            // number the row was given. A client that receives these in order
            // can tell a gap from a quiet moment; one that received them before
            // the write could be told about an event that never landed.
            Ok(event) => (self.emit_durable)(event.envelope()),
            // The run ended while a tool call was still in flight — an ordinary
            // race after an abort, not a fault.
            Err(events::AppendError::AlreadyEnded { .. })
            | Err(events::AppendError::Duplicate { .. }) => {}
            Err(error) => {
                log::warn!("[tasks] run {run_id}: {error}");
            }
        }
    }
}

/// Records a refusal.
///
/// One place rather than five, because a refusal that reaches the model and not
/// the history is the failure mode that makes a trace say the policy never did
/// anything.
pub(super) fn remember_refusal(deps: &Arc<RuntimeDeps>, call: &CallParams, reason: &str) {
    deps.remember(
        &call.run_id,
        events::TaskEventType::ToolRefused,
        json!({
            "toolCallId": call.tool_call_id,
            "tool": call.tool,
            "reason": reason,
        }),
    );
}

/// Refuses a call, and records the refusal before returning it.
pub(super) fn refused(deps: &Arc<RuntimeDeps>, call: &CallParams, reason: String) -> Value {
    remember_refusal(deps, call, &reason);
    json!({ "outcome": "refuse", "reason": reason })
}

/// Writes how a tool call went into the run's durable history.
///
/// Separate from `super::record_call`, which fills the in-memory table the task
/// record is built from at the end. The two look redundant and are not: the
/// in-memory one dies with the process, and this one is what a screen reads
/// after a restart. A run interrupted halfway still shows the calls it made.
pub(super) fn remember_outcome(
    deps: &Arc<RuntimeDeps>,
    call: &CallParams,
    tool: ToolName,
    resolved_path: Option<&Path>,
    outcome: &Result<String, String>,
) {
    let (event_type, payload) = match outcome {
        // `detail` is redacted on the way in — a search result carries the
        // passage it found, and that is exactly what must not be copied here.
        Ok(text) => (
            events::TaskEventType::ToolSucceeded,
            json!({
                "toolCallId": call.tool_call_id,
                "tool": tool.as_str(),
                "detail": text,
            }),
        ),
        Err(reason) => (
            events::TaskEventType::ToolFailed,
            json!({
                "toolCallId": call.tool_call_id,
                "tool": tool.as_str(),
                "reason": reason,
            }),
        ),
    };
    deps.remember(&call.run_id, event_type, payload);

    // The file is a reference: its name, so the Tasks screen can list what a
    // run produced without opening anything.
    if outcome.is_ok() {
        if let Some(name) = resolved_path
            .filter(|_| matches!(tool, ToolName::CreateDocx | ToolName::CreateXlsx | ToolName::WriteScopedFile))
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
        {
            deps.remember(
                &call.run_id,
                events::TaskEventType::ArtifactProduced,
                json!({ "name": name, "tool": tool.as_str() }),
            );
        }
    }
}

/// Keeps the two loop events a recovered trace would otherwise be missing.
///
/// Everything else the loop publishes is progress — a message part, a tool
/// starting — and this side already records the tool calls themselves from the
/// authorisation path, which is the account that cannot be dropped. These two
/// have no other source: the turn count and the compactions are only ever
/// announced by the loop.
pub(super) fn remember_loop_event(deps: &Arc<RuntimeDeps>, params: &Value) {
    let Some(run_id) = params.get("runId").and_then(Value::as_str) else {
        return;
    };
    let Some(event) = params.get("event") else {
        return;
    };
    match event.get("type").and_then(Value::as_str) {
        Some("turn_end") => deps.remember(run_id, events::TaskEventType::TurnEnded, json!({})),
        Some("context_compacted") => deps.remember(
            run_id,
            events::TaskEventType::ContextCompacted,
            json!({
                "tokensBefore": event.get("tokensBefore").cloned().unwrap_or(Value::Null),
                "tokensAfter": event.get("tokensAfter").cloned().unwrap_or(Value::Null),
                "messagesSummarised": event
                    .get("messagesSummarised")
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
        ),
        _ => {}
    }
}


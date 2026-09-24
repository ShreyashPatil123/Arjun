//! Receipts, resolved in the store that wrote them; and the outbox's road back.
//!
//! ## What a receipt is, and what it is not
//!
//! A receipt is a row in the durable task event log that says a named tool
//! succeeded in a named run and returned output with a named hash. It is **not**
//! a number a writer puts on an item. [`super::runtime_memory::admit`] used to
//! accept any positive `event_seq` with a tool and a run id beside it; every
//! worker then published every finding with `event_seq: 0` against the
//! *parent's* run, attributed to whichever tool the child happened to call
//! first. The zero was refused, correctly -- and changing it to a positive
//! number would have been manufactured corroboration.
//!
//! So a receipt is now *resolved*: [`ReceiptLedger::verify`] reads the event
//! back and checks that it exists, is a success, names the same tool, and
//! records the same output hash. The store asks before it admits anything.
//!
//! ## Where the events come from
//!
//! Two places, and both are real calls:
//!
//! - A child that runs a model loop calls tools through the same gateway as
//!   its parent, under its own run id, and `agent_runtime::recording` writes a
//!   `tool_succeeded` event for each with the output hashed by redaction.
//! - A worker on the mechanical path (no model) performs the tool's work
//!   directly -- a search, a file read, a calculation, a re-open -- and records
//!   that action here with [`record_tool_receipt`], under the child's run id,
//!   before any finding is built from it.
//!
//! In both, each finding carries the receipt of the call that produced *it*.
//!
//! ## The outbox's consumer
//!
//! [`OutboxConsumer`] is how a graph commit reaches the event log, which is a
//! separate connection and cannot share a transaction with it. The graph writes
//! the item and an outbox row together; a deliverer later appends a
//! `memory_published` event whose id is derived from the outbox key, so
//! delivering the same row twice is refused by the event log as a duplicate.

use serde_json::{json, Value};

use crate::agent_runtime::events::{AppendError, EventDraft, TaskEventLog, TaskEventType};
use crate::orchestrator::tools::ToolName;

use super::runtime_memory::Provenance;
use super::runtime_store::OutboxRow;

/// Where a receipt is resolved.
///
/// A trait so the graph does not depend on the event log's storage, and so a
/// test can hand it the real [`TaskEventLog`] rather than a stand-in that would
/// prove only that the stand-in agrees.
pub trait ReceiptLedger: Send + Sync {
    /// `Ok` when the event exists, succeeded, names `tool` and recorded exactly
    /// `output_sha256`. Otherwise the part that failed, in words.
    fn verify(
        &self,
        run_id: &str,
        event_seq: i64,
        tool: &str,
        output_sha256: &str,
    ) -> Result<(), String>;
}

/// The same tool, whichever spelling each side wrote.
fn same_tool(recorded: &str, claimed: &str) -> bool {
    match (ToolName::from_str(recorded), ToolName::from_str(claimed)) {
        (Some(a), Some(b)) => a == b,
        _ => recorded == claimed,
    }
}

impl ReceiptLedger for TaskEventLog {
    fn verify(
        &self,
        run_id: &str,
        event_seq: i64,
        tool: &str,
        output_sha256: &str,
    ) -> Result<(), String> {
        let event = self
            .event_at(run_id, event_seq)?
            .ok_or_else(|| format!("run {run_id} has no event {event_seq}"))?;

        // Only a success establishes anything. A failed call established only
        // that it failed, and a refusal that it was never made.
        if event.event_type != TaskEventType::ToolSucceeded {
            return Err(format!(
                "event {event_seq} is {}, not a successful tool call",
                event.event_type.as_str()
            ));
        }

        let recorded_tool = event
            .payload
            .get("tool")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("event {event_seq} does not name a tool"))?;
        if !same_tool(recorded_tool, tool) {
            return Err(format!(
                "event {event_seq} is a {recorded_tool} call, and the item claims {tool}"
            ));
        }

        // The output as the event recorded it: redaction replaced it with its
        // own hash on the way in, which is what makes this comparable without
        // the event ever holding the content.
        let recorded_output = event
            .payload
            .get("detail")
            .and_then(|detail| detail.get("sha256"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("event {event_seq} records no output hash"))?;
        if recorded_output != output_sha256 {
            return Err(format!(
                "event {event_seq} recorded different output from the one this item rests on"
            ));
        }
        Ok(())
    }
}

/// One tool action, as recorded: enough to name it in a provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// The run that made the call. For a worker, the child's own run.
    pub run_id: String,
    pub tool: String,
    pub event_seq: i64,
    pub output_sha256: String,
}

impl Receipt {
    pub fn provenance(&self) -> Provenance {
        Provenance::ToolReceipt {
            run_id: self.run_id.clone(),
            tool: self.tool.clone(),
            event_seq: self.event_seq,
            output_sha256: Some(self.output_sha256.clone()),
        }
    }
}

/// Records a tool action a worker performed itself, as the durable receipt its
/// findings rest on.
///
/// `call_key` identifies the action within the run -- the same action recorded
/// twice (a retry after a worker died) is one event, and the receipt names it.
/// `output` is what the tool produced; the event keeps only its hash.
pub fn record_tool_receipt(
    events: &TaskEventLog,
    run_id: &str,
    tool: ToolName,
    call_key: &str,
    actor: &str,
    output: &str,
) -> Result<Receipt, String> {
    let draft = EventDraft::idempotent(run_id, TaskEventType::ToolSucceeded, actor, call_key).with(
        json!({
            "toolCallId": call_key,
            "tool": tool.as_str(),
            "detail": output,
            // Said in the record, so a reader can tell a worker's own action
            // from a call a model made through the gateway.
            "performedBy": "worker",
        }),
    );
    let output_sha256 = crate::agent_runtime::events::digest(output);
    let event_seq = match events.append(draft) {
        Ok(event) => event.seq,
        Err(AppendError::Duplicate { seq, .. }) => seq,
        Err(error) => {
            return Err(format!(
                "the receipt for this {} call could not be recorded: {error}",
                tool.as_str()
            ))
        }
    };
    Ok(Receipt {
        run_id: run_id.to_string(),
        tool: tool.as_str().to_string(),
        event_seq,
        output_sha256,
    })
}

/// Something an outbox row is delivered to.
pub trait OutboxConsumer: Send + Sync {
    /// The `target` an outbox row names for this consumer.
    fn target(&self) -> &'static str;
    /// Delivers one row. Must be idempotent on `row.idempotency_key`: the
    /// deliverer retries whatever it could not mark delivered.
    fn deliver(&self, row: &OutboxRow) -> Result<(), String>;
}

/// The outbox target the task event log answers to.
pub const EVENTS_TARGET: &str = "events";

impl OutboxConsumer for TaskEventLog {
    fn target(&self) -> &'static str {
        EVENTS_TARGET
    }

    fn deliver(&self, row: &OutboxRow) -> Result<(), String> {
        let payload: Value = serde_json::from_str(&row.payload)
            .map_err(|error| format!("outbox row {} is not readable: {error}", row.outbox_id))?;
        let run_id = payload
            .get("runId")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("outbox row {} names no run", row.outbox_id))?;
        let draft = EventDraft::idempotent(
            run_id,
            TaskEventType::MemoryPublished,
            crate::agent_runtime::events::SYSTEM_ACTOR,
            &row.idempotency_key,
        )
        .with(payload.get("event").cloned().unwrap_or(Value::Null));
        // Past an ending, deliberately: the commit happened while the run was
        // live, and the run finishing before the deliverer caught up does not
        // make the publication not have happened. The ending is not rewritten.
        match events_append_past_ending(self, draft) {
            Ok(()) => Ok(()),
            Err(error) => Err(error),
        }
    }
}

fn events_append_past_ending(events: &TaskEventLog, draft: EventDraft) -> Result<(), String> {
    match events.append_past_ending(draft) {
        Ok(_) => Ok(()),
        // Delivered before. The consumer is idempotent by construction.
        Err(AppendError::Duplicate { .. }) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> TaskEventLog {
        TaskEventLog::in_memory().expect("an event log")
    }

    #[test]
    fn a_recorded_action_verifies_and_every_mismatch_is_refused() {
        let events = log();
        let receipt = record_tool_receipt(
            &events,
            "child-1",
            ToolName::RunCalculation,
            "child-1:calc:1",
            "priya",
            "9.0 mm - 8.2 mm = 0.8 mm",
        )
        .expect("recorded");
        assert!(receipt.event_seq > 0);

        events
            .verify(&receipt.run_id, receipt.event_seq, &receipt.tool, &receipt.output_sha256)
            .expect("the receipt verifies");
        // Either spelling of the tool is the tool.
        events
            .verify(&receipt.run_id, receipt.event_seq, "run_calculation", &receipt.output_sha256)
            .expect("a legacy spelling is the same tool");

        for (run, seq, tool, hash, why) in [
            ("child-1", receipt.event_seq + 5, receipt.tool.as_str(), receipt.output_sha256.as_str(), "no such event"),
            ("child-2", receipt.event_seq, receipt.tool.as_str(), receipt.output_sha256.as_str(), "another run"),
            ("child-1", receipt.event_seq, "knowledge.search_authorized", receipt.output_sha256.as_str(), "another tool"),
            ("child-1", receipt.event_seq, receipt.tool.as_str(), "not-the-output", "another output"),
        ] {
            assert!(events.verify(run, seq, tool, hash).is_err(), "{why} verified");
        }
    }

    #[test]
    fn a_failed_call_is_not_a_receipt() {
        let events = log();
        let failed = events
            .append(
                EventDraft::new("child-1", TaskEventType::ToolFailed, "priya").with(json!({
                    "toolCallId": "tc-1",
                    "tool": "knowledge.search_authorized",
                    "reason": "the index could not be searched",
                })),
            )
            .expect("appended");
        let refusal = events
            .verify("child-1", failed.seq, "knowledge.search_authorized", "anything")
            .expect_err("a failure verified");
        assert!(refusal.contains("tool_failed"), "{refusal}");
    }

    #[test]
    fn recording_the_same_action_twice_is_one_event() {
        let events = log();
        let first = record_tool_receipt(&events, "child-1", ToolName::ReadScopedFile, "k", "a", "x")
            .expect("first");
        let again = record_tool_receipt(&events, "child-1", ToolName::ReadScopedFile, "k", "a", "x")
            .expect("again");
        assert_eq!(first, again);
        assert_eq!(events.events_since("child-1", 0).expect("reads").events.len(), 1);
    }
}

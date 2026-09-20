//! Commands behind the Approvals surface.
//!
//! Thin on purpose: every rule that matters — who may decide, that a rejection
//! carries a reason, that a decision is final — lives in
//! [`crate::orchestrator::approvals`], where it is tested without a UI. A rule
//! enforced only in a command is a rule that stops applying the moment anything
//! else calls the same code.

use std::sync::Arc;

use tauri::State;

use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::orchestrator::approvals::{ApprovalItem, ApprovalQueue, Decision};

/// Everything raised this session, newest first, settled ones included.
#[tauri::command]
pub async fn list_approvals(
    queue: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<ApprovalItem>, String> {
    // The approvals queue is reviewer's work. The matrix puts the
    // ability to see it under `ApproveOutput`. A `User` or `Auditor`
    // does not get to see what they are not allowed to decide.
    require_permission(&session, Permission::ApproveOutput)?;
    Ok(queue.all())
}

/// Approves or rejects one request.
#[tauri::command]
pub async fn decide_approval(
    queue: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    events: State<'_, crate::commands::agent::TaskEvents>,
    run_to_conversation: State<'_, crate::commands::conversations::RunToConversationState>,
    id: String,
    approve: bool,
    because: Option<String>,
    // `always` is "Always approve", from the third button. An `Option` because
    // every older caller omits it, and `None` means "just this one".
    always: Option<bool>,
) -> Result<Decision, String> {
    let signed_in = require_session(&session)?;

    let decision = queue
        .decide(&signed_in, &id, approve, because.as_deref())
        .map_err(|e| {
            // A refused decision is recorded too. "Who tried to approve their
            // own work" is exactly the question an auditor asks later.
            let _ = audit.record(
                &signed_in.user.id,
                AuditKind::PolicyDecision,
                format!("Approval decision refused: {}", e.message),
                Some(serde_json::json!({ "approvalId": id, "allowed": false })),
            );
            e.message
        })?;

    let item = queue.find(&id);

    // "Always approve" is recorded only when the answer was actually yes.
    //
    // `always: true` with `approve: false` is not a state the buttons can
    // produce, and treating it as a standing grant would turn a rejection into
    // a blanket permission. Read as a conjunction rather than trusted.
    //
    // The conversation is resolved here rather than sent by the caller. The
    // request carries the run id in `task_id`, and the index from run to
    // conversation is authoritative on this side — a scope supplied by the
    // caller would be a permission boundary named by the thing being bounded.
    let standing = approve && always.unwrap_or(false);
    if standing {
        match item.as_ref() {
            Some(held) => match run_to_conversation.0.lookup(&held.request.task_id) {
                Some(conversation_id) => {
                    queue.grant_standing(&conversation_id, &held.request.tool);
                    let _ = audit.record(
                        &signed_in.user.id,
                        AuditKind::Approval,
                        format!(
                            "Standing approval granted for {} in this conversation",
                            held.request.tool
                        ),
                        Some(serde_json::json!({
                            "approvalId": id,
                            "taskId": held.request.task_id,
                            "conversationId": conversation_id,
                            "tool": held.request.tool,
                            "scope": "conversation",
                        })),
                    );
                }
                // A run outside any conversation has no scope to hold the
                // grant. The single approval above still stands; only the
                // "always" part is dropped, and saying so beats a silent
                // no-op that looks to the person like it worked.
                None => log::warn!(
                    "[approvals] {id}: \"always\" was asked for but run {} is not in a \
                     conversation, so only this one call was approved",
                    held.request.task_id
                ),
            },
            None => log::warn!(
                "[approvals] {id}: \"always\" was asked for but the request could not be \
                 found, so only this one call was approved"
            ),
        }
    }

    let _ = audit.record(
        &signed_in.user.id,
        AuditKind::Approval,
        format!(
            "{} {} for {}",
            if approve { "Approved" } else { "Rejected" },
            item.as_ref().map(|i| i.request.tool.as_str()).unwrap_or("an action"),
            item.as_ref().map(|i| i.request.target.as_str()).unwrap_or("an unknown target"),
        ),
        Some(serde_json::json!({
            "approvalId": id,
            "taskId": item.as_ref().map(|i| i.request.task_id.clone()),
            "approved": approve,
        })),
    );

    // Recorded durably as well as in memory, so a restart does not lose the
    // answer. Written after the queue has accepted it: the queue is what
    // enforces who may decide, and a decision it refused must not appear here
    // as one that was taken.
    let status = if approve {
        crate::agent_runtime::events::ApprovalStatus::Approved
    } else {
        crate::agent_runtime::events::ApprovalStatus::Rejected
    };
    if let Err(error) = events.resolve_approval(
        &id,
        status,
        &signed_in.user.id,
        because.as_deref(),
        chrono::Utc::now(),
    ) {
        log::warn!("[approvals] decision on {id} was not recorded durably: {error}");
    }

    Ok(decision)
}

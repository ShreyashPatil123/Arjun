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
use crate::commands::governance::{require_session, CurrentSession};
use crate::orchestrator::approvals::{ApprovalItem, ApprovalQueue, Decision};

/// Everything raised this session, newest first, settled ones included.
#[tauri::command]
pub async fn list_approvals(
    queue: State<'_, Arc<ApprovalQueue>>,
) -> Result<Vec<ApprovalItem>, String> {
    Ok(queue.all())
}

/// Approves or rejects one request.
#[tauri::command]
pub async fn decide_approval(
    queue: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    id: String,
    approve: bool,
    because: Option<String>,
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

    Ok(decision)
}

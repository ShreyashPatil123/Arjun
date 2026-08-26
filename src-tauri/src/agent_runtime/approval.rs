//! Waiting for a person.
//!
//! Three of the eight tools leave a trace outside the task — a file, a
//! document, an execution — and the gateway marks those `needs_approval`. Until
//! now the runtime turned that verdict into a refusal, because there was
//! nothing to wait on. That is the wrong shape: the model asked whether it may
//! act, and "a person has not looked yet" is not the same answer as "no".
//!
//! ## Why the waiting happens here and not in the runtime
//!
//! The obvious alternative is to answer `needsApproval` immediately and let the
//! Node runtime poll. That would put the pending state in two places and give
//! the runtime a reason to know about approvals at all — and the runtime is the
//! side that must not be trusted with policy.
//!
//! So `tool.authorize` simply takes longer when a person is involved. From the
//! loop's point of view a slow authorisation is indistinguishable from a slow
//! anything else; from the operator's, the request appears on the approvals
//! screen and the run continues when they decide. Neither side has to
//! understand the other's model of waiting.
//!
//! ## Why it polls
//!
//! [`ApprovalQueue`] is a plain mutex-guarded list, deliberately: it is read by
//! Tauri commands, the health panel and this module, and a notification channel
//! would make its lock ordering something to reason about. A quarter-second
//! poll costs nothing next to the minutes a human takes, and the queue stays
//! the simple thing that many callers can read safely.

use std::sync::Arc;
use std::time::Duration;

use crate::identity::Session;
use crate::orchestrator::approvals::{ApprovalQueue, ApprovalRequest, Decision};
use crate::orchestrator::tools::ToolName;

/// How often the queue is checked.
const POLL: Duration = Duration::from_millis(250);

/// How long a run waits before giving up on a person.
///
/// Long enough that an approver can finish what they were doing and come back;
/// short enough that a run started before lunch does not hold a model server
/// resident all afternoon. On expiry the model is told plainly that nobody
/// answered, which is something it can report rather than something it has to
/// infer from a hang.
const WAIT_LIMIT: Duration = Duration::from_secs(15 * 60);

/// What waiting came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    Approved { by: String },
    Rejected { by: String, because: String },
    /// Nobody decided within the limit.
    TimedOut,
}

impl ApprovalOutcome {
    /// The sentence handed back to the model when the answer is not yes.
    ///
    /// Written for a model to act on: it says what happened, and what the model
    /// should do about it. "Rejected" alone invites the same action again.
    pub fn refusal(&self) -> String {
        match self {
            ApprovalOutcome::Approved { .. } => String::new(),
            ApprovalOutcome::Rejected { by, because } => format!(
                "{by} did not approve this action, because: {because}. It did not happen. Do not \
                 propose the same action again — address the objection first, or explain to the \
                 user why you cannot."
            ),
            ApprovalOutcome::TimedOut =>
                "Nobody responded to the approval request in time, so the action did not happen. \
                 Say so plainly rather than describing what it would have produced."
                    .to_string(),
        }
    }
}

/// Puts a proposed action in front of a person and waits for their answer.
pub async fn await_decision(
    queue: &Arc<ApprovalQueue>,
    session: &Session,
    run_id: &str,
    tool: ToolName,
    summary: String,
    target: String,
    arguments: Vec<String>,
) -> ApprovalOutcome {
    let id = uuid::Uuid::new_v4().to_string();
    queue.request(ApprovalRequest {
        id: id.clone(),
        task_id: run_id.to_string(),
        tool: tool.as_str().to_string(),
        target,
        arguments,
        // Populated in a later phase, when a run carries the passages it relied
        // on. Empty is honest; inventing evidence to fill the field would make
        // the approval screen less trustworthy, not more.
        evidence: Vec::new(),
        expected_output: summary,
        consequences: tool.describe().to_string(),
        requested_by: session.user.id.clone(),
        requested_at: chrono::Utc::now(),
    });

    let deadline = tokio::time::Instant::now() + WAIT_LIMIT;
    loop {
        if let Some(item) = queue.find(&id) {
            match item.decision {
                Some(Decision::Approved { by, .. }) => return ApprovalOutcome::Approved { by },
                Some(Decision::Rejected { by, because, .. }) => {
                    return ApprovalOutcome::Rejected { by, because }
                }
                None => {}
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return ApprovalOutcome::TimedOut;
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};

    fn reviewer() -> Session {
        Session::open(User::new("ravi", "Ravi Menon", vec![Role::Reviewer]))
    }

    fn author() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::User]))
    }

    #[tokio::test]
    async fn an_approved_action_names_who_approved_it() {
        let queue = Arc::new(ApprovalQueue::new());
        let waiting = {
            let queue = queue.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &author(),
                    "run-1",
                    ToolName::WriteScopedFile,
                    "Write 5 bytes to note.txt".into(),
                    "note.txt".into(),
                    vec!["path=note.txt".into()],
                )
                .await
            })
        };

        // Let the request reach the queue before deciding it.
        let id = loop {
            if let Some(item) = queue.pending().first() {
                break item.request.id.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        queue.decide(&reviewer(), &id, true, None).expect("approved");

        assert_eq!(
            waiting.await.expect("task finished"),
            ApprovalOutcome::Approved { by: "ravi".into() }
        );
    }

    #[tokio::test]
    async fn a_rejection_carries_the_reason_back_to_the_model() {
        let queue = Arc::new(ApprovalQueue::new());
        let waiting = {
            let queue = queue.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &author(),
                    "run-1",
                    ToolName::WriteScopedFile,
                    "Write 5 bytes to note.txt".into(),
                    "note.txt".into(),
                    vec![],
                )
                .await
            })
        };

        let id = loop {
            if let Some(item) = queue.pending().first() {
                break item.request.id.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        queue
            .decide(&reviewer(), &id, false, Some("the seal figure is unsourced"))
            .expect("rejected");

        let outcome = waiting.await.expect("task finished");
        let refusal = outcome.refusal();
        assert!(refusal.contains("the seal figure is unsourced"), "{refusal}");
        // A model told only "rejected" proposes the same thing again.
        assert!(refusal.contains("Do not propose the same action again"), "{refusal}");
    }

    #[test]
    fn a_timeout_tells_the_model_to_say_so_rather_than_invent_a_result() {
        let refusal = ApprovalOutcome::TimedOut.refusal();
        assert!(refusal.contains("did not happen"));
        assert!(refusal.contains("rather than describing what it would have produced"));
    }

    #[tokio::test]
    async fn the_request_reaches_the_queue_with_what_an_approver_needs_to_judge_it() {
        let queue = Arc::new(ApprovalQueue::new());
        let handle = {
            let queue = queue.clone();
            tokio::spawn(async move {
                await_decision(
                    &queue,
                    &author(),
                    "run-42",
                    ToolName::CreateDocx,
                    "Produce an approval note at note.docx".into(),
                    "note.docx".into(),
                    vec!["template=approval-note".into()],
                )
                .await
            })
        };

        let item = loop {
            if let Some(item) = queue.pending().first().cloned() {
                break item;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };

        assert_eq!(item.request.task_id, "run-42");
        assert_eq!(item.request.tool, "create_docx");
        assert_eq!(item.request.target, "note.docx");
        assert_eq!(item.request.requested_by, "priya");
        // The consequence, not just the name — an approver reading "create_docx"
        // learns nothing they did not already know.
        assert_eq!(item.request.consequences, "produce a Word document");

        handle.abort();
    }
}

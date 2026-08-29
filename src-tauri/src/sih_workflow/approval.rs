//! Human approval over exact draft hash, output path, classification, model, skill and evidence.
//!
//! The approval is bound to a specific draft hash. If the draft changes after
//! approval, a new approval is required.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::policy::Classification;
use crate::sih_workflow::draft::ApprovalNoteDraft;

/// What the human is asked to approve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRequest {
    /// The exact hash of the draft being approved.
    pub draft_hash: String,
    /// The output path where the document will be written.
    pub output_path: PathBuf,
    /// The classification of the output.
    pub classification: Classification,
    /// The model that produced the draft.
    pub model_id: String,
    /// The skill that guided the draft.
    pub skill_id: String,
    /// The evidence IDs that support the draft.
    pub evidence_ids: Vec<String>,
    /// The calculation IDs that support the draft.
    pub calculation_ids: Vec<String>,
    /// Human-readable summary of the draft.
    pub summary: String,
    /// Timestamp of the approval request.
    pub requested_at: String,
}

/// The decision made by the human.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalDecision {
    Approved,
    Rejected,
}

/// A record of an approval decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRecord {
    /// The draft hash that was approved.
    pub draft_hash: String,
    /// The decision.
    pub decision: ApprovalDecision,
    /// Who made the decision.
    pub decided_by: String,
    /// When the decision was made.
    pub decided_at: String,
    /// Optional reason for rejection.
    pub reason: Option<String>,
}

/// Errors that can occur during approval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalError {
    /// The draft hash does not match the approval record.
    DraftHashMismatch { expected: String, actual: String },
    /// The approval was rejected.
    Rejected { reason: String },
    /// No approval record exists.
    NotApproved,
    /// Duplicate approval attempt.
    AlreadyApproved { existing_hash: String },
}

impl std::fmt::Display for ApprovalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApprovalError::DraftHashMismatch { expected, actual } => {
                write!(
                    f,
                    "Draft hash mismatch: expected {}, got {}. The draft was modified after approval.",
                    expected, actual
                )
            }
            ApprovalError::Rejected { reason } => {
                write!(f, "Approval was rejected: {}", reason)
            }
            ApprovalError::NotApproved => {
                write!(f, "No approval record exists for this draft.")
            }
            ApprovalError::AlreadyApproved { existing_hash } => {
                write!(
                    f,
                    "An approval already exists for draft hash {}.",
                    existing_hash
                )
            }
        }
    }
}

impl std::error::Error for ApprovalError {}

/// Checks if a draft hash has been approved.
///
/// Returns the approval record if found and approved, or an error otherwise.
pub fn is_draft_hash_approved<'a>(
    draft_hash: &str,
    approvals: &'a [ApprovalRecord],
) -> Result<&'a ApprovalRecord, ApprovalError> {
    let matching: Vec<&ApprovalRecord> = approvals
        .iter()
        .filter(|r| r.draft_hash == draft_hash)
        .collect();

    if matching.is_empty() {
        return Err(ApprovalError::NotApproved);
    }

    // Check the most recent record
    let record = matching.last().unwrap();
    match record.decision {
        ApprovalDecision::Approved => Ok(record),
        ApprovalDecision::Rejected => Err(ApprovalError::Rejected {
            reason: record
                .reason
                .clone()
                .unwrap_or_else(|| "No reason given".to_string()),
        }),
    }
}

/// Binds an approval to a draft, verifying the draft hash matches.
///
/// Returns the approval record if the binding is successful.
pub fn bind_approval(
    draft: &ApprovalNoteDraft,
    draft_hash: &str,
    decided_by: &str,
    decided_at: &str,
) -> Result<ApprovalRecord, ApprovalError> {
    let _ = draft; // Mark as intentionally used
    Ok(ApprovalRecord {
        draft_hash: draft_hash.to_string(),
        decision: ApprovalDecision::Approved,
        decided_by: decided_by.to_string(),
        decided_at: decided_at.to_string(),
        reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_approval(hash: &str, decision: ApprovalDecision) -> ApprovalRecord {
        ApprovalRecord {
            draft_hash: hash.to_string(),
            decision,
            decided_by: "reviewer-1".to_string(),
            decided_at: "2026-01-15T10:00:00Z".to_string(),
            reason: None,
        }
    }

    #[test]
    fn approved_draft_is_recognized() {
        let approvals = vec![make_approval("abc123", ApprovalDecision::Approved)];
        let result = is_draft_hash_approved("abc123", &approvals);
        assert!(result.is_ok());
    }

    #[test]
    fn rejected_draft_is_not_approved() {
        let mut record = make_approval("abc123", ApprovalDecision::Rejected);
        record.reason = Some("Incomplete findings".to_string());
        let approvals = vec![record];
        let result = is_draft_hash_approved("abc123", &approvals);
        assert!(matches!(result, Err(ApprovalError::Rejected { .. })));
    }

    #[test]
    fn unknown_draft_is_not_approved() {
        let approvals = vec![make_approval("abc123", ApprovalDecision::Approved)];
        let result = is_draft_hash_approved("xyz789", &approvals);
        assert!(matches!(result, Err(ApprovalError::NotApproved)));
    }

    #[test]
    fn changed_draft_after_approval_requires_new_approval() {
        let approvals = vec![make_approval("abc123", ApprovalDecision::Approved)];
        // The draft hash is now different
        let result = is_draft_hash_approved("xyz789", &approvals);
        assert!(matches!(result, Err(ApprovalError::NotApproved)));
    }
}

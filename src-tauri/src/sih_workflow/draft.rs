//! Typed ApprovalNoteDraft structure.
//!
//! The model fills this typed draft. Trusted local code validates it and
//! renders the Word document. The model never writes DOCX XML directly.

use serde::{Deserialize, Serialize};

use crate::policy::Classification;

/// Status of the draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DraftStatus {
    Draft,
    PendingApproval,
    Approved,
    Rejected,
    Rendered,
}

/// Severity level of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Low => "Low",
            Severity::Medium => "Medium",
            Severity::High => "High",
            Severity::Critical => "Critical",
        }
    }
}

/// A single finding from the inspection report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// Unique ID for this finding within the draft.
    pub id: String,
    /// Description of what was found.
    pub description: String,
    /// Severity of the finding.
    pub severity: Severity,
    /// Optional location on the equipment.
    pub location: Option<String>,
    /// Page in the source report where this was found.
    pub source_page: Option<u32>,
    /// Evidence IDs that support this finding.
    pub evidence_ids: Vec<String>,
}

/// Reference to a calculation record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CalculationRef {
    /// The calculation ID from the calculation engine.
    pub calculation_id: String,
    /// The expression that was evaluated.
    pub expression: String,
    /// The result with units.
    pub result: String,
    /// Which finding this calculation supports.
    pub finding_id: Option<String>,
}

/// Reference to an authorized evidence passage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    /// The evidence ID (matches a passage in the knowledge base).
    pub evidence_id: String,
    /// Document SHA-256.
    pub document_sha256: String,
    /// Page number in the source document.
    pub page: u32,
    /// The text of the passage.
    pub passage_text: String,
}

/// An uncertainty note from the extraction process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UncertaintyNote {
    /// What is uncertain.
    pub what: String,
    /// Why it is uncertain.
    pub reason: String,
    /// Which finding this uncertainty applies to.
    pub finding_id: Option<String>,
    /// Whether this should block approval.
    pub blocks_approval: bool,
}

/// The typed ApprovalNoteDraft that the model fills in.
///
/// All fields are required to be present (non-null) for the draft to be
/// valid for rendering, even if empty. The `status` field tracks the
/// lifecycle of the draft.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalNoteDraft {
    /// Equipment ID being inspected.
    pub equipment_id: String,
    /// Date of inspection (ISO 8601).
    pub inspection_date: String,
    /// Findings from the inspection.
    pub findings: Vec<Finding>,
    /// Overall severity (highest severity among findings).
    pub severity: Severity,
    /// Evidence IDs from authorized passages.
    pub evidence_ids: Vec<String>,
    /// Proposed action based on the findings.
    pub proposed_action: String,
    /// References to calculation records.
    pub calculation_ids: Vec<String>,
    /// Uncertainty notes from the extraction.
    pub uncertainty_notes: Vec<UncertaintyNote>,
    /// Classification of the material.
    pub classification: Classification,
    /// Status of the draft.
    pub status: DraftStatus,
    /// Optional: model ID that produced this draft.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Optional: skill ID that guided this draft.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_id: Option<String>,
}

impl ApprovalNoteDraft {
    /// Creates a new draft with the DRAFT status.
    pub fn new(
        equipment_id: String,
        inspection_date: String,
        classification: Classification,
    ) -> Self {
        Self {
            equipment_id,
            inspection_date,
            findings: Vec::new(),
            severity: Severity::Low,
            evidence_ids: Vec::new(),
            proposed_action: String::new(),
            calculation_ids: Vec::new(),
            uncertainty_notes: Vec::new(),
            classification,
            status: DraftStatus::Draft,
            model_id: None,
            skill_id: None,
        }
    }

    /// Returns the highest severity among all findings.
    pub fn highest_severity(&self) -> Severity {
        self.findings
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(Severity::Low)
    }

    /// Returns true if any uncertainty note blocks approval.
    pub fn has_blocking_uncertainty(&self) -> bool {
        self.uncertainty_notes.iter().any(|u| u.blocks_approval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_draft_has_draft_status() {
        let draft = ApprovalNoteDraft::new(
            "EQ-001".to_string(),
            "2026-01-15".to_string(),
            Classification::Internal,
        );
        assert_eq!(draft.status, DraftStatus::Draft);
        assert_eq!(draft.equipment_id, "EQ-001");
    }

    #[test]
    fn highest_severity_returns_max() {
        let mut draft = ApprovalNoteDraft::new(
            "EQ-001".to_string(),
            "2026-01-15".to_string(),
            Classification::Internal,
        );
        draft.findings = vec![
            Finding {
                id: "F1".to_string(),
                description: "Minor corrosion".to_string(),
                severity: Severity::Low,
                location: None,
                source_page: Some(1),
                evidence_ids: vec![],
            },
            Finding {
                id: "F2".to_string(),
                description: "Crack detected".to_string(),
                severity: Severity::High,
                location: None,
                source_page: Some(2),
                evidence_ids: vec![],
            },
        ];
        assert_eq!(draft.highest_severity(), Severity::High);
    }

    #[test]
    fn has_blocking_uncertainty_checks_notes() {
        let mut draft = ApprovalNoteDraft::new(
            "EQ-001".to_string(),
            "2026-01-15".to_string(),
            Classification::Internal,
        );
        draft.uncertainty_notes = vec![
            UncertaintyNote {
                what: "Page 3 unclear".to_string(),
                reason: "Low OCR confidence".to_string(),
                finding_id: None,
                blocks_approval: false,
            },
        ];
        assert!(!draft.has_blocking_uncertainty());

        draft.uncertainty_notes.push(UncertaintyNote {
            what: "Critical value unreadable".to_string(),
            reason: "Scan quality too low".to_string(),
            finding_id: Some("F1".to_string()),
            blocks_approval: true,
        });
        assert!(draft.has_blocking_uncertainty());
    }
}

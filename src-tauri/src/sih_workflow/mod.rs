//! SIH Inspection-Report-to-Word-Approval-Note Workflow
//!
//! Implements the exact sequence required by SIH26117:
//!
//! 1. Upload scanned inspection report and optional photograph.
//! 2. Validate type, size, classification and workspace scope.
//! 3. Run local OCR/VLM extraction.
//! 4. Search authorized SOP/manual collections.
//! 5. Run deterministic calculations with units where needed.
//! 6. Produce a typed ApprovalNoteDraft object.
//! 7. Validate required fields and evidence IDs.
//! 8. Ask for human approval over the exact draft hash, output path,
//!    classification, model, skill and evidence.
//! 9. Render a Word document from a trusted local template.
//! 10. Reopen and validate the DOCX structure.
//! 11. Verify citations, figures, classification and approval binding.
//! 12. Compute artifact hash and export the evidence package.
//!
//! The model never directly writes DOCX XML. It fills a typed draft, and
//! trusted local code renders the document.

pub mod draft;
pub mod pipeline;
pub mod approval;
pub mod evidence_package;

pub use draft::{
    ApprovalNoteDraft, DraftStatus, Finding, Severity, CalculationRef,
    EvidenceRef, UncertaintyNote,
};
pub use pipeline::{
    SihPipeline, PipelineInput, PipelineOutput, PipelineError,
    calculate_draft_hash, validate_draft, render_document, verify_document,
};
pub use approval::{
    ApprovalRequest, ApprovalDecision, ApprovalRecord, ApprovalError,
    is_draft_hash_approved, bind_approval,
};
pub use evidence_package::{
    EvidencePackage, PackageArtifact, Provenance, export_evidence_package,
};

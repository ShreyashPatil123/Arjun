//! SIH Inspection-Report-to-Word-Approval-Note Workflow
//!
//! Implements ARJUN's worked example of the end-to-end agentic task PS 26117
//! names: *"reading a scanned inspection report, pulling out key findings and
//! drafting an approval note as a Word file"*. The numbered sequence below is
//! ARJUN's decomposition, not a sequence the problem statement specifies:
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
//!
//! ## This module owns the workflow
//!
//! There was a second implementation, in Python, at `sidecars/sih_workflow/` —
//! 2,888 lines covering the same twelve steps, with its own approval and
//! evidence-package code. It has been removed, and this is the reasoning, kept
//! here because the next person to reach for a sidecar deserves to find it.
//!
//! Nothing called it. It was not imported by any Rust module, not launched by
//! any command, not referenced from `package.json`, and its tests were not in
//! `npm run test:sidecar` — which runs only the document sidecar. Its own suite
//! had 46 failures: 41 errors from a draft API that had moved underneath it and
//! 5 assertions that no longer held.
//!
//! Two implementations of approval and evidence packaging is the specific thing
//! this product cannot have. An approval binds a person's decision to an exact
//! draft hash, and an evidence package is what somebody is asked to stand
//! behind; if there are two of each and they disagree, the question "what was
//! approved?" has two answers and no way to choose. A contract test between
//! them would be the minimum price of keeping both, and there was none.
//!
//! So: Rust is canonical, because Rust is what runs. If a Python surface is
//! wanted later — for a notebook, for an integration — it should call this
//! through the sidecar protocol rather than reimplement it.

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

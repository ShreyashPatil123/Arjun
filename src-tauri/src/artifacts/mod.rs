//! Turning a task's findings into something somebody can hand to their manager.
//!
//! PS 26117: *"Output should be real deliverables, approval notes, PPT/Word/Excel
//! files, working code, calculations with steps shown, not just chat replies."*
//!
//! - [`docx`]: Word documents from templates the model cannot improvise around.
//! - [`xlsx`]: the calculation workbook, where Excel recomputes and can disagree.
//! - [`pptx`]: the briefing deck, the most tightly templated of the three.
//! - [`ooxml`]: escaping and packaging, shared by both.
//! - [`production`]: compose, render, re-open, verify — revising, never
//!   overwriting.
//! - [`verifier`]: the check between a draft and something somebody signs.

pub mod docx;
pub mod ooxml;
pub mod pptx;
pub mod production;
pub mod xlsx;
pub mod verifier;

pub use docx::{check_document, write_document, DocumentCheck, DocumentMetadata};
pub use pptx::{check_deck, write_deck, DeckCheck, Slide};
pub use production::{produce, ContentSource, ProductionOutcome, Revision};
pub use xlsx::{check_workbook, write_workbook, WorkbookCheck};
pub use verifier::{verify, Evidence, Severity, Standing, VerificationReport};

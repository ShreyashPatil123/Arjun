//! Visible watermark stamped onto every generated document.
//!
//! ## What this is, honestly
//!
//! A visible watermark is a printed claim, in the document body itself, that
//! says "this artifact was produced by ARJUN at <time> by <model> for <task>,
//! and is (or is not) a draft." It is **the only watermark this build ships**.
//!
//! ## What this is NOT
//!
//! It is not a forensic mark. A reader who has the source `.docx` can edit the
//! text, or strip the watermark paragraph outright. It is also not a defense
//! against an attacker who controls the runtime — a process that controls the
//! runtime already controls everything that gets written. The honest security
//! claim is *traceability*, not *attribution*:
//!
//! - A reviewer reading the document can see who made it, when, and under what
//!   task, without leaving the page.
//! - The claim is generated from the *metadata struct the verifier already
//!   wrote into the audit log*, so a watermark that disagrees with the audit
//!   entry is itself a discrepancy worth investigating.
//! - Steganographic or cryptographic watermarks (Item 5) are deliberately not
//!   implemented in this build — see the stego module for the written-out
//!   reasoning.
//!
//! ## Gating
//!
//! The PDF / Word / Excel / PowerPoint generators are partially built (the
//! approval-note template exists; the rest is scaffolded). For document
//! formats whose generator is not yet wired up, [`stamp_for_format`] returns
//! `None` and callers must omit the stamp rather than fabricate one. The
//! [`apply_to_docx_body`] helper exists for the one path that *is* wired up.

use serde::{Deserialize, Serialize};

/// The shape of a visible stamp, before it is rendered into a particular
/// document format. Carrying a typed struct (rather than a free-form string)
/// means a generated document's watermark can be machine-checked against the
/// audit log later.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VisibleStamp {
    pub task_id: String,
    pub model: String,
    pub created_at: String,
    pub classification: String,
    pub is_draft: bool,
    /// Free-text claim, e.g. "verified by ARJUN's calculation engine".
    /// Kept short and factual on purpose — this is the *visible* mark, so
    /// it must read as a sentence a person can understand at a glance.
    pub claim: String,
}

impl VisibleStamp {
    /// One-line summary that fits the header band of a Word document.
    pub fn header_line(&self) -> String {
        let draft = if self.is_draft { "DRAFT — " } else { "" };
        format!(
            "{draft}task {} · {} · model {} · {}",
            self.task_id, self.created_at, self.model, self.classification
        )
    }

    /// The footer sentence printed at the foot of every generated document.
    pub fn footer_sentence(&self) -> String {
        format!(
            "Produced by ARJUN for task {} at {} using {}. {}. {}",
            self.task_id, self.created_at, self.model, self.claim, self.classification_caveat()
        )
    }

    fn classification_caveat(&self) -> &'static str {
        if self.is_draft {
            "DRAFT — not verified; do not act on this document until reviewed."
        } else {
            "Verified by ARJUN's calculation engine, not by the model."
        }
    }
}

/// Builds a stamp from the document metadata the `docx` writer already
/// carries. The `claim` is a short, factual statement about provenance.
pub fn stamp_from_metadata(
    task_id: &str,
    model: &str,
    created_at: &str,
    classification: &str,
    is_draft: bool,
) -> VisibleStamp {
    VisibleStamp {
        task_id: task_id.to_string(),
        model: model.to_string(),
        created_at: created_at.to_string(),
        classification: classification.to_string(),
        is_draft,
        claim: "Content supplied by the named model; structure and required \
                fields enforced by the template engine."
            .to_string(),
    }
}

/// Returns `Some(stamp)` for the formats whose renderer is wired up, and
/// `None` for formats whose renderer is still scaffolded. Callers MUST
/// refuse to emit an unstamped document, so the absence of a stamp is a
/// loud failure rather than a silent gap.
///
/// The full set of formats ARJUN is meant to produce is documented in
/// `artifacts/mod.rs`; the ones not yet implemented are not silenced here
/// — they error.
pub fn stamp_for_format(
    format: &str,
    task_id: &str,
    model: &str,
    created_at: &str,
    classification: &str,
    is_draft: bool,
) -> Option<VisibleStamp> {
    match format {
        // The approval-note template is implemented in `docx.rs`.
        "docx" | "approval_note" => Some(stamp_from_metadata(
            task_id,
            model,
            created_at,
            classification,
            is_draft,
        )),
        // Other formats are scaffolded but not yet wired. We surface that
        // as `None` so the caller errors with a clear message instead of
        // producing an unstamped file.
        _ => None,
    }
}

/// Returns the document-XML snippet for the watermark band at the top of a
/// Word document. Intended to be prepended to the body the `docx` writer
/// builds, after the draft banner (if any) and before the classification
/// paragraph.
///
/// The snippet is the same shape as the `paragraph` helper in `docx.rs`,
/// kept inline so this module does not pull a dependency on the XML
/// formatting helpers — the goal is for the watermark to be readable in
/// isolation while a reviewer is reading it.
pub fn apply_to_docx_body(body: &str, stamp: &VisibleStamp) -> String {
    // The watermark band is two paragraphs: the header line and a faint
    // separator. We deliberately do NOT use a header/footer relationship
    // part, because not every reader renders them, and a watermark that
    // only lives in the page margin is a watermark that disappears on
    // copy-paste.
    let header = stamp.header_line();
    let footer = stamp.footer_sentence();
    format!(
        "<w:p><w:pPr><w:pStyle w:val=\"Heading2\"/></w:pPr>\
         <w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{header}</w:t></w:r></w:p>\
         <w:p><w:pPr><w:pStyle w:val=\"Heading2\"/></w:pPr>\
         <w:r><w:rPr><w:i/></w:rPr><w:t xml:space=\"preserve\">{footer}</w:t></w:r></w:p>{body}",
        header = xml_escape(&header),
        footer = xml_escape(&footer),
        body = body,
    )
}

/// Minimal XML escape. Identical in behaviour to `super::ooxml::escape` but
/// inlined so this module compiles in isolation if `ooxml` is reworked.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VisibleStamp {
        stamp_from_metadata(
            "task-42",
            "gemma-3-12b-it",
            "2026-08-29T12:00:00Z",
            "Internal",
            true,
        )
    }

    #[test]
    fn header_line_marks_drafts() {
        let s = sample();
        let h = s.header_line();
        assert!(h.starts_with("DRAFT — "), "draft must be visible in the header: {h}");
        assert!(h.contains("task-42"));
        assert!(h.contains("gemma-3-12b-it"));
    }

    #[test]
    fn header_line_omits_draft_marker_for_verified() {
        let mut s = sample();
        s.is_draft = false;
        assert!(!s.header_line().starts_with("DRAFT"));
    }

    #[test]
    fn footer_sentence_mentions_the_model_and_task() {
        let s = sample();
        let f = s.footer_sentence();
        assert!(f.contains("task-42"));
        assert!(f.contains("gemma-3-12b-it"));
    }

    #[test]
    fn apply_to_docx_body_keeps_the_original_body() {
        let s = sample();
        let out = apply_to_docx_body("<w:p>original</w:p>", &s);
        assert!(out.contains("original"));
        assert!(out.contains("DRAFT"));
        assert!(out.contains("task-42"));
    }

    #[test]
    fn apply_to_docx_body_xml_escapes_unsafe_characters() {
        // A classification label that contains a `<` would otherwise
        // break the document XML. The watermark must escape it.
        let stamp = VisibleStamp {
            task_id: "<malicious>".to_string(),
            model: "m&m".to_string(),
            created_at: "2026-08-29T12:00:00Z".to_string(),
            classification: "Internal".to_string(),
            is_draft: false,
            claim: "x".to_string(),
        };
        let out = apply_to_docx_body("", &stamp);
        assert!(out.contains("&lt;malicious&gt;"));
        assert!(out.contains("m&amp;m"));
        // And, of course, no raw `<malicious>` survives in the output.
        assert!(!out.contains("<malicious>"));
    }

    #[test]
    fn stamp_for_format_returns_some_for_docx() {
        let s = stamp_for_format(
            "docx",
            "task-1",
            "m",
            "2026-08-29T12:00:00Z",
            "Internal",
            false,
        );
        assert!(s.is_some());
    }

    #[test]
    fn stamp_for_format_returns_none_for_unwired_format() {
        // Scaffolds the rule: a renderer that is not yet wired must
        // return `None` so callers error loudly rather than silently
        // emit an unstamped file.
        let s = stamp_for_format(
            "pdf",
            "task-1",
            "m",
            "2026-08-29T12:00:00Z",
            "Internal",
            false,
        );
        assert!(s.is_none());
    }

    #[test]
    fn stamp_serialization_round_trips() {
        let s = sample();
        let json = serde_json::to_string(&s).unwrap();
        let back: VisibleStamp = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }
}

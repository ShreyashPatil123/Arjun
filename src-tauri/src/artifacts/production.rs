//! Producing an artifact: compose, render, re-open, verify — and revise rather
//! than overwrite.
//!
//! Three PS requirements meet here, and they only work together.
//!
//! **Step 29 — the renderer's complaints go back to the model.** When a template
//! refuses because a required field is missing, that refusal is not an error
//! message for a log. It is the most useful thing that could be said to the
//! model: *this exact field, by name, was not supplied*. So [`produce`] hands it
//! straight back and asks again. This is the loop `altr-oss` demonstrated, and
//! it is why a small model can hit a document schema it would otherwise miss.
//!
//! **Step 30 — the produced file is re-opened.** Not the code that wrote it, the
//! file. A bug between the template and the ZIP would pass every test of the
//! template and still produce something Word refuses.
//!
//! **Step 32 — a correction creates a new revision.** Attempt two is written to
//! its own path. Nothing overwrites anything. The refused attempt stays on disk
//! with the reason it was refused, because "what did it get wrong the first
//! time" is a question a reviewer of an AI system will ask, and an overwriting
//! pipeline destroys the only evidence that answers it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::docx::{self, DocumentCheck, DocumentMetadata, FieldSpec};
use super::verifier::{self, Evidence, Grounding, VerificationReport};

/// How cool the model samples when it is filling in a document template.
///
/// Not a tuning knob. `altr-oss` reported that sampling cooler makes small
/// models dramatically more reliable at producing complete, schema-correct
/// documents, and that matches what this loop is for: the reasoning already
/// happened in an earlier, unconstrained turn. This turn is transcription into
/// fields, and creativity in transcription is just a different word for drift.
pub const COMPOSITION_TEMPERATURE: f32 = 0.3;

/// How many times the loop will ask the model to fix its own output.
///
/// Three, because a model that has been told the exact missing field twice and
/// still has not supplied it is not going to on the fourth ask — it is missing
/// the information, not the instruction. At that point the honest outcome is a
/// refusal a person can act on.
pub const MAX_ATTEMPTS: usize = 3;

/// What the model is being asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionRequest {
    pub template: String,
    /// Fields the template will accept, with their headings — so the model is
    /// told the shape rather than left to infer it.
    pub fields: Vec<CompositionField>,
    /// Sampling temperature. Carried on the request so an implementation cannot
    /// quietly sample hot.
    pub temperature: f32,
    /// On a retry: exactly what the renderer or the verifier objected to last
    /// time. Empty on the first attempt.
    pub corrections: Vec<String>,
    /// Which attempt this is, from 1.
    pub attempt: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionField {
    pub key: String,
    pub heading: String,
    pub required: bool,
}

impl CompositionField {
    fn from_spec(spec: &FieldSpec) -> Self {
        Self {
            key: spec.key.to_string(),
            heading: spec.heading.to_string(),
            required: spec.required,
        }
    }
}

/// Whatever fills in a template — a model in production, a fixture in tests.
pub trait ContentSource {
    /// Returns field values for the request, or explains why it cannot.
    fn compose(&mut self, request: &CompositionRequest) -> Result<BTreeMap<String, String>, String>;
}

/// One attempt at producing the artifact, kept whether or not it worked.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Revision {
    /// From 1.
    pub number: usize,
    /// Where it was written. `None` when composition failed before rendering.
    pub path: Option<PathBuf>,
    /// Why this revision was superseded. `None` on the one that stood.
    pub superseded_because: Option<String>,
}

/// What producing an artifact came to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProductionOutcome {
    /// The revision that stands. `None` when every attempt failed.
    pub artifact: Option<PathBuf>,
    /// Every attempt, in order, including the ones that were superseded.
    pub revisions: Vec<Revision>,
    /// The check on the file that stands.
    pub check: Option<DocumentCheck>,
    /// The verification of the text that stands.
    pub verification: Option<VerificationReport>,
    /// Set when nothing was produced, in the words a person can act on.
    pub failure: Option<String>,
}

impl ProductionOutcome {
    pub fn succeeded(&self) -> bool {
        self.artifact.is_some()
    }

    /// Whether the artifact may be presented as finished rather than as a draft.
    pub fn is_ready(&self) -> bool {
        self.verification.as_ref().is_some_and(|v| v.is_ready())
            && self.check.as_ref().is_some_and(|c| c.is_sound())
    }
}

/// `note.docx` at revision 2 becomes `note.r2.docx`.
///
/// The revision is in the filename rather than a sidecar record so that a
/// directory full of artifacts is self-describing to somebody who found it
/// without this application.
fn revision_path(base: &Path, revision: usize) -> PathBuf {
    let stem = base.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let extension = base.extension().map(|e| e.to_string_lossy().to_string());

    let name = match extension {
        Some(ext) => format!("{stem}.r{revision}.{ext}"),
        None => format!("{stem}.r{revision}"),
    };

    base.with_file_name(name)
}

/// Assembles the draft text that gets verified.
///
/// Verification runs on what the document *says*, not on its XML, so the fields
/// are joined in reading order. A citation the model put in the findings section
/// counts exactly as much as one in the recommendation.
fn draft_text(template: &[FieldSpec], content: &BTreeMap<String, String>) -> String {
    template
        .iter()
        .filter_map(|field| content.get(field.key).map(|v| v.as_str()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Produces an artifact, correcting the model against the renderer's own
/// objections, and writing each attempt as its own revision.
pub fn produce(
    base_path: &Path,
    template_name: &str,
    source: &mut dyn ContentSource,
    metadata: &DocumentMetadata,
    evidence: &Evidence<'_>,
) -> ProductionOutcome {
    let Some(template) = docx::template_for(template_name) else {
        return ProductionOutcome {
            artifact: None,
            revisions: Vec::new(),
            check: None,
            verification: None,
            failure: Some(format!("There is no {template_name:?} template.")),
        };
    };

    let fields: Vec<CompositionField> = template.iter().map(CompositionField::from_spec).collect();
    let mut revisions = Vec::new();
    let mut corrections: Vec<String> = Vec::new();

    for attempt in 1..=MAX_ATTEMPTS {
        let request = CompositionRequest {
            template: template_name.to_string(),
            fields: fields.clone(),
            temperature: COMPOSITION_TEMPERATURE,
            corrections: corrections.clone(),
            attempt,
        };

        let content = match source.compose(&request) {
            Ok(content) => content,
            Err(why) => {
                revisions.push(Revision {
                    number: attempt,
                    path: None,
                    superseded_because: Some(format!("the content could not be composed: {why}")),
                });
                corrections = vec![why];
                continue;
            }
        };

        // The draft's standing decides how the document presents itself, so it
        // is settled before the file is written rather than stamped on after.
        let verification = verifier::verify(&draft_text(template, &content), evidence);
        let stamped = DocumentMetadata { is_draft: !verification.is_ready(), ..metadata.clone() };

        let path = revision_path(base_path, attempt);
        if let Err(error) = docx::write_document(&path, template_name, &content, &stamped) {
            // The renderer's refusal names the exact fields. Handing that back
            // is the whole point of the loop.
            revisions.push(Revision {
                number: attempt,
                path: None,
                superseded_because: Some(error.message.clone()),
            });
            corrections = if error.missing.is_empty() {
                vec![error.message]
            } else {
                error
                    .missing
                    .iter()
                    .map(|field| {
                        format!(
                            "The {field:?} field was required and was not supplied. Write it in \
                             full; do not leave it blank and do not fold it into another field."
                        )
                    })
                    .collect()
            };
            continue;
        }

        // Check the file, not the code that wrote it.
        let check = docx::check_document(&path, template_name);
        if !check.is_sound() {
            revisions.push(Revision {
                number: attempt,
                path: Some(path),
                superseded_because: Some(format!(
                    "the produced file did not survive being re-opened: {}",
                    check.problems.join("; ")
                )),
            });
            corrections = check.problems.clone();
            continue;
        }

        revisions.push(Revision { number: attempt, path: Some(path.clone()), superseded_because: None });

        return ProductionOutcome {
            artifact: Some(path),
            revisions,
            check: Some(check),
            verification: Some(verification),
            failure: None,
        };
    }

    let failure = revisions
        .last()
        .and_then(|r| r.superseded_because.clone())
        .unwrap_or_else(|| "the artifact could not be produced".to_string());

    ProductionOutcome {
        artifact: None,
        revisions,
        check: None,
        verification: None,
        failure: Some(format!(
            "After {MAX_ATTEMPTS} attempts the artifact was still not usable. Last problem: \
             {failure}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::index::{Retrieval, SearchResult};
    use crate::orchestrator::calculation::CalculationRecord;
    use crate::policy::Classification;

    fn metadata() -> DocumentMetadata {
        DocumentMetadata {
            task_id: "task-7".into(),
            created_at: "2026-08-26T10:00:00Z".into(),
            model: "Qwen2.5-7B-Instruct".into(),
            classification: "Inspection report".into(),
            is_draft: false,
        }
    }

    fn passage(id: &str, text: &str) -> SearchResult {
        SearchResult {
            chunk_id: id.to_string(),
            document_sha256: "sop".into(),
            document_name: "Maintenance SOP".into(),
            text: text.to_string(),
            page: 4,
            section_path: vec!["4.2 Wall Thickness".into()],
            classification: Classification::Internal,
            score: -1.0,
            retrieval: Retrieval::Keyword,
        }
    }

    /// A source that supplies whatever it is told to, one scripted reply per
    /// attempt, and records the requests it saw.
    struct Scripted {
        replies: Vec<Result<BTreeMap<String, String>, String>>,
        seen: Vec<CompositionRequest>,
    }

    impl Scripted {
        fn new(replies: Vec<Result<BTreeMap<String, String>, String>>) -> Self {
            Self { replies, seen: Vec::new() }
        }
    }

    impl ContentSource for Scripted {
        fn compose(
            &mut self,
            request: &CompositionRequest,
        ) -> Result<BTreeMap<String, String>, String> {
            self.seen.push(request.clone());
            if self.replies.is_empty() {
                return Err("the fixture ran out of replies".to_string());
            }
            self.replies.remove(0)
        }
    }

    fn complete() -> BTreeMap<String, String> {
        let mut content = BTreeMap::new();
        content.insert("title".into(), "Approval note — PV-2201".into());
        content.insert("recipient".into(), "M. Rao, Maintenance".into());
        content.insert("subject".into(), "Thickness below minimum".into());
        content.insert("findings".into(), "Measured 8.2 mm against 9.0 mm [E1].".into());
        content.insert("recommendation".into(), "Replace within 90 days [E1].".into());
        content.insert("references".into(), "Maintenance SOP rev C, page 4 [E1].".into());
        content.insert("assumptions".into(), "None.".into());
        content
    }

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn a_first_attempt_that_works_produces_revision_one() {
        let dir = temp();
        let passages = [passage("c1", "Minimum wall thickness is 9.0 mm; measured 8.2 mm.")];
        let calculations: [CalculationRecord; 0] = [];
        let evidence =
            Evidence { grounding: Grounding::OrganisationRecord, passages: &passages, calculations: &calculations, unread_pages: &[] };

        let mut source = Scripted::new(vec![Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert_eq!(outcome.revisions.len(), 1);
        assert!(outcome.is_ready());
        assert!(outcome.artifact.as_ref().unwrap().ends_with("note.r1.docx"));
    }

    /// The renderer's refusal names the exact field. That is what goes back.
    #[test]
    fn a_missing_field_is_named_back_to_the_model_and_the_retry_succeeds() {
        let dir = temp();
        let passages = [passage("c1", "Minimum wall thickness is 9.0 mm; measured 8.2 mm.")];
        let calculations: [CalculationRecord; 0] = [];
        let evidence =
            Evidence { grounding: Grounding::OrganisationRecord, passages: &passages, calculations: &calculations, unread_pages: &[] };

        let mut incomplete = complete();
        incomplete.remove("recommendation");

        let mut source = Scripted::new(vec![Ok(incomplete), Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert_eq!(outcome.revisions.len(), 2);

        let second = &source.seen[1];
        assert_eq!(second.attempt, 2);
        assert_eq!(second.corrections.len(), 1);
        assert!(second.corrections[0].contains("\"recommendation\""));
        assert!(second.corrections[0].contains("do not leave it blank"));
    }

    /// A pipeline that overwrote its first attempt would destroy the only
    /// evidence of what the model got wrong.
    #[test]
    fn a_correction_writes_a_new_revision_and_leaves_the_first_alone() {
        let dir = temp();
        let passages = [passage("c1", "SOP text.")];
        let calculations: [CalculationRecord; 0] = [];
        let evidence =
            Evidence { grounding: Grounding::OrganisationRecord, passages: &passages, calculations: &calculations, unread_pages: &[] };

        // First attempt renders but cites a passage that was never retrieved,
        // so it stands as a draft; the file still exists at r1.
        let mut source = Scripted::new(vec![Ok(complete()), Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        let artifact = outcome.artifact.clone().unwrap();
        assert!(artifact.exists());
        assert!(artifact.to_string_lossy().contains(".r1."));

        // Producing again against the same base path does not touch r1.
        let mut second = Scripted::new(vec![Ok(complete())]);
        let again = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut second,
            &metadata(),
            &evidence,
        );
        assert!(artifact.exists(), "the first revision must survive a second production");
        assert_eq!(again.artifact.unwrap(), artifact);
    }

    #[test]
    fn revision_numbering_survives_a_path_with_no_extension() {
        assert!(revision_path(Path::new("/a/note.docx"), 2).ends_with("note.r2.docx"));
        assert!(revision_path(Path::new("/a/note"), 3).ends_with("note.r3"));
    }

    /// Three refusals is where asking again stops being useful.
    #[test]
    fn a_model_that_never_supplies_the_field_fails_in_words_a_person_can_act_on() {
        let dir = temp();
        let calculations: [CalculationRecord; 0] = [];
        let evidence = Evidence { grounding: Grounding::OrganisationRecord, passages: &[], calculations: &calculations, unread_pages: &[] };

        let mut incomplete = complete();
        incomplete.remove("references");

        let mut source = Scripted::new(vec![
            Ok(incomplete.clone()),
            Ok(incomplete.clone()),
            Ok(incomplete),
        ]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(!outcome.succeeded());
        assert_eq!(outcome.revisions.len(), MAX_ATTEMPTS);
        let failure = outcome.failure.unwrap();
        assert!(failure.contains("3 attempts"));
        assert!(failure.contains("references"));
    }

    #[test]
    fn a_source_that_cannot_compose_is_asked_again_with_its_own_reason() {
        let dir = temp();
        let passages = [passage("c1", "SOP text.")];
        let calculations: [CalculationRecord; 0] = [];
        let evidence =
            Evidence { grounding: Grounding::OrganisationRecord, passages: &passages, calculations: &calculations, unread_pages: &[] };

        let mut source =
            Scripted::new(vec![Err("the context window was exceeded".into()), Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(outcome.succeeded());
        assert_eq!(source.seen[1].corrections, vec!["the context window was exceeded"]);
        assert_eq!(outcome.revisions[0].path, None);
    }

    /// The reasoning turn already happened. This one is transcription.
    #[test]
    fn every_composition_request_carries_the_cool_sampling_temperature() {
        let dir = temp();
        let calculations: [CalculationRecord; 0] = [];
        let evidence = Evidence { grounding: Grounding::OrganisationRecord, passages: &[], calculations: &calculations, unread_pages: &[] };

        let mut source = Scripted::new(vec![Ok(complete())]);
        let _ = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert_eq!(source.seen[0].temperature, COMPOSITION_TEMPERATURE);
        assert_eq!(source.seen[0].temperature, 0.3);
    }

    /// The model is told the shape rather than left to infer it.
    #[test]
    fn the_request_names_every_field_the_template_will_accept() {
        let dir = temp();
        let calculations: [CalculationRecord; 0] = [];
        let evidence = Evidence { grounding: Grounding::OrganisationRecord, passages: &[], calculations: &calculations, unread_pages: &[] };

        let mut source = Scripted::new(vec![Ok(complete())]);
        let _ = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        let keys: Vec<&str> = source.seen[0].fields.iter().map(|f| f.key.as_str()).collect();
        assert!(keys.contains(&"findings"));
        assert!(keys.contains(&"assumptions"));
        assert!(source.seen[0].fields.iter().any(|f| f.key == "calculation" && !f.required));
    }

    /// An unverifiable draft is still produced — marked, not withheld.
    #[test]
    fn a_draft_that_fails_verification_is_still_written_and_says_so() {
        let dir = temp();
        let calculations: [CalculationRecord; 0] = [];
        // No passages retrieved, so every citation is to something that was
        // never read.
        let evidence = Evidence { grounding: Grounding::OrganisationRecord, passages: &[], calculations: &calculations, unread_pages: &[] };

        let mut source = Scripted::new(vec![Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(outcome.succeeded());
        assert!(!outcome.is_ready());

        let check = docx::check_document(&outcome.artifact.unwrap(), "approval_note");
        assert!(check.sections[0].contains("DRAFT"));
    }

    /// The loop corrects against the *validator* too, not only the renderer:
    /// a placeholder renders perfectly well and is still not a document.
    #[test]
    fn a_placeholder_supersedes_the_revision_and_the_retry_stands() {
        let dir = temp();
        let passages = [passage("c1", "Minimum wall thickness is 9.0 mm; measured 8.2 mm.")];
        let calculations: [CalculationRecord; 0] = [];
        let evidence =
            Evidence { grounding: Grounding::OrganisationRecord, passages: &passages, calculations: &calculations, unread_pages: &[] };

        let mut lazy = complete();
        lazy.insert("recommendation".into(), "TBD — pending review [E1].".into());

        let mut source = Scripted::new(vec![Ok(lazy), Ok(complete())]);
        let outcome = produce(
            &dir.path().join("note.docx"),
            "approval_note",
            &mut source,
            &metadata(),
            &evidence,
        );

        assert!(outcome.succeeded(), "{:?}", outcome.failure);
        assert_eq!(outcome.revisions.len(), 2);

        // The refused attempt stays on disk with the reason it was refused.
        let first = &outcome.revisions[0];
        assert!(first.path.as_ref().unwrap().exists());
        assert!(first.superseded_because.as_ref().unwrap().contains("placeholder"));

        // And the reason went back to the model.
        assert!(source.seen[1].corrections.iter().any(|c| c.contains("placeholder")));

        assert!(outcome.artifact.as_ref().unwrap().ends_with("note.r2.docx"));
    }

    #[test]
    fn an_unknown_template_fails_before_asking_the_model_anything() {
        let dir = temp();
        let calculations: [CalculationRecord; 0] = [];
        let evidence = Evidence { grounding: Grounding::OrganisationRecord, passages: &[], calculations: &calculations, unread_pages: &[] };

        let mut source = Scripted::new(vec![]);
        let outcome =
            produce(&dir.path().join("x.docx"), "invoice", &mut source, &metadata(), &evidence);

        assert!(!outcome.succeeded());
        assert!(source.seen.is_empty());
        assert!(outcome.failure.unwrap().contains("no \"invoice\" template"));
    }
}

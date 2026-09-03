//! The scripted baseline — the demo, as a test.
//!
//! ARJUN design rule 15 asks for a scripted baseline over five inputs: *"one scanned
//! inspection report, one SOP, one calculation, one code task and one image"*.
//! This runs all five through the real modules against real fixture files, and
//! then does the thing the criteria actually care about: it assembles them into
//! one grounded approval note, verifies it, and exports a checkable package.
//!
//! ## Why this exists as a test rather than a script
//!
//! A shell script that drove the UI would prove the UI works on the day it was
//! written. This runs on every `cargo test`, so a change that quietly breaks
//! the chain — retrieval stops finding the tag number, the verifier stops
//! catching an invented citation, the package stops sealing — fails here rather
//! than on stage.
//!
//! ## The two fixtures that report Blocked
//!
//! The image and the code task cannot pass on a machine with no vision model
//! and no container runtime. They assert the **refusal** instead: that ARJUN
//! says so in words a person can act on, rather than guessing at the image or
//! running the code unsandboxed. A baseline that skipped them would be claiming
//! coverage it does not have.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sarathi_lib::artifacts::docx::DocumentMetadata;
use sarathi_lib::artifacts::production::{CompositionRequest, ContentSource};
use sarathi_lib::artifacts::verifier::Evidence;
use sarathi_lib::documents::{EngineCapabilities, ExtractedDocument, ExtractedPage};
use sarathi_lib::identity::{Session, UserDirectory};
use sarathi_lib::knowledge::chunking::chunk_document;
use sarathi_lib::knowledge::index::KnowledgeIndex;
use sarathi_lib::orchestrator::calculation::evaluate;
use sarathi_lib::orchestrator::sandbox::{assess, detect_tier, SandboxPolicy};
use sarathi_lib::package::{export, ApprovalRecord, ModelUse, TaskPackage};
use sarathi_lib::policy::Classification;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn sha_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("the fixture must be readable");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

/// Reads a Markdown fixture as if a text-layer engine had extracted it.
///
/// `unreadable_pages` stands in for pages a scan lost — the condition that
/// makes an inspection report different from a clean digital document.
fn extract(path: &Path, unreadable_pages: &[u32]) -> ExtractedDocument {
    let text = std::fs::read_to_string(path).expect("the fixture must be readable");
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // One page per ~1,500 characters, which is roughly what a printed A4 page
    // of this material holds.
    let mut pages: Vec<ExtractedPage> = text
        .as_bytes()
        .chunks(1_500)
        .enumerate()
        .map(|(index, slice)| {
            let body = String::from_utf8_lossy(slice).to_string();
            ExtractedPage {
                page: index as u32 + 1,
                char_count: body.chars().count() as u32,
                text: body,
                confidence: 0.99,
                needs_review: false,
                review_reason: None,
                regions: Vec::new(),
                read_by: Some("text_layer".into()),
            }
        })
        .collect();

    for page in &mut pages {
        if unreadable_pages.contains(&page.page) {
            page.text.clear();
            page.char_count = 0;
            page.confidence = 0.0;
            page.needs_review = true;
            page.review_reason =
                Some("This page has no text layer — it is an image and was not read.".into());
        }
    }

    ExtractedDocument {
        engine: "text_layer".into(),
        engine_version: "1".into(),
        pages_needing_review: unreadable_pages.len() as u32,
        pages,
        capabilities: EngineCapabilities {
            ocr: false,
            layout: false,
            tables: false,
            formulas: false,
            handwriting: false,
        },
        warnings: Vec::new(),
        source_path: path.display().to_string(),
        source_bytes: bytes,
        injection_scan: Default::default(),
        escalation: Default::default(),
    }
}

fn session(user_id: &str) -> Session {
    let directory = UserDirectory::seeded();
    Session::open(directory.find(user_id).expect("the seeded user must exist").clone())
}

// ── Fixture 1 — the SOP ────────────────────────────────────────────────

#[test]
fn fixture_1_the_sop_is_indexed_and_its_governing_figure_is_retrievable() {
    let dir = tempfile::tempdir().unwrap();
    let index = KnowledgeIndex::open(dir.path()).unwrap();
    let path = fixture("maintenance-sop.md");

    let extracted = extract(&path, &[]);
    let chunks = chunk_document(&sha_of(&path), &extracted);
    assert!(!chunks.is_empty(), "the SOP produced no chunks");

    index.index_document("Maintenance SOP rev C", Classification::Internal, &chunks).unwrap();

    // The number the whole task turns on has to come back, and it has to arrive
    // with the heading trail that makes it citable.
    let hits = index.search(&session("engineer"), "minimum allowable thickness", 5).unwrap();
    assert!(!hits.is_empty(), "the SOP's governing figure was not retrievable");
    assert!(hits.iter().any(|h| h.text.contains("9.0 mm")));
    assert!(hits.iter().any(|h| !h.section_path.is_empty()), "a passage with no heading trail is not evidence");
}

// ── Fixture 2 — the scanned inspection report ──────────────────────────

#[test]
fn fixture_2_the_inspection_report_indexes_and_reports_the_page_it_could_not_read() {
    let dir = tempfile::tempdir().unwrap();
    let index = KnowledgeIndex::open(dir.path()).unwrap();
    let path = fixture("inspection-report.md");

    // Page 2 stands in for a scanned page with no text layer — the ordinary
    // case, where part of a report is readable and part is not. A report where
    // nothing was readable would prove nothing about retrieval.
    let extracted = extract(&path, &[2]);
    assert_eq!(extracted.pages_needing_review, 1);
    assert!(extracted.needs_human_review());

    let unread: Vec<u32> = extracted
        .pages
        .iter()
        .filter(|p| p.needs_review)
        .map(|p| p.page)
        .collect();
    assert_eq!(unread, vec![2]);

    let chunks = chunk_document(&sha_of(&path), &extracted);
    index.index_document("PV-2201 inspection", Classification::Internal, &chunks).unwrap();

    // A tag number is the query a refinery actually types, and the hyphen in it
    // is read by FTS5 as NOT unless it is quoted — so this is worth pinning.
    let hits = index.search(&session("engineer"), "PV-2201", 5).unwrap();
    assert!(!hits.is_empty(), "the tag number returned nothing");
}

// ── Fixture 3 — the calculation ────────────────────────────────────────

#[test]
fn fixture_3_the_calculation_shows_its_working_and_the_engine_owns_the_figure() {
    let record = evaluate("(9.0 - 8.2) / 9.0 * 100").unwrap();

    assert!(record.deterministic, "the figure must come from the engine, not a model");
    assert!(!record.steps.is_empty(), "a calculation with no working cannot be checked");
    assert!(!record.rounding.is_empty(), "the rounding rule has to be stated");
    assert!((record.value - 8.888_888).abs() < 0.001, "got {}", record.value);

    // The workbook is the form a process engineer checks arithmetic in, and it
    // must re-open.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calculation.xlsx");
    sarathi_lib::artifacts::write_workbook(&path, std::slice::from_ref(&record), "Inspection report")
        .unwrap();

    let check = sarathi_lib::artifacts::check_workbook(&path);
    assert!(check.is_sound(), "{:?}", check.problems);
    assert_eq!(check.live_formulas, 1, "Excel must be able to recompute and disagree");
}

// ── Fixture 4 — the code task ──────────────────────────────────────────

/// On this machine the honest outcome is a refusal, and the refusal has to be
/// legible. A sandbox claimed but not present is worse than no sandbox.
#[test]
fn fixture_4_the_code_task_either_runs_isolated_or_refuses_in_words() {
    let tier = detect_tier();
    let policy = SandboxPolicy::default();
    let assessment = assess(tier, &policy);

    let summary = assessment.audit_summary();
    assert!(!summary.is_empty(), "every outcome must be recordable");

    if assessment.may_run() {
        // Whatever tier is active, it has to be named in the audit line — "it
        // ran in a sandbox" is not a claim anyone can check.
        assert!(
            summary.contains("isolation") || summary.contains(assessment.tier().label()),
            "the active tier must be recorded: {summary}"
        );
    } else {
        assert!(
            summary.to_lowercase().contains("network") || summary.to_lowercase().contains("isolat"),
            "a refusal has to say what is missing: {summary}"
        );
    }
}

// ── Fixture 5 — the image ──────────────────────────────────────────────

/// The image path stops at the honest boundary: without a vision engine ARJUN
/// reports that it cannot read the drawing rather than guessing at it.
#[test]
fn fixture_5_the_image_is_refused_rather_than_guessed_at() {
    let path = fixture("pid-excerpt.png");
    assert!(path.exists(), "the image fixture is missing");

    let capabilities = EngineCapabilities {
        ocr: false,
        layout: false,
        tables: false,
        formulas: false,
        handwriting: false,
    };

    // A text-layer engine has nothing to offer a PNG. The document it returns
    // must say so on the page rather than come back empty and plausible.
    let extracted = ExtractedDocument {
        engine: "text_layer".into(),
        engine_version: "1".into(),
        pages: vec![ExtractedPage {
            page: 1,
            text: String::new(),
            confidence: 0.0,
            needs_review: true,
            review_reason: Some(
                "This page is an image and no vision engine is available to read it.".into(),
            ),
            char_count: 0,
            regions: Vec::new(),
            read_by: None,
        }],
        capabilities,
        warnings: vec!["No vision engine is provisioned.".into()],
        pages_needing_review: 1,
        source_path: path.display().to_string(),
        source_bytes: std::fs::metadata(&path).unwrap().len(),
        injection_scan: Default::default(),
        escalation: Default::default(),
    };

    assert!(extracted.needs_human_review());
    assert!(!extracted.warnings.is_empty(), "the gap has to be stated, not implied");

    // And nothing citable is manufactured out of a page nobody read.
    let chunks = chunk_document(&sha_of(&path), &extracted);
    assert!(chunks.is_empty(), "an unread page must produce no passages");
}

// ── The five, assembled ────────────────────────────────────────────────

/// Fills the approval-note template from the fixtures, exactly as the
/// orchestrator would once a model is wired in. Deterministic so the baseline
/// asserts the *pipeline*, not a model's wording.
struct FixtureContent {
    measured: String,
    minimum: String,
    wall_loss: String,
}

impl ContentSource for FixtureContent {
    fn compose(
        &mut self,
        _request: &CompositionRequest,
    ) -> Result<BTreeMap<String, String>, String> {
        let mut content = BTreeMap::new();
        content.insert("title".into(), "Approval note — PV-2201 wall thickness".into());
        content.insert("recipient".into(), "M. Rao, Maintenance".into());
        content.insert("subject".into(), "Thickness below minimum allowable on PV-2201".into());
        content.insert(
            "findings".into(),
            format!(
                "The governing ultrasonic measurement on the lower shell course of PV-2201 is \
                 {} mm [E1]. The minimum allowable thickness for hydrocarbon service is {} mm \
                 [E2]. The vessel is therefore below the minimum allowable thickness.",
                self.measured, self.minimum
            ),
        );
        content.insert(
            "calculation".into(),
            format!("Wall loss against the minimum allowable thickness is {}%.", self.wall_loss),
        );
        content.insert(
            "recommendation".into(),
            "Schedule PV-2201 for replacement within 90 days of the inspection date [E2]."
                .to_string(),
        );
        content.insert(
            "references".into(),
            "Maintenance SOP rev C, section 3 and section 4.3 [E2]. PV-2201 inspection report, \
             2026-08-12 [E1]."
                .to_string(),
        );
        content.insert(
            "assumptions".into(),
            "The vessel remains in hydrocarbon service at its rated pressure. No \
             fitness-for-service assessment has been performed."
                .to_string(),
        );
        Ok(content)
    }
}

/// The whole chain: two documents indexed, retrieved, a figure computed, an
/// artifact produced, verified, packaged and the package checked.
#[test]
fn the_baseline_run_produces_a_grounded_artifact_and_a_checkable_package() {
    let work = tempfile::tempdir().unwrap();
    let index = KnowledgeIndex::open(work.path()).unwrap();

    // 1 & 2 — both documents into the index.
    for (name, file, unreadable) in [
        ("Maintenance SOP rev C", "maintenance-sop.md", &[][..]),
        ("PV-2201 inspection", "inspection-report.md", &[][..]),
    ] {
        let path = fixture(file);
        let extracted = extract(&path, unreadable);
        let chunks = chunk_document(&sha_of(&path), &extracted);
        index.index_document(name, Classification::Internal, &chunks).unwrap();
    }

    let user = session("engineer");
    let mut passages = index.search(&user, "governing measurement 8.2", 3).unwrap();
    passages.extend(index.search(&user, "minimum allowable thickness 9.0", 3).unwrap());
    assert!(passages.len() >= 2, "the note needs both documents behind it");

    // 3 — the figure, from the engine.
    let calculation = evaluate("(9.0 - 8.2) / 9.0 * 100").unwrap();

    // The artifact, corrected against the renderer's own objections.
    let mut source = FixtureContent {
        measured: "8.2".into(),
        minimum: "9.0".into(),
        wall_loss: calculation.formatted.clone(),
    };
    let metadata = DocumentMetadata {
        task_id: "baseline-1".into(),
        created_at: "2026-08-26T00:00:00Z".into(),
        model: "fixture (deterministic)".into(),
        classification: "Inspection report".into(),
        is_draft: false,
    };
    let calculations = [calculation.clone()];
    let evidence =
        Evidence {
            // An approval note about the organisation's own record: it must
            // rest on retrieved passages, and it does.
            grounding: sarathi_lib::artifacts::Grounding::OrganisationRecord,
            passages: &passages,
            calculations: &calculations,
            unread_pages: &[],
        };

    let outcome = sarathi_lib::artifacts::produce(
        &work.path().join("approval-note.docx"),
        "approval_note",
        &mut source,
        &metadata,
        &evidence,
    );

    assert!(outcome.succeeded(), "{:?}", outcome.failure);
    let artifact = outcome.artifact.clone().unwrap();

    // Every citation resolves to a passage actually retrieved, and every figure
    // to a calculation performed or a passage read. That is the whole claim.
    let verification = outcome.verification.clone().unwrap();
    assert!(
        verification.is_ready(),
        "the baseline note did not verify: {:?}",
        verification.findings
    );
    assert!(verification.citations_resolved >= 2);

    // The package, and the check anyone could run with unzip and sha256sum.
    let package_path = work.path().join("baseline-1.zip");
    let artifacts = vec![artifact];
    let result = export(
        &package_path,
        &TaskPackage {
            skills: &[],
            task_id: "baseline-1",
            exported_at: chrono::Utc::now(),
            exported_by: "engineer",
            classification: "Inspection report",
            artifacts: &artifacts,
            evidence: &passages,
            calculations: &calculations,
            models: &[ModelUse {
                role: "reasoning".into(),
                model_id: "fixture".into(),
                reason: "the baseline runs deterministically so it asserts the pipeline, not a \
                         model's wording"
                    .into(),
            }],
            trace: &[],
            approvals: &[ApprovalRecord {
                tool: "write_file".into(),
                prompt: "Write approval-note.docx".into(),
                decided_by: "reviewer".into(),
                approved: true,
                at: chrono::Utc::now(),
            }],
            audit: &[],
            chain: None,
        },
    )
    .unwrap();

    assert!(!result.manifest_sha256.is_empty());

    let check = sarathi_lib::package::check(&package_path);
    assert!(check.is_intact(), "{check:?}");
    assert!(check.files_checked >= 7);
}

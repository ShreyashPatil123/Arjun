//! The task package — everything needed to check a piece of work, in one file.
//!
//! ARJUN design rule 33 asks for an export carrying *"the artifact, the evidence map, the
//! calculation record, model metadata, the execution trace, the approval record
//! and hashes"*. The reason it matters is narrower than it sounds, and worth
//! stating plainly: an AI system that produces a document nobody can audit has
//! produced a liability. The package is what turns "the assistant wrote this"
//! into a claim somebody can check six months later, after the machine has been
//! reimaged and the person who ran the task has left.
//!
//! ## The package must be checkable without this application
//!
//! It is a plain ZIP of JSON and the artifact itself. Every entry's SHA-256 is
//! recorded in `manifest.json`, and the manifest's own SHA-256 is returned to
//! the caller to be recorded elsewhere — in the audit log, or written down.
//! Anyone with `unzip` and `sha256sum` can verify the whole thing; nothing in
//! the check requires ARJUN to be installed, or trusted.
//!
//! That is deliberate. A tamper check that only the tool being audited can
//! perform is not a tamper check.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::audit::{AuditEntry, ChainVerification};
use crate::skills::SkillUse;
use crate::knowledge::index::SearchResult;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::executor::StepOutcome;

/// Which model did what, and why it was the one chosen.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelUse {
    pub role: String,
    pub model_id: String,
    /// The router's own words. ARJUN design rule 20 asks that the reason be recorded, not
    /// just the choice.
    pub reason: String,
}

/// One approval, as it was given.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRecord {
    pub tool: String,
    /// What the approver was shown — not a summary written afterwards.
    pub prompt: String,
    pub decided_by: String,
    pub approved: bool,
    pub at: DateTime<Utc>,
}

/// Everything that goes into a package.
pub struct TaskPackage<'a> {
    pub task_id: &'a str,
    pub exported_at: DateTime<Utc>,
    pub exported_by: &'a str,
    /// The classification of the work, carried on the package itself so the
    /// file can be handled correctly by somebody who has not opened it.
    pub classification: &'a str,
    /// Files produced by the task. Read from disk and copied in whole.
    pub artifacts: &'a [std::path::PathBuf],
    pub evidence: &'a [SearchResult],
    pub calculations: &'a [CalculationRecord],
    pub models: &'a [ModelUse],
    /// Skills the run loaded, with the hash and version of each.
    ///
    /// On the manifest rather than only in a part file, so that checking which
    /// instructions a task was working from does not require unpacking it. A
    /// skill is guidance the model was given, and "what was it told" is one of
    /// the first questions asked when an output looks wrong.
    pub skills: &'a [SkillUse],
    pub trace: &'a [StepOutcome],
    pub approvals: &'a [ApprovalRecord],
    pub audit: &'a [AuditEntry],
    pub chain: Option<&'a ChainVerification>,
}

/// One entry in the package, with the hash that proves it has not changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackagedFile {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}

/// The package's index, and the thing worth keeping a copy of.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageManifest {
    pub task_id: String,
    pub exported_at: DateTime<Utc>,
    pub exported_by: String,
    pub classification: String,
    pub files: Vec<PackagedFile>,
    /// Whether the audit chain verified at the moment of export. A package
    /// exported from a broken chain still exports — and says so, because
    /// refusing would destroy the evidence of the break.
    pub audit_chain_intact: Option<bool>,
    /// What ARJUN produced this. Recorded so a package outlives the version.
    pub produced_by: String,
    /// Skills the run loaded. Each carries its content hash and version, so a
    /// reader can tell whether the instructions a task followed are the ones
    /// installed today.
    pub skills: Vec<SkillUse>,
}

/// Serialises one part of the package, pretty-printed so a person opening the
/// package in a text editor can read it.
fn json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(value).map_err(|e| format!("could not serialise: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// The note a person finds when they open the package without context.
fn readme(manifest: &PackageManifest) -> String {
    let files: String = manifest
        .files
        .iter()
        .map(|f| format!("  {}  {}  ({} bytes)\n", f.sha256, f.name, f.bytes))
        .collect();

    format!(
        "ARJUN task package\n\
         ==================\n\n\
         Task:            {task}\n\
         Classification:  {classification}\n\
         Exported:        {at}\n\
         Exported by:     {by}\n\
         Produced by:     {produced}\n\n\
         What this is\n\
         ------------\n\
         Everything needed to check one piece of AI-assisted work: the document\n\
         that was produced, the passages it was grounded in, the calculations\n\
         behind its figures, which models were used and why, every step the\n\
         agent took, every approval a person gave, and the audit trail.\n\n\
         How to check it without ARJUN\n\
         -----------------------------\n\
         Unzip this file and run sha256sum against each entry. The hashes below\n\
         are also in manifest.json. If one disagrees, that entry has been\n\
         altered since export.\n\n\
         {files}\n\
         The manifest's own hash was recorded separately at export time. Compare\n\
         it against sha256sum of manifest.json.\n\n\
         What this does not contain\n\
         --------------------------\n\
         The source documents themselves. Only the passages actually retrieved\n\
         appear here, with their document hash and page, so the originals can be\n\
         identified without this package carrying confidential material into\n\
         wherever it is sent.\n",
        task = manifest.task_id,
        classification = manifest.classification,
        at = manifest.exported_at,
        by = manifest.exported_by,
        produced = manifest.produced_by,
    )
}

/// What the export produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub manifest: PackageManifest,
    /// The manifest's own hash. Record this somewhere outside the package —
    /// it is what makes the package tamper-evident rather than merely
    /// self-consistent.
    pub manifest_sha256: String,
}

/// Writes a task package.
pub fn export(path: &Path, package: &TaskPackage<'_>) -> Result<ExportResult, String> {
    if package.artifacts.is_empty() && package.trace.is_empty() {
        return Err(
            "There is nothing to package: the task produced no artifact and took no steps."
                .to_string(),
        );
    }

    // Collect every entry's bytes first, so the manifest can hash them all
    // before anything is written. A manifest built as we go could describe a
    // package that failed halfway.
    let mut entries: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    for artifact in package.artifacts {
        let bytes = std::fs::read(artifact)
            .map_err(|e| format!("the artifact {} could not be read: {e}", artifact.display()))?;
        let name = artifact
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "artifact".to_string());
        entries.insert(format!("artifact/{name}"), bytes);
    }

    entries.insert("evidence.json".into(), json(&package.evidence)?);
    entries.insert("calculations.json".into(), json(&package.calculations)?);
    entries.insert("models.json".into(), json(&package.models)?);
    entries.insert("trace.json".into(), json(&package.trace)?);
    entries.insert("approvals.json".into(), json(&package.approvals)?);
    entries.insert("audit.json".into(), json(&package.audit)?);
    if let Some(chain) = package.chain {
        entries.insert("audit-chain.json".into(), json(chain)?);
    }

    let files: Vec<PackagedFile> = entries
        .iter()
        .map(|(name, bytes)| PackagedFile {
            name: name.clone(),
            bytes: bytes.len(),
            sha256: sha256_hex(bytes),
        })
        .collect();

    let manifest = PackageManifest {
        task_id: package.task_id.to_string(),
        exported_at: package.exported_at,
        exported_by: package.exported_by.to_string(),
        classification: package.classification.to_string(),
        files,
        audit_chain_intact: package.chain.map(|c| c.intact),
        produced_by: format!("ARJUN {}", env!("CARGO_PKG_VERSION")),
        skills: package.skills.to_vec(),
    };

    let manifest_bytes = json(&manifest)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let readme_bytes = readme(&manifest).into_bytes();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }

    let file = std::fs::File::create(path)
        .map_err(|e| format!("the package could not be created at {}: {e}", path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options: zip::write::FileOptions<'_, ()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    let mut write = |name: &str, bytes: &[u8]| -> Result<(), String> {
        zip.start_file(name, options).map_err(|e| format!("{name}: {e}"))?;
        zip.write_all(bytes).map_err(|e| format!("{name}: {e}"))
    };

    // README first, so it is what somebody sees at the top of the listing.
    write("README.txt", &readme_bytes)?;
    write("manifest.json", &manifest_bytes)?;
    for (name, bytes) in &entries {
        write(name, bytes)?;
    }

    zip.finish().map_err(|e| format!("the package could not be closed: {e}"))?;

    Ok(ExportResult { manifest, manifest_sha256 })
}

/// What checking a package found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageCheck {
    pub opens: bool,
    pub files_checked: usize,
    /// Entries whose bytes no longer hash to what the manifest recorded.
    pub altered: Vec<String>,
    /// Entries the manifest names that are not in the package.
    pub missing: Vec<String>,
    /// Entries in the package the manifest does not name.
    pub unexpected: Vec<String>,
    pub problems: Vec<String>,
}

impl PackageCheck {
    pub fn is_intact(&self) -> bool {
        self.opens && self.altered.is_empty() && self.missing.is_empty() && self.problems.is_empty()
    }
}

/// Re-opens a package and checks every entry against the manifest.
///
/// This is the same check anyone could run with `unzip` and `sha256sum`. It
/// exists in the application for convenience, not because the package depends
/// on it.
pub fn check(path: &Path) -> PackageCheck {
    let mut problems = Vec::new();

    let Ok(file) = std::fs::File::open(path) else {
        return PackageCheck {
            opens: false,
            files_checked: 0,
            altered: Vec::new(),
            missing: Vec::new(),
            unexpected: Vec::new(),
            problems: vec![format!("{} could not be opened", path.display())],
        };
    };

    let Ok(mut archive) = zip::ZipArchive::new(file) else {
        return PackageCheck {
            opens: false,
            files_checked: 0,
            altered: Vec::new(),
            missing: Vec::new(),
            unexpected: Vec::new(),
            problems: vec![format!("{} is not a readable package", path.display())],
        };
    };

    let mut manifest_bytes = Vec::new();
    match archive.by_name("manifest.json") {
        Ok(mut entry) => {
            let _ = entry.read_to_end(&mut manifest_bytes);
        }
        Err(_) => {
            return PackageCheck {
                opens: true,
                files_checked: 0,
                altered: Vec::new(),
                missing: vec!["manifest.json".into()],
                unexpected: Vec::new(),
                problems: vec![
                    "the package has no manifest, so nothing in it can be checked".to_string(),
                ],
            }
        }
    }

    let manifest: PackageManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(e) => {
            return PackageCheck {
                opens: true,
                files_checked: 0,
                altered: Vec::new(),
                missing: Vec::new(),
                unexpected: Vec::new(),
                problems: vec![format!("the manifest could not be read: {e}")],
            }
        }
    };

    let present: Vec<String> = archive.file_names().map(|n| n.to_string()).collect();
    let mut altered = Vec::new();
    let mut missing = Vec::new();

    for recorded in &manifest.files {
        let mut bytes = Vec::new();
        match archive.by_name(&recorded.name) {
            Ok(mut entry) => {
                let _ = entry.read_to_end(&mut bytes);
                if sha256_hex(&bytes) != recorded.sha256 {
                    altered.push(recorded.name.clone());
                }
            }
            Err(_) => missing.push(recorded.name.clone()),
        }
    }

    // An entry nobody recorded is not proof of tampering — but it is something
    // a person checking the package should be told about rather than left to
    // notice.
    let named: Vec<&str> = manifest.files.iter().map(|f| f.name.as_str()).collect();
    let unexpected: Vec<String> = present
        .iter()
        .filter(|name| !name.ends_with('/'))
        .filter(|name| name.as_str() != "manifest.json" && name.as_str() != "README.txt")
        .filter(|name| !named.contains(&name.as_str()))
        .cloned()
        .collect();

    if manifest.audit_chain_intact == Some(false) {
        problems.push(
            "the audit chain was already broken when this package was exported — the trail in it \
             is not trustworthy past the break"
                .to_string(),
        );
    }

    PackageCheck {
        opens: true,
        files_checked: manifest.files.len(),
        altered,
        missing,
        unexpected,
        problems,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditKind;
    use crate::knowledge::index::Retrieval;
    use crate::orchestrator::calculation::evaluate;
    use crate::policy::Classification;

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn passage() -> SearchResult {
        SearchResult {
            chunk_id: "c1".into(),
            document_sha256: "abc123".into(),
            document_name: "Maintenance SOP".into(),
            text: "Minimum wall thickness is 9.0 mm.".into(),
            page: 4,
            section_path: vec!["4.2 Wall Thickness".into()],
            classification: Classification::Internal,
            score: -1.0,
            retrieval: Retrieval::Keyword,
        }
    }

    fn audit_entry() -> AuditEntry {
        AuditEntry {
            seq: 1,
            at: Utc::now(),
            actor: "r.iyer".into(),
            kind: AuditKind::Task,
            summary: "Task started".into(),
            detail: None,
            hash: "deadbeef".into(),
        }
    }

    fn step() -> StepOutcome {
        StepOutcome {
            tool: "search_documents".into(),
            result: "3 passages".into(),
            permitted: true,
            took_ms: 12,
        }
    }

    fn approval() -> ApprovalRecord {
        ApprovalRecord {
            tool: "write_file".into(),
            prompt: "Write approval-note.docx to the task folder".into(),
            decided_by: "s.menon".into(),
            approved: true,
            at: Utc::now(),
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        artifact: std::path::PathBuf,
        evidence: Vec<SearchResult>,
        calculations: Vec<CalculationRecord>,
        models: Vec<ModelUse>,
        trace: Vec<StepOutcome>,
        approvals: Vec<ApprovalRecord>,
        audit: Vec<AuditEntry>,
    }

    fn fixture() -> Fixture {
        let dir = temp();
        let artifact = dir.path().join("approval-note.r1.docx");
        std::fs::write(&artifact, b"pretend this is a document").unwrap();

        Fixture {
            _dir: dir,
            artifact,
            evidence: vec![passage()],
            calculations: vec![evaluate("(9.0 - 8.2) / 9.0 * 100").unwrap()],
            models: vec![ModelUse {
                role: "reasoning".into(),
                model_id: "qwen2.5-7b-instruct".into(),
                reason: "largest model meeting the planning floor that fits 6.1 GB of VRAM".into(),
            }],
            trace: vec![step()],
            approvals: vec![approval()],
            audit: vec![audit_entry()],
        }
    }

    fn package<'a>(f: &'a Fixture, artifacts: &'a [std::path::PathBuf]) -> TaskPackage<'a> {
        TaskPackage {
            skills: &[],
            task_id: "task-42",
            exported_at: Utc::now(),
            exported_by: "r.iyer",
            classification: "Inspection report",
            artifacts,
            evidence: &f.evidence,
            calculations: &f.calculations,
            models: &f.models,
            trace: &f.trace,
            approvals: &f.approvals,
            audit: &f.audit,
            chain: None,
        }
    }

    #[test]
    fn a_package_carries_every_part_the_problem_statement_asks_for() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];

        let result = export(&path, &package(&f, &artifacts)).unwrap();

        let names: Vec<&str> = result.manifest.files.iter().map(|e| e.name.as_str()).collect();
        for required in [
            "artifact/approval-note.r1.docx",
            "evidence.json",
            "calculations.json",
            "models.json",
            "trace.json",
            "approvals.json",
            "audit.json",
        ] {
            assert!(names.contains(&required), "missing {required} — have {names:?}");
        }
    }

    #[test]
    fn a_fresh_package_checks_out_intact() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];

        export(&path, &package(&f, &artifacts)).unwrap();

        let check = check(&path);
        assert!(check.is_intact(), "{check:?}");
        assert_eq!(check.files_checked, 7);
    }

    /// The point of the hashes: an altered entry is detectable afterwards.
    #[test]
    fn an_altered_entry_is_caught_by_its_hash() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];
        export(&path, &package(&f, &artifacts)).unwrap();

        // Rewrite the package with one entry's contents changed, leaving the
        // manifest exactly as it was — the tamper a naive check would miss.
        let original = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(original).unwrap();
        let mut contents: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).unwrap();
            let name = entry.name().to_string();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            if name == "artifact/approval-note.r1.docx" {
                bytes = b"a different document entirely".to_vec();
            }
            contents.push((name, bytes));
        }
        drop(archive);

        let tampered = out.path().join("tampered.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&tampered).unwrap());
        let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
        for (name, bytes) in contents {
            zip.start_file(name, options).unwrap();
            zip.write_all(&bytes).unwrap();
        }
        zip.finish().unwrap();

        let check = check(&tampered);
        assert!(!check.is_intact());
        assert_eq!(check.altered, vec!["artifact/approval-note.r1.docx"]);
    }

    #[test]
    fn the_manifest_hash_changes_when_anything_in_the_package_does() {
        let out = temp();
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];

        let first =
            export(&out.path().join("a.zip"), &package(&f, &artifacts)).unwrap().manifest_sha256;

        let changed = out.path().join("changed.docx");
        std::fs::write(&changed, b"a different document").unwrap();
        let other = vec![changed];
        let second = export(&out.path().join("b.zip"), &package(&f, &other)).unwrap().manifest_sha256;

        assert_ne!(first, second);
    }

    #[test]
    fn the_readme_tells_someone_how_to_check_it_without_arjun() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];
        export(&path, &package(&f, &artifacts)).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut readme = String::new();
        archive.by_name("README.txt").unwrap().read_to_string(&mut readme).unwrap();

        assert!(readme.contains("sha256sum"));
        assert!(readme.contains("without ARJUN"));
        assert!(readme.contains("Inspection report"));
    }

    /// The package names the source documents without carrying them.
    #[test]
    fn evidence_identifies_its_source_without_shipping_the_document() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];
        export(&path, &package(&f, &artifacts)).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut evidence = String::new();
        archive.by_name("evidence.json").unwrap().read_to_string(&mut evidence).unwrap();

        assert!(evidence.contains("abc123"));
        assert!(evidence.contains("Maintenance SOP"));
        let names: Vec<String> = archive.file_names().map(|n| n.to_string()).collect();
        assert!(!names.iter().any(|n| n.starts_with("documents/")));
    }

    /// Refusing to export would destroy the evidence of the break.
    #[test]
    fn a_broken_audit_chain_still_exports_and_the_package_says_so() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let artifacts = vec![f.artifact.clone()];

        let broken = ChainVerification {
            entries_checked: 12,
            intact: false,
            first_broken_seq: Some(7),
            detail: "row 7 does not match its recorded hash".into(),
        };
        let mut p = package(&f, &artifacts);
        p.chain = Some(&broken);

        let result = export(&path, &p).unwrap();
        assert_eq!(result.manifest.audit_chain_intact, Some(false));

        let check = check(&path);
        assert!(!check.is_intact());
        assert!(check.problems[0].contains("already broken"));
    }

    #[test]
    fn a_task_that_did_nothing_produces_no_package() {
        let out = temp();
        let f = fixture();
        let empty: Vec<std::path::PathBuf> = Vec::new();
        let mut p = package(&f, &empty);
        p.trace = &[];

        let error = export(&out.path().join("x.zip"), &p).unwrap_err();
        assert!(error.contains("nothing to package"));
    }

    #[test]
    fn a_missing_artifact_fails_before_a_package_exists() {
        let out = temp();
        let path = out.path().join("task-42.zip");
        let f = fixture();
        let absent = vec![out.path().join("was-never-written.docx")];

        let error = export(&path, &package(&f, &absent)).unwrap_err();
        assert!(error.contains("could not be read"));
        assert!(!path.exists());
    }

    #[test]
    fn a_file_that_is_not_a_package_is_reported_rather_than_panicking() {
        let out = temp();
        let path = out.path().join("nope.zip");
        std::fs::write(&path, b"not a zip").unwrap();

        let check = check(&path);
        assert!(!check.opens);
        assert!(!check.is_intact());
    }

    #[test]
    fn a_package_with_no_manifest_says_nothing_can_be_checked() {
        let out = temp();
        let path = out.path().join("bare.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
        zip.start_file("something.txt", options).unwrap();
        zip.write_all(b"hello").unwrap();
        zip.finish().unwrap();

        let check = check(&path);
        assert!(!check.is_intact());
        assert!(check.problems[0].contains("no manifest"));
    }
}

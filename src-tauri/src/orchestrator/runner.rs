//! The tools, actually doing things.
//!
//! Everything here runs only after [`super::gateway`] has permitted it, so this
//! module contains no permission checks — a second, weaker copy of a rule
//! enforced properly elsewhere is how the two drift apart and the weaker one
//! becomes the real policy.
//!
//! What it does contain is the checks the gateway *cannot* make, because they
//! depend on what is on disk at the moment of the call rather than on what the
//! model asked for: whether a file exists, whether it is text, whether it is
//! bigger than it claimed to be.
//!
//! ## Output is written for the model to read
//!
//! Every tool returns a string that goes straight back into the conversation as
//! that call's result. So results say what happened in words the model can act
//! on — "no passages matched" rather than an empty list, and "the file is
//! 40 MB, above the limit" rather than a truncated read. A result the model
//! cannot interpret costs a step and teaches it nothing.

use std::path::Path;

use super::calculation;
use super::executor::ToolRunner;
use super::sandbox::{assess, SandboxPolicy, SandboxTier};
use super::tools::{spec_for, ToolCall, ToolName};
use crate::identity::Session;
use crate::knowledge::KnowledgeIndex;

/// How many passages a search returns to the model.
///
/// Enough to answer a question, few enough to leave room for the task itself.
/// A model handed forty passages spends its attention on them rather than on
/// what it was asked to do.
const SEARCH_LIMIT: usize = 6;

/// Longest file content handed back in one read.
///
/// Below the gateway's own ceiling on purpose: the gateway stops a file from
/// exhausting memory, this stops one from exhausting the context window and
/// pushing the task's own instructions out of it.
const READ_CHARS: usize = 24_000;

pub struct LocalToolRunner<'a> {
    pub index: &'a KnowledgeIndex,
    pub session: &'a Session,
    pub sandbox_tier: SandboxTier,
    pub sandbox_policy: SandboxPolicy,
}

impl<'a> LocalToolRunner<'a> {
    pub fn new(index: &'a KnowledgeIndex, session: &'a Session) -> Self {
        Self {
            index,
            session,
            sandbox_tier: super::sandbox::detect_tier(),
            sandbox_policy: SandboxPolicy::default(),
        }
    }

    fn search(&self, call: &ToolCall) -> Result<String, String> {
        let query = call.text("query").unwrap_or_default();

        let hits = self
            .index
            .search(self.session, query, SEARCH_LIMIT)
            .map_err(|e| format!("the knowledge base could not be searched: {e}"))?;

        if hits.is_empty() {
            // Said explicitly. An empty result the model has to infer from
            // silence is how a summary ends up citing something that was never
            // found — PS Part C asks for exactly this behaviour.
            return Ok(format!(
                "No passages matched {query:?}. Nothing in the connected collections says this, \
                 so do not assert it. Either try different wording or state that no source was found."
            ));
        }

        let mut out = format!("{} passage(s) found.\n\n", hits.len());
        for (i, hit) in hits.iter().enumerate() {
            out.push_str(&format!("[{}] {}\n{}\n\n", i + 1, hit.citation(), hit.text));
        }
        Ok(out)
    }

    fn read(&self, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this read")?;

        if !path.exists() {
            return Err(format!(
                "{} does not exist. List what is in the workspace before reading from it.",
                path.display()
            ));
        }

        let bytes = std::fs::read(path).map_err(|e| format!("{} could not be read: {e}", path.display()))?;

        // Checked here rather than at the gateway because the size on disk is
        // not knowable from the call the model wrote.
        let limit = spec_for(ToolName::ReadScopedFile)
            .max_bytes
            .unwrap_or(u64::MAX);
        if bytes.len() as u64 > limit {
            return Err(format!(
                "{} is {} MB, above the {} MB read limit.",
                path.display(),
                bytes.len() / 1024 / 1024,
                limit / 1024 / 1024
            ));
        }

        let text = String::from_utf8(bytes).map_err(|_| {
            format!(
                "{} is not text. Use the document tools to read a PDF or an image.",
                path.display()
            )
        })?;

        if text.chars().count() > READ_CHARS {
            let kept: String = text.chars().take(READ_CHARS).collect();
            // Truncation is stated, never silent: a model that believes it has
            // the whole file will confidently answer from the half it got.
            return Ok(format!(
                "{kept}\n\n[This file was longer than {READ_CHARS} characters and was cut off \
                 here. What you have above is the beginning of it, not the whole thing.]"
            ));
        }

        Ok(text)
    }

    fn write(&self, call: &ToolCall, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this write")?;
        let content = call.text("content").unwrap_or_default();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not prepare {}: {e}", parent.display()))?;
        }

        std::fs::write(path, content)
            .map_err(|e| format!("{} could not be written: {e}", path.display()))?;

        Ok(format!(
            "Wrote {} byte(s) to {}.",
            content.len(),
            path.display()
        ))
    }

    fn calculate(&self, call: &ToolCall) -> Result<String, String> {
        let expression = call.text("expression").unwrap_or_default();

        match calculation::evaluate(expression) {
            Ok(record) => {
                let mut out = format!("{} = {}\n", record.expression, record.formatted);
                out.push_str("Working:\n");
                for step in &record.steps {
                    out.push_str(&format!("  {} = {}\n", step.description, step.result));
                }
                out.push_str(&format!(
                    "Rounded to {}. Use this figure exactly as written; do not recompute it.",
                    record.rounding
                ));
                Ok(out)
            }
            // Returned as an error so the model sees it as a failed call and
            // corrects the expression, rather than as a result it might quote.
            Err(problem) => Err(problem.message),
        }
    }

    fn execute_code(&self) -> Result<String, String> {
        let assessment = assess(self.sandbox_tier, &self.sandbox_policy);
        match assessment {
            super::sandbox::SandboxAssessment::Refused { reason } => Err(format!(
                "Code was not run: {reason} Nothing was executed, so no result exists — do not \
                 describe what the code would have produced."
            )),
            _ => Err(
                "Running code is not implemented yet, even though this machine could isolate it. \
                 No result exists."
                    .to_string(),
            ),
        }
    }

    fn validate(&self, path: Option<&Path>) -> Result<String, String> {
        let path = path.ok_or("no path was resolved for this check")?;

        if !path.exists() {
            return Err(format!("{} does not exist, so there is nothing to check.", path.display()));
        }

        let size = std::fs::metadata(path)
            .map_err(|e| format!("{} could not be inspected: {e}", path.display()))?
            .len();

        if size == 0 {
            return Err(format!(
                "{} exists but is empty, so it is not a usable file.",
                path.display()
            ));
        }

        Ok(format!("{} exists and holds {size} byte(s).", path.display()))
    }
}

impl ToolRunner for LocalToolRunner<'_> {
    fn run(
        &self,
        tool: ToolName,
        call: &ToolCall,
        resolved_path: Option<&Path>,
    ) -> Result<String, String> {
        match tool {
            ToolName::SearchDocuments => self.search(call),
            ToolName::ReadScopedFile => self.read(resolved_path),
            ToolName::WriteScopedFile => self.write(call, resolved_path),
            ToolName::RunCalculation => self.calculate(call),
            ToolName::ExecuteCode => self.execute_code(),
            ToolName::ValidateArtifact => self.validate(resolved_path),
            // Phase 6. Said plainly so a model does not describe a document it
            // has not produced.
            ToolName::CreateDocx | ToolName::CreateXlsx => Err(format!(
                "Producing a {} is not built yet, so no file was created and none exists.",
                if tool == ToolName::CreateDocx { "Word document" } else { "spreadsheet" }
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::{Chunk, ChunkKind};
    use crate::policy::Classification;
    use serde_json::json;
    use std::path::PathBuf;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        index: KnowledgeIndex,
        session: Session,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let index = KnowledgeIndex::open(&root).unwrap();
        index
            .index_document(
                "Maintenance SOP",
                Classification::Internal,
                &[Chunk {
                    id: "c1".into(),
                    document_sha256: "sop".into(),
                    ordinal: 0,
                    text: "Minimum acceptable wall thickness is 9.0 mm.".into(),
                    page: 4,
                    section_path: vec!["4.2 Wall Thickness".into()],
                    kind: ChunkKind::Prose,
                    char_count: 44,
                }],
            )
            .unwrap();

        Fixture {
            _dir: dir,
            root,
            index,
            session: Session::open(User::new("kiran", "Kiran", vec![Role::User])),
        }
    }

    fn runner(f: &Fixture) -> LocalToolRunner<'_> {
        LocalToolRunner {
            index: &f.index,
            session: &f.session,
            sandbox_tier: SandboxTier::JobObject,
            sandbox_policy: SandboxPolicy::default(),
        }
    }

    // ── Search ───────────────────────────────────────────────────────────

    #[test]
    fn a_search_returns_passages_with_their_citations() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::SearchDocuments,
                &ToolCall::new("search_documents", json!({ "query": "wall thickness" })),
                None,
            )
            .unwrap();

        assert!(out.contains("1 passage(s) found"));
        assert!(out.contains("Maintenance SOP"));
        assert!(out.contains("9.0 mm"));
    }

    /// PS Part C: no source found must be said, not left as silence for the
    /// model to fill in.
    #[test]
    fn finding_nothing_says_so_and_tells_the_model_not_to_assert_it() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::SearchDocuments,
                &ToolCall::new("search_documents", json!({ "query": "sasquatch" })),
                None,
            )
            .unwrap();

        assert!(out.contains("No passages matched"));
        assert!(out.contains("do not assert it"));
    }

    // ── Read ─────────────────────────────────────────────────────────────

    #[test]
    fn reading_a_file_returns_its_text() {
        let f = fixture();
        let path = f.root.join("note.txt");
        std::fs::write(&path, "Wall thickness measured at 8.2 mm.").unwrap();

        let out = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path))
            .unwrap();

        assert_eq!(out, "Wall thickness measured at 8.2 mm.");
    }

    #[test]
    fn reading_a_missing_file_says_what_to_do_instead() {
        let f = fixture();
        let missing = f.root.join("absent.txt");

        let error = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&missing))
            .unwrap_err();

        assert!(error.contains("does not exist"));
        assert!(error.contains("List what is in the workspace"));
    }

    /// A model that believes it has the whole file will confidently answer from
    /// the half it got.
    #[test]
    fn a_long_file_is_truncated_and_says_so() {
        let f = fixture();
        let path = f.root.join("long.txt");
        std::fs::write(&path, "x".repeat(READ_CHARS + 5_000)).unwrap();

        let out = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path))
            .unwrap();

        assert!(out.contains("was cut off here"));
        assert!(out.contains("not the whole thing"));
    }

    #[test]
    fn reading_a_binary_file_suggests_the_document_tools() {
        let f = fixture();
        let path = f.root.join("image.bin");
        std::fs::write(&path, [0xff, 0xfe, 0x00, 0x80]).unwrap();

        let error = runner(&f)
            .run(ToolName::ReadScopedFile, &ToolCall::new("read_scoped_file", json!({})), Some(&path))
            .unwrap_err();

        assert!(error.contains("is not text"));
        assert!(error.contains("document tools"));
    }

    // ── Write ────────────────────────────────────────────────────────────

    #[test]
    fn writing_creates_the_file_and_any_missing_directories() {
        let f = fixture();
        let path = f.root.join("out/deep/note.txt");

        let out = runner(&f)
            .run(
                ToolName::WriteScopedFile,
                &ToolCall::new("write_scoped_file", json!({ "content": "hello" })),
                Some(&path),
            )
            .unwrap();

        assert!(out.contains("Wrote 5 byte(s)"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    // ── Calculation ──────────────────────────────────────────────────────

    #[test]
    fn a_calculation_returns_the_figure_and_its_working() {
        let f = fixture();
        let out = runner(&f)
            .run(
                ToolName::RunCalculation,
                &ToolCall::new("run_calculation", json!({ "expression": "(8.2 mm - 9.0 mm) / 9.0 mm * 100" })),
                None,
            )
            .unwrap();

        assert!(out.contains("-8.889"));
        assert!(out.contains("Working:"));
        // The model must quote this figure, not produce its own.
        assert!(out.contains("do not recompute it"));
    }

    /// A bad expression comes back as a failure so the model fixes it, rather
    /// than as a result it might quote.
    #[test]
    fn an_impossible_calculation_is_an_error_not_a_result() {
        let f = fixture();
        let error = runner(&f)
            .run(
                ToolName::RunCalculation,
                &ToolCall::new("run_calculation", json!({ "expression": "8.2 mm + 9.0 kg" })),
                None,
            )
            .unwrap_err();

        assert!(error.contains("units do not match"));
    }

    // ── Things that are not built, said plainly ──────────────────────────

    /// The machine cannot isolate code, so nothing runs — and the model is told
    /// not to describe output that does not exist.
    #[test]
    fn running_code_on_a_weak_sandbox_refuses_and_forbids_inventing_output() {
        let f = fixture();
        let error = runner(&f)
            .run(
                ToolName::ExecuteCode,
                &ToolCall::new("execute_code", json!({ "language": "python", "source": "print(1)" })),
                None,
            )
            .unwrap_err();

        assert!(error.contains("Code was not run"));
        assert!(error.contains("do not describe what the code would have produced"));
    }

    #[test]
    fn an_unbuilt_document_tool_says_no_file_exists() {
        let f = fixture();
        let error = runner(&f)
            .run(
                ToolName::CreateDocx,
                &ToolCall::new("create_docx", json!({})),
                Some(&f.root.join("note.docx")),
            )
            .unwrap_err();

        assert!(error.contains("not built yet"));
        assert!(error.contains("none exists"));
    }

    // ── Validation ───────────────────────────────────────────────────────

    #[test]
    fn validating_reports_a_real_file() {
        let f = fixture();
        let path = f.root.join("note.txt");
        std::fs::write(&path, "content").unwrap();

        let out = runner(&f)
            .run(ToolName::ValidateArtifact, &ToolCall::new("validate_artifact", json!({})), Some(&path))
            .unwrap();

        assert!(out.contains("7 byte(s)"));
    }

    /// An empty file exists but is not a usable artifact, and saying "it exists"
    /// would let a task report success on nothing.
    #[test]
    fn an_empty_file_fails_validation() {
        let f = fixture();
        let path = f.root.join("empty.docx");
        std::fs::write(&path, "").unwrap();

        let error = runner(&f)
            .run(ToolName::ValidateArtifact, &ToolCall::new("validate_artifact", json!({})), Some(&path))
            .unwrap_err();

        assert!(error.contains("empty"));
    }
}

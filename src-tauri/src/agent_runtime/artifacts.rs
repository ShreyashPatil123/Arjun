//! Producing the two deliverables a run can hand to somebody.
//!
//! PS 26117: *"Output should be real deliverables, approval notes, PPT/Word/Excel
//! files, working code, calculations with steps shown, not just chat replies."*
//! [`crate::artifacts`] already renders those files. What was missing was a way
//! for a run to reach it, and `runner.rs` said so plainly rather than pretending
//! otherwise.
//!
//! ## Why the model supplies the content directly
//!
//! [`crate::artifacts::production::produce`] runs its own correction loop: ask a
//! model for the fields, render, and if the renderer objects, ask again with the
//! objection. That was the right design when the model was reachable from Rust.
//!
//! It is not any more. The model lives in the Node runtime, and a Rust-side
//! correction loop would have to call back into it — Rust asking the runtime to
//! ask the model, while the runtime waits on Rust. The agent loop already *is* a
//! correction loop: a failed render comes back as a tool error, the model reads
//! the objection and calls again. So the tool takes the fields as arguments and
//! `produce`'s loop is simply not the one used here.
//!
//! That is why `ToolSpec` for these tools already declares a `content` object
//! argument. The contract anticipated this shape.
//!
//! ## Why the workbook is not composed by the model at all
//!
//! A calculation workbook is written from what the calculation engine actually
//! computed during the run, not from what the model says it computed. The engine
//! exists precisely because a model is usually about right, and "usually" is not
//! a property a pump specification can have.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::artifacts::{check_document, check_workbook, write_document, write_workbook, DocumentMetadata};
use crate::identity::Session;
use crate::orchestrator::calculation::CalculationRecord;
use crate::orchestrator::tools::ToolCall;

use super::CallParams;

/// What kind of file a run produced, and therefore how it is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Document,
    Workbook,
    /// A note or draft. Checked for being present and non-empty, no more —
    /// there is no structure to check it against.
    Text,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Document => "Word document",
            Kind::Workbook => "Workbook",
            Kind::Text => "Text file",
        }
    }
}

/// A file this run produced, remembered so it can be re-opened afterwards.
///
/// The template is kept because a document cannot really be checked without it:
/// the check asks whether the sections the template promised are in the file,
/// and a checker that does not know which template was used can only ask
/// whether the ZIP opens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Produced {
    /// What to call it. Relative to the run's workspace, which is where it is.
    pub name: String,
    pub path: String,
    pub kind: Kind,
    pub template: Option<String>,
    /// RFC 3339, UTC.
    pub produced_at: String,
}

/// Files produced so far, keyed by run id — the same shape as the calculation
/// and evidence tables, and per run for the same reason.
pub type RunArtifacts = Arc<Mutex<HashMap<String, Vec<Produced>>>>;

/// Records a produced file against its run.
///
/// A path written twice replaces the earlier entry rather than appearing twice:
/// the model correcting a document it just produced is ordinary, and a list
/// showing one file as two deliverables would misreport what the run made.
pub fn remember(table: &RunArtifacts, run_id: &str, produced: Produced) {
    if let Ok(mut table) = table.lock() {
        let entries = table.entry(run_id.to_string()).or_default();
        if let Some(existing) = entries.iter_mut().find(|kept| kept.path == produced.path) {
            *existing = produced;
        } else {
            entries.push(produced);
        }
    }
}

/// Everything this run produced, in the order it was produced.
pub fn for_run(table: &RunArtifacts, run_id: &str) -> Vec<Produced> {
    table
        .lock()
        .ok()
        .and_then(|table| table.get(run_id).cloned())
        .unwrap_or_default()
}

/// Drops a finished run's list once its report has been written.
pub fn forget(table: &RunArtifacts, run_id: &str) {
    if let Ok(mut table) = table.lock() {
        table.remove(run_id);
    }
}

/// Builds the record of a produced file.
///
/// The name is relative to the workspace, so the name the model wrote is the
/// name that comes back — an absolute path with a UUID in it tells the person
/// reading the task nothing they wanted to know.
pub fn produced_from(
    path: &Path,
    root: Option<&Path>,
    kind: Kind,
    template: Option<String>,
) -> Produced {
    let name = root
        .and_then(|root| path.strip_prefix(root).ok())
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|| {
            Path::new(path.file_name().unwrap_or_default())
                .display()
                .to_string()
        });
    Produced {
        name,
        path: path.display().to_string(),
        kind,
        template,
        produced_at: chrono::Utc::now().to_rfc3339(),
    }
}

/// What re-opening a produced file found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactReport {
    pub name: String,
    pub path: String,
    pub kind: Kind,
    /// Carried through so a later re-check asks the same question this one did.
    /// A document checked against a different template than it was rendered
    /// from reports missing sections that were never promised.
    #[serde(default)]
    pub template: Option<String>,
    pub bytes: u64,
    /// False when the file is missing, empty, will not open, or is missing
    /// something the template promised.
    pub sound: bool,
    /// One line, in the words somebody reading the task would use.
    pub detail: String,
    pub problems: Vec<String>,
    pub produced_at: String,
}

/// Re-opens a produced file and reports what is actually in it.
///
/// PS step 30 asks that the application open the generated file locally and
/// confirm it is not corrupt and that required sections exist. Checking the
/// file rather than the code that wrote it is the whole point: a bug between
/// the template and the ZIP passes every test of the template and still
/// produces a document that opens to a page of placeholders.
pub fn check(produced: &Produced) -> ArtifactReport {
    let path = PathBuf::from(&produced.path);
    let report = |sound: bool, detail: String, problems: Vec<String>, bytes: u64| ArtifactReport {
        name: produced.name.clone(),
        path: produced.path.clone(),
        kind: produced.kind,
        template: produced.template.clone(),
        bytes,
        sound,
        detail,
        problems,
        produced_at: produced.produced_at.clone(),
    };

    let Ok(metadata) = std::fs::metadata(&path) else {
        return report(
            false,
            "The file is not where the task said it wrote it.".to_string(),
            vec!["the file does not exist".to_string()],
            0,
        );
    };
    let bytes = metadata.len();
    if bytes == 0 {
        return report(
            false,
            "The file was created but nothing was written into it.".to_string(),
            vec!["the file is empty".to_string()],
            0,
        );
    }

    match produced.kind {
        Kind::Document => {
            let template = produced.template.as_deref().unwrap_or("approval_note");
            let check = check_document(&path, template);
            let detail = if check.is_sound() {
                format!(
                    "Opens, and holds the {} section(s) the {template} template promises.",
                    check.sections.len()
                )
            } else if check.opens {
                "Opens, but does not hold everything the template promises.".to_string()
            } else {
                "Does not open as a Word document.".to_string()
            };
            report(check.is_sound(), detail, check.problems, bytes)
        }
        Kind::Workbook => {
            let check = check_workbook(&path);
            let detail = if check.is_sound() {
                format!(
                    "Opens, with {} calculation(s), {} of them live formulas Excel recomputes.",
                    check.calculations, check.live_formulas
                )
            } else if check.opens {
                "Opens, but the working in it is not sound.".to_string()
            } else {
                "Does not open as a workbook.".to_string()
            };
            report(check.is_sound(), detail, check.problems, bytes)
        }
        // Nothing to check it against beyond being there and having content,
        // and claiming more than that would be inventing a standard.
        Kind::Text => report(true, format!("Present, {bytes} byte(s)."), Vec::new(), bytes),
    }
}

/// Re-opens everything a run produced.
pub fn report_for_run(table: &RunArtifacts, run_id: &str) -> Vec<ArtifactReport> {
    for_run(table, run_id).iter().map(check).collect()
}

/// Renders a Word document from fields the model supplied.
pub fn create_docx(
    call: &CallParams,
    resolved_path: Option<&Path>,
    session: &Session,
    tool_call: &ToolCall,
) -> Result<String, String> {
    let path = resolved_path.ok_or_else(|| {
        "No path was resolved for the document, so nothing was written.".to_string()
    })?;
    let template = tool_call
        .text("template")
        .ok_or_else(|| "The document needs a template. Available templates: approval_note.".to_string())?;

    let content = fields_from(tool_call)?;

    let metadata = DocumentMetadata {
        task_id: call.run_id.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        // The model that produced the content, so a reader knows what wrote it.
        // Recorded per run by the caller; unknown here rather than guessed.
        model: call.model.clone().unwrap_or_else(|| "unrecorded".to_string()),
        classification: "Internal".to_string(),
        // Every document a run produces is a draft until a person signs it. The
        // word is printed on the page rather than only stored, so a file that
        // escapes into an inbox still says what it is.
        is_draft: true,
    };

    write_document(path, template, &content, &metadata).map_err(|error| error.message)?;

    // Re-opened and checked, not assumed. A renderer that wrote a placeholder
    // through produces a file that opens and says nothing — the failure a
    // person only finds in the meeting.
    let check = check_document(path, template);
    if !check.problems.is_empty() {
        return Err(format!(
            "{} was written but did not pass its own check: {}. Correct the content and produce it again.",
            path.display(),
            check.problems.join("; ")
        ));
    }

    Ok(format!(
        "Wrote {} from the {template} template ({} field(s)). It is marked DRAFT until somebody approves it.",
        path.display(),
        content.len()
    ))
}

/// Reads the `content` object into template fields.
///
/// Values are required to be strings. A model that supplies a number or a
/// nested object for a document field has misunderstood the template, and
/// silently stringifying it would put `{"value":9}` into an approval note.
fn fields_from(tool_call: &ToolCall) -> Result<BTreeMap<String, String>, String> {
    let object = tool_call
        .arguments
        .get("content")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            "The document's content must be an object of field name to text, for example \
             {\"title\": \"...\", \"recommendation\": \"...\"}."
                .to_string()
        })?;

    let mut fields = BTreeMap::new();
    for (key, value) in object {
        match value.as_str() {
            Some(text) => {
                fields.insert(key.clone(), text.to_string());
            }
            None => {
                return Err(format!(
                    "The field {key:?} must be text, but was {}. Supply each field as a string.",
                    match value {
                        serde_json::Value::Null => "null",
                        serde_json::Value::Bool(_) => "a boolean",
                        serde_json::Value::Number(_) => "a number",
                        serde_json::Value::Array(_) => "a list",
                        _ => "an object",
                    }
                ))
            }
        }
    }
    Ok(fields)
}

/// Writes the run's calculations into a workbook Excel can recompute.
pub fn create_xlsx(
    resolved_path: Option<&Path>,
    calculations: &Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>>,
    run_id: &str,
) -> Result<String, String> {
    let path = resolved_path.ok_or_else(|| {
        "No path was resolved for the workbook, so nothing was written.".to_string()
    })?;

    let records = calculations
        .lock()
        .map_err(|_| "the calculation record is unavailable".to_string())?
        .get(run_id)
        .cloned()
        .unwrap_or_default();

    if records.is_empty() {
        // Said as a refusal rather than an empty file: a workbook with no rows
        // looks like a calculation that produced nothing, which is a different
        // and worse claim than one that was never run.
        return Err(
            "No calculations have been run in this task, so there is nothing to put in a \
             workbook. Use run_calculation first; the workbook shows the working from those \
             calls, not figures written out again."
                .to_string(),
        );
    }

    write_workbook(path, &records, "Internal")?;

    let check = check_workbook(path);
    if !check.problems.is_empty() {
        return Err(format!(
            "{} was written but did not pass its own check: {}.",
            path.display(),
            check.problems.join("; ")
        ));
    }

    Ok(format!(
        "Wrote {} with {} calculation(s), {} of them as live formulas Excel recomputes. \
         If Excel disagrees with a figure, Excel is right and the note needs correcting.",
        path.display(),
        check.calculations,
        check.live_formulas
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::orchestrator::calculation::evaluate;
    use serde_json::json;

    fn call_params(run_id: &str) -> CallParams {
        CallParams {
            run_id: run_id.to_string(),
            tool_call_id: "tc-1".into(),
            tool: "create_docx".into(),
            args: json!({}),
            model: Some("qwen2.5-coder-7b".into()),
        }
    }

    fn author() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
    }

    #[test]
    fn a_document_is_written_and_says_it_is_a_draft() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("note.docx");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({
                "path": "note.docx",
                "template": "approval_note",
                "content": {
                    "title": "Pump seal replacement",
                    "recipient": "Maintenance Manager",
                    "subject": "P-101 mechanical seal",
                    "findings": "The seal is worn beyond the 9.0 mm limit.",
                    "recommendation": "Replace the seal at the next shutdown.",
                    "references": "Maintenance SOP p.4.",
                    "assumptions": "None."
                }
            }),
        );

        let message = create_docx(&call_params("run-1"), Some(&path), &author(), &tool_call)
            .expect("the document is written");

        assert!(path.exists());
        assert!(message.contains("DRAFT"), "{message}");
    }

    #[test]
    fn a_field_that_is_not_text_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "approval_note", "content": { "title": 9 } }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("\"title\""), "{error}");
        assert!(error.contains("a number"), "{error}");
    }

    #[test]
    fn content_that_is_not_an_object_says_what_shape_is_wanted() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "approval_note", "content": "just some text" }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("field name to text"), "{error}");
    }

    #[test]
    fn an_unknown_template_names_the_ones_that_exist() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tool_call = ToolCall::new(
            "create_docx",
            json!({ "template": "invoice", "content": { "title": "x" } }),
        );

        let error = create_docx(
            &call_params("run-1"),
            Some(&dir.path().join("note.docx")),
            &author(),
            &tool_call,
        )
        .unwrap_err();

        assert!(error.contains("approval_note"), "{error}");
    }

    #[test]
    fn a_workbook_holds_the_calculations_the_engine_actually_ran() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("working.xlsx");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
        table.lock().unwrap().insert(
            "run-1".into(),
            vec![
                evaluate("2 m * 3 m").expect("evaluates"),
                evaluate("10 kg / 2 s").expect("evaluates"),
            ],
        );

        let message = create_xlsx(Some(&path), &table, "run-1").expect("the workbook is written");

        assert!(path.exists());
        assert!(message.contains("2 calculation(s)"), "{message}");
        // The point of the workbook: Excel recomputes and may disagree.
        assert!(message.contains("Excel is right"), "{message}");
    }

    #[test]
    fn a_workbook_with_nothing_to_show_is_refused_rather_than_written_empty() {
        let dir = tempfile::tempdir().expect("temp dir");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();

        let error = create_xlsx(Some(&dir.path().join("working.xlsx")), &table, "run-1").unwrap_err();

        assert!(error.contains("run_calculation first"), "{error}");
        assert!(!dir.path().join("working.xlsx").exists());
    }

    #[test]
    fn one_runs_calculations_do_not_appear_in_anothers_workbook() {
        let dir = tempfile::tempdir().expect("temp dir");
        let table: Arc<Mutex<HashMap<String, Vec<CalculationRecord>>>> = Arc::default();
        table
            .lock()
            .unwrap()
            .insert("run-1".into(), vec![evaluate("2 m * 3 m").expect("evaluates")]);

        let error = create_xlsx(Some(&dir.path().join("other.xlsx")), &table, "run-2").unwrap_err();
        assert!(error.contains("No calculations have been run in this task"), "{error}");
    }
}

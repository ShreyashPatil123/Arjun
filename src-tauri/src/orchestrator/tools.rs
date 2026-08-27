//! What the assistant is allowed to ask for.
//!
//! A model in ARJUN cannot open a file, run a command, or reach the network. It
//! can only emit a [`ToolCall`] — a name and some arguments — and something else
//! decides whether that happens. This module is the catalogue of what may be
//! asked for; [`super::gateway`] is what decides.
//!
//! ## Every tool declares its own cost
//!
//! PS step 13 asks that each tool carry a schema, permitted inputs, permitted
//! directories, size and time limits, and whether a human has to approve it.
//! Those live on the [`ToolSpec`] rather than in the code that runs the tool,
//! so the limits can be read without reading an implementation — and so a new
//! tool cannot be added without stating them.
//!
//! ## Approval is a property of the tool, not of the moment
//!
//! Whether something needs a human is decided here, once, by what the tool can
//! do — not by how risky a particular call looks. A judgement made per-call
//! would depend on the arguments, and the arguments come from the model.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::identity::Permission;

/// Every tool ARJUN exposes. Nothing outside this list can be requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    /// Search the organisation's own documents.
    SearchDocuments,
    /// Read a named page range of a document already retrieved from.
    ///
    /// The counterpart to search, and the reason a run does not need whole
    /// documents in its context: a passage that stops mid-clause is followed by
    /// a request for the two pages around it, not for the file.
    LoadMoreEvidence,
    /// Read a file inside the task workspace.
    ReadScopedFile,
    /// Write a file inside the task workspace.
    WriteScopedFile,
    /// Arithmetic with units, done deterministically rather than by a model.
    RunCalculation,
    /// Produce a Word document from a template.
    CreateDocx,
    /// Produce a spreadsheet from a template.
    CreateXlsx,
    /// Run code in the sandbox.
    ExecuteCode,
    /// Re-open a produced file and check it is sound.
    ValidateArtifact,
}

impl ToolName {
    pub const ALL: &'static [ToolName] = &[
        ToolName::SearchDocuments,
        ToolName::LoadMoreEvidence,
        ToolName::ReadScopedFile,
        ToolName::WriteScopedFile,
        ToolName::RunCalculation,
        ToolName::CreateDocx,
        ToolName::CreateXlsx,
        ToolName::ExecuteCode,
        ToolName::ValidateArtifact,
    ];

    /// The wire name a model emits.
    pub const fn as_str(self) -> &'static str {
        match self {
            ToolName::SearchDocuments => "search_documents",
            ToolName::LoadMoreEvidence => "load_more_evidence",
            ToolName::ReadScopedFile => "read_scoped_file",
            ToolName::WriteScopedFile => "write_scoped_file",
            ToolName::RunCalculation => "run_calculation",
            ToolName::CreateDocx => "create_docx",
            ToolName::CreateXlsx => "create_xlsx",
            ToolName::ExecuteCode => "execute_code",
            ToolName::ValidateArtifact => "validate_artifact",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.as_str() == raw)
    }

    /// How the action reads in an approval prompt or an audit line.
    pub const fn describe(self) -> &'static str {
        match self {
            ToolName::SearchDocuments => "search the knowledge base",
            ToolName::LoadMoreEvidence => "read a specific page range of a document",
            ToolName::ReadScopedFile => "read a file from the task workspace",
            ToolName::WriteScopedFile => "write a file into the task workspace",
            ToolName::RunCalculation => "run a calculation",
            ToolName::CreateDocx => "produce a Word document",
            ToolName::CreateXlsx => "produce a spreadsheet",
            ToolName::ExecuteCode => "run code in the sandbox",
            ToolName::ValidateArtifact => "check a produced file",
        }
    }
}

/// One required argument, and what it has to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgumentSpec {
    pub name: &'static str,
    pub kind: ArgumentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentKind {
    Text,
    /// A path. Always checked against the task's permitted directories.
    Path,
    Integer,
    /// A nested object, validated by the tool itself rather than here.
    Object,
}

/// Everything the gateway needs to know about a tool before allowing it.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: ToolName,
    /// What the user must hold for this to be permitted at all.
    pub permission: Permission,
    pub arguments: &'static [ArgumentSpec],
    /// Whether a person has to say yes before this runs.
    ///
    /// True for anything that leaves a trace outside the task — a file, a
    /// document, an execution — and false for reading and computing, which can
    /// be undone by ignoring the result.
    pub needs_approval: bool,
    /// Largest input or output this tool may handle.
    pub max_bytes: Option<u64>,
    /// Wall-clock ceiling. Every tool has one: a tool that can hang has a
    /// budget of infinity, and a plan containing it can never be bounded.
    pub timeout: Duration,
    /// Whether the call touches a path that must be inside the workspace.
    pub scoped_to_workspace: bool,
}

/// The catalogue.
///
/// Deliberately a function over a static table rather than data loaded at
/// runtime: a tool the code does not know about cannot be enabled by editing a
/// file, which is the correct trade for the one surface a model can reach.
pub fn spec_for(name: ToolName) -> ToolSpec {
    use ArgumentKind::*;
    use Permission::*;

    match name {
        ToolName::SearchDocuments => ToolSpec {
            name,
            permission: SearchKnowledge,
            arguments: &[ArgumentSpec { name: "query", kind: Text }],
            needs_approval: false,
            max_bytes: None,
            timeout: Duration::from_secs(30),
            scoped_to_workspace: false,
        },
        ToolName::LoadMoreEvidence => ToolSpec {
            name,
            // The same permission as search, because it reads the same shelf
            // through the same clearance checks. A weaker permission here would
            // be a way to read by page number what may not be read by searching.
            permission: SearchKnowledge,
            arguments: &[
                ArgumentSpec { name: "documentSha256", kind: Text },
                ArgumentSpec { name: "fromPage", kind: Integer },
                ArgumentSpec { name: "toPage", kind: Integer },
            ],
            needs_approval: false,
            max_bytes: None,
            timeout: Duration::from_secs(30),
            scoped_to_workspace: false,
        },
        ToolName::ReadScopedFile => ToolSpec {
            name,
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "path", kind: Path }],
            needs_approval: false,
            // Large enough for a long report, small enough that one file cannot
            // fill the context window and push the task's own instructions out.
            max_bytes: Some(8 * 1024 * 1024),
            timeout: Duration::from_secs(15),
            scoped_to_workspace: true,
        },
        ToolName::WriteScopedFile => ToolSpec {
            name,
            permission: GenerateArtifact,
            arguments: &[
                ArgumentSpec { name: "path", kind: Path },
                ArgumentSpec { name: "content", kind: Text },
            ],
            needs_approval: true,
            max_bytes: Some(32 * 1024 * 1024),
            timeout: Duration::from_secs(30),
            scoped_to_workspace: true,
        },
        ToolName::RunCalculation => ToolSpec {
            name,
            permission: UseModel,
            arguments: &[ArgumentSpec { name: "expression", kind: Text }],
            needs_approval: false,
            max_bytes: None,
            timeout: Duration::from_secs(5),
            scoped_to_workspace: false,
        },
        ToolName::CreateDocx | ToolName::CreateXlsx => ToolSpec {
            name,
            permission: GenerateArtifact,
            arguments: &[
                ArgumentSpec { name: "path", kind: Path },
                ArgumentSpec { name: "template", kind: Text },
                ArgumentSpec { name: "content", kind: Object },
            ],
            needs_approval: true,
            max_bytes: Some(64 * 1024 * 1024),
            timeout: Duration::from_secs(120),
            scoped_to_workspace: true,
        },
        ToolName::ExecuteCode => ToolSpec {
            name,
            permission: ExecuteCode,
            arguments: &[
                ArgumentSpec { name: "language", kind: Text },
                ArgumentSpec { name: "source", kind: Text },
            ],
            needs_approval: true,
            max_bytes: Some(1024 * 1024),
            // Short on purpose. A calculation that needs longer than this is
            // not a calculation, and an agent loop that can wait five minutes
            // per step will exhaust a person's patience before its own budget.
            timeout: Duration::from_secs(60),
            scoped_to_workspace: false,
        },
        ToolName::ValidateArtifact => ToolSpec {
            name,
            permission: GenerateArtifact,
            arguments: &[ArgumentSpec { name: "path", kind: Path }],
            needs_approval: false,
            max_bytes: Some(64 * 1024 * 1024),
            timeout: Duration::from_secs(60),
            scoped_to_workspace: true,
        },
    }
}

/// A request from the model. Nothing more than a name and some arguments —
/// deliberately inert until the gateway has looked at it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(tool: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool: tool.into(),
            arguments,
        }
    }

    /// The argument as a string, if present and of the right shape.
    pub fn text(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    /// The argument as a non-negative whole number.
    ///
    /// A string of digits is accepted as well as a JSON number: local models
    /// emit `"fromPage": "11"` often enough that refusing it would spend a turn
    /// on a formatting quarrel rather than on the work. Anything else — a
    /// negative, a fraction, a word — is absent, because a page number this
    /// could not read is not a page number it should guess at.
    pub fn integer(&self, key: &str) -> Option<u32> {
        let value = self.arguments.get(key)?;
        if let Some(number) = value.as_u64() {
            return u32::try_from(number).ok();
        }
        value.as_str()?.trim().parse::<u32>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_round_trips_through_its_wire_name() {
        for tool in ToolName::ALL {
            assert_eq!(ToolName::from_str(tool.as_str()), Some(*tool));
        }
    }

    #[test]
    fn an_unknown_name_is_not_a_tool() {
        assert_eq!(ToolName::from_str("delete_everything"), None);
        assert_eq!(ToolName::from_str(""), None);
    }

    /// A tool without a time limit makes every plan containing it unbounded.
    #[test]
    fn every_tool_has_a_time_limit() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            assert!(spec.timeout > Duration::ZERO, "{} has no timeout", tool.as_str());
            assert!(
                spec.timeout <= Duration::from_secs(120),
                "{} may hang for too long",
                tool.as_str()
            );
        }
    }

    /// Anything that leaves a trace outside the task needs a person.
    #[test]
    fn tools_that_write_or_execute_all_require_approval() {
        for tool in [
            ToolName::WriteScopedFile,
            ToolName::CreateDocx,
            ToolName::CreateXlsx,
            ToolName::ExecuteCode,
        ] {
            assert!(spec_for(tool).needs_approval, "{} should need approval", tool.as_str());
        }
    }

    /// Reading and computing can be undone by ignoring the result, so making a
    /// person confirm them would only train them to click through.
    #[test]
    fn reading_and_computing_do_not_interrupt_anyone() {
        for tool in [
            ToolName::SearchDocuments,
            ToolName::LoadMoreEvidence,
            ToolName::ReadScopedFile,
            ToolName::RunCalculation,
            ToolName::ValidateArtifact,
        ] {
            assert!(!spec_for(tool).needs_approval, "{} should not need approval", tool.as_str());
        }
    }

    #[test]
    fn every_tool_that_takes_a_path_is_scoped_to_the_workspace() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            let takes_path = spec.arguments.iter().any(|a| a.kind == ArgumentKind::Path);
            assert_eq!(
                takes_path,
                spec.scoped_to_workspace,
                "{} disagrees about whether its path is scoped",
                tool.as_str()
            );
        }
    }

    /// A tool that could read or write without limit could exhaust the machine.
    #[test]
    fn every_tool_touching_a_file_has_a_size_ceiling() {
        for tool in ToolName::ALL {
            let spec = spec_for(*tool);
            if spec.scoped_to_workspace {
                assert!(spec.max_bytes.is_some(), "{} has no size limit", tool.as_str());
            }
        }
    }

    #[test]
    fn arguments_are_read_out_of_a_call_safely() {
        let call = ToolCall::new("search_documents", serde_json::json!({ "query": "wall thickness" }));
        assert_eq!(call.text("query"), Some("wall thickness"));
        assert_eq!(call.text("missing"), None);

        // A wrong-typed argument reads as absent rather than panicking.
        let wrong = ToolCall::new("search_documents", serde_json::json!({ "query": 42 }));
        assert_eq!(wrong.text("query"), None);
    }
}

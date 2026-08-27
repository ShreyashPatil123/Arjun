//! What a subagent role is, declared in Markdown and enforced in Rust.
//!
//! ## Why profiles are files and enforcement is not
//!
//! A profile says what a `document-extractor` is for, which tools it needs and
//! how many turns it gets. Writing that in Markdown means an operator can read
//! it, review it and diff it — which is the whole reason a role is a document
//! rather than a `const` in a source file.
//!
//! And it means a profile is **untrusted input**, exactly like a skill. So the
//! file is a *declaration*, compiled here into an [`AgentProfile`], and the
//! profile is then only ever used to **narrow** what a child may do. A profile
//! that asked for a tool the parent does not hold gets nothing; a profile that
//! set `max-turns: 10000` is capped at the ceiling below. Nothing a profile says
//! can make a child more capable than its parent.
//!
//! ## Allowed and disallowed, both
//!
//! `allowed-tools` is the request. `disallowed-tools` is a denylist that wins
//! over it, over the parent's grant, and over everything else.
//!
//! Two lists rather than one is deliberate redundancy. The allow list is what
//! somebody edits when adding a capability, and the moment for a mistake is
//! exactly then. A `code-worker` that names `create_docx` in its denylist stays
//! unable to write documents even if a later edit adds it to the allow list by
//! accident — and the contradiction is visible in the file rather than latent.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::registry::ModelRole;
use crate::skills::frontmatter::{self, Document, Node};

/// Hard ceilings no profile may exceed.
///
/// A profile is a file on disk, so its numbers are a request. These are the
/// answer to "what if somebody writes a very large one" — not a policy a
/// profile can raise, and not a default it can lower past usefulness.
pub mod ceiling {
    /// Most turns any child may take, whatever its profile asks for.
    pub const MAX_TURNS: u32 = 24;
    /// Most output tokens per child.
    pub const MAX_OUTPUT_TOKENS: u32 = 8192;
    /// Deepest a child may be. One means a parent may spawn a child and that
    /// child may spawn nothing — requirement 4, expressed as a number so the
    /// check is an inequality rather than a special case.
    pub const MAX_DEPTH: u8 = 1;
}

/// How a child is isolated, and therefore whether it may run alongside others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Isolation {
    /// Touches nothing outside its own reading. Several may run at once: they
    /// cannot affect each other's results, and the operator waits for the
    /// slowest rather than the sum.
    ReadOnly,
    /// Writes into its own directory. Runs alone, because two writers to one
    /// workspace have an order and it should not be whichever finished last.
    Writer,
    /// Proposes something a person must approve. Runs alone, for the same
    /// reason and one more: an approver shown three requests at once cannot
    /// tell which run each belongs to.
    ApprovalSensitive,
}

impl Isolation {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "read-only" | "readOnly" => Isolation::ReadOnly,
            "writer" => Isolation::Writer,
            "approval-sensitive" | "approvalSensitive" => Isolation::ApprovalSensitive,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Isolation::ReadOnly => "read-only",
            Isolation::Writer => "writer",
            Isolation::ApprovalSensitive => "approval-sensitive",
        }
    }

    /// Whether children of this kind may run concurrently with each other.
    pub const fn is_concurrent(self) -> bool {
        matches!(self, Isolation::ReadOnly)
    }
}

/// What a child may remember, and for how long.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryScope {
    /// Nothing survives the child. The default, and what every read-only role
    /// should use: a worker that remembers across tasks is a worker whose
    /// answer depends on something the person asking cannot see.
    None,
    /// Survives for this child only.
    Task,
    /// Survives for the parent run.
    Run,
}

impl MemoryScope {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "none" => MemoryScope::None,
            "task" => MemoryScope::Task,
            "run" => MemoryScope::Run,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            MemoryScope::None => "none",
            MemoryScope::Task => "task",
            MemoryScope::Run => "run",
        }
    }
}

/// Where a child may write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WritePolicy {
    /// Nowhere.
    None,
    /// Its own directory under the parent run's workspace, and nowhere else.
    OwnDirectory,
}

impl WritePolicy {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "none" => WritePolicy::None,
            "own-directory" | "ownDirectory" => WritePolicy::OwnDirectory,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            WritePolicy::None => "none",
            WritePolicy::OwnDirectory => "own-directory",
        }
    }
}

/// The shape a child must return.
///
/// Named rather than free-form so the parent can refuse a result that does not
/// answer the question it asked. A worker that returns prose where an
/// extraction was wanted has failed, and should be recorded as failing rather
/// than having its prose folded into the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SchemaKind {
    /// Pages, with what was read and what was not.
    Extraction,
    /// Passages, with citations.
    Retrieval,
    /// Figures, with the working behind each.
    Calculation,
    /// Findings about a produced file.
    Review,
    /// A program, and whether it ran.
    Code,
}

impl SchemaKind {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "extraction" => SchemaKind::Extraction,
            "retrieval" => SchemaKind::Retrieval,
            "calculation" => SchemaKind::Calculation,
            "review" => SchemaKind::Review,
            "code" => SchemaKind::Code,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            SchemaKind::Extraction => "extraction",
            SchemaKind::Retrieval => "retrieval",
            SchemaKind::Calculation => "calculation",
            SchemaKind::Review => "review",
            SchemaKind::Code => "code",
        }
    }
}

/// The limits a child runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Limits {
    pub max_turns: u32,
    pub max_output_tokens: u32,
    /// How many children this child may spawn. Zero for every shipped profile.
    pub max_children: u8,
    pub max_duration_seconds: u64,
}

/// A subagent role, validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentProfile {
    pub name: String,
    pub description: String,
    pub version: String,
    /// The role a model must be registered for to take this work.
    pub model_role: ModelRole,
    /// Specific model ids, when the profile names them. Empty means any model
    /// registered for `model_role` that passes the certification check.
    pub eligible_models: Vec<String>,
    pub allowed_tools: Vec<ToolName>,
    /// Wins over `allowed_tools`, over the parent's grant, over everything.
    pub disallowed_tools: Vec<ToolName>,
    pub limits: Limits,
    pub isolation: Isolation,
    pub memory_scope: MemoryScope,
    /// Always `false`. Present as a field so the declaration is explicit in
    /// every profile rather than implied by absence.
    pub network_permitted: bool,
    pub write_policy: WritePolicy,
    /// The most sensitive material this role may be given.
    pub classification_ceiling: Classification,
    pub required_schema: SchemaKind,
    /// SHA-256 of the profile file.
    pub sha256: String,
}

impl AgentProfile {
    /// The tools this profile asks for, with its own denylist already applied.
    ///
    /// Applied here rather than at the call site so there is no path by which a
    /// caller reads `allowed_tools` and forgets the other list.
    pub fn requested_tools(&self) -> Vec<ToolName> {
        self.allowed_tools
            .iter()
            .copied()
            .filter(|tool| !self.disallowed_tools.contains(tool))
            .collect()
    }
}

/// Why a profile could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "reason")]
pub enum ProfileError {
    Malformed { detail: String },
    MissingField { field: String },
    InvalidField { field: String, detail: String },
    NameMismatch { declared: String, file: String },
    UnknownTool { tool: String },
    /// The same tool is in both lists. Refused rather than resolved, because
    /// either reading is a guess about what the author meant.
    Contradiction { tool: String },
    /// It asks for something above a hard ceiling.
    AboveCeiling { field: String, asked: u64, ceiling: u64 },
}

impl ProfileError {
    pub fn explain(&self) -> String {
        match self {
            ProfileError::Malformed { detail } => format!("Its frontmatter could not be read: {detail}"),
            ProfileError::MissingField { field } => format!("It has no {field:?}, which is required."),
            ProfileError::InvalidField { field, detail } => format!("Its {field:?} is not usable: {detail}"),
            ProfileError::NameMismatch { declared, file } => format!(
                "It calls itself {declared:?} but is in a file called {file:?}. Rename one to match."
            ),
            ProfileError::UnknownTool { tool } => format!(
                "It names a tool called {tool:?}, which this build does not have. A profile cannot \
                 introduce a tool."
            ),
            ProfileError::Contradiction { tool } => format!(
                "{tool:?} is in both allowed-tools and disallowed-tools. Neither reading is safe to \
                 guess at, so the profile is refused."
            ),
            ProfileError::AboveCeiling { field, asked, ceiling } => format!(
                "Its {field} is {asked}, above the hard ceiling of {ceiling}. A profile cannot raise it."
            ),
        }
    }
}

/// Compiles a profile from the Markdown file's frontmatter.
///
/// `file_stem` is the file name without its extension, and the declared name
/// must match it — the file is what an operator audits.
pub fn compile(source: &str, file_stem: &str, sha256: &str) -> Result<AgentProfile, ProfileError> {
    let split = frontmatter::split(source).map_err(|error| ProfileError::Malformed {
        detail: error.to_string(),
    })?;
    let document = frontmatter::parse(split.frontmatter).map_err(|error| ProfileError::Malformed {
        detail: error.to_string(),
    })?;

    let required = |field: &str| -> Result<String, ProfileError> {
        document
            .scalar(field)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| ProfileError::MissingField {
                field: field.to_string(),
            })
    };

    let name = required("name")?;
    if !crate::skills::is_valid_name(&name) {
        return Err(ProfileError::InvalidField {
            field: "name".to_string(),
            detail: format!("{name:?} must be lowercase letters, digits and single hyphens"),
        });
    }
    if name != file_stem {
        return Err(ProfileError::NameMismatch {
            declared: name,
            file: file_stem.to_string(),
        });
    }

    let description = required("description")?;
    let version = required("version")?;

    let model_role = parse_enum::<ModelRole>(&required("model-role")?).ok_or_else(|| {
        ProfileError::InvalidField {
            field: "model-role".to_string(),
            detail: "must be one of: reasoning, coding, vision, documentOcr, embedding, rerank"
                .to_string(),
        }
    })?;
    let eligible_models = document
        .list("eligible-models")
        .map(|items| items.to_vec())
        .unwrap_or_default();

    let allowed_tools = tool_list(&document, "allowed-tools", true)?;
    let disallowed_tools = tool_list(&document, "disallowed-tools", false)?;
    for tool in &allowed_tools {
        if disallowed_tools.contains(tool) {
            return Err(ProfileError::Contradiction {
                tool: tool.as_str().to_string(),
            });
        }
    }

    let limits_map = document
        .map("limits")
        .ok_or_else(|| ProfileError::MissingField {
            field: "limits".to_string(),
        })?;
    let number = |field: &str| -> Result<u64, ProfileError> {
        limits_map
            .get(field)
            .and_then(Node::as_scalar)
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .ok_or_else(|| ProfileError::MissingField {
                field: format!("limits.{field}"),
            })
    };

    let max_turns = number("max-turns")?;
    capped("limits.max-turns", max_turns, ceiling::MAX_TURNS as u64)?;
    let max_output_tokens = number("max-output-tokens")?;
    capped(
        "limits.max-output-tokens",
        max_output_tokens,
        ceiling::MAX_OUTPUT_TOKENS as u64,
    )?;
    let max_children = number("max-children")?;
    // A profile that wanted grandchildren is refused rather than clamped: the
    // author asked for something the model does not support, and silently
    // giving them zero would hide that.
    capped("limits.max-children", max_children, 0)?;
    let max_duration_seconds = number("max-duration-seconds")?;

    let isolation = Isolation::parse(&required("isolation")?).ok_or_else(|| {
        ProfileError::InvalidField {
            field: "isolation".to_string(),
            detail: "must be read-only, writer or approval-sensitive".to_string(),
        }
    })?;
    let memory_scope = MemoryScope::parse(&required("memory-scope")?).ok_or_else(|| {
        ProfileError::InvalidField {
            field: "memory-scope".to_string(),
            detail: "must be none, task or run".to_string(),
        }
    })?;

    let network = required("network")?;
    if network.trim() != "none" {
        // There is no other legal value. A profile asking for anything else is
        // refused rather than downgraded, so the request is visible.
        return Err(ProfileError::InvalidField {
            field: "network".to_string(),
            detail: "a subagent may only declare `none`; there is no network for it to use"
                .to_string(),
        });
    }

    let write_policy = WritePolicy::parse(&required("write-policy")?).ok_or_else(|| {
        ProfileError::InvalidField {
            field: "write-policy".to_string(),
            detail: "must be none or own-directory".to_string(),
        }
    })?;
    let classification_ceiling = parse_enum::<Classification>(&required("classification-ceiling")?)
        .ok_or_else(|| ProfileError::InvalidField {
            field: "classification-ceiling".to_string(),
            detail: "must be one of the seven classifications, in camelCase".to_string(),
        })?;
    let required_schema =
        SchemaKind::parse(&required("required-schema")?).ok_or_else(|| ProfileError::InvalidField {
            field: "required-schema".to_string(),
            detail: "must be extraction, retrieval, calculation, review or code".to_string(),
        })?;

    // A read-only worker that could write would be neither. Checked because the
    // two fields are edited independently and the combination is what decides
    // whether several of these run at once.
    if isolation == Isolation::ReadOnly && write_policy != WritePolicy::None {
        return Err(ProfileError::InvalidField {
            field: "isolation".to_string(),
            detail: "a read-only worker may not have a write policy; several run at once"
                .to_string(),
        });
    }

    Ok(AgentProfile {
        name,
        description,
        version,
        model_role,
        eligible_models,
        allowed_tools,
        disallowed_tools,
        limits: Limits {
            max_turns: max_turns as u32,
            max_output_tokens: max_output_tokens as u32,
            max_children: max_children as u8,
            max_duration_seconds,
        },
        isolation,
        memory_scope,
        network_permitted: false,
        write_policy,
        classification_ceiling,
        required_schema,
        sha256: sha256.to_string(),
    })
}

fn capped(field: &str, asked: u64, ceiling: u64) -> Result<(), ProfileError> {
    if asked > ceiling {
        return Err(ProfileError::AboveCeiling {
            field: field.to_string(),
            asked,
            ceiling,
        });
    }
    Ok(())
}

fn tool_list(
    document: &Document,
    field: &str,
    required: bool,
) -> Result<Vec<ToolName>, ProfileError> {
    let Some(names) = document.list(field) else {
        if required {
            return Err(ProfileError::MissingField {
                field: field.to_string(),
            });
        }
        return Ok(Vec::new());
    };
    let mut seen = BTreeSet::new();
    let mut tools = Vec::new();
    for raw in names {
        let tool = ToolName::from_str(raw.trim()).ok_or_else(|| ProfileError::UnknownTool {
            tool: raw.clone(),
        })?;
        if seen.insert(tool.as_str()) {
            tools.push(tool);
        }
    }
    if required && tools.is_empty() {
        return Err(ProfileError::InvalidField {
            field: field.to_string(),
            detail: "a worker with no tools cannot do anything".to_string(),
        });
    }
    Ok(tools)
}

/// Reads an enum from its camelCase wire name.
fn parse_enum<T: serde::de::DeserializeOwned>(raw: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(raw.trim().to_string())).ok()
}

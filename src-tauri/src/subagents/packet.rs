//! What a parent hands a child, and deliberately what it does not.
//!
//! ## A fresh context, not a transcript
//!
//! Requirement 2 is that a child receives a fresh context rather than the whole
//! parent transcript. That is partly about cost — a 7B model with a 8k window
//! cannot hold twenty turns of parent history and do useful work — and mostly
//! about blast radius.
//!
//! A parent transcript contains every passage the parent retrieved, including
//! ones a narrower child has no business seeing, and every instruction the
//! parent was given, including any a poisoned document managed to insert. Handing
//! that down would make every child as exposed as the parent and defeat the
//! point of a narrower worker.
//!
//! So a packet carries an **objective** in the parent's own words and
//! **references** to inputs. Not contents. A child that needs a passage
//! retrieves it itself, under its own clearance, through the same gateway — and
//! what comes back is what *it* is allowed to see, not what the parent was.
//!
//! ## Why the policy hash travels with it
//!
//! So the result can be checked against the constraints the child was actually
//! given. A result that arrives claiming a tool the packet did not grant is a
//! result from a child that was not running the policy this parent sent, and
//! that is worth being able to detect rather than assume away.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;

use super::inherit::EffectivePolicy;
use super::profile::{Limits, SchemaKind};

/// One thing a child is pointed at.
///
/// Every variant is a **reference**. There is deliberately no variant carrying
/// text: a packet that could hold a passage is a packet through which a
/// parent's retrieved material reaches a child that may not be cleared for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum InputRef {
    /// A document in the knowledge base, by content hash. The child retrieves
    /// it under its own clearance and may legitimately get less than the parent.
    Document {
        sha256: String,
        /// Absent means the whole document.
        page: Option<u32>,
    },
    /// A file in the parent run's workspace, relative to the run root.
    WorkspaceFile { path: String },
    /// A passage the parent already cited, by its marker. The child resolves it
    /// itself; the marker is a name, not the text.
    Evidence { marker: usize },
    /// An expression to check, which is the one case where the value *is* the
    /// reference — a calculation has nothing behind it to point at.
    Expression { expression: String },
}

impl InputRef {
    /// How this reads in a trace, without resolving anything.
    pub fn describe(&self) -> String {
        match self {
            InputRef::Document { sha256, page } => match page {
                Some(page) => format!("document {}… page {page}", &sha256[..8.min(sha256.len())]),
                None => format!("document {}…", &sha256[..8.min(sha256.len())]),
            },
            InputRef::WorkspaceFile { path } => format!("workspace file {path}"),
            InputRef::Evidence { marker } => format!("[E{marker}]"),
            InputRef::Expression { expression } => format!("expression {expression:?}"),
        }
    }
}

/// The work order handed to a child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildTaskPacket {
    pub child_id: String,
    pub parent_run_id: String,
    /// What makes creating the same child twice harmless. Derived by the parent
    /// from the work rather than generated, so two attempts at one piece of
    /// work agree without coordinating. See [`derive_idempotency_key`].
    pub idempotency_key: String,
    pub profile: String,
    /// What the child is for, in the parent's words. Prose, not instructions
    /// copied out of a document.
    pub objective: String,
    pub inputs: Vec<InputRef>,
    /// Exactly what this child may call. Already the intersection.
    pub allowed_tools: Vec<ToolName>,
    pub limits: Limits,
    pub required_schema: SchemaKind,
    /// The most sensitive material this child may touch.
    pub classification_ceiling: Classification,
    /// The digest of the policy this was derived from, so a result can be
    /// checked against the constraints the child was actually sent.
    pub policy_hash: String,
    pub created_at: DateTime<Utc>,
    /// When this child must stop.
    pub deadline: DateTime<Utc>,
}

impl ChildTaskPacket {
    /// Builds the packet from a policy that has already been narrowed.
    ///
    /// Takes the policy rather than the profile, so there is no path by which a
    /// packet is built from a profile's requests instead of from what the child
    /// was actually granted.
    pub fn new(
        child_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        idempotency_key: impl Into<String>,
        objective: impl Into<String>,
        inputs: Vec<InputRef>,
        policy: &EffectivePolicy,
        now: DateTime<Utc>,
    ) -> Self {
        let deadline = now
            + chrono::Duration::try_seconds(policy.limits.max_duration_seconds as i64)
                .unwrap_or_else(|| chrono::Duration::minutes(5));
        Self {
            child_id: child_id.into(),
            parent_run_id: parent_run_id.into(),
            idempotency_key: idempotency_key.into(),
            profile: policy.profile.clone(),
            objective: objective.into(),
            inputs,
            allowed_tools: policy.tools.clone(),
            limits: policy.limits,
            required_schema: policy.required_schema,
            classification_ceiling: policy.classification_ceiling,
            policy_hash: policy.inherited_hash.clone(),
            created_at: now,
            deadline,
        }
    }

    /// Whether the deadline has passed.
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.deadline
    }

    /// One line for the trace. Carries no objective text and no input contents,
    /// because a trace is read by more people than a run is.
    pub fn describe(&self) -> String {
        format!(
            "{} on {} input(s), {} tool(s), {} turn(s) at most",
            self.profile,
            self.inputs.len(),
            self.allowed_tools.len(),
            self.limits.max_turns
        )
    }
}

/// The key two attempts at the same piece of work compute independently.
///
/// Over the parent run, the profile and the work itself — so retrying after an
/// ambiguous failure finds the existing child rather than starting a second
/// one, and two different objectives under one profile are two children.
pub fn derive_idempotency_key(
    parent_run_id: &str,
    profile: &str,
    objective: &str,
    inputs: &[InputRef],
) -> String {
    let mut hasher = Sha256::new();
    let mut field = |value: &str| {
        hasher.update(value.as_bytes());
        hasher.update(b"\x1f");
    };
    field(parent_run_id);
    field(profile);
    field(objective);
    for input in inputs {
        field(&serde_json::to_string(input).unwrap_or_default());
    }
    format!("{:x}", hasher.finalize())
}

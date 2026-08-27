//! What a child inherits, and the fact that it can only ever be less.
//!
//! ## The shape of the guarantee
//!
//! A child's permissions are computed by **intersecting** what the parent holds
//! with what the profile asks for. Every field either narrows or stays the
//! same, and the narrowing runs one way: [`EffectivePolicy`] is built here and
//! nowhere else, from an [`InheritedPolicy`] that the child never gets to
//! construct for itself.
//!
//! That is the same technique as [`crate::skills::narrowing`], applied to more
//! fields. Where a set is involved the child's is built by *filtering the
//! parent's*, so a value the parent does not hold has no path into the child's
//! whatever the profile declared. Where a scalar is involved it is a `min`.
//!
//! ## Depth, and why it is a number
//!
//! Requirement 4 is "a child cannot spawn another child by default". Expressed
//! as a boolean that would be a special case somebody could forget; expressed
//! as a depth against [`ceiling::MAX_DEPTH`] it is an inequality that holds for
//! grandchildren, great-grandchildren and anything else a later phase invents.
//!
//! ## The policy hash
//!
//! [`InheritedPolicy::hash`] is over every field that constrains the child. It
//! travels in the task packet, so the parent can tell — when the result comes
//! back — whether the constraints the child ran under are the ones it was sent.
//! It is not a secret and it is not a signature: it detects drift between two
//! records of the same thing, which is what it is for.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{Role, Session};
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;

use super::profile::{
    ceiling, AgentProfile, Isolation, Limits, MemoryScope, SchemaKind, WritePolicy,
};

/// Everything a child takes from its parent and cannot exceed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InheritedPolicy {
    /// Who the work is attributed to. A child never acquires an identity of its
    /// own: every tool call it makes is authorised against this person, and the
    /// audit record names them.
    pub user_id: String,
    /// Their roles, which is what clearance is decided from.
    pub roles: Vec<Role>,
    /// The most sensitive material anything under this run may touch.
    pub classification_ceiling: Classification,
    /// Always false. A field rather than an absence so it appears in the hash
    /// and in the packet, where somebody reading a record can see it was false.
    pub network_permitted: bool,
    /// Whether a person must approve consequential actions. Inherited so a
    /// child cannot be the way round an approval the parent owed.
    pub approval_required: bool,
    /// The run's own directory. A child writes under this or nowhere.
    pub workspace_root: PathBuf,
    /// What the parent may call. A child's list is filtered from this.
    pub permitted_tools: Vec<ToolName>,
    /// 0 for the parent run. A child is 1, and 1 is the ceiling.
    pub depth: u8,
}

impl InheritedPolicy {
    /// The policy a top-level run passes to its children.
    pub fn of_run(
        session: &Session,
        classification_ceiling: Classification,
        workspace_root: impl Into<PathBuf>,
        permitted_tools: &[ToolName],
    ) -> Self {
        Self {
            user_id: session.user.id.clone(),
            roles: session.user.roles.clone(),
            classification_ceiling,
            network_permitted: false,
            approval_required: true,
            workspace_root: workspace_root.into(),
            permitted_tools: permitted_tools.to_vec(),
            depth: 0,
        }
    }

    /// A stable digest of everything that constrains a child.
    ///
    /// Fields are joined with a separator that cannot occur inside them, so no
    /// rearrangement of contents produces the same digest as a different
    /// policy.
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        let mut field = |value: &str| {
            hasher.update(value.as_bytes());
            hasher.update(b"\x1f");
        };
        field(&self.user_id);
        let mut roles: Vec<&str> = self.roles.iter().map(|role| role.label()).collect();
        roles.sort_unstable();
        field(&roles.join(","));
        field(self.classification_ceiling.label());
        field(if self.network_permitted { "network" } else { "no-network" });
        field(if self.approval_required { "approval" } else { "no-approval" });
        field(&self.workspace_root.display().to_string());
        let mut tools: Vec<&str> = self.permitted_tools.iter().map(|t| t.as_str()).collect();
        tools.sort_unstable();
        field(&tools.join(","));
        field(&self.depth.to_string());
        format!("{:x}", hasher.finalize())
    }
}

/// Why a child could not be given a policy at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "refusal")]
pub enum InheritRefusal {
    /// This would be a grandchild.
    TooDeep { depth: u8, ceiling: u8 },
    /// The profile and the parent have no tool in common, so the child could
    /// not do anything. Reported rather than run into one refusal at a time.
    NoToolsInCommon {
        profile_wants: Vec<String>,
        parent_permits: Vec<String>,
    },
    /// The signed-in person is not cleared for the ceiling this profile needs.
    NotCleared { classification: String },
}

impl InheritRefusal {
    pub fn explain(&self) -> String {
        match self {
            InheritRefusal::TooDeep { depth, ceiling } => format!(
                "A subagent may not start another subagent. This would be depth {depth}, and the \
                 limit is {ceiling}."
            ),
            InheritRefusal::NoToolsInCommon {
                profile_wants,
                parent_permits,
            } => format!(
                "This worker needs {} and the task permits {}. They have nothing in common, so it \
                 would be able to do nothing.",
                profile_wants.join(", "),
                if parent_permits.is_empty() {
                    "no tools".to_string()
                } else {
                    parent_permits.join(", ")
                }
            ),
            InheritRefusal::NotCleared { classification } => format!(
                "This worker handles {classification} material, which the signed-in user is not \
                 cleared for."
            ),
        }
    }
}

/// What a child may actually do. Built only by [`InheritedPolicy::narrow_for`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectivePolicy {
    pub profile: String,
    pub inherited: InheritedPolicy,
    /// The intersection. Never wider than `inherited.permitted_tools`.
    pub tools: Vec<ToolName>,
    /// Tools the profile asked for and did not get, recorded so the trace can
    /// say why a worker did less than its profile describes.
    pub refused_tools: Vec<ToolName>,
    pub limits: Limits,
    pub isolation: Isolation,
    pub memory_scope: MemoryScope,
    pub write_policy: WritePolicy,
    /// The lower of the parent's ceiling and the profile's.
    pub classification_ceiling: Classification,
    pub required_schema: SchemaKind,
    /// Where this child may write, when it may write at all.
    pub write_root: Option<PathBuf>,
    /// The hash of the policy this was derived from.
    pub inherited_hash: String,
}

impl InheritedPolicy {
    /// Derives a child's policy. Every field narrows or stays the same.
    pub fn narrow_for(
        &self,
        profile: &AgentProfile,
        child_id: &str,
    ) -> Result<EffectivePolicy, InheritRefusal> {
        let depth = self.depth + 1;
        if depth > ceiling::MAX_DEPTH {
            return Err(InheritRefusal::TooDeep {
                depth,
                ceiling: ceiling::MAX_DEPTH,
            });
        }

        // Built by filtering the parent's list. A tool the parent does not hold
        // has no path into this vector, whatever the profile declared — the
        // same technique, and for the same reason, as `skills::narrowing`.
        let wanted = profile.requested_tools();
        let tools: Vec<ToolName> = self
            .permitted_tools
            .iter()
            .copied()
            .filter(|tool| wanted.contains(tool))
            // The profile's denylist wins here too, so a tool that reached the
            // parent's list some other way still cannot reach the child's.
            .filter(|tool| !profile.disallowed_tools.contains(tool))
            .collect();

        if tools.is_empty() {
            return Err(InheritRefusal::NoToolsInCommon {
                profile_wants: wanted.iter().map(|t| t.as_str().to_string()).collect(),
                parent_permits: self
                    .permitted_tools
                    .iter()
                    .map(|t| t.as_str().to_string())
                    .collect(),
            });
        }

        let refused_tools: Vec<ToolName> = wanted
            .iter()
            .copied()
            .filter(|tool| !self.permitted_tools.contains(tool))
            .collect();

        // The lower of the two. A profile declaring a higher ceiling than the
        // run does not raise it; it is simply capped, and the run's ceiling is
        // what applies.
        let classification_ceiling = if profile
            .classification_ceiling
            .sensitivity()
            <= self.classification_ceiling.sensitivity()
        {
            profile.classification_ceiling
        } else {
            self.classification_ceiling
        };

        // The person has to be cleared for what the child would handle. Checked
        // here as well as at every tool call: this gives a clear refusal before
        // anything starts, and the gateway gives one on each call regardless.
        let cleared = classification_ceiling
            .cleared_roles()
            .iter()
            .any(|role| self.roles.contains(role));
        if !cleared {
            return Err(InheritRefusal::NotCleared {
                classification: classification_ceiling.label().to_string(),
            });
        }

        let limits = Limits {
            max_turns: profile.limits.max_turns.min(ceiling::MAX_TURNS),
            max_output_tokens: profile
                .limits
                .max_output_tokens
                .min(ceiling::MAX_OUTPUT_TOKENS),
            // Whatever the profile said, a child of a child is not permitted.
            max_children: 0,
            max_duration_seconds: profile.limits.max_duration_seconds,
        };

        let write_root = match profile.write_policy {
            WritePolicy::None => None,
            // Its own directory, under the parent's. Never the parent's own
            // root: two workers writing beside each other's deliverables is the
            // thing per-child directories exist to prevent.
            WritePolicy::OwnDirectory => {
                Some(self.workspace_root.join("children").join(child_id))
            }
        };

        Ok(EffectivePolicy {
            profile: profile.name.clone(),
            inherited: InheritedPolicy {
                depth,
                // Narrowed on the way down, so a child's own inherited view is
                // already the reduced one — there is no copy of the parent's
                // wider policy inside the child to be found and used.
                permitted_tools: tools.clone(),
                classification_ceiling,
                ..self.clone()
            },
            tools,
            refused_tools,
            limits,
            isolation: profile.isolation,
            memory_scope: profile.memory_scope,
            write_policy: profile.write_policy,
            classification_ceiling,
            required_schema: profile.required_schema,
            write_root,
            inherited_hash: self.hash(),
        })
    }
}

impl EffectivePolicy {
    /// Whether this child may touch material of a given kind.
    pub fn may_handle(&self, classification: Classification) -> bool {
        classification.within(self.classification_ceiling)
            && classification
                .cleared_roles()
                .iter()
                .any(|role| self.inherited.roles.contains(role))
    }

    /// Whether this child may call a tool.
    pub fn may_call(&self, tool: ToolName) -> bool {
        self.tools.contains(&tool)
    }

    /// Whether a path is somewhere this child may write.
    ///
    /// Textual `..` resolution and a containment check against its own
    /// directory — the same rule the tool gateway uses, applied to the second
    /// boundary a child introduces. A child with no write policy may write
    /// nowhere, and that is the common case.
    pub fn may_write(&self, path: &Path) -> bool {
        let Some(root) = &self.write_root else {
            return false;
        };
        let Some(candidate) = normalise(path) else {
            return false;
        };
        let Some(root) = normalise(root) else {
            return false;
        };
        candidate.starts_with(&root)
    }

    /// Whether this child may run alongside others.
    pub fn is_concurrent(&self) -> bool {
        self.isolation.is_concurrent()
    }
}

/// Resolves `..` textually. `None` when the path climbs above its own root.
fn normalise(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

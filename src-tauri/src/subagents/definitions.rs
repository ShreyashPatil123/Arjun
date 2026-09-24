//! Which definition a new child runs under, decided at the moment it is sent.
//!
//! ## The gap this closes
//!
//! Until this module, delegation never read the agent registry. The manager
//! was built once at start-up from the Markdown profiles in `agents/`, keyed by
//! file name, and each worker held the instructions it was constructed with.
//! An administrator editing an agent on the Agents screen changed a JSON row
//! that no dispatch path consulted — so the edit reached nothing, not merely
//! "not the running child". Plan §3 finding 9, and it was wider than stated.
//!
//! ## What "pinned" means now
//!
//! [`DefinitionSource::resolve`] is called once per dispatch, and what it
//! returns is copied into the child's packet. A child that is already running
//! holds its copy; an edit saved while it runs changes the registry and not the
//! packet. The next dispatch resolves again and gets the edit. That is the
//! whole of the pinning rule, and it needs no lock held across the work.
//!
//! A retry of the *same* work in the same run is not a new dispatch: the
//! manager's idempotency ledger returns the first child's answer, which ran
//! under the first child's pin. That is deliberate — the key names the work,
//! and an edit between two attempts at one piece of work does not make it a
//! second piece of work.
//!
//! ## Keys that do not move when somebody renames an agent
//!
//! A dispatch names either an `agent_id` (`ag-…`, stable for the life of the
//! agent) or a bundled role key (`knowledge-retriever`, the profile file's own
//! name, recorded on the imported row as `imported_from.profile_name`). Neither
//! is the display name, so renaming an agent cannot break delegation to it and
//! cannot make a second agent answer to the first one's name.
//!
//! Which *worker* performs a definition is decided by its output schema —
//! [`capability_for`] — and not by any name. That is what lets a cloned agent,
//! whose id and display name are both new, be performed by the same code as
//! the role it was cloned from, and what keeps a custom agent from claiming a
//! capability no code implements.

use sha2::{Digest, Sha256};

use super::profile::{AgentProfile, SchemaKind};
use crate::agents::store::{AgentRegistry, Visibility};
use crate::agents::{AgentDefinition, ModelBinding, SkillBinding};

/// Where a resolved definition came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionOrigin {
    /// A registry row, at an exact `definition_version`.
    Registry,
    /// A Markdown profile loaded at start-up, because no registry was in reach.
    ///
    /// Kept as a separate origin rather than dressed up as version 0 of a
    /// registry row: a trace that says "registry" for something the registry
    /// never saw would be a provenance claim nothing backs.
    BundledProfile,
}

impl DefinitionOrigin {
    pub const fn as_str(self) -> &'static str {
        match self {
            DefinitionOrigin::Registry => "registry",
            DefinitionOrigin::BundledProfile => "bundled-profile",
        }
    }
}

/// The definition one dispatch runs under, copied at the moment it was sent.
#[derive(Debug, Clone)]
pub struct ResolvedDefinition {
    /// The registry id, or the bundled role key when there is no registry.
    pub agent_id: String,
    /// The registry's version counter. `None` for a bundled profile, which has
    /// a file hash and no version, and saying `0` would invent one.
    pub definition_version: Option<u64>,
    pub display_name: String,
    /// The worker that performs this definition. See [`capability_for`].
    pub capability: String,
    /// The profile the narrowing runs over. Built by
    /// [`AgentDefinition::to_profile`] for a registry row, so a definition can
    /// only ever be narrowed against the parent's grant and never added to it.
    pub profile: AgentProfile,
    /// Whether what this child publishes is readable by the other agents on its
    /// task. Carried, not yet enforced at publication — plan §3 finding 8 is
    /// P02's to close, and the packet records the policy so that closing it is a
    /// change to the publisher rather than to every dispatch site.
    pub shared_with_task: bool,
    pub origin: DefinitionOrigin,
    /// The skills the definition is bound to, at the bytes it was bound to.
    /// Empty for a bundled profile, which binds none.
    pub skills: Vec<SkillBinding>,
    /// Which models the definition may be routed to, and which it prefers.
    /// Carried into the packet's model policy; see
    /// [`super::packet::ModelPolicy`] for why it is recorded and not yet held.
    pub model_binding: ModelBinding,
}

impl ResolvedDefinition {
    /// What the child's own model is told it is for.
    pub fn instructions(&self) -> &str {
        &self.profile.instructions
    }

    /// The sha-256 of those instructions, so a trace can show which text a
    /// child actually received without copying the text into the trace.
    pub fn instructions_sha256(&self) -> String {
        format!("{:x}", Sha256::digest(self.profile.instructions.as_bytes()))
    }

    /// The same for a bundled profile, which is the only source when a
    /// deployment has no registry.
    pub fn from_bundled(profile: &AgentProfile) -> Self {
        Self {
            agent_id: profile.name.clone(),
            definition_version: None,
            display_name: profile.name.clone(),
            capability: capability_for(profile.required_schema)
                .unwrap_or(profile.name.as_str())
                .to_string(),
            profile: profile.clone(),
            // What the bundled workers have always done. See the field doc.
            shared_with_task: true,
            origin: DefinitionOrigin::BundledProfile,
            skills: Vec::new(),
            // A profile names only its eligible set; it has no default or
            // fallbacks to prefer.
            model_binding: ModelBinding {
                default_model_id: None,
                fallback_model_ids: Vec::new(),
                eligible_model_ids: profile.eligible_models.clone(),
            },
        }
    }
}

/// Why no definition could be resolved for a dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unresolved {
    /// Nothing in the registry answers to this key.
    NoDefinition { key: String },
    /// The agent exists and an administrator has taken it out of service.
    ///
    /// Refused rather than falling back to the bundled profile of the same
    /// role: disabling an agent is a decision, and quietly running the
    /// original in its place would be the setting having no effect.
    NotRunnable { agent_id: String, state: String },
    /// The definition asks for an output no worker in this build produces.
    NoCapability { agent_id: String, schema: String },
    /// The registry could not be read.
    Registry { detail: String },
}

impl Unresolved {
    pub fn explain(&self) -> String {
        match self {
            Unresolved::NoDefinition { key } => format!(
                "No agent in this deployment's registry answers to {key:?}. Name an agent id \
                 (ag-…) or a bundled role such as knowledge-retriever."
            ),
            Unresolved::NotRunnable { agent_id, state } => format!(
                "Agent {agent_id} is {state}, so it is not given new work. An administrator \
                 can re-enable it on the Agents screen."
            ),
            Unresolved::NoCapability { agent_id, schema } => format!(
                "Agent {agent_id} is defined to return {schema}, and no worker in this build \
                 produces that. It is registered and cannot yet be run."
            ),
            Unresolved::Registry { detail } => {
                format!("The agent registry could not be read: {detail}")
            }
        }
    }
}

/// Where a dispatch finds the definition it runs under.
///
/// A trait so the manager does not depend on the registry's storage, and so a
/// test can hand it a real [`AgentRegistry`] in a temporary directory rather
/// than a stand-in that would only prove the stand-in works.
pub trait DefinitionSource: Send + Sync {
    /// The definition a *new* dispatch for `key` should run under, read now.
    fn resolve(&self, key: &str) -> Result<ResolvedDefinition, Unresolved>;
}

/// The worker that produces a given output.
///
/// The one place a capability is tied to code. Every other name in the path —
/// agent id, display name, role key — can change or multiply without moving
/// this, and a schema with no entry here is a capability nothing implements.
///
/// `None` for a schema that has a contract and no worker yet. That is the
/// state plan P01 asks for: a role's result shape registered before its
/// handler, and not reported as runnable until the handler exists.
pub const fn capability_for(schema: SchemaKind) -> Option<&'static str> {
    match schema {
        SchemaKind::Extraction => Some("document-extractor"),
        SchemaKind::Retrieval => Some("knowledge-retriever"),
        SchemaKind::Calculation => Some("calculation-checker"),
        SchemaKind::Review => Some("artifact-reviewer"),
        SchemaKind::Code => Some("code-worker"),
        // Contracts without a worker. Listed rather than covered by a wildcard,
        // so the day a writer worker lands, adding it here is a compile error
        // away from being forgotten.
        SchemaKind::Document | SchemaKind::Deck | SchemaKind::Workbook => None,
    }
}

impl DefinitionSource for AgentRegistry {
    fn resolve(&self, key: &str) -> Result<ResolvedDefinition, Unresolved> {
        let key = key.trim();
        // Administrator visibility because this is the runtime asking what an
        // agent *is*, not a person asking what they may see. What the child may
        // then *do* is still decided by the narrowing against the parent's
        // grant, which this does not touch.
        let held = self
            .list(Visibility::Administrator)
            .map_err(|error| Unresolved::Registry {
                detail: error.explain(),
            })?;

        // An id first, because an id is unambiguous. A role key second, and only
        // against the row that was imported from that role's file — a clone has
        // `imported_from: None`, so it can never answer to its source's name.
        let found = held
            .iter()
            .find(|agent| agent.agent_id == key)
            .or_else(|| {
                held.iter().find(|agent| {
                    agent
                        .imported_from
                        .as_ref()
                        .is_some_and(|origin| origin.profile_name == key)
                })
            })
            .ok_or_else(|| Unresolved::NoDefinition {
                key: key.to_string(),
            })?;

        resolved_from(found)
    }
}

/// A registry row, checked and converted.
pub fn resolved_from(agent: &AgentDefinition) -> Result<ResolvedDefinition, Unresolved> {
    if !agent.state.is_runnable() {
        return Err(Unresolved::NotRunnable {
            agent_id: agent.agent_id.clone(),
            state: agent.state.as_str().to_string(),
        });
    }
    let capability =
        capability_for(agent.output_schema).ok_or_else(|| Unresolved::NoCapability {
            agent_id: agent.agent_id.clone(),
            schema: agent.output_schema.as_str().to_string(),
        })?;

    Ok(ResolvedDefinition {
        agent_id: agent.agent_id.clone(),
        definition_version: Some(agent.definition_version),
        display_name: agent.display_name.clone(),
        capability: capability.to_string(),
        profile: agent.to_profile(),
        shared_with_task: agent.memory.shared_with_task,
        origin: DefinitionOrigin::Registry,
        skills: agent.skills.clone(),
        model_binding: agent.models.clone(),
    })
}

impl ResolvedDefinition {
    /// The model policy a child dispatched under this definition carries, given
    /// the routing decision actually made for it.
    pub fn model_policy(
        &self,
        decision: &super::certification::Decision,
    ) -> super::packet::ModelPolicy {
        let eligible = self.model_binding.eligible_model_ids.clone();
        super::packet::ModelPolicy {
            role: self.profile.model_role.label().to_string(),
            preferred_model_ids: self
                .model_binding
                .default_model_id
                .iter()
                .chain(self.model_binding.fallback_model_ids.iter())
                .cloned()
                .collect(),
            within_eligible: eligible.is_empty()
                || eligible.iter().any(|id| *id == decision.model_id),
            eligible_model_ids: eligible,
            routing_reason: decision.reason.clone(),
            cheaper_than_parent: decision.cheaper_than_parent,
        }
    }
}

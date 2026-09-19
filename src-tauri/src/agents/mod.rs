//! Agents as things a deployment owns, rather than files an installer shipped.
//!
//! ## What this replaces
//!
//! [`crate::subagents::profile`] compiles a Markdown file into an
//! [`AgentProfile`]. That is a good compiler and it is not a registry: the
//! *file name* is the identity, the files live in the installer's resource
//! directory, and nothing about an agent can be changed without editing an
//! installed program's own files. So there was no way to rename an agent, give
//! it a colour, disable one, or point it at a different model — and no stable
//! thing to attach durable memory to, because renaming the file would have
//! renamed the agent.
//!
//! This module keeps the compiler and adds the part that was missing: a record
//! in the application's own data directory, with an identity that survives
//! everything a person is allowed to change about it.
//!
//! ## The identities, and why each is separate
//!
//! | Identity | Lifetime | Changed by |
//! |---|---|---|
//! | `agent_id` | forever | nothing |
//! | `definition_version` | one revision | any accepted edit |
//! | [`ModelBinding`] | until reassigned | an administrator |
//! | `task_id` / `run_id` | one task | nothing |
//! | `attempt_id` | one attempt at a run | a resumption |
//! | `worker_instance_id` | one process | a restart |
//!
//! The separation is the point. Memory is owned by `agent_id`, so changing an
//! agent's model does not change what it remembers; history is attributed to
//! `agent_id`, so renaming one does not orphan its past; and an in-flight run
//! pins a [`PinnedDefinition`], so an edit made while it is working does not
//! change the rules under it mid-task.
//!
//! **Model identity is not memory ownership.** Nothing here is keyed by a model
//! id, and nothing should be. A model is a thing an agent currently uses.
//!
//! ## Why this cannot grant anything
//!
//! A definition compiles *down* to an [`AgentProfile`] — see
//! [`AgentDefinition::to_profile`] — and is then handed to
//! [`crate::subagents::inherit::InheritedPolicy::narrow_for`], unchanged. That
//! function intersects every set with the parent's and takes the `min` of every
//! scalar, so a definition asking for a tool the operator does not hold gets
//! the tool refused and recorded, not granted.
//!
//! Writing the registry's own check instead would have been a second opinion
//! about authority, and the second opinion is the one that eventually disagrees.

use serde::{Deserialize, Serialize};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::registry::ModelRole;
use crate::subagents::profile::{
    ceiling, AgentProfile, Isolation, Limits, MemoryScope, SchemaKind, WritePolicy,
};

pub mod store;

/// The layout of the stored registry.
///
/// Bumped when a field changes meaning. A registry written under a version this
/// build does not know is refused rather than partly read — half-understanding
/// a record that decides what an agent may do is the case where being wrong is
/// silent.
pub const AGENT_SCHEMA_VERSION: u32 = 1;

/// The colours an agent may be given.
///
/// ## Why a fixed palette rather than any colour
///
/// Because these are drawn on a black canvas beside each other, and the
/// question a person asks of the graph is "which of these did agent B write".
/// Two agents a few degrees of hue apart answer that question wrongly, and
/// nothing warns anybody: the picture simply reads as one agent.
///
/// These eight are separable at the sizes the canvas draws, and each holds
/// enough contrast against `#000` to be legible as a thin edge. Colour carries
/// *ownership* here and nothing else — type stays a shape and status stays
/// luminance, which is the rule `GraphCanvas` already documents.
pub const AGENT_PALETTE: [&str; 8] = [
    "#60A5FA", "#2DD4BF", "#A78BFA", "#F472B6", "#FBBF24", "#FB923C", "#4ADE80", "#F87171",
];

/// The neutral used for an item no single agent owns.
pub const SHARED_COLOR: &str = "#94A3B8";

/// Where an agent stands with the deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AgentState {
    /// Available to be given work.
    Enabled,
    /// Kept, and not offered. A pause, not a deletion.
    Disabled,
    /// Retired. Still resolvable, because history attributes work to it and a
    /// record that cannot name its author is not a record.
    Archived,
}

impl AgentState {
    /// Whether an agent in this state may be given new work.
    pub fn is_runnable(self) -> bool {
        matches!(self, AgentState::Enabled)
    }

    /// Whether somebody who is not an administrator should see it at all.
    ///
    /// A disabled or archived agent is deployment configuration. An employee
    /// shown one in a picker would be offered something that cannot run.
    pub fn is_visible_to_operators(self) -> bool {
        matches!(self, AgentState::Enabled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AgentState::Enabled => "enabled",
            AgentState::Disabled => "disabled",
            AgentState::Archived => "archived",
        }
    }
}

/// One skill an agent is bound to, pinned to the bytes it was bound to.
///
/// The version *and* the hash, because they answer different questions. The
/// version is what an author meant; the hash is what is on disk. A skill edited
/// without its version being bumped is the case that makes a pinned version a
/// fiction, and the hash is what notices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillBinding {
    pub name: String,
    pub version: String,
    /// SHA-256 of the whole `SKILL.md`, as [`crate::skills::manifest`] computes
    /// it. A run pins this, so a skill that changes under a working agent is
    /// detectable rather than silently in effect.
    pub sha256: String,
}

/// Which models an agent may use, and which it prefers.
///
/// Mutable, and deliberately the only mutable identity-adjacent thing here.
/// Reassigning it changes what an agent runs on and changes nothing about who
/// it is, what it remembers or what it has done.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelBinding {
    /// The model to use when nothing says otherwise.
    ///
    /// `None` means "whatever routing picks for the role", which is what every
    /// bundled profile meant before this existed.
    pub default_model_id: Option<String>,
    /// Tried, in order, when the default cannot be served.
    pub fallback_model_ids: Vec<String>,
    /// The whole set this agent may be routed to. Empty means any model
    /// registered for its role — the existing behaviour, said explicitly.
    pub eligible_model_ids: Vec<String>,
}

impl ModelBinding {
    /// The same binding with a different default model.
    ///
    /// The fallbacks and the eligible set are carried across untouched. They are
    /// an administrator's statement about what this agent may *ever* be routed
    /// to, and a handoff that widened them on the way past would be a handoff
    /// that granted something. A target outside a non-empty eligible set is
    /// refused by
    /// [`crate::agent_runtime::model_transition::validate_target`] rather than
    /// quietly added here.
    pub fn rebound_to(&self, model_id: &str) -> Self {
        Self {
            default_model_id: Some(model_id.to_string()),
            fallback_model_ids: self.fallback_model_ids.clone(),
            eligible_model_ids: self.eligible_model_ids.clone(),
        }
    }

    /// Whether two bindings name the same models, in the same preference.
    ///
    /// Exists so [`store::AgentRegistry::update`] can tell an ordinary edit from
    /// a model reassignment and refuse the second. Order-sensitive on purpose:
    /// moving a model from third fallback to first changes what this agent runs
    /// on when its default cannot be served, which is a binding change even
    /// though the set is identical.
    pub fn names_same_models(&self, other: &Self) -> bool {
        self.default_model_id == other.default_model_id
            && self.fallback_model_ids == other.fallback_model_ids
            && self.eligible_model_ids == other.eligible_model_ids
    }

    /// Every model id named here, in preference order, without duplicates.
    pub fn preference_order(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for id in self
            .default_model_id
            .iter()
            .chain(self.fallback_model_ids.iter())
            .chain(self.eligible_model_ids.iter())
        {
            if !out.iter().any(|held| held == id) {
                out.push(id.clone());
            }
        }
        out
    }
}

/// What an agent may remember, and who may read it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryPolicy {
    /// How long anything this agent records survives.
    pub scope: MemoryScope,
    /// Whether what it records is readable by the other agents on a task.
    ///
    /// Separate from `scope` because they are different questions: `Run` scope
    /// says "this outlives the child", and this says "another agent on the same
    /// task may read it". A worker that remembers privately and a worker that
    /// publishes to the task are both reasonable, and conflating them would
    /// make one of the two unavailable.
    pub shared_with_task: bool,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        // Nothing survives and nothing is shared. The narrow reading, because
        // widening it later is a decision somebody makes and narrowing it later
        // is a decision that silently breaks a working agent.
        Self {
            scope: MemoryScope::None,
            shared_with_task: false,
        }
    }
}

/// Where a definition came from, when it was not written here.
///
/// The durable half of idempotent import: an agent imported from a bundled
/// profile carries the profile's name and hash, so a second import recognises
/// it as the same agent rather than making another one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportOrigin {
    /// `bundled` for a profile the installer shipped.
    pub source: String,
    /// The profile's own name, which is what the import is keyed on.
    pub profile_name: String,
    /// The hash of the profile file this was last imported from. A re-import
    /// with a different hash updates the shipped fields; the same hash is a
    /// no-op.
    pub profile_sha256: String,
}

/// Why a definition was not accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "camelCase")]
pub enum InvalidDefinition {
    MissingField {
        field: String,
    },
    /// A colour outside the palette. Refused rather than rounded to the nearest
    /// one: a colour nobody chose is worse than a refusal somebody can answer.
    UnknownColor {
        color: String,
    },
    /// The same tool in both lists. Refused rather than resolved, because
    /// either reading is a guess about what the author meant — the same rule
    /// `subagents::profile` holds.
    ContradictoryTool {
        tool: String,
    },
    /// Above a hard ceiling. See [`ceiling`].
    AboveCeiling {
        field: String,
        asked: u64,
        ceiling: u64,
    },
    /// A model id, skill name or other reference that resolves to nothing.
    UnresolvedReference {
        kind: String,
        value: String,
    },
}

impl InvalidDefinition {
    pub fn explain(&self) -> String {
        match self {
            Self::MissingField { field } => format!("{field} is required and was not given."),
            Self::UnknownColor { color } => format!(
                "{color} is not one of the agent colours. Choose one of: {}.",
                AGENT_PALETTE.join(", ")
            ),
            Self::ContradictoryTool { tool } => format!(
                "{tool} is both allowed and denied. Which was meant cannot be guessed, so the \
                 definition is refused rather than resolved one way."
            ),
            Self::AboveCeiling {
                field,
                asked,
                ceiling,
            } => format!("{field} asks for {asked}, and the hard ceiling is {ceiling}."),
            Self::UnresolvedReference { kind, value } => {
                format!("the {kind} {value:?} does not resolve on this machine.")
            }
        }
    }
}

/// An agent, as the deployment holds it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AgentDefinition {
    /// Stable for the life of the agent. Never derived from the name, the
    /// model, the file it came from or anything else a person may change.
    pub agent_id: String,
    /// Which revision of this agent's definition this is.
    ///
    /// Immutable in the sense that matters: a given version describes one set
    /// of rules for ever. An edit produces the *next* version rather than
    /// changing this one, which is what lets an in-flight run pin a number and
    /// know what it pinned.
    pub definition_version: u64,

    pub display_name: String,
    pub description: String,
    /// What the agent is told it is for. Reaches the model; everything else
    /// here is for Rust.
    pub instructions: String,
    pub role: ModelRole,
    pub state: AgentState,
    /// One of [`AGENT_PALETTE`].
    pub color: String,

    pub skills: Vec<SkillBinding>,
    pub allowed_tools: Vec<ToolName>,
    /// Wins over `allowed_tools`, over the parent's grant, over everything.
    pub denied_tools: Vec<ToolName>,

    pub memory: MemoryPolicy,
    pub models: ModelBinding,
    pub output_schema: SchemaKind,

    pub limits: Limits,
    /// How many of this agent may work at once.
    ///
    /// One by default. A GPU holds one model at a time in the ordinary case, so
    /// "logically parallel" and "simultaneously resident" are different claims
    /// and this is the second one.
    pub max_concurrent: u8,

    pub isolation: Isolation,
    pub write_policy: WritePolicy,
    pub classification_ceiling: Classification,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<ImportOrigin>,
    /// RFC 3339, UTC.
    pub created_at: String,
    pub updated_at: String,
}

impl AgentDefinition {
    /// Checks everything that can be checked without touching the machine.
    ///
    /// Reference resolution — does this model exist, is this skill installed —
    /// is deliberately not here: it needs the registries, it changes over time,
    /// and a definition that is *currently* unresolvable is still a definition
    /// worth keeping. [`Self::unresolved_against`] answers that separately.
    pub fn validate(&self) -> Result<(), InvalidDefinition> {
        if self.agent_id.trim().is_empty() {
            return Err(InvalidDefinition::MissingField {
                field: "agentId".into(),
            });
        }
        if self.display_name.trim().is_empty() {
            return Err(InvalidDefinition::MissingField {
                field: "displayName".into(),
            });
        }
        if !AGENT_PALETTE.contains(&self.color.as_str()) {
            return Err(InvalidDefinition::UnknownColor {
                color: self.color.clone(),
            });
        }
        for tool in &self.denied_tools {
            if self.allowed_tools.contains(tool) {
                return Err(InvalidDefinition::ContradictoryTool {
                    tool: tool.as_str().to_string(),
                });
            }
        }
        if self.limits.max_turns > ceiling::MAX_TURNS {
            return Err(InvalidDefinition::AboveCeiling {
                field: "limits.maxTurns".into(),
                asked: u64::from(self.limits.max_turns),
                ceiling: u64::from(ceiling::MAX_TURNS),
            });
        }
        if self.limits.max_output_tokens > ceiling::MAX_OUTPUT_TOKENS {
            return Err(InvalidDefinition::AboveCeiling {
                field: "limits.maxOutputTokens".into(),
                asked: u64::from(self.limits.max_output_tokens),
                ceiling: u64::from(ceiling::MAX_OUTPUT_TOKENS),
            });
        }
        if self.limits.max_children > ceiling::MAX_DEPTH {
            return Err(InvalidDefinition::AboveCeiling {
                field: "limits.maxChildren".into(),
                asked: u64::from(self.limits.max_children),
                ceiling: u64::from(ceiling::MAX_DEPTH),
            });
        }
        if self.max_concurrent == 0 {
            return Err(InvalidDefinition::AboveCeiling {
                field: "maxConcurrent".into(),
                asked: 0,
                ceiling: 1,
            });
        }
        Ok(())
    }

    /// References this definition names that do not resolve right now.
    ///
    /// Reported rather than refused. A model uninstalled after an agent was
    /// configured is an ordinary state of a deployment, and deleting the
    /// agent's configuration over it would lose work. The administration screen
    /// shows these so an agent is not presented as ready because its record
    /// parsed.
    pub fn unresolved_against(
        &self,
        known_models: &[String],
        known_skills: &[SkillBinding],
    ) -> Vec<InvalidDefinition> {
        let mut out = Vec::new();
        for id in self.models.preference_order() {
            if !known_models.iter().any(|known| known == &id) {
                out.push(InvalidDefinition::UnresolvedReference {
                    kind: "model".into(),
                    value: id,
                });
            }
        }
        for binding in &self.skills {
            match known_skills.iter().find(|known| known.name == binding.name) {
                None => out.push(InvalidDefinition::UnresolvedReference {
                    kind: "skill".into(),
                    value: binding.name.clone(),
                }),
                // Installed, but not the bytes this agent was bound to. Named
                // separately from "missing", because the remedy is different:
                // one is an install, the other is a decision about whether the
                // change is wanted.
                Some(known) if known.sha256 != binding.sha256 => {
                    out.push(InvalidDefinition::UnresolvedReference {
                        kind: "skill revision".into(),
                        value: format!(
                            "{} pinned at {} and installed at {}",
                            binding.name,
                            &binding.sha256[..binding.sha256.len().min(12)],
                            &known.sha256[..known.sha256.len().min(12)]
                        ),
                    })
                }
                Some(_) => {}
            }
        }
        out
    }

    /// The compiler's shape, so the existing narrowing can be reused unchanged.
    ///
    /// ## Why this exists rather than a `narrow` of its own
    ///
    /// [`crate::subagents::inherit::InheritedPolicy::narrow_for`] is where
    /// authority is decided: it intersects every set with the parent's and
    /// takes the `min` of every scalar, and it is the only place an
    /// `EffectivePolicy` can be constructed. Writing a second narrowing here
    /// would be a second opinion about authority, and the second opinion is the
    /// one that eventually disagrees.
    ///
    /// So a definition is converted into the thing that function already takes,
    /// and the answer is computed by the code that has always computed it. A
    /// definition therefore *cannot* grant a tool, a clearance or a depth: the
    /// widest it can be is what the operator already holds.
    pub fn to_profile(&self) -> AgentProfile {
        AgentProfile {
            // The stable id, not the display name. Narrowing records the
            // profile it derived from, and a record that named a display name
            // would stop being true the first time somebody renamed the agent.
            name: self.agent_id.clone(),
            description: self.description.clone(),
            version: self.definition_version.to_string(),
            model_role: self.role,
            eligible_models: self.models.eligible_model_ids.clone(),
            allowed_tools: self.allowed_tools.clone(),
            disallowed_tools: self.denied_tools.clone(),
            limits: self.limits.clone(),
            isolation: self.isolation,
            memory_scope: self.memory.scope,
            // Always false, and not configurable here. A registry field that
            // could turn this on would be a registry that grants network
            // access, which is the one thing this product's design says nothing
            // may do without a reviewed decision elsewhere.
            network_permitted: false,
            write_policy: self.write_policy,
            classification_ceiling: self.classification_ceiling,
            required_schema: self.output_schema,
            // The definition's own identity, so a trace can join a narrowing
            // back to the exact revision it was derived from.
            sha256: self.revision_key(),
            // What a worker for this agent is told it is for. The registry's
            // own field, so an agent created on the administration screen
            // carries instructions exactly as a bundled profile does.
            instructions: self.instructions.clone(),
        }
    }

    /// A stable key for this exact revision, for pinning and for traces.
    pub fn revision_key(&self) -> String {
        format!("{}@{}", self.agent_id, self.definition_version)
    }

    /// What an in-flight run holds on to.
    pub fn pin(&self) -> PinnedDefinition {
        PinnedDefinition {
            agent_id: self.agent_id.clone(),
            definition_version: self.definition_version,
            display_name: self.display_name.clone(),
            color: self.color.clone(),
            skills: self.skills.clone(),
            model_binding: self.models.clone(),
        }
    }
}

/// The definition a run is held to, captured when it started.
///
/// ## Why a run pins rather than re-reads
///
/// Because an administrator editing an agent while it is working would
/// otherwise change the rules under it mid-task: the tool it was about to call
/// becomes denied, the skill it has been using changes underneath it, the model
/// it is mid-conversation with is reassigned. None of those are wrong things to
/// want; all of them are wrong to apply halfway through a task.
///
/// So a run reads the definition once, at a boundary where nothing is in
/// flight, and is held to that. A later edit takes effect at the next such
/// boundary — the next run, or the next resumption — which is a rule somebody
/// can predict.
///
/// The colour and display name are pinned too, for a different reason: the
/// trace and the graph should show what this run *was* attributed to, and
/// re-reading them later would silently rewrite history to match the present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PinnedDefinition {
    pub agent_id: String,
    pub definition_version: u64,
    pub display_name: String,
    pub color: String,
    pub skills: Vec<SkillBinding>,
    pub model_binding: ModelBinding,
}

impl PinnedDefinition {
    /// Whether a stored definition is still the one this run pinned.
    pub fn matches(&self, current: &AgentDefinition) -> bool {
        current.agent_id == self.agent_id && current.definition_version == self.definition_version
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A valid definition, for tests in this crate that need one.
    ///
    /// Visible to the crate rather than to this module: the administration
    /// commands need exactly this fixture, and a second copy of a struct with
    /// twenty fields is a second thing to update when one of them changes.
    pub(crate) fn definition(id: &str) -> AgentDefinition {
        AgentDefinition {
            agent_id: id.to_string(),
            definition_version: 1,
            display_name: "Knowledge retriever".into(),
            description: "Finds passages and cites them.".into(),
            instructions: "Answer only from the passages you retrieve.".into(),
            role: ModelRole::Reasoning,
            state: AgentState::Enabled,
            color: AGENT_PALETTE[0].to_string(),
            skills: Vec::new(),
            allowed_tools: vec![ToolName::SearchDocuments],
            denied_tools: Vec::new(),
            memory: MemoryPolicy::default(),
            models: ModelBinding::default(),
            output_schema: SchemaKind::Retrieval,
            limits: Limits {
                max_turns: 8,
                max_output_tokens: 2048,
                max_children: 0,
                max_duration_seconds: 300,
            },
            max_concurrent: 1,
            isolation: Isolation::ReadOnly,
            write_policy: WritePolicy::None,
            classification_ceiling: Classification::Internal,
            imported_from: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn a_well_formed_definition_validates() {
        assert!(definition("ag-1").validate().is_ok());
    }

    #[test]
    fn a_colour_outside_the_palette_is_refused_rather_than_rounded() {
        let mut agent = definition("ag-1");
        agent.color = "#123456".into();
        let refusal = agent.validate().expect_err("an unknown colour is refused");
        assert_eq!(
            refusal,
            InvalidDefinition::UnknownColor {
                color: "#123456".into()
            }
        );
        // And the message names the alternatives, so somebody can act on it.
        assert!(refusal.explain().contains("#60A5FA"));
    }

    #[test]
    fn a_tool_in_both_lists_is_refused_rather_than_resolved() {
        let mut agent = definition("ag-1");
        agent.denied_tools = vec![ToolName::SearchDocuments];
        assert!(matches!(
            agent.validate(),
            Err(InvalidDefinition::ContradictoryTool { .. })
        ));
    }

    #[test]
    fn limits_above_the_hard_ceiling_are_refused() {
        let mut agent = definition("ag-1");
        agent.limits.max_turns = ceiling::MAX_TURNS + 1;
        assert!(matches!(
            agent.validate(),
            Err(InvalidDefinition::AboveCeiling { .. })
        ));

        let mut deep = definition("ag-1");
        deep.limits.max_children = ceiling::MAX_DEPTH + 1;
        assert!(matches!(
            deep.validate(),
            Err(InvalidDefinition::AboveCeiling { .. })
        ));
    }

    #[test]
    fn an_agent_that_can_never_run_is_refused() {
        let mut agent = definition("ag-1");
        agent.max_concurrent = 0;
        assert!(agent.validate().is_err());
    }

    /// The record is read strictly. A field this build does not know is a
    /// record written by a newer one, and acting on the half it understands is
    /// how an agent ends up with permissions nobody granted.
    #[test]
    fn an_unknown_field_in_a_stored_record_is_refused() {
        let mut value = serde_json::to_value(definition("ag-1")).expect("serialises");
        value["networkPermitted"] = serde_json::Value::Bool(true);
        let parsed: Result<AgentDefinition, _> = serde_json::from_value(value);
        assert!(
            parsed.is_err(),
            "a field this build does not know must not be ignored"
        );
    }

    /// The compiled profile is named by the stable id, so a narrowing recorded
    /// today still names the right agent after a rename.
    #[test]
    fn the_compiled_profile_is_named_by_the_stable_id() {
        let mut agent = definition("ag-7f3c");
        agent.display_name = "Renamed entirely".into();
        let profile = agent.to_profile();
        assert_eq!(profile.name, "ag-7f3c");
        assert_eq!(profile.sha256, "ag-7f3c@1");
    }

    /// The registry cannot turn the network on, whatever a record says.
    #[test]
    fn a_compiled_profile_never_permits_the_network() {
        assert!(!definition("ag-1").to_profile().network_permitted);
    }

    #[test]
    fn the_denied_list_reaches_the_compiler() {
        let mut agent = definition("ag-1");
        agent.allowed_tools = vec![ToolName::SearchDocuments];
        agent.denied_tools = vec![ToolName::ExecuteCode];
        let profile = agent.to_profile();
        assert!(profile.disallowed_tools.contains(&ToolName::ExecuteCode));
        assert!(!profile
            .requested_tools()
            .contains(&ToolName::ExecuteCode));
    }

    #[test]
    fn a_pin_notices_when_the_definition_has_moved_on() {
        let agent = definition("ag-1");
        let pinned = agent.pin();
        assert!(pinned.matches(&agent));

        let mut edited = agent.clone();
        edited.definition_version = 2;
        assert!(
            !pinned.matches(&edited),
            "an edited definition must not satisfy a pin taken before it"
        );
    }

    /// A pin carries what the run should be *attributed* to, so a later rename
    /// does not rewrite what a finished run's trace says.
    #[test]
    fn a_pin_carries_the_name_and_colour_the_run_ran_under() {
        let agent = definition("ag-1");
        let pinned = agent.pin();
        assert_eq!(pinned.display_name, "Knowledge retriever");
        assert_eq!(pinned.color, AGENT_PALETTE[0]);
    }

    #[test]
    fn model_preference_is_ordered_and_deduplicated() {
        let binding = ModelBinding {
            default_model_id: Some("a".into()),
            fallback_model_ids: vec!["b".into(), "a".into()],
            eligible_model_ids: vec!["a".into(), "b".into(), "c".into()],
        };
        assert_eq!(binding.preference_order(), vec!["a", "b", "c"]);
    }

    #[test]
    fn unresolved_models_and_skills_are_reported_not_refused() {
        let mut agent = definition("ag-1");
        agent.models.default_model_id = Some("a-model-nobody-installed".into());
        agent.skills.push(SkillBinding {
            name: "hazop-analyzer".into(),
            version: "1.0.0".into(),
            sha256: "aaaa".into(),
        });

        // Still a valid definition: an uninstalled model is a state of the
        // machine, not a fault in the record.
        assert!(agent.validate().is_ok());

        let problems = agent.unresolved_against(&[], &[]);
        assert_eq!(problems.len(), 2);
        assert!(problems.iter().any(|problem| matches!(
            problem,
            InvalidDefinition::UnresolvedReference { kind, .. } if kind == "model"
        )));
        assert!(problems.iter().any(|problem| matches!(
            problem,
            InvalidDefinition::UnresolvedReference { kind, .. } if kind == "skill"
        )));
    }

    /// A skill installed at different bytes than the agent pinned is its own
    /// finding: the remedy is a decision, not an install.
    #[test]
    fn a_skill_installed_at_a_different_revision_is_named_as_such() {
        let mut agent = definition("ag-1");
        agent.skills.push(SkillBinding {
            name: "hazop-analyzer".into(),
            version: "1.0.0".into(),
            sha256: "aaaaaaaaaaaaaaaa".into(),
        });
        let installed = vec![SkillBinding {
            name: "hazop-analyzer".into(),
            version: "1.0.0".into(),
            sha256: "bbbbbbbbbbbbbbbb".into(),
        }];
        let problems = agent.unresolved_against(&[], &installed);
        assert_eq!(problems.len(), 1);
        assert!(matches!(
            &problems[0],
            InvalidDefinition::UnresolvedReference { kind, .. } if kind == "skill revision"
        ));
    }

    #[test]
    fn only_an_enabled_agent_is_runnable_or_offered() {
        assert!(AgentState::Enabled.is_runnable());
        assert!(!AgentState::Disabled.is_runnable());
        assert!(!AgentState::Archived.is_runnable());
        assert!(AgentState::Enabled.is_visible_to_operators());
        assert!(!AgentState::Archived.is_visible_to_operators());
    }

    #[test]
    fn the_palette_is_eight_distinct_colours_and_excludes_the_shared_neutral() {
        let mut seen = std::collections::BTreeSet::new();
        for colour in AGENT_PALETTE {
            assert!(seen.insert(colour), "{colour} appears twice");
            assert!(colour.starts_with('#') && colour.len() == 7);
        }
        assert!(
            !AGENT_PALETTE.contains(&SHARED_COLOR),
            "the neutral for shared items must not be assignable to an agent"
        );
    }
}

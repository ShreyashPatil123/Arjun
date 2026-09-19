//! Changing the model an agent is bound to, as an act with its own states.
//!
//! ## Why a router configuration change is not a handoff
//!
//! Before this existed there were two ways an agent's model could change, and
//! neither was a transition.
//!
//! The first was [`crate::agents::store::AgentRegistry::update`], which took a
//! whole [`ModelBinding`] among an agent's other editable fields. Saving that
//! form was a one-line write: the next turn routed somewhere else, the
//! conversation carried on, and nothing recorded that the model had changed —
//! let alone checked that the new one could serve the window the old turn was
//! budgeted against, or that the notes being handed over were written by a model
//! whose tokenizer the new one does not share.
//!
//! The second was `drive_run`'s resumption path, which re-ran routing on every
//! resumption. A run interrupted under one model could continue under another
//! whenever the registry, the free VRAM or the conversation's sticky preference
//! had moved. That was closed by refusing it outright — see
//! [`crate::commands::agent`] — and a refusal is the right answer only while
//! there is no honest third one. This module is the third one.
//!
//! ## What a handoff has to establish, and in what order
//!
//! The ordering is the design. Each step is where it is because doing it later
//! would make one of the others a lie.
//!
//! | Phase | What it establishes | Why it cannot come later |
//! |---|---|---|
//! | [`Draining`](TransitionPhase::Draining) | no model or tool step is in flight | a model swapped under a running round hands half a turn to a stranger |
//! | [`AwaitingSettlement`](TransitionPhase::AwaitingSettlement) | every effect is settled or classified | an unsettled effect is a question, and the new model cannot answer it |
//! | [`Checkpointed`](TransitionPhase::Checkpointed) | the task state is on disk | the next phase may unload the source model, and a crash then loses it |
//! | [`Validating`](TransitionPhase::Validating) | the target may hold this work | loading first spends a model load on a refusal |
//! | [`Loading`](TransitionPhase::Loading) | the target is actually up | a binding to a model that cannot be served is not a binding |
//! | [`Recompiling`](TransitionPhase::Recompiling) | the context fits the target's real window | the served window is not knowable until the server reports it |
//! | [`CommitPending`](TransitionPhase::CommitPending) | the intent to rebind is durable | a crash between the two stores would otherwise be unresolvable |
//! | [`Committed`](TransitionPhase::Committed) | the agent is on the target | — |
//!
//! ## Why the binding is committed last, and what a crash means at each point
//!
//! The registry write is the commit point, and it happens after the target is
//! serving and the context has been recompiled for it. That makes both crash
//! windows readable without a guess:
//!
//! - **Crashed before the commit.** The agent still names the source model, and
//!   the checkpoint taken at [`Checkpointed`](TransitionPhase::Checkpointed) is
//!   the resume point. The run continues where it was, on the model it was on.
//! - **Crashed after the commit.** The agent names the target, the record says
//!   so, and the recompiled manifest is what a resumption rebuilds from.
//! - **Crashed between the two stores.** The one ambiguous window, and
//!   [`CommitPending`](TransitionPhase::CommitPending) removes the ambiguity: the
//!   row is written *before* the registry write and carries the version the
//!   registry is expected to be at afterwards. Reconciliation then reads the
//!   registry and gets a fact rather than a preference — see
//!   [`PendingCommit::reconcile`].
//!
//! ## What is not carried across, and why that is not a loss
//!
//! Nothing opaque. A KV cache belongs to one model's attention geometry, a
//! prompt cache belongs to one server process, and a provider's compaction
//! object is a summary produced by — and only readable by — the model that made
//! it. Handing any of them to a different tokenizer produces text that is
//! fluent and wrong, which is worse than producing nothing.
//!
//! So [`PortableState`] is *reconstructed* rather than transferred: graph
//! records at a frozen revision, source references by content hash, receipts the
//! durable event log corroborates, and the typed notes
//! [`super::state_commit`] has already checked. What was left behind is named in
//! the record — see [`DroppedState`] — because an operator reading a different
//! answer after a handoff deserves to know what the new model was not given.

use serde::{Deserialize, Serialize};

use super::context_manifest::SelectedItem;
use super::events::machine::RunState;
use super::events::model::digest;
use super::memory::{CompletedEffect, RunMemory};
use crate::agents::{AgentDefinition, ModelBinding};
use crate::policy::Classification;
use crate::registry::{Modality, ModelEntry, ModelRole};

/// The layout of a stored transition record.
///
/// Bumped when a field changes meaning. A record written under a version this
/// build does not know is reported rather than partly read: it decides which
/// model an agent is on, and half-understanding that is the case where being
/// wrong is silent.
pub const TRANSITION_SCHEMA_VERSION: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// The states
// ─────────────────────────────────────────────────────────────────────────────

/// Where a model-binding transition has got to.
///
/// Explicit states rather than a boolean, because "it did not work" splits into
/// four situations with four different remedies: refused before anything moved,
/// parked waiting for a person, undone back to the source, and undone
/// unsuccessfully. One word for all four would send somebody to the wrong place
/// every time but one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransitionPhase {
    /// Accepted and recorded. Nothing has been touched.
    Requested,
    /// No new model or tool step is being scheduled; whatever is running is
    /// being allowed to finish.
    Draining,
    /// Something is in flight that only a person can settle — a pending
    /// approval, an effect whose outcome nobody recorded. Not a failure: a
    /// question waiting for an answer.
    AwaitingSettlement,
    /// The task state is durably written, and the source memory and context
    /// revision are frozen.
    Checkpointed,
    /// The target is being checked against this agent's policy, capabilities,
    /// template, tokenizer, context capacity and serving admission.
    Validating,
    /// The target is being started and health-checked.
    Loading,
    /// Context is being recompiled for the window the target actually holds.
    Recompiling,
    /// The intent to rebind is durable and the registry write is next. The one
    /// phase that exists purely so a crash has a determinate meaning.
    CommitPending,
    /// The agent is bound to the target. Terminal.
    Committed,
    /// Undoing back to the source binding.
    RollingBack,
    /// Back on the source binding, with the last committed task state intact.
    /// Terminal.
    RolledBack,
    /// Refused or abandoned before anything changed. Terminal.
    Failed,
    /// The rollback itself did not complete. Terminal, and needs a person: the
    /// agent may be bound to a model that is not serving.
    RollbackFailed,
}

impl TransitionPhase {
    pub const ALL: &'static [TransitionPhase] = &[
        TransitionPhase::Requested,
        TransitionPhase::Draining,
        TransitionPhase::AwaitingSettlement,
        TransitionPhase::Checkpointed,
        TransitionPhase::Validating,
        TransitionPhase::Loading,
        TransitionPhase::Recompiling,
        TransitionPhase::CommitPending,
        TransitionPhase::Committed,
        TransitionPhase::RollingBack,
        TransitionPhase::RolledBack,
        TransitionPhase::Failed,
        TransitionPhase::RollbackFailed,
    ];

    /// Stable database spelling, written out rather than derived from the
    /// variant name so renaming a variant cannot rewrite history.
    pub const fn as_str(self) -> &'static str {
        match self {
            TransitionPhase::Requested => "requested",
            TransitionPhase::Draining => "draining",
            TransitionPhase::AwaitingSettlement => "awaiting_settlement",
            TransitionPhase::Checkpointed => "checkpointed",
            TransitionPhase::Validating => "validating",
            TransitionPhase::Loading => "loading",
            TransitionPhase::Recompiling => "recompiling",
            TransitionPhase::CommitPending => "commit_pending",
            TransitionPhase::Committed => "committed",
            TransitionPhase::RollingBack => "rolling_back",
            TransitionPhase::RolledBack => "rolled_back",
            TransitionPhase::Failed => "failed",
            TransitionPhase::RollbackFailed => "rollback_failed",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|phase| phase.as_str() == raw)
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            TransitionPhase::Committed
                | TransitionPhase::RolledBack
                | TransitionPhase::Failed
                | TransitionPhase::RollbackFailed
        )
    }

    /// Whether reaching this phase could have taken the source model's memory.
    ///
    /// Only the phases at or after the load. `serving::admission::admit` is what
    /// evicts, and nothing before [`Self::Loading`] has called it — so a
    /// transition abandoned at [`Self::Validating`] has released nothing, and an
    /// undo that reloaded "the source" there would be starting a model that
    /// nothing had stopped.
    ///
    /// Whether the source *was in fact* released is narrower still, and is
    /// recorded per transition: see [`TransitionRecord::released`].
    pub const fn may_have_released_memory(self) -> bool {
        matches!(
            self,
            TransitionPhase::Loading
                | TransitionPhase::Recompiling
                | TransitionPhase::CommitPending
                | TransitionPhase::RollingBack
        )
    }

    /// What a screen shows for this phase.
    pub const fn outcome(self) -> TransitionOutcome {
        match self {
            TransitionPhase::Committed => TransitionOutcome::Succeeded,
            TransitionPhase::Failed => TransitionOutcome::Failed,
            TransitionPhase::RolledBack => TransitionOutcome::RolledBack,
            TransitionPhase::RollbackFailed => TransitionOutcome::RollbackFailed,
            // Everything else is still going, including the parked state. A
            // transition waiting for a person has not failed.
            _ => TransitionOutcome::Pending,
        }
    }

    /// The phases this one may legally move to.
    ///
    /// Written as a list rather than as a chain of conditions so the machine can
    /// be enumerated in a test: every phase, every target, and the edges that
    /// are refused are refused on purpose. A transition that could skip
    /// [`Self::Checkpointed`] on its way to [`Self::Loading`] would be one that
    /// unloads the source model with the task state still only in memory.
    pub fn may_advance_to(self, next: TransitionPhase) -> bool {
        use TransitionPhase::*;

        // Abandoning is reachable from wherever it is discovered, but only while
        // nothing has been *disturbed*. The line is the load, not the
        // checkpoint: writing a run's state down is not something an undo has
        // to reverse — a rollback deliberately *keeps* it — whereas admitting
        // the target to memory can evict the model the run was using, and that
        // has to be put back.
        if next == Failed {
            return matches!(
                self,
                Requested | Draining | AwaitingSettlement | Checkpointed | Validating
            );
        }
        if next == RollingBack {
            return matches!(self, Loading | Recompiling | CommitPending);
        }
        matches!(
            (self, next),
            (Requested, Draining)
                | (Draining, AwaitingSettlement)
                | (Draining, Checkpointed)
                // Re-attempted once a person has settled what was in flight.
                | (AwaitingSettlement, Draining)
                | (Checkpointed, Validating)
                | (Validating, Loading)
                // An inactive agent that is not verified by loading goes
                // straight to the commit: there is no served window to
                // recompile against, and the record says so rather than
                // inventing one.
                | (Validating, CommitPending)
                | (Loading, Recompiling)
                | (Recompiling, CommitPending)
                | (CommitPending, Committed)
                | (RollingBack, RolledBack)
                | (RollingBack, RollbackFailed)
        )
    }
}

/// What a transition came to, in the words a surface needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransitionOutcome {
    Succeeded,
    /// Still going, or parked waiting for a person.
    Pending,
    /// Refused or abandoned. Nothing changed.
    Failed,
    /// Undone. The agent is on the model it started on and the last committed
    /// task state is intact.
    RolledBack,
    /// Undone unsuccessfully. Needs a person.
    RollbackFailed,
}

impl TransitionOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            TransitionOutcome::Succeeded => "succeeded",
            TransitionOutcome::Pending => "pending",
            TransitionOutcome::Failed => "failed",
            TransitionOutcome::RolledBack => "rolledBack",
            TransitionOutcome::RollbackFailed => "rollbackFailed",
        }
    }

    /// Whether somebody has to do something before this agent is usable.
    pub const fn needs_human(self) -> bool {
        matches!(self, TransitionOutcome::RollbackFailed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The drain boundary
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the run may be handed over from where it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "camelCase")]
pub enum DrainVerdict {
    /// Nothing is in flight. The handoff may proceed.
    Safe,
    /// A model call or a tool is running. Stop scheduling new steps and come
    /// back at the next boundary.
    Drain { because: String },
    /// A person has to settle something first. Carries what.
    NeedsSettlement { because: String, keys: Vec<String> },
}

impl DrainVerdict {
    pub fn is_safe(&self) -> bool {
        matches!(self, DrainVerdict::Safe)
    }

    /// Whether waiting would help. A drain resolves itself; a settlement does
    /// not, and telling the two apart is what stops a screen spinning forever on
    /// a question nobody has been asked.
    pub fn resolves_by_waiting(&self) -> bool {
        matches!(self, DrainVerdict::Drain { .. })
    }

    pub fn because(&self) -> Option<&str> {
        match self {
            DrainVerdict::Safe => None,
            DrainVerdict::Drain { because } | DrainVerdict::NeedsSettlement { because, .. } => {
                Some(because)
            }
        }
    }
}

/// Whether a run in this state is safe to hand over from.
///
/// Read off the run state rather than from a flag the loop sets, because the
/// loop is on the far side of a JSON-RPC channel and a flag it owns is a flag it
/// can be wrong about. These states are ones Rust itself wrote down.
pub fn boundary_of(state: RunState) -> DrainVerdict {
    match state {
        // Nothing has started, or the last thing finished and was recorded.
        // `ToolResultRecorded` is the important one: it is *defined* as the
        // point a side effect is known to have settled, which is exactly the
        // property a handoff needs.
        RunState::Created
        | RunState::Classified
        | RunState::Routed
        | RunState::Planned
        | RunState::ToolResultRecorded
        | RunState::Paused => DrainVerdict::Safe,

        RunState::Running => DrainVerdict::Drain {
            because: "a model round is in progress; the handoff waits for it to finish rather \
                      than handing half a turn to a different model"
                .into(),
        },
        RunState::ExecutingTool => DrainVerdict::Drain {
            because: "a tool is running; the handoff waits for its outcome to be recorded, \
                      because an effect that settles after the switch cannot be attributed to \
                      either model"
                .into(),
        },
        RunState::Verifying => DrainVerdict::Drain {
            because: "the answer is being checked against the evidence this run has; switching \
                      now would check one model's answer against another model's reading"
                .into(),
        },
        RunState::Compacting => DrainVerdict::Drain {
            because: "the context is being compacted, so it is half-projected; a switch here \
                      would hand the new model a transcript that is neither shape"
                .into(),
        },
        RunState::Recovering => DrainVerdict::Drain {
            because: "this run is being picked back up after an interruption; the handoff waits \
                      until the recovery has settled what it found"
                .into(),
        },
        RunState::AwaitingApproval => DrainVerdict::NeedsSettlement {
            because: "somebody has been asked to allow an action and has not answered. The \
                      approval authorises one specific call, and the person who would allow it \
                      does so knowing which model will make it — so it is not carried over. \
                      Answer it, and the handoff can proceed from the boundary after."
                .into(),
            keys: Vec::new(),
        },
        RunState::WaitingForExternalEvent => DrainVerdict::NeedsSettlement {
            because: "this run is waiting on something outside ARJUN. Nothing here can tell \
                      whether what arrives will answer the old model's request or the new one's, \
                      so the wait is settled first."
                .into(),
            keys: Vec::new(),
        },

        // Terminal. There is no work to hand over, so as far as this run is
        // concerned the binding change is an ordinary reassignment. Listed
        // rather than caught by a guard so that adding a state to `RunState` is
        // a compile error here instead of a silent "safe".
        RunState::Completed
        | RunState::Cancelled
        | RunState::Failed
        | RunState::StoppedByBudget
        | RunState::StoppedByLength
        | RunState::StoppedByPolicy
        | RunState::DegradedNeedsHuman => DrainVerdict::Safe,
    }
}

/// The drain verdict for a run, given its state, its unsettled effects and any
/// approval it is waiting on.
///
/// Three inputs because they fail separately. A run sitting at a safe boundary
/// can still have an effect from three steps ago that recovery promoted to
/// `Unknown`, and handing that to a new model would be asking it to decide
/// whether a document it cannot see was written.
pub fn drain_for(
    state: RunState,
    unsettled_effects: &[String],
    pending_approvals: &[String],
) -> DrainVerdict {
    if !unsettled_effects.is_empty() {
        return DrainVerdict::NeedsSettlement {
            because: format!(
                "{} action(s) this run started have no recorded outcome. A different model \
                 cannot be told whether they took effect, and guessing either way is how the \
                 same document gets written twice. Reconcile them first.",
                unsettled_effects.len()
            ),
            keys: unsettled_effects.to_vec(),
        };
    }
    if !pending_approvals.is_empty() {
        return DrainVerdict::NeedsSettlement {
            because: format!(
                "{} action(s) are waiting for somebody to allow them. An approval authorises one \
                 specific call and is not carried across a model change — answer them, and the \
                 handoff can proceed from the boundary after.",
                pending_approvals.len()
            ),
            keys: pending_approvals.to_vec(),
        };
    }
    boundary_of(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Identities carried in the record
// ─────────────────────────────────────────────────────────────────────────────

/// Which bytes a model id referred to at the moment of a transition.
///
/// The id alone is not enough. A registry entry is re-importable and a download
/// alias moves — [`ModelEntry::revision`] exists for exactly that reason — so a
/// record naming only `qwen3.5-9b` cannot answer "was the model this agent was
/// on in March the same weights as the one it is on now".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFingerprint {
    pub model_id: String,
    /// SHA-256 of the weights where the registry holds one, otherwise a digest
    /// over the fields that identify the file.
    pub digest: String,
    /// `weightsSha256`, `derived` or `unregistered`. Named rather than folded
    /// away, because "these are the same bytes" and "these are the same registry
    /// entry" are different claims and only the first is a proof.
    pub basis: String,
    pub weights_bytes: u64,
    /// The window the registry declares, which is not the window a server is
    /// started with. See [`ServedBinding::served_window`].
    pub declared_window: u32,
    pub roles: Vec<String>,
}

impl ModelFingerprint {
    /// The fingerprint of a registered entry.
    pub fn of(entry: &ModelEntry) -> Self {
        let (value, basis) = match entry.sha256.as_deref() {
            Some(sha) if !sha.trim().is_empty() => (sha.to_string(), "weightsSha256"),
            _ => (
                digest(&format!(
                    "{}|{}|{}|{}",
                    entry.id,
                    entry.path.display(),
                    entry.weights_bytes,
                    entry.version
                )),
                "derived",
            ),
        };
        Self {
            model_id: entry.id.clone(),
            digest: value,
            basis: basis.to_string(),
            weights_bytes: entry.weights_bytes,
            declared_window: entry.context_length,
            roles: entry
                .roles
                .iter()
                .map(|role| role.label().to_string())
                .collect(),
        }
    }

    /// A fingerprint for a model that is named but not registered here.
    ///
    /// Used when a record has to name the model an agent *was* on and that entry
    /// has since been removed. `basis` says `unregistered`, so nothing reads the
    /// digest as a claim about bytes.
    pub fn unregistered(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            digest: digest(model_id),
            basis: "unregistered".to_string(),
            weights_bytes: 0,
            declared_window: 0,
            roles: Vec::new(),
        }
    }

    /// Whether two fingerprints name the same bytes.
    ///
    /// `false` whenever either was derived rather than hashed, because a derived
    /// digest proves the registry entry matches and says nothing about the file.
    pub fn same_weights_as(&self, other: &Self) -> bool {
        self.basis == "weightsSha256"
            && other.basis == "weightsSha256"
            && self.digest == other.digest
    }
}

/// Where the number a turn is budgeted against came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WindowSource {
    /// The server said so. The only figure that is a measurement.
    ServerReported,
    /// The server would not say, so the registry's declared window was used.
    RegistryDeclared,
    /// Nothing was loaded, so nobody knows. Recorded as this rather than filled
    /// in from the registry, because a window nobody measured is not a window a
    /// context was fitted to.
    NotMeasured,
}

impl WindowSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            WindowSource::ServerReported => "serverReported",
            WindowSource::RegistryDeclared => "registryDeclared",
            WindowSource::NotMeasured => "notMeasured",
        }
    }

    pub const fn is_measured(self) -> bool {
        matches!(self, WindowSource::ServerReported)
    }
}

/// The string a tokenizer probe counts.
///
/// Deliberately full of what this product actually carries — a tag number, a
/// unit, a decimal, an Indian-grouped currency figure — because those are where
/// tokenizers disagree most, and a probe of plain English would often come back
/// equal from two genuinely different vocabularies.
pub const TOKENIZER_PROBE: &str = "PV-2201 gasket torque 47.5 N\u{b7}m; rev C; \u{20b9}12,40,000";

/// What a server's own tokenizer makes of [`TOKENIZER_PROBE`].
///
/// The cheapest honest evidence that two models tokenise differently, and the
/// only one available without shipping their vocabularies: the same string sent
/// to each server's own `POST /tokenize`. Two different counts is a fact; two
/// equal counts is not a proof of sameness, and [`Self::differs_from`] says so.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenizerProbe {
    /// The string that was counted. Part of the record, so two probes are only
    /// ever compared when they counted the same thing.
    pub probe: String,
    pub tokens: u32,
}

impl TokenizerProbe {
    pub fn new(tokens: u32) -> Self {
        Self {
            probe: TOKENIZER_PROBE.to_string(),
            tokens,
        }
    }

    /// Whether these two tokenizers demonstrably disagree.
    pub fn differs_from(&self, other: &Self) -> bool {
        self.probe == other.probe && self.tokens != other.tokens
    }
}

/// The destination as it actually came up.
///
/// Every field here is something a server reported or a probe measured. Nothing
/// is copied from the registry except where [`Self::window_source`] says so.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServedBinding {
    pub model_id: String,
    /// The id the *server* knows it by, which is not always ARJUN's id.
    pub served_model_id: String,
    /// The window the server was started with. Never the trained maximum.
    pub served_window: u32,
    pub window_source: WindowSource,
    /// The chat template identity, when the serving side knows it. Part of the
    /// context's identity because framing costs tokens and differs per template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerProbe>,
    pub supports_toggled_reasoning: bool,
    /// True when the server answered a readiness probe.
    pub healthy: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validating the destination
// ─────────────────────────────────────────────────────────────────────────────

/// Why a model may not take an agent's work.
///
/// Each variant names one thing, because the remedies differ: a model outside
/// the eligible set is a decision to widen, a model that cannot hold the
/// mandatory context is a different model, and a model that cannot call tools is
/// a different agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "refusal", rename_all = "camelCase")]
pub enum ValidationRefusal {
    /// Nothing in the registry has that id.
    NotRegistered { model_id: String },
    /// Registered and turned off.
    Disabled { model_id: String },
    /// Already the agent's model. Refused rather than treated as a no-op,
    /// because a record saying an agent moved from X to X makes the history
    /// harder to read rather than more complete.
    SameModel { model_id: String },
    /// Outside the set this agent may be routed to.
    NotEligible {
        model_id: String,
        eligible: Vec<String>,
    },
    /// Not registered for the role this agent works in.
    WrongRole {
        model_id: String,
        needs: String,
        serves: Vec<String>,
    },
    /// Not reviewed for material this sensitive.
    ClassificationNotPermitted {
        model_id: String,
        classification: String,
    },
    /// The agent's work needs a modality this model does not have.
    ModalityUnsupported { model_id: String, modality: String },
    /// The agent is bound to tools, and this model cannot be asked to call one.
    ToolsUnsupported {
        model_id: String,
        tools: Vec<String>,
    },
    /// ARJUN cannot start this model's runtime, or the installed build is too
    /// old for its architecture.
    RuntimeUnsupported { model_id: String, detail: String },
    /// The window the destination can actually be served at does not hold the
    /// part of the context that may not be dropped.
    ContextTooSmall {
        model_id: String,
        affords: u32,
        mandatory: u32,
    },
    /// The machine cannot serve it at all.
    AdmissionRefused { model_id: String, detail: String },
    /// It came up and never answered.
    NeverHealthy { model_id: String, detail: String },
}

impl ValidationRefusal {
    /// The sentence an operator is shown.
    pub fn explain(&self) -> String {
        match self {
            Self::NotRegistered { model_id } => format!(
                "{model_id} is not in the model registry on this machine, so an agent cannot be \
                 bound to it. Import it first."
            ),
            Self::Disabled { model_id } => format!(
                "{model_id} is registered and turned off. Enable it before binding an agent to \
                 it — a binding to a disabled model is a binding that cannot run."
            ),
            Self::SameModel { model_id } => format!(
                "This agent is already on {model_id}. Nothing was recorded, because a transition \
                 from a model to itself would put a line in the history saying nothing happened."
            ),
            Self::NotEligible { model_id, eligible } => format!(
                "{model_id} is not in this agent's eligible set ({}). Widening that set is a \
                 change to the agent — made deliberately and version-checked — rather than \
                 something a handoff does on the way past.",
                if eligible.is_empty() {
                    "every model registered for its role".to_string()
                } else {
                    eligible.join(", ")
                }
            ),
            Self::WrongRole {
                model_id,
                needs,
                serves,
            } => format!(
                "{model_id} is registered for {} and this agent works in {needs}. Routing it here \
                 would produce work of the wrong kind rather than worse work of the right kind.",
                if serves.is_empty() {
                    "no role".to_string()
                } else {
                    serves.join(", ")
                }
            ),
            Self::ClassificationNotPermitted {
                model_id,
                classification,
            } => format!(
                "{model_id} has not been reviewed for {classification} material, which is this \
                 agent's ceiling. A model nobody reviewed is not usable on everything."
            ),
            Self::ModalityUnsupported { model_id, modality } => format!(
                "This agent's work needs {modality}, and {model_id} does not support it. It would \
                 answer about a page it cannot see."
            ),
            Self::ToolsUnsupported { model_id, tools } => format!(
                "This agent is bound to {} tool(s) — {} — and {model_id} is not registered as \
                 supporting structured output, so it cannot be asked to call one. The agent would \
                 keep its tools and lose the ability to use them, which is worse than a refusal.",
                tools.len(),
                tools.join(", ")
            ),
            Self::RuntimeUnsupported { model_id, detail } => {
                format!("{model_id} cannot be served on this machine: {detail}")
            }
            Self::ContextTooSmall {
                model_id,
                affords,
                mandatory,
            } => format!(
                "{model_id} can be served with {affords} usable tokens on this machine, and the \
                 part of this task's context that may not be dropped needs {mandatory}. Handing \
                 the work over would mean dropping an objective, a correction or a receipt, so it \
                 is refused instead."
            ),
            Self::AdmissionRefused { model_id, detail } => {
                format!("{model_id} cannot be admitted to memory on this machine: {detail}")
            }
            Self::NeverHealthy { model_id, detail } => format!(
                "{model_id} was started and never became ready ({detail}). The agent is still on \
                 the model it was on."
            ),
        }
    }

    /// Whether this refusal happened before anything was disturbed.
    ///
    /// The ones that did become [`TransitionPhase::Failed`]; the ones that did
    /// not need a rollback, because by then the machine has been changed.
    ///
    /// Three are not pre-flight, and [`Self::ContextTooSmall`] is the one worth
    /// explaining. It could be checked early against the window the registry
    /// *declares*, and that check would pass in exactly the cases that matter:
    /// [`crate::ai_engine::vram_planner`] buys GPU layers by walking the
    /// context ladder down, so a model declaring 32 768 is routinely served at
    /// 8 192. The only figure worth refusing on is the one the server came up
    /// with, and by the time that is known the target is loaded and the source
    /// may have been evicted to make room for it.
    pub fn is_pre_flight(&self) -> bool {
        !matches!(
            self,
            Self::AdmissionRefused { .. } | Self::NeverHealthy { .. } | Self::ContextTooSmall { .. }
        )
    }
}

/// What the agent needs of whatever model it is bound to.
///
/// Derived from the definition rather than passed in, so a caller cannot ask for
/// a laxer check than the agent's own record justifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingRequirements {
    pub role: ModelRole,
    pub eligible: Vec<String>,
    pub classification_ceiling: Classification,
    /// Tool names the agent may use, with its own denylist already applied.
    pub tools: Vec<String>,
    /// The modality the agent's role implies, when it implies one.
    pub modality: Option<Modality>,
}

impl BindingRequirements {
    pub fn of(definition: &AgentDefinition) -> Self {
        Self {
            role: definition.role,
            eligible: definition.models.eligible_model_ids.clone(),
            classification_ceiling: definition.classification_ceiling,
            tools: definition
                .allowed_tools
                .iter()
                .filter(|tool| !definition.denied_tools.contains(tool))
                .map(|tool| tool.as_str().to_string())
                .collect(),
            modality: match definition.role {
                ModelRole::Vision | ModelRole::DocumentOcr => Some(Modality::Image),
                _ => None,
            },
        }
    }
}

/// Everything about a target that can be checked without touching the machine.
///
/// Split from the admission and the load for the same reason
/// [`super::events::checkpoint::RunCheckpoint::resumable_against`] is split from
/// gathering the world: a check that also gathers its own inputs cannot be
/// tested against inputs it would refuse, and the refusals are the point.
pub fn validate_target(
    target: &ModelEntry,
    current_model_id: Option<&str>,
    needs: &BindingRequirements,
) -> Result<(), ValidationRefusal> {
    if current_model_id == Some(target.id.as_str()) {
        return Err(ValidationRefusal::SameModel {
            model_id: target.id.clone(),
        });
    }
    if !target.enabled {
        return Err(ValidationRefusal::Disabled {
            model_id: target.id.clone(),
        });
    }
    // Empty means "any model registered for the role", which is what every
    // bundled profile meant before an eligible set existed. A non-empty set that
    // does not name the target is a decision not to allow it.
    if !needs.eligible.is_empty() && !needs.eligible.iter().any(|id| id == &target.id) {
        return Err(ValidationRefusal::NotEligible {
            model_id: target.id.clone(),
            eligible: needs.eligible.clone(),
        });
    }
    if !target.serves(needs.role) {
        return Err(ValidationRefusal::WrongRole {
            model_id: target.id.clone(),
            needs: needs.role.label().to_string(),
            serves: target
                .roles
                .iter()
                .map(|role| role.label().to_string())
                .collect(),
        });
    }
    if !target.permits(needs.classification_ceiling) {
        return Err(ValidationRefusal::ClassificationNotPermitted {
            model_id: target.id.clone(),
            classification: needs.classification_ceiling.label().to_string(),
        });
    }
    if let Some(modality) = needs.modality {
        if !target.supports_modality(modality) {
            return Err(ValidationRefusal::ModalityUnsupported {
                model_id: target.id.clone(),
                modality: modality.label().to_string(),
            });
        }
    }
    // Refused rather than narrowed. An agent bound to `create_docx` that quietly
    // stopped being able to call it would produce a turn describing the document
    // it was asked to write.
    if !needs.tools.is_empty() && !target.supports_structured_output() {
        return Err(ValidationRefusal::ToolsUnsupported {
            model_id: target.id.clone(),
            tools: needs.tools.clone(),
        });
    }
    if let Err(unsupported) = crate::serving::check_runtime_supports(target) {
        return Err(ValidationRefusal::RuntimeUnsupported {
            model_id: target.id.clone(),
            detail: unsupported.explain(),
        });
    }
    Ok(())
}

/// Whether the destination's real window holds what may not be dropped.
///
/// Checked against the *served* window rather than the declared one, because on
/// a constrained card a model declaring 32 768 is routinely started with 8 192 —
/// and a handoff budgeted against the declared figure would send a turn the
/// server refuses.
pub fn fits_mandatory(
    model_id: &str,
    served_window: u32,
    mandatory_tokens: u32,
    reserved: u32,
) -> Result<(), ValidationRefusal> {
    let affords = served_window.saturating_sub(reserved);
    if mandatory_tokens > affords {
        return Err(ValidationRefusal::ContextTooSmall {
            model_id: model_id.to_string(),
            affords,
            mandatory: mandatory_tokens,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// What crosses, and what does not
// ─────────────────────────────────────────────────────────────────────────────

/// State that is deliberately not carried to the new model.
///
/// Named rather than silently discarded. Dropping a prompt cache costs the next
/// turn a prefill; dropping a provider's compaction object costs the run a
/// summary it will have to re-derive. Both are correct and both change what the
/// next turn looks like, so both are in the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DroppedState {
    /// The attention cache. Belongs to one model's layer and head geometry.
    KvCache,
    /// The server's prefix cache. Belongs to one process.
    PromptCache,
    /// A summary the previous model produced of its own transcript, in whatever
    /// shape its provider chose.
    ProviderCompaction,
    /// Token ids. A different vocabulary makes them a different sentence.
    TokenizedTranscript,
    /// Logit bias, grammar state, sampler seeds — anything whose meaning is
    /// defined by the vocabulary it indexes.
    SamplerState,
}

impl DroppedState {
    pub const ALL: &'static [DroppedState] = &[
        DroppedState::KvCache,
        DroppedState::PromptCache,
        DroppedState::ProviderCompaction,
        DroppedState::TokenizedTranscript,
        DroppedState::SamplerState,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            DroppedState::KvCache => "kvCache",
            DroppedState::PromptCache => "promptCache",
            DroppedState::ProviderCompaction => "providerCompaction",
            DroppedState::TokenizedTranscript => "tokenizedTranscript",
            DroppedState::SamplerState => "samplerState",
        }
    }

    pub const fn explain(self) -> &'static str {
        match self {
            DroppedState::KvCache => {
                "The attention cache was not carried over: it is keyed to the previous model's \
                 layer and head geometry, and reinterpreting it under a different one produces \
                 text that is fluent and unrelated to the conversation."
            }
            DroppedState::PromptCache => {
                "The server's prefix cache was not carried over, so the first round after the \
                 handoff pays a full prefill. Nothing was lost except time."
            }
            DroppedState::ProviderCompaction => {
                "The previous model's compaction of its own transcript was not carried over. It \
                 is a summary in a shape only that provider defines; the facts it summarised are \
                 in the graph and in the receipts, which is what the new model was given instead."
            }
            DroppedState::TokenizedTranscript => {
                "Token ids were not carried over. The same ids index a different vocabulary in \
                 the new model, so the transcript was handed over as text and re-tokenised."
            }
            DroppedState::SamplerState => {
                "Grammar, logit-bias and sampler state were not carried over: each of them names \
                 positions in a vocabulary the new model does not share."
            }
        }
    }
}

/// What the new model is given, reconstructed from things Rust wrote down.
///
/// ## Why every field has a corroborating source
///
/// Because the alternative is asking the previous model what the new one should
/// know — and one of the things it would be asked is which side effects have
/// already happened. A resumed run reads that to decide what not to do again, so
/// a claim there is functionally an instruction to skip work. Which is why each
/// field comes from somewhere a model cannot write:
///
/// | Field | Comes from |
/// |---|---|
/// | `notes` | [`super::state_commit`], which checked every claim that could excuse work |
/// | `graph_items` | the authorised graph snapshot at the frozen revision |
/// | `source_refs` | content hashes in the source [`super::context_manifest::ContextManifest`] |
/// | `receipts` | `ToolSucceeded` / `ArtifactProduced` rows in the durable log |
/// | `artifact_hashes` | the run's produced-file table |
///
/// The struct also *cannot* hold a KV cache, a compaction blob or a token array:
/// there is no field for one. That is the enforcement, rather than a rule in a
/// comment somebody has to remember.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableState {
    /// The typed notes, as Rust last accepted them.
    pub notes: RunMemory,
    /// Graph records the source turn was authorised against, at the frozen
    /// revision.
    pub graph_items: Vec<SelectedItem>,
    /// Content addresses of everything the source turn read.
    pub source_refs: Vec<String>,
    /// Effects the durable log corroborates. Never widened here: this is a copy
    /// of what was already accepted, and the new model cannot add to it.
    pub receipts: Vec<CompletedEffect>,
    pub artifact_hashes: Vec<String>,
    /// What was deliberately left behind.
    pub dropped: Vec<DroppedState>,
    /// A hash over everything above, so a record can be checked against the
    /// state it claims to describe.
    pub state_hash: String,
}

impl PortableState {
    /// Assembles the carry-over and seals it.
    pub fn new(
        notes: RunMemory,
        graph_items: Vec<SelectedItem>,
        source_refs: Vec<String>,
        receipts: Vec<CompletedEffect>,
        artifact_hashes: Vec<String>,
    ) -> Self {
        let mut state = Self {
            notes,
            graph_items,
            source_refs,
            receipts,
            artifact_hashes,
            // Every one of them, every time. There is no path through a model
            // change on which any of these is safe to keep, so the list is a
            // constant rather than a decision somebody makes per transition.
            dropped: DroppedState::ALL.to_vec(),
            state_hash: String::new(),
        };
        state.state_hash = state.compute_hash();
        state
    }

    /// The hash of everything except the hash.
    ///
    /// Built from a canonical string rather than from serialised JSON, because
    /// field order is a property of the serialiser and this has to be stable
    /// across builds of it.
    pub fn compute_hash(&self) -> String {
        let notes = serde_json::to_string(&self.notes).unwrap_or_default();
        let items: Vec<String> = self
            .graph_items
            .iter()
            .map(|item| format!("{}@{}", item.item_id, item.revision))
            .collect();
        let receipts: Vec<String> = self
            .receipts
            .iter()
            .map(|effect| format!("{}:{}", effect.tool, effect.target))
            .collect();
        digest(&format!(
            "v{}|{}|{}|{}|{}|{}",
            TRANSITION_SCHEMA_VERSION,
            digest(&notes),
            items.join(","),
            self.source_refs.join(","),
            receipts.join(","),
            self.artifact_hashes.join(","),
        ))
    }

    pub fn is_intact(&self) -> bool {
        self.state_hash == self.compute_hash()
    }

    /// Whether an effect this state records was already done.
    ///
    /// The one question the new model's first round has to be able to ask, and
    /// the reason the receipts are carried at all.
    pub fn has_done(&self, tool: &str, target: &str) -> bool {
        self.notes.has_done(tool, target)
            || self
                .receipts
                .iter()
                .any(|effect| effect.tool == tool && effect.target == target)
    }
}

/// The source revision a handoff is frozen against.
///
/// Frozen before the target is touched, so the two sides of the transition are
/// comparable. The graph moves while a run works, and a record naming "the
/// graph" without saying which revision would describe a state that no longer
/// exists by the time anybody reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFreeze {
    /// The changefeed position this handoff is authorised against.
    pub graph_revision: i64,
    /// The durable event the checkpoint was taken after.
    pub last_event_seq: i64,
    /// The manifest the source turn was built from, by its own seal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_manifest_hash: Option<String>,
    /// The checkpoint that is the resume point if this transition is undone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_hash: Option<String>,
    pub at: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Rollback
// ─────────────────────────────────────────────────────────────────────────────

/// How far back a transition can be undone, and what must survive it.
///
/// Bounded on purpose. "Undo the transition" cannot mean "undo the work": the
/// run may have written a document between the checkpoint and the failure, and a
/// rollback that tried to unwrite it would be inventing an undo for an effect
/// that has none. So a rollback restores exactly two things — the binding and
/// the resume point — and *asserts* that everything else was left alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RollbackPlan {
    /// The binding to put back.
    pub restore_binding: ModelBinding,
    /// The definition version the registry must be at for the restore to be
    /// safe. A concurrent edit moves it, and the rollback then refuses rather
    /// than overwriting somebody's change.
    pub expected_version: u64,
    /// The checkpoint the run keeps. Never rewritten by a rollback: it is the
    /// last state the run itself established, and a rollback is about the
    /// binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_checkpoint_hash: Option<String>,
    /// Whether the source model has to be served again for the run to continue.
    pub reload_source: bool,
    /// Effects completed before or during the transition. Named so a rollback
    /// can be *checked* for having preserved them.
    pub preserve_effects: Vec<CompletedEffect>,
}

impl RollbackPlan {
    /// Whether this rollback would lose an effect the run already had.
    ///
    /// The assertion a rollback is checked against rather than trusted for. An
    /// undo that quietly dropped a receipt would let the resumed run write the
    /// same document a second time — the exact failure
    /// [`super::state_commit`] exists to prevent, arriving by a different door.
    pub fn preserves(&self, after: &RunMemory) -> Result<(), Vec<CompletedEffect>> {
        let lost: Vec<CompletedEffect> = self
            .preserve_effects
            .iter()
            .filter(|effect| !after.has_done(&effect.tool, &effect.target))
            .cloned()
            .collect();
        if lost.is_empty() {
            Ok(())
        } else {
            Err(lost)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The durable record
// ─────────────────────────────────────────────────────────────────────────────

/// The intent to rebind, written before the registry is touched.
///
/// The whole of the answer to "what does a crash between the two stores mean".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingCommit {
    pub agent_id: String,
    pub target_model_id: String,
    /// The version the registry held when this was written.
    pub expected_version: u64,
    /// The version it will hold once the write lands.
    pub next_version: u64,
}

/// What reading the registry after a crash says about a pending commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Reconciliation {
    /// The registry holds the target at the expected next version. The write
    /// landed; finish the record as committed.
    Landed,
    /// The registry still holds the source at the version before the write. It
    /// did not land; the agent is on the model it was on.
    DidNotLand,
    /// Neither. Somebody edited the agent between the crash and now, so what
    /// happened to this write cannot be read off the record. Needs a person.
    Indeterminate,
}

impl Reconciliation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Reconciliation::Landed => "landed",
            Reconciliation::DidNotLand => "didNotLand",
            Reconciliation::Indeterminate => "indeterminate",
        }
    }
}

impl PendingCommit {
    /// Reads a crashed commit off the registry as it is now.
    ///
    /// Deterministic, and deliberately three-valued. Collapsing
    /// [`Reconciliation::Indeterminate`] into either of the others would be a
    /// guess about which model an agent is on, made by code, in the one place
    /// where being wrong changes what the agent does next.
    pub fn reconcile(&self, current: &AgentDefinition) -> Reconciliation {
        let on_target =
            current.models.default_model_id.as_deref() == Some(self.target_model_id.as_str());
        match (current.definition_version, on_target) {
            (version, true) if version == self.next_version => Reconciliation::Landed,
            (version, false) if version == self.expected_version => Reconciliation::DidNotLand,
            _ => Reconciliation::Indeterminate,
        }
    }
}

/// The longest a reason may be. Bounded for the same reason an operator intent
/// is: it goes into a durable record that a screen reads.
pub const MAX_REASON: usize = 400;

/// One transition, as the deployment keeps it.
///
/// Read by an administration screen, by the reconciliation that runs at start-up
/// and by whoever asks in six months why this agent's answers changed shape. It
/// holds identifiers, hashes and counts — never a passage, an argument or an
/// answer — for the same reason a checkpoint does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransitionRecord {
    pub transition_id: String,
    pub schema_version: u32,

    /// Never changes across a transition. The point of the whole module.
    pub agent_id: String,
    /// The run being handed over, when there is one. `None` is an inactive-agent
    /// reassignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// The task the run belongs to. Preserved, like `agent_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// The attempt in flight when the handoff began.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,

    pub from_model: ModelFingerprint,
    pub to_model: ModelFingerprint,
    pub from_definition_version: u64,
    /// The version the registry holds once the binding is committed. Recorded
    /// before the write, so a crash is readable.
    pub to_definition_version: u64,
    pub from_binding: ModelBinding,
    pub to_binding: ModelBinding,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freeze: Option<SourceFreeze>,
    /// What the target actually came up as. `None` when nothing was loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served: Option<ServedBinding>,
    /// The manifest the source turn was built from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_manifest_hash: Option<String>,
    /// The manifest recompiled for the target's actual window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_manifest_hash: Option<String>,
    /// The carry-over, by its seal. The state itself lives with the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub portable_state_hash: Option<String>,
    /// Artifact content hashes as they stood at the freeze. A handoff must not
    /// change one, and this is what makes that checkable rather than assumed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_hashes: Vec<String>,
    /// Whether the two models demonstrably tokenise differently. `None` when one
    /// of them could not be asked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_changed: Option<bool>,
    /// Model servers that were stopped to make room for the target.
    ///
    /// Recorded because it is the only thing that makes a rollback's reload
    /// step honest. "Put the source model back" is right when the source was
    /// evicted to make room and wrong when it was never running — and starting
    /// a model nothing had stopped, on a card that has just been filled, is a
    /// worse answer than doing nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub released: Vec<String>,
    /// True when the in-process model was unloaded to make room as well.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub released_in_process: bool,

    pub phase: TransitionPhase,
    /// Every phase this transition passed through, in order, with its time.
    /// `phase` is where it is; this is how it got there.
    pub history: Vec<PhaseEntry>,
    /// Why it is not committed, when it is not. One sentence, in an operator's
    /// terms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub because: Option<String>,
    /// The typed refusal, when a validation produced one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<ValidationRefusal>,
    /// What has to be settled, when the transition is parked. Carried so a
    /// screen can send somebody to the right place rather than to "try again".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub awaiting: Vec<String>,

    pub requested_by: String,
    /// What the administrator said they were doing. Bounded by [`MAX_REASON`].
    pub reason: String,
    pub requested_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settled_at: Option<String>,
}

/// One step of a transition's history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseEntry {
    pub phase: TransitionPhase,
    pub at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl TransitionRecord {
    /// Opens a transition. Nothing has been touched at this point.
    pub fn begin(
        transition_id: impl Into<String>,
        agent: &AgentDefinition,
        from_model: ModelFingerprint,
        to_model: ModelFingerprint,
        to_binding: ModelBinding,
        requested_by: impl Into<String>,
        reason: &str,
    ) -> Self {
        let at = chrono::Utc::now().to_rfc3339();
        let reason: String = reason.trim().chars().take(MAX_REASON).collect();
        Self {
            transition_id: transition_id.into(),
            schema_version: TRANSITION_SCHEMA_VERSION,
            agent_id: agent.agent_id.clone(),
            run_id: None,
            task_id: None,
            attempt_id: None,
            from_model,
            to_model,
            from_definition_version: agent.definition_version,
            to_definition_version: agent.definition_version + 1,
            from_binding: agent.models.clone(),
            to_binding,
            freeze: None,
            served: None,
            source_manifest_hash: None,
            target_manifest_hash: None,
            portable_state_hash: None,
            artifact_hashes: Vec::new(),
            tokenizer_changed: None,
            released: Vec::new(),
            released_in_process: false,
            phase: TransitionPhase::Requested,
            history: vec![PhaseEntry {
                phase: TransitionPhase::Requested,
                at: at.clone(),
                detail: None,
            }],
            because: None,
            refusal: None,
            awaiting: Vec::new(),
            requested_by: requested_by.into(),
            reason,
            requested_at: at,
            settled_at: None,
        }
    }

    /// Moves to the next phase, or says why that edge does not exist.
    ///
    /// Refused rather than forced, because a record that can be moved anywhere
    /// is a record that proves nothing about the order things happened in — and
    /// the order is what makes the crash windows readable.
    pub fn advance(&mut self, next: TransitionPhase, detail: Option<String>) -> Result<(), String> {
        if !self.phase.may_advance_to(next) {
            return Err(format!(
                "a transition cannot go from {} to {}; this is a defect in the handoff rather \
                 than something an operator did",
                self.phase.as_str(),
                next.as_str()
            ));
        }
        let at = chrono::Utc::now().to_rfc3339();
        self.phase = next;
        self.history.push(PhaseEntry {
            phase: next,
            at: at.clone(),
            detail,
        });
        if next.is_terminal() {
            self.settled_at = Some(at);
        }
        Ok(())
    }

    /// The commit intent this record represents, for reconciliation after a
    /// crash.
    pub fn pending_commit(&self) -> PendingCommit {
        PendingCommit {
            agent_id: self.agent_id.clone(),
            target_model_id: self.to_model.model_id.clone(),
            expected_version: self.from_definition_version,
            next_version: self.to_definition_version,
        }
    }

    /// The bounded undo for wherever this transition has got to.
    pub fn rollback_plan(&self, preserve_effects: Vec<CompletedEffect>) -> RollbackPlan {
        RollbackPlan {
            restore_binding: self.from_binding.clone(),
            expected_version: self.from_definition_version,
            keep_checkpoint_hash: self
                .freeze
                .as_ref()
                .and_then(|freeze| freeze.checkpoint_hash.clone()),
            // The source comes back only if this handoff is what stopped it.
            // Anything else — reloading a model that was never running, or one
            // somebody else's turn is deliberately not using — would be an undo
            // inventing work rather than reversing it.
            reload_source: self.phase.may_have_released_memory()
                && self
                    .from_binding
                    .default_model_id
                    .as_deref()
                    .is_some_and(|source| {
                        self.released.iter().any(|stopped| stopped == source)
                    }),
            preserve_effects,
        }
    }

    pub fn outcome(&self) -> TransitionOutcome {
        self.phase.outcome()
    }

    /// Whether this row describes a transition that is still going.
    pub fn is_open(&self) -> bool {
        !self.phase.is_terminal()
    }

    /// The sentence a screen shows for where this got to.
    pub fn describe(&self) -> String {
        if let Some(because) = &self.because {
            return because.clone();
        }
        match self.phase {
            TransitionPhase::Committed => format!(
                "This agent is now on {}. Its id, its memory and its history are unchanged.",
                self.to_model.model_id
            ),
            TransitionPhase::RolledBack => format!(
                "The move to {} was undone. The agent is on {} and its last saved state is \
                 intact.",
                self.to_model.model_id, self.from_model.model_id
            ),
            TransitionPhase::RollbackFailed => format!(
                "The agent could not be put back on {}. Somebody has to check which model it is \
                 bound to before it is given work.",
                self.from_model.model_id
            ),
            TransitionPhase::Failed => format!(
                "The agent was not moved to {}. It is still on {}.",
                self.to_model.model_id, self.from_model.model_id
            ),
            TransitionPhase::AwaitingSettlement => format!(
                "The move to {} is waiting: {} thing(s) have to be settled by a person first.",
                self.to_model.model_id,
                self.awaiting.len().max(1)
            ),
            phase => format!(
                "Moving from {} to {}: {}.",
                self.from_model.model_id,
                self.to_model.model_id,
                phase.as_str().replace('_', " ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    //! The rules, tested without a machine.
    //!
    //! Everything here is a pure function over states, refusals and records, so
    //! every case can be enumerated — including the ones that must be *refused*,
    //! which is where the value is. The end-to-end behaviour of the handoff is
    //! `super::model_transition_tests`, and the two real models are
    //! `tests/model_binding_handoff_live.rs`.

    use super::*;
    use crate::agents::{AgentDefinition, AgentState, MemoryPolicy, AGENT_PALETTE};
    use crate::orchestrator::tools::ToolName;
    use crate::policy::Classification;
    use crate::registry::tests::entry as registered;
    use crate::subagents::profile::{Isolation, Limits, SchemaKind, WritePolicy};

    fn agent(model: Option<&str>) -> AgentDefinition {
        AgentDefinition {
            agent_id: "ag-1".into(),
            definition_version: 4,
            display_name: "Knowledge retriever".into(),
            description: "Finds passages and cites them.".into(),
            instructions: "Answer only from the passages you retrieve.".into(),
            role: ModelRole::Reasoning,
            state: AgentState::Enabled,
            color: AGENT_PALETTE[0].into(),
            skills: Vec::new(),
            allowed_tools: vec![ToolName::SearchDocuments],
            denied_tools: Vec::new(),
            memory: MemoryPolicy::default(),
            models: ModelBinding {
                default_model_id: model.map(str::to_string),
                fallback_model_ids: Vec::new(),
                eligible_model_ids: Vec::new(),
            },
            output_schema: SchemaKind::Retrieval,
            limits: Limits {
                max_turns: 8,
                max_output_tokens: 2_048,
                max_children: 0,
                max_duration_seconds: 600,
            },
            max_concurrent: 1,
            isolation: Isolation::ReadOnly,
            write_policy: WritePolicy::None,
            classification_ceiling: Classification::Internal,
            imported_from: None,
            created_at: "2026-09-18T00:00:00Z".into(),
            updated_at: "2026-09-18T00:00:00Z".into(),
        }
    }

    fn record() -> TransitionRecord {
        let from = ModelFingerprint::unregistered("model-a");
        let to = ModelFingerprint::unregistered("model-b");
        let agent = agent(Some("model-a"));
        TransitionRecord::begin(
            "tr-1",
            &agent,
            from,
            to,
            agent.models.rebound_to("model-b"),
            "priya",
            "moving to the larger model",
        )
    }

    // -- The state machine ------------------------------------------------

    /// The one path that commits, in full. Written out rather than derived, so
    /// that reordering the phases in `may_advance_to` fails here.
    #[test]
    fn the_committing_path_is_the_one_documented_order() {
        let mut record = record();
        for next in [
            TransitionPhase::Draining,
            TransitionPhase::Checkpointed,
            TransitionPhase::Validating,
            TransitionPhase::Loading,
            TransitionPhase::Recompiling,
            TransitionPhase::CommitPending,
            TransitionPhase::Committed,
        ] {
            record
                .advance(next, None)
                .unwrap_or_else(|error| panic!("{next:?}: {error}"));
        }
        assert_eq!(record.outcome(), TransitionOutcome::Succeeded);
        assert!(record.settled_at.is_some(), "a terminal phase settles the row");
        assert_eq!(record.history.len(), 8, "every phase is in the history");
    }

    /// The headline refusal. A handoff cannot reach the phase that unloads the
    /// source model without having written the task state down first.
    #[test]
    fn nothing_reaches_the_load_without_a_checkpoint() {
        for phase in [
            TransitionPhase::Requested,
            TransitionPhase::Draining,
            TransitionPhase::AwaitingSettlement,
        ] {
            assert!(
                !phase.may_advance_to(TransitionPhase::Loading),
                "{phase:?} must not be able to start a load"
            );
            assert!(
                !phase.may_advance_to(TransitionPhase::Validating),
                "{phase:?} must not be able to validate before the state is written down"
            );
        }
        // The only way in is through the checkpoint, and then the validation.
        assert!(TransitionPhase::Checkpointed.may_advance_to(TransitionPhase::Validating));
        assert!(TransitionPhase::Validating.may_advance_to(TransitionPhase::Loading));
    }

    /// The binding is only ever written from the one phase that made the intent
    /// durable first. This is what makes a crash between the two stores
    /// readable.
    #[test]
    fn only_a_durable_intent_may_be_committed() {
        for phase in TransitionPhase::ALL {
            let allowed = phase.may_advance_to(TransitionPhase::Committed);
            assert_eq!(
                allowed,
                *phase == TransitionPhase::CommitPending,
                "{phase:?} -> committed should be {}",
                *phase == TransitionPhase::CommitPending
            );
        }
    }

    /// Giving up is "failed" while nothing has been *disturbed*, and a rollback
    /// once something has.
    ///
    /// The line is the load, not the checkpoint, and that distinction is the
    /// one this test exists to hold. Writing a run's state down is not
    /// something an undo reverses — a rollback deliberately keeps it — so a
    /// transition refused at the validation has nothing to put back, and an
    /// undo there would try to start a model nothing had stopped.
    #[test]
    fn abandoning_before_the_load_is_a_failure_and_after_it_is_a_rollback() {
        for phase in [
            TransitionPhase::Requested,
            TransitionPhase::Draining,
            TransitionPhase::AwaitingSettlement,
            TransitionPhase::Checkpointed,
            TransitionPhase::Validating,
        ] {
            assert!(
                phase.may_advance_to(TransitionPhase::Failed),
                "{phase:?} disturbed nothing and may simply fail"
            );
            assert!(
                !phase.may_advance_to(TransitionPhase::RollingBack),
                "{phase:?} has nothing to undo"
            );
        }
        for phase in [
            TransitionPhase::Loading,
            TransitionPhase::Recompiling,
            TransitionPhase::CommitPending,
        ] {
            assert!(
                !phase.may_advance_to(TransitionPhase::Failed),
                "{phase:?} may have taken a model's memory and must not simply fail"
            );
            assert!(phase.may_advance_to(TransitionPhase::RollingBack), "{phase:?}");
        }
    }

    #[test]
    fn a_terminal_phase_goes_nowhere() {
        for phase in TransitionPhase::ALL.iter().filter(|p| p.is_terminal()) {
            for next in TransitionPhase::ALL {
                assert!(
                    !phase.may_advance_to(*next),
                    "{phase:?} is terminal and must not move to {next:?}"
                );
            }
        }
    }

    /// A parked handoff is picked up again, and that is the only way back into
    /// the machine from a waiting state.
    #[test]
    fn a_settled_question_lets_the_handoff_resume() {
        let mut record = record();
        record.advance(TransitionPhase::Draining, None).expect("drain");
        record
            .advance(TransitionPhase::AwaitingSettlement, None)
            .expect("park");
        assert_eq!(record.outcome(), TransitionOutcome::Pending);
        assert!(record.is_open(), "a parked handoff has not settled");
        record
            .advance(TransitionPhase::Draining, None)
            .expect("a retry re-drains");
    }

    /// Refused rather than forced. A record that could be moved anywhere proves
    /// nothing about the order things happened in.
    #[test]
    fn an_edge_that_does_not_exist_is_refused_and_leaves_the_record_alone() {
        let mut record = record();
        let error = record
            .advance(TransitionPhase::Committed, None)
            .expect_err("requested cannot commit");
        assert!(error.contains("requested"), "{error}");
        assert!(error.contains("committed"), "{error}");
        assert_eq!(record.phase, TransitionPhase::Requested);
        assert_eq!(record.history.len(), 1, "a refused edge writes no history");
    }

    #[test]
    fn every_phase_round_trips_through_its_stored_spelling() {
        for phase in TransitionPhase::ALL {
            assert_eq!(TransitionPhase::from_str(phase.as_str()), Some(*phase));
        }
        assert_eq!(TransitionPhase::from_str("nonsense"), None);
    }

    /// Only one phase means "needs a person", and it is the one where the agent
    /// may be bound to a model nothing is serving.
    #[test]
    fn only_a_failed_rollback_asks_for_a_person() {
        for phase in TransitionPhase::ALL {
            assert_eq!(
                phase.outcome().needs_human(),
                *phase == TransitionPhase::RollbackFailed,
                "{phase:?}"
            );
        }
    }

    // -- The drain boundary -----------------------------------------------

    /// The acceptance case. A tool result is recorded, which is *defined* as
    /// the point an effect has settled, so the handoff may proceed.
    #[test]
    fn a_handoff_after_a_tool_is_at_a_safe_boundary() {
        assert!(boundary_of(RunState::ToolResultRecorded).is_safe());
    }

    /// The other acceptance case. A person has been asked to allow something,
    /// and the approval authorises a specific call by a model they were told
    /// about — so it is not carried over.
    #[test]
    fn a_pending_approval_parks_the_handoff_rather_than_failing_it() {
        let verdict = drain_for(RunState::AwaitingApproval, &[], &[]);
        match &verdict {
            DrainVerdict::NeedsSettlement { because, .. } => {
                assert!(because.contains("allow"), "{because}");
            }
            other => panic!("expected a settlement, got {other:?}"),
        }
        assert!(!verdict.is_safe());
        assert!(
            !verdict.resolves_by_waiting(),
            "waiting does not answer an approval; a person does"
        );
    }

    /// An approval raised while the run sits at an otherwise safe boundary
    /// still parks the handoff. The state alone is not the whole question.
    #[test]
    fn an_approval_at_a_safe_boundary_still_parks_it() {
        let verdict = drain_for(
            RunState::ToolResultRecorded,
            &[],
            &["approval-1".to_string()],
        );
        match verdict {
            DrainVerdict::NeedsSettlement { keys, .. } => assert_eq!(keys, vec!["approval-1"]),
            other => panic!("expected a settlement, got {other:?}"),
        }
    }

    /// The most important one. An effect nobody accounted for cannot be
    /// described to a different model as having happened or not.
    #[test]
    fn an_unaccounted_effect_outranks_a_safe_state() {
        let verdict = drain_for(
            RunState::ToolResultRecorded,
            &["key-1".to_string(), "key-2".to_string()],
            &[],
        );
        match verdict {
            DrainVerdict::NeedsSettlement { keys, because } => {
                assert_eq!(keys.len(), 2);
                assert!(because.contains("twice"), "{because}");
            }
            other => panic!("expected a settlement, got {other:?}"),
        }
    }

    /// An unaccounted effect is reported before a pending approval, because it
    /// is the one that cannot be answered by simply deciding.
    #[test]
    fn an_unaccounted_effect_is_reported_before_an_approval() {
        let verdict = drain_for(
            RunState::AwaitingApproval,
            &["key-1".to_string()],
            &["approval-1".to_string()],
        );
        match verdict {
            DrainVerdict::NeedsSettlement { keys, .. } => assert_eq!(keys, vec!["key-1"]),
            other => panic!("expected a settlement, got {other:?}"),
        }
    }

    /// Mid-flight states drain rather than park: waiting genuinely resolves
    /// them, and a screen should say so rather than sending somebody to answer
    /// a question nobody asked.
    #[test]
    fn work_in_flight_drains_and_says_waiting_will_help() {
        for state in [
            RunState::Running,
            RunState::ExecutingTool,
            RunState::Verifying,
            RunState::Compacting,
            RunState::Recovering,
        ] {
            let verdict = boundary_of(state);
            assert!(!verdict.is_safe(), "{state:?}");
            assert!(verdict.resolves_by_waiting(), "{state:?}");
            assert!(
                verdict.because().is_some_and(|line| line.len() > 40),
                "{state:?} needs a reason somebody can act on"
            );
        }
    }

    /// A finished run has nothing to hand over, so the binding change is an
    /// ordinary reassignment as far as it is concerned.
    #[test]
    fn a_finished_run_does_not_block_a_reassignment() {
        for state in RunState::ALL.iter().filter(|state| state.is_terminal()) {
            assert!(boundary_of(*state).is_safe(), "{state:?}");
        }
    }

    // -- Validating the destination ---------------------------------------

    /// A registered model, built by the registry's own test constructor so the
    /// shape here cannot drift from the shape production reads.
    fn entry(id: &str) -> ModelEntry {
        let mut entry = registered(id, 9.0, vec![ModelRole::Reasoning]);
        entry.supports_structured_output = true;
        entry
    }

    #[test]
    fn a_suitable_target_is_accepted() {
        let agent = agent(Some("model-a"));
        validate_target(&entry("model-b"), Some("model-a"), &BindingRequirements::of(&agent))
            .expect("a reasoning model with tools and clearance is acceptable");
    }

    /// The acceptance case. An agent bound to a tool cannot be moved to a model
    /// that cannot be asked to call one: it would keep its tools and lose the
    /// ability to use them, and produce a turn describing the document it was
    /// asked to write.
    #[test]
    fn a_model_that_cannot_call_tools_is_refused_rather_than_narrowed() {
        let agent = agent(Some("model-a"));
        let mut target = entry("model-b");
        target.supports_structured_output = false;

        let refusal = validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("a model with no structured output cannot take a tool-bound agent");
        match &refusal {
            ValidationRefusal::ToolsUnsupported { tools, .. } => {
                // The tool's *stable* spelling, not the variant name. See
                // `ToolName::as_str`: the two were deliberately decoupled so a
                // rename could not rewrite records, and a refusal an operator
                // reads has to name the tool the way the rest of the product
                // does.
                assert_eq!(
                    tools,
                    &vec![ToolName::SearchDocuments.as_str().to_string()]
                );
                assert_eq!(tools[0], "knowledge.search_authorized");
            }
            other => panic!("expected tools unsupported, got {other:?}"),
        }
        assert!(refusal.explain().contains("worse than a refusal"), "{}", refusal.explain());
        assert!(
            refusal.is_pre_flight(),
            "nothing was touched, so this is a plain failure"
        );
    }

    /// An agent with no tools may move to a model that cannot call them.
    #[test]
    fn an_agent_with_no_tools_may_move_to_a_model_without_them() {
        let mut agent = agent(Some("model-a"));
        agent.allowed_tools.clear();
        let mut target = entry("model-b");
        target.supports_structured_output = false;
        validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect("no tools, nothing to lose");
    }

    /// A denied tool is not a bound tool, so it does not constrain the target.
    #[test]
    fn a_denied_tool_does_not_hold_an_agent_to_a_model() {
        let mut agent = agent(Some("model-a"));
        agent.denied_tools = vec![ToolName::SearchDocuments];
        let mut target = entry("model-b");
        target.supports_structured_output = false;
        validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect("the only tool is denied, so the agent is not bound to it");
    }

    #[test]
    fn a_model_outside_the_eligible_set_is_refused_and_not_added_to_it() {
        let mut agent = agent(Some("model-a"));
        agent.models.eligible_model_ids = vec!["model-a".into(), "model-c".into()];

        let refusal = validate_target(&entry("model-b"), Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("model-b is not eligible");
        assert!(matches!(refusal, ValidationRefusal::NotEligible { .. }));
        // And the rebinding helper does not quietly widen it either.
        let rebound = agent.models.rebound_to("model-b");
        assert_eq!(rebound.eligible_model_ids, agent.models.eligible_model_ids);
    }

    #[test]
    fn an_empty_eligible_set_means_any_model_for_the_role() {
        let agent = agent(Some("model-a"));
        assert!(agent.models.eligible_model_ids.is_empty());
        validate_target(&entry("model-b"), Some("model-a"), &BindingRequirements::of(&agent))
            .expect("an empty set is not an empty allowance");
    }

    #[test]
    fn a_model_for_a_different_role_is_refused() {
        let agent = agent(Some("model-a"));
        let mut target = entry("model-b");
        target.roles = vec![ModelRole::Embedding];
        let refusal = validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("an embedding model cannot hold a reasoning agent");
        assert!(matches!(refusal, ValidationRefusal::WrongRole { .. }));
    }

    #[test]
    fn a_model_not_reviewed_for_the_material_is_refused() {
        let agent = agent(Some("model-a"));
        let mut target = entry("model-b");
        target.permitted_classifications = vec![Classification::Financial];
        let refusal = validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("this agent's ceiling is Internal, which this model is not reviewed for");
        assert!(matches!(
            refusal,
            ValidationRefusal::ClassificationNotPermitted { .. }
        ));
    }

    #[test]
    fn a_disabled_model_is_refused() {
        let agent = agent(Some("model-a"));
        let mut target = entry("model-b");
        target.enabled = false;
        let refusal = validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("a disabled model is not a binding");
        assert!(matches!(refusal, ValidationRefusal::Disabled { .. }));
    }

    /// Refused rather than treated as a no-op: a record saying an agent moved
    /// from X to X makes the history harder to read rather than more complete.
    #[test]
    fn moving_an_agent_to_the_model_it_is_already_on_is_refused() {
        let agent = agent(Some("model-b"));
        let refusal = validate_target(&entry("model-b"), Some("model-b"), &BindingRequirements::of(&agent))
            .expect_err("already there");
        assert!(matches!(refusal, ValidationRefusal::SameModel { .. }));
    }

    /// A vision agent needs a model that can see. Derived from the role rather
    /// than asked of the caller, so a laxer check cannot be requested.
    #[test]
    fn a_vision_agent_needs_a_model_with_the_modality() {
        let mut agent = agent(Some("model-a"));
        agent.role = ModelRole::Vision;
        let mut target = entry("model-b");
        target.roles = vec![ModelRole::Vision];
        target.modalities = vec![Modality::Text];

        let refusal = validate_target(&target, Some("model-a"), &BindingRequirements::of(&agent))
            .expect_err("a text-only model would answer about a page it cannot see");
        assert!(matches!(refusal, ValidationRefusal::ModalityUnsupported { .. }));
    }

    // -- Context capacity -------------------------------------------------

    /// Checked against the served window, and the refusal names both figures so
    /// an operator can see how far short it is.
    #[test]
    fn a_window_that_cannot_hold_the_mandatory_context_is_refused() {
        let refusal = fits_mandatory("model-b", 8_192, 6_000, 4_000)
            .expect_err("6000 mandatory tokens do not fit in 8192 minus 4000");
        match &refusal {
            ValidationRefusal::ContextTooSmall {
                affords, mandatory, ..
            } => {
                assert_eq!(*affords, 4_192);
                assert_eq!(*mandatory, 6_000);
            }
            other => panic!("expected context too small, got {other:?}"),
        }
        // Not pre-flight: the figure it is checked against is only knowable
        // once the target is serving, by which point the source may be gone.
        assert!(!refusal.is_pre_flight());
    }

    #[test]
    fn a_window_that_holds_it_exactly_is_accepted() {
        fits_mandatory("model-b", 8_192, 4_192, 4_000).expect("exactly enough is enough");
    }

    /// Reserves larger than the window afford nothing rather than wrapping
    /// around to an enormous budget.
    #[test]
    fn reserves_larger_than_the_window_afford_nothing() {
        let refusal = fits_mandatory("model-b", 1_000, 1, 4_000).expect_err("nothing fits");
        match refusal {
            ValidationRefusal::ContextTooSmall { affords, .. } => assert_eq!(affords, 0),
            other => panic!("expected context too small, got {other:?}"),
        }
    }

    // -- Fingerprints -----------------------------------------------------

    #[test]
    fn a_hashed_fingerprint_proves_the_bytes_and_a_derived_one_does_not() {
        let mut hashed = entry("model-b");
        hashed.sha256 = Some("a".repeat(64));
        let one = ModelFingerprint::of(&hashed);
        let two = ModelFingerprint::of(&hashed);
        assert_eq!(one.basis, "weightsSha256");
        assert!(one.same_weights_as(&two));

        let mut unhashed = entry("model-b");
        unhashed.sha256 = None;
        let derived = ModelFingerprint::of(&unhashed);
        assert_eq!(derived.basis, "derived");
        // Same entry, same digest — and still not a claim about the file.
        assert_eq!(derived.digest, ModelFingerprint::of(&unhashed).digest);
        assert!(
            !derived.same_weights_as(&ModelFingerprint::of(&unhashed)),
            "a derived digest must never read as proof of identical bytes"
        );
    }

    #[test]
    fn a_different_file_under_the_same_id_fingerprints_differently() {
        let mut first = entry("model-b");
        first.weights_bytes = 4_000_000_000;
        let mut second = entry("model-b");
        second.weights_bytes = 9_000_000_000;
        assert_ne!(
            ModelFingerprint::of(&first).digest,
            ModelFingerprint::of(&second).digest
        );
    }

    // -- Tokenizer comparison ---------------------------------------------

    #[test]
    fn two_different_counts_are_a_change_and_two_equal_ones_are_not() {
        let before = TokenizerProbe::new(21);
        assert!(before.differs_from(&TokenizerProbe::new(19)));
        assert!(!before.differs_from(&TokenizerProbe::new(21)));
    }

    /// Two probes of different strings are not comparable, and saying "the
    /// same" about them would be a measurement nobody made.
    #[test]
    fn probes_of_different_strings_are_not_compared() {
        let mine = TokenizerProbe::new(21);
        let theirs = TokenizerProbe {
            probe: "something else".into(),
            tokens: 19,
        };
        assert!(!mine.differs_from(&theirs));
    }

    // -- What crosses, and what does not ----------------------------------

    fn effect(tool: &str, target: &str) -> CompletedEffect {
        CompletedEffect {
            tool: tool.into(),
            target: target.into(),
            at: "2026-09-18T00:00:00Z".into(),
        }
    }

    /// The structural guarantee. There is no field for a KV cache, and every
    /// opaque thing is named as dropped on every transition rather than being a
    /// decision somebody makes per handoff.
    #[test]
    fn nothing_opaque_ever_crosses_and_all_of_it_is_named() {
        let state = PortableState::new(
            RunMemory::default(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(state.dropped.len(), DroppedState::ALL.len());
        for dropped in DroppedState::ALL {
            assert!(state.dropped.contains(dropped), "{dropped:?} is not named");
            assert!(
                dropped.explain().len() > 60,
                "{dropped:?} needs a reason an operator can read"
            );
        }
    }

    /// Receipts cross, because the new model's first round has to be able to
    /// ask whether a document was already written.
    #[test]
    fn a_receipt_survives_the_handoff() {
        let notes = RunMemory {
            completed: vec![effect("artifact.create_docx", "note.docx")],
            ..RunMemory::default()
        };
        let state = PortableState::new(
            notes,
            Vec::new(),
            vec!["sha-1".into()],
            vec![effect("artifact.create_docx", "note.docx")],
            vec!["artifact-1".into()],
        );
        assert!(state.has_done("artifact.create_docx", "note.docx"));
        assert!(!state.has_done("artifact.create_docx", "other.docx"));
        assert!(state.is_intact());
    }

    #[test]
    fn a_portable_state_notices_being_edited() {
        let mut state = PortableState::new(
            RunMemory::default(),
            Vec::new(),
            vec!["sha-1".into()],
            Vec::new(),
            Vec::new(),
        );
        assert!(state.is_intact());
        state.source_refs.push("sha-2".into());
        assert!(!state.is_intact(), "a changed body must not keep its seal");
    }

    /// Two assemblies of the same state hash the same way, so a record can be
    /// checked against the state it claims to describe.
    #[test]
    fn the_same_state_seals_the_same_way_twice() {
        let build = || {
            PortableState::new(
                RunMemory {
                    goal: "Compare the two tenders.".into(),
                    ..RunMemory::default()
                },
                Vec::new(),
                vec!["sha-1".into(), "sha-2".into()],
                vec![effect("artifact.create_xlsx", "working.xlsx")],
                vec!["artifact-1".into()],
            )
        };
        assert_eq!(build().state_hash, build().state_hash);
    }

    // -- Rollback ---------------------------------------------------------

    /// The assertion a rollback is checked against rather than trusted for.
    #[test]
    fn a_rollback_that_lost_a_receipt_is_not_a_clean_rollback() {
        let mut record = record();
        record.advance(TransitionPhase::Draining, None).expect("drain");
        record
            .advance(TransitionPhase::Checkpointed, None)
            .expect("checkpoint");

        let plan = record.rollback_plan(vec![effect("artifact.create_docx", "note.docx")]);
        assert_eq!(plan.restore_binding, record.from_binding);
        assert_eq!(plan.expected_version, 4);
        assert!(
            !plan.reload_source,
            "nothing has been loaded yet, so nothing was evicted to put back"
        );

        // The receipt is still there: a clean undo.
        let kept = RunMemory {
            completed: vec![effect("artifact.create_docx", "note.docx")],
            ..RunMemory::default()
        };
        plan.preserves(&kept).expect("nothing was lost");

        // The receipt is gone: a resumption could write the document twice.
        let lost = plan
            .preserves(&RunMemory::default())
            .expect_err("a dropped receipt has to be reported");
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].target, "note.docx");
    }

    /// Only the phases at or after the load can have taken a model's memory, so
    /// only those can owe a reload. Abandoning at the validation has released
    /// nothing, and an undo that started "the source" there would be starting a
    /// model nothing had stopped.
    #[test]
    fn only_the_phases_that_could_have_evicted_owe_a_reload() {
        for phase in TransitionPhase::ALL {
            let expected = matches!(
                phase,
                TransitionPhase::Loading
                    | TransitionPhase::Recompiling
                    | TransitionPhase::CommitPending
                    | TransitionPhase::RollingBack
            );
            assert_eq!(phase.may_have_released_memory(), expected, "{phase:?}");
        }
    }

    /// And even then, only what this handoff actually stopped is put back.
    #[test]
    fn a_rollback_reloads_only_the_model_this_handoff_stopped() {
        let mut record = record();
        record.advance(TransitionPhase::Draining, None).expect("drain");
        record
            .advance(TransitionPhase::Checkpointed, None)
            .expect("checkpoint");
        record
            .advance(TransitionPhase::Validating, None)
            .expect("validate");
        record.advance(TransitionPhase::Loading, None).expect("load");

        // Nothing was evicted: the card had room.
        assert!(
            !record.rollback_plan(Vec::new()).reload_source,
            "an undo must not start a model nothing stopped"
        );

        // Something else was evicted, but not this agent's model.
        record.released = vec!["some-other-model".into()];
        assert!(!record.rollback_plan(Vec::new()).reload_source);

        // The source itself was stopped to make room. Now it owes a reload.
        record.released = vec!["model-a".into()];
        assert!(record.rollback_plan(Vec::new()).reload_source);
    }

    // -- Reconciling a crash ----------------------------------------------

    /// The write landed. Read off the registry, not guessed.
    #[test]
    fn a_landed_write_is_recognised_by_the_version_and_the_model() {
        let record = record();
        let pending = record.pending_commit();
        let mut after = agent(Some("model-b"));
        after.definition_version = 5;
        assert_eq!(pending.reconcile(&after), Reconciliation::Landed);
    }

    /// The write did not land. The agent is on the model it was on.
    #[test]
    fn a_write_that_never_happened_is_recognised_too() {
        let record = record();
        let pending = record.pending_commit();
        let before = agent(Some("model-a"));
        assert_eq!(pending.reconcile(&before), Reconciliation::DidNotLand);
    }

    /// The acceptance case that must not be guessed: somebody edited the agent
    /// between the crash and the sweep, so what happened to this write cannot
    /// be read off the record.
    #[test]
    fn a_concurrent_edit_makes_a_crashed_write_indeterminate() {
        let record = record();
        let pending = record.pending_commit();

        // Edited to a third model.
        let mut elsewhere = agent(Some("model-c"));
        elsewhere.definition_version = 5;
        assert_eq!(pending.reconcile(&elsewhere), Reconciliation::Indeterminate);

        // Edited twice, and it does hold the target — still not this write's
        // doing, and still not something to conclude.
        let mut moved_on = agent(Some("model-b"));
        moved_on.definition_version = 6;
        assert_eq!(pending.reconcile(&moved_on), Reconciliation::Indeterminate);
    }

    // -- The record -------------------------------------------------------

    /// Identity is preserved by construction: the record names one agent, and
    /// there is no field through which a transition could change it.
    #[test]
    fn a_transition_never_changes_who_the_agent_is() {
        let mut record = record();
        let agent_id = record.agent_id.clone();
        for next in [
            TransitionPhase::Draining,
            TransitionPhase::Checkpointed,
            TransitionPhase::Validating,
            TransitionPhase::Loading,
            TransitionPhase::Recompiling,
            TransitionPhase::CommitPending,
            TransitionPhase::Committed,
        ] {
            record.advance(next, None).expect("advance");
            assert_eq!(record.agent_id, agent_id);
        }
        // And the memory policy is not in the record at all, so it cannot move.
        assert_eq!(record.from_definition_version, 4);
        assert_eq!(record.to_definition_version, 5);
    }

    #[test]
    fn a_long_reason_is_bounded_rather_than_refused() {
        let agent = agent(Some("model-a"));
        let record = TransitionRecord::begin(
            "tr-1",
            &agent,
            ModelFingerprint::unregistered("model-a"),
            ModelFingerprint::unregistered("model-b"),
            agent.models.rebound_to("model-b"),
            "priya",
            &"x".repeat(MAX_REASON * 3),
        );
        assert_eq!(record.reason.chars().count(), MAX_REASON);
    }

    /// A stored record survives a round trip through its own JSON, because that
    /// is how the ledger keeps it.
    #[test]
    fn a_record_round_trips_through_json() {
        let mut record = record();
        record.advance(TransitionPhase::Draining, None).expect("drain");
        record.freeze = Some(SourceFreeze {
            graph_revision: 42,
            last_event_seq: 17,
            source_manifest_hash: Some("manifest-a".into()),
            checkpoint_hash: Some("checkpoint-a".into()),
            at: "2026-09-18T00:00:00Z".into(),
        });
        let text = serde_json::to_string(&record).expect("serialises");
        let back: TransitionRecord = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, record);
    }

    /// Every phase has a sentence, and none of them is the enum's own spelling
    /// leaked to a screen.
    #[test]
    fn every_phase_describes_itself_in_words() {
        for phase in TransitionPhase::ALL {
            let mut record = record();
            record.phase = *phase;
            let line = record.describe();
            assert!(line.len() > 20, "{phase:?}: {line}");
            assert!(!line.contains('_'), "{phase:?} leaked a stored spelling: {line}");
        }
    }
}

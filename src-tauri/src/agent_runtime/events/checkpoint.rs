//! The point a run can honestly be continued from.
//!
//! ## Why the event history is not enough
//!
//! [`super::store`] already records everything a run did, and a snapshot folds
//! that into what a screen draws. Neither answers the question this module
//! exists for: *may this run be picked up again, and from where?*
//!
//! The history says what happened. It does not say whether the world those
//! events described still holds — whether the workspace is still there, whether
//! the person still has the clearance they had, whether the plan it was held to
//! is the plan it would be held to now. A resumption decided from history alone
//! assumes nothing changed while the run was not running, and the reason a run
//! stopped is usually that something did.
//!
//! So a checkpoint is a *claim about the world*, hashed, taken at a moment the
//! run was safe to interrupt. Resuming re-derives each of those hashes from the
//! world as it is now and refuses on any disagreement. That is the difference
//! between reattaching to a run and continuing one.
//!
//! ## What is deliberately not in here
//!
//! Any value that could be confidential. A checkpoint carries the *shape* of a
//! run — identifiers, hashes, counts, bounded notes that are themselves only
//! markers — and never a passage, an argument, an answer or a document. It is
//! read by the recovery path before anybody has been authenticated, so it must
//! contain nothing that authentication would have protected.
//!
//! ## Why one row per run and not an append-only log
//!
//! The events are the append-only record; this is the resume point, and there is
//! exactly one worth having — the latest safe one. Keeping every checkpoint
//! would mean choosing between them at resume time, and the only defensible
//! choice is the newest, which is the one this stores. Writes are guarded by
//! sequence so a late write from a dying process cannot move the point
//! backwards.

use serde::{Deserialize, Serialize};

use super::machine::RunState;
use super::model::digest;
use crate::agent_runtime::memory::RunMemory;
use crate::agent_runtime::tasks::ContextLedgerRecord;

/// The layout of a stored checkpoint.
///
/// Bumped when a field changes meaning. A checkpoint written under a version
/// this build does not know is refused rather than guessed at: continuing a run
/// from a record you cannot fully read is exactly the case where being wrong is
/// silent.
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Why a run cannot be safely continued.
///
/// Each variant is a sentence an operator can act on. "Cannot resume" with no
/// reason sends somebody to try again; naming the workspace that moved sends
/// them to look at the workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "camelCase")]
pub enum NotResumable {
    /// Nothing was ever checkpointed for this run.
    NoCheckpoint,
    /// A checkpoint exists but this build cannot read it.
    UnreadableCheckpoint { detail: String },
    /// The stored hash does not match the stored body.
    CorruptCheckpoint,
    /// Written by a build whose checkpoint layout this one does not know.
    UnknownSchema { found: u32, understood: u32 },
    /// A side effect was in flight when the run stopped and nobody has said what
    /// happened to it. The most important refusal here: continuing would either
    /// repeat it or assume it worked, and neither is knowable.
    UnknownEffects { keys: Vec<String> },
    /// The run already finished. Reattaching to look at it is fine; continuing
    /// it is not a thing that means anything.
    AlreadyEnded { state: RunState },
    /// The plan is not the plan this checkpoint was taken under.
    PlanChanged,
    /// The person's clearance, the material's classification, or the machine's
    /// mode is not what it was.
    PolicyChanged,
    /// The working directory this run owned is gone or is somewhere else.
    WorkspaceChanged,
    /// The model this run was routed to is not available now.
    ModelUnavailable { model: String },
    /// Somebody else is signed in, or nobody is.
    DifferentOperator,
}

impl NotResumable {
    /// The sentence shown to a person.
    pub fn explain(&self) -> String {
        match self {
            Self::NoCheckpoint => {
                "This run was never checkpointed, so there is no safe point to continue from. \
                 You can look at what it did, and start a new task."
                    .to_string()
            }
            Self::UnreadableCheckpoint { detail } => format!(
                "This run's checkpoint could not be read ({detail}), so continuing it would be \
                 guesswork. Review what it did and start again."
            ),
            Self::CorruptCheckpoint => {
                "This run's checkpoint does not match its own hash, so something altered or \
                 truncated it. It is not safe to continue from."
                    .to_string()
            }
            Self::UnknownSchema { found, understood } => format!(
                "This run was checkpointed by a newer build (format {found}; this build reads \
                 {understood}). Continuing it could mean acting on a record this build only \
                 partly understands."
            ),
            Self::UnknownEffects { keys } => format!(
                "{} action(s) were in flight when this run stopped and nobody has said whether \
                 they took effect. Someone has to check and record what happened before it can \
                 continue.",
                keys.len()
            ),
            Self::AlreadyEnded { state } => format!(
                "This run has already ended ({}). You can read its record; there is nothing to \
                 continue.",
                state.describe()
            ),
            Self::PlanChanged => {
                "The plan this run was held to is not the plan it would be held to now, so \
                 continuing would carry it on under rules it never agreed to."
                    .to_string()
            }
            Self::PolicyChanged => {
                "The permissions, the material's classification, or this machine's mode have \
                 changed since this run stopped. Continuing would apply the old decision to the \
                 new situation."
                    .to_string()
            }
            Self::WorkspaceChanged => {
                "The working directory this run owned is missing or has moved, so the files it \
                 was working on cannot be identified. Continuing could write over something else."
                    .to_string()
            }
            Self::ModelUnavailable { model } => format!(
                "The model this run was routed to ({model}) is not available now. Continuing on a \
                 different model would produce work the first half of the run does not match."
            ),
            Self::DifferentOperator => {
                "This run belongs to a different person, or nobody is signed in. Sign in as the \
                 person who started it to continue it."
                    .to_string()
            }
        }
    }

    /// Whether this refusal means a person has to reconcile something.
    ///
    /// Distinct from "cannot resume" generally: an unknown effect is not a
    /// closed door, it is a question waiting for an answer, and the screen
    /// should send somebody to answer it rather than to start over.
    pub fn needs_human_reconciliation(&self) -> bool {
        matches!(self, Self::UnknownEffects { .. })
    }
}

/// The state of the world a run was safe to continue from.
///
/// Every field is an identifier, a hash, a count, or bounded notes that are
/// themselves only markers. See the module note on what is deliberately absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunCheckpoint {
    /// The logical task. Stable across every attempt.
    pub run_id: String,
    /// This attempt at it. A resumption gets a new one, so the trace can show
    /// that the same task was picked up again rather than that it restarted.
    pub attempt_id: String,
    /// The last lifecycle state the run was known to be safely at.
    pub state: RunState,
    /// The durable event this checkpoint was taken after. Guards the write
    /// order and tells a resumption where the history it already knows ends.
    pub last_event_seq: i64,
    /// The run's own bounded notes — goal, stage, marker lists, completed
    /// effects. Markers, never content. See [`RunMemory`].
    pub notes: RunMemory,
    /// Where the context window stood. Absent before the first measurement.
    pub ledger: Option<ContextLedgerRecord>,
    /// The plan the run is held to, hashed. A different plan is a different run.
    pub plan_hash: String,
    /// The person's roles, the material's classification and the machine's mode,
    /// hashed together. One hash rather than three fields because the question
    /// at resume time is "is any of this different", and three comparisons
    /// invite two of them being written and the third forgotten.
    pub policy_hash: String,
    /// The run's working directory, hashed. Identity, not a path: the path is on
    /// the machine and does not belong in a record read before sign-in.
    pub workspace_hash: String,
    /// The model this run was routed to.
    pub model_id: String,
    /// Idempotency keys of side effects nobody has settled. Non-empty means the
    /// run is not resumable until a person says what happened.
    pub unknown_effects: Vec<String>,
    /// RFC 3339, UTC.
    pub at: String,
    pub schema_version: u32,
    /// Over every field above. Checked on load, so a truncated or edited row is
    /// refused rather than acted on.
    pub checkpoint_hash: String,
}

impl RunCheckpoint {
    /// Builds a checkpoint and seals it with its own hash.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        attempt_id: impl Into<String>,
        state: RunState,
        last_event_seq: i64,
        notes: RunMemory,
        ledger: Option<ContextLedgerRecord>,
        plan_hash: impl Into<String>,
        policy_hash: impl Into<String>,
        workspace_hash: impl Into<String>,
        model_id: impl Into<String>,
        unknown_effects: Vec<String>,
    ) -> Self {
        let mut checkpoint = Self {
            run_id: run_id.into(),
            attempt_id: attempt_id.into(),
            state,
            last_event_seq,
            notes,
            ledger,
            plan_hash: plan_hash.into(),
            policy_hash: policy_hash.into(),
            workspace_hash: workspace_hash.into(),
            model_id: model_id.into(),
            unknown_effects,
            at: chrono::Utc::now().to_rfc3339(),
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_hash: String::new(),
        };
        checkpoint.checkpoint_hash = checkpoint.compute_hash();
        checkpoint
    }

    /// The hash of everything except the hash itself.
    ///
    /// Built from a canonical string rather than from serialised JSON, because
    /// JSON field order is a property of the serialiser and this has to be
    /// stable across builds of it. Every field is included: a field left out of
    /// the hash is a field an editor could change undetected.
    pub fn compute_hash(&self) -> String {
        let notes = serde_json::to_string(&self.notes).unwrap_or_default();
        let ledger = serde_json::to_string(&self.ledger).unwrap_or_default();
        digest(&format!(
            "v{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.schema_version,
            self.run_id,
            self.attempt_id,
            self.state.as_str(),
            self.last_event_seq,
            digest(&notes),
            digest(&ledger),
            self.plan_hash,
            self.policy_hash,
            self.workspace_hash,
            self.model_id,
            self.unknown_effects.join(","),
            self.at,
        ))
    }

    /// Whether the stored hash still matches the stored body.
    pub fn is_intact(&self) -> bool {
        self.checkpoint_hash == self.compute_hash()
    }

    /// Whether this checkpoint could be continued from, given the world now.
    ///
    /// Takes the freshly-derived facts rather than deriving them, so the caller
    /// that knows how to compute each one owns that, and this owns only the
    /// comparison. The order is deliberate: the refusal that needs a person
    /// comes before the ones that need a new task, so an operator is sent to the
    /// more specific remedy first.
    pub fn resumable_against(&self, now: &WorldNow) -> Result<(), NotResumable> {
        if !self.is_intact() {
            return Err(NotResumable::CorruptCheckpoint);
        }
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(NotResumable::UnknownSchema {
                found: self.schema_version,
                understood: CHECKPOINT_SCHEMA_VERSION,
            });
        }
        // Before anything else about the world: an effect nobody settled is a
        // question, and every other check is moot until it is answered.
        if !self.unknown_effects.is_empty() {
            return Err(NotResumable::UnknownEffects {
                keys: self.unknown_effects.clone(),
            });
        }
        if now.ended {
            return Err(NotResumable::AlreadyEnded { state: now.state });
        }
        if !now.same_operator {
            return Err(NotResumable::DifferentOperator);
        }
        if now.policy_hash != self.policy_hash {
            return Err(NotResumable::PolicyChanged);
        }
        if now.plan_hash != self.plan_hash {
            return Err(NotResumable::PlanChanged);
        }
        if now.workspace_hash.as_deref() != Some(self.workspace_hash.as_str()) {
            return Err(NotResumable::WorkspaceChanged);
        }
        if !now.model_available {
            return Err(NotResumable::ModelUnavailable {
                model: self.model_id.clone(),
            });
        }
        Ok(())
    }
}

/// The world as it is at the moment somebody asks to resume.
///
/// A plain record of freshly-derived facts. Assembling it is the caller's job
/// and comparing it is [`RunCheckpoint::resumable_against`]'s — the split exists
/// so the comparison can be tested exhaustively without a running application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldNow {
    /// Re-derived from the signed-in session, the classification and the mode.
    pub policy_hash: String,
    /// Re-derived from the prompt, deterministically, the same way the run's
    /// plan was derived in the first place.
    pub plan_hash: String,
    /// `None` when the directory is gone.
    pub workspace_hash: Option<String>,
    /// Whether the routed model can be served right now.
    pub model_available: bool,
    /// Whether the person asking is the person the run belongs to.
    pub same_operator: bool,
    /// Whether the run has already reached a terminal state.
    pub ended: bool,
    /// The state it is in, for the message when it has ended.
    pub state: RunState,
}

/// How resumable a run is, as a screen needs to draw it.
///
/// Three states rather than a boolean, because "cannot continue" splits into two
/// genuinely different situations with different remedies: one wants a person to
/// reconcile something, and the other wants them to read the record and move on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum Resumability {
    /// The backend has checked and this run can be safely continued.
    Resumable {
        attempt_id: String,
        /// The event the resumption would carry on after.
        from_seq: i64,
        /// What the run was doing when it stopped.
        state: RunState,
    },
    /// A person has to settle something first. Carries what.
    NeedsReconciliation { because: String, keys: Vec<String> },
    /// Read-only. Reattaching shows what happened; there is nothing to continue.
    ViewOnly { because: String },
}

impl Resumability {
    /// Turns a check into the three-way answer a screen draws.
    pub fn of(checkpoint: Option<&RunCheckpoint>, now: &WorldNow) -> Self {
        let Some(checkpoint) = checkpoint else {
            return Self::ViewOnly {
                because: NotResumable::NoCheckpoint.explain(),
            };
        };
        match checkpoint.resumable_against(now) {
            Ok(()) => Self::Resumable {
                attempt_id: checkpoint.attempt_id.clone(),
                from_seq: checkpoint.last_event_seq,
                state: checkpoint.state,
            },
            Err(refusal) if refusal.needs_human_reconciliation() => {
                let keys = match &refusal {
                    NotResumable::UnknownEffects { keys } => keys.clone(),
                    _ => Vec::new(),
                };
                Self::NeedsReconciliation {
                    because: refusal.explain(),
                    keys,
                }
            }
            Err(refusal) => Self::ViewOnly {
                because: refusal.explain(),
            },
        }
    }

    pub fn is_resumable(&self) -> bool {
        matches!(self, Self::Resumable { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> RunCheckpoint {
        RunCheckpoint::new(
            "run-1",
            "attempt-1",
            RunState::ToolResultRecorded,
            12,
            RunMemory::default(),
            None,
            "plan-hash",
            "policy-hash",
            "workspace-hash",
            "qwen2.5-7b",
            Vec::new(),
        )
    }

    fn world() -> WorldNow {
        WorldNow {
            policy_hash: "policy-hash".to_string(),
            plan_hash: "plan-hash".to_string(),
            workspace_hash: Some("workspace-hash".to_string()),
            model_available: true,
            same_operator: true,
            ended: false,
            state: RunState::ToolResultRecorded,
        }
    }

    #[test]
    fn a_checkpoint_seals_itself_and_notices_being_edited() {
        let mut point = checkpoint();
        assert!(point.is_intact());

        // Every field is in the hash, so any of these is caught. The one that
        // matters most is `unknown_effects`: clearing it is exactly how a
        // corrupted record would claim a run is safe to continue.
        let mut moved_seq = point.clone();
        moved_seq.last_event_seq = 99;
        assert!(!moved_seq.is_intact());

        let mut cleared_effects = point.clone();
        cleared_effects.unknown_effects = vec!["k".to_string()];
        assert!(!cleared_effects.is_intact());

        point.policy_hash = "something-else".to_string();
        assert!(!point.is_intact());
    }

    #[test]
    fn an_intact_checkpoint_against_an_unchanged_world_is_resumable() {
        assert!(checkpoint().resumable_against(&world()).is_ok());
        assert!(Resumability::of(Some(&checkpoint()), &world()).is_resumable());
    }

    #[test]
    fn a_corrupt_checkpoint_is_refused_before_anything_else_is_considered() {
        // Including when the world is otherwise perfect: a record that does not
        // match its own hash is not evidence of anything.
        let mut point = checkpoint();
        point.checkpoint_hash = "tampered".to_string();
        assert_eq!(
            point.resumable_against(&world()),
            Err(NotResumable::CorruptCheckpoint)
        );
    }

    #[test]
    fn an_unsettled_side_effect_blocks_resumption_and_asks_for_a_person() {
        // The refusal that matters most. Continuing would either repeat the
        // effect or assume it worked, and nothing here can tell which.
        let point = RunCheckpoint::new(
            "run-1",
            "attempt-1",
            RunState::ExecutingTool,
            12,
            RunMemory::default(),
            None,
            "plan-hash",
            "policy-hash",
            "workspace-hash",
            "qwen2.5-7b",
            vec!["create_docx:abc123".to_string()],
        );

        let refusal = point.resumable_against(&world()).expect_err("must refuse");
        assert!(refusal.needs_human_reconciliation());

        let answer = Resumability::of(Some(&point), &world());
        match answer {
            Resumability::NeedsReconciliation { keys, .. } => {
                assert_eq!(keys, vec!["create_docx:abc123".to_string()]);
            }
            other => panic!("expected reconciliation, got {other:?}"),
        }
    }

    #[test]
    fn an_unsettled_effect_outranks_every_other_disagreement() {
        // If the policy also changed, the person still needs to settle the
        // effect: telling them to start over would leave the effect unresolved.
        let point = RunCheckpoint::new(
            "run-1",
            "attempt-1",
            RunState::ExecutingTool,
            12,
            RunMemory::default(),
            None,
            "plan-hash",
            "policy-hash",
            "workspace-hash",
            "qwen2.5-7b",
            vec!["create_docx:abc123".to_string()],
        );
        let mut now = world();
        now.policy_hash = "different".to_string();
        now.workspace_hash = None;

        let refusal = point.resumable_against(&now).expect_err("must refuse");
        assert!(matches!(refusal, NotResumable::UnknownEffects { .. }));
    }

    #[test]
    fn a_changed_policy_blocks_resumption() {
        let mut now = world();
        now.policy_hash = "roles-changed".to_string();
        assert_eq!(
            checkpoint().resumable_against(&now),
            Err(NotResumable::PolicyChanged)
        );
    }

    #[test]
    fn a_changed_plan_blocks_resumption() {
        let mut now = world();
        now.plan_hash = "different-plan".to_string();
        assert_eq!(
            checkpoint().resumable_against(&now),
            Err(NotResumable::PlanChanged)
        );
    }

    #[test]
    fn a_missing_or_moved_workspace_blocks_resumption() {
        let mut gone = world();
        gone.workspace_hash = None;
        assert_eq!(
            checkpoint().resumable_against(&gone),
            Err(NotResumable::WorkspaceChanged)
        );

        let mut moved = world();
        moved.workspace_hash = Some("somewhere-else".to_string());
        assert_eq!(
            checkpoint().resumable_against(&moved),
            Err(NotResumable::WorkspaceChanged)
        );
    }

    #[test]
    fn an_unavailable_model_blocks_resumption() {
        let mut now = world();
        now.model_available = false;
        assert_eq!(
            checkpoint().resumable_against(&now),
            Err(NotResumable::ModelUnavailable {
                model: "qwen2.5-7b".to_string()
            })
        );
    }

    #[test]
    fn another_persons_run_is_not_resumable_by_this_one() {
        let mut now = world();
        now.same_operator = false;
        assert_eq!(
            checkpoint().resumable_against(&now),
            Err(NotResumable::DifferentOperator)
        );
    }

    #[test]
    fn a_finished_run_is_view_only_rather_than_resumable() {
        let mut now = world();
        now.ended = true;
        now.state = RunState::Completed;
        assert!(matches!(
            checkpoint().resumable_against(&now),
            Err(NotResumable::AlreadyEnded { .. })
        ));
        assert!(matches!(
            Resumability::of(Some(&checkpoint()), &now),
            Resumability::ViewOnly { .. }
        ));
    }

    #[test]
    fn a_run_with_no_checkpoint_is_view_only_and_says_why() {
        let answer = Resumability::of(None, &world());
        match answer {
            Resumability::ViewOnly { because } => assert!(because.contains("never checkpointed")),
            other => panic!("expected view-only, got {other:?}"),
        }
    }

    #[test]
    fn a_checkpoint_from_a_newer_build_is_refused_rather_than_guessed_at() {
        let mut point = checkpoint();
        point.schema_version = CHECKPOINT_SCHEMA_VERSION + 1;
        point.checkpoint_hash = point.compute_hash();

        assert!(matches!(
            point.resumable_against(&world()),
            Err(NotResumable::UnknownSchema { .. })
        ));
    }

    #[test]
    fn every_refusal_explains_itself_in_a_sentence_a_person_can_act_on() {
        let refusals = [
            NotResumable::NoCheckpoint,
            NotResumable::CorruptCheckpoint,
            NotResumable::UnknownSchema {
                found: 2,
                understood: 1,
            },
            NotResumable::UnknownEffects {
                keys: vec!["k".to_string()],
            },
            NotResumable::AlreadyEnded {
                state: RunState::Completed,
            },
            NotResumable::PlanChanged,
            NotResumable::PolicyChanged,
            NotResumable::WorkspaceChanged,
            NotResumable::ModelUnavailable {
                model: "m".to_string(),
            },
            NotResumable::DifferentOperator,
            NotResumable::UnreadableCheckpoint {
                detail: "truncated".to_string(),
            },
        ];
        for refusal in refusals {
            let sentence = refusal.explain();
            assert!(sentence.len() > 40, "too terse: {sentence}");
            assert!(sentence.ends_with('.'), "not a sentence: {sentence}");
        }
    }

    #[test]
    fn a_checkpoint_carries_no_free_text_that_could_hold_a_document() {
        // The confidentiality claim, asserted structurally. Every field is an
        // id, a hash, a count, an enum or bounded notes of markers — so a
        // serialised checkpoint of an empty run is small, and stays small.
        let serialised = serde_json::to_string(&checkpoint()).expect("serialises");
        assert!(
            serialised.len() < 1_200,
            "a checkpoint grew big enough to hold content: {} bytes",
            serialised.len()
        );
    }
}

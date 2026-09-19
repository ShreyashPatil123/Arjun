//! What the loop proposes, and what Rust is prepared to write down.
//!
//! ## The failure this exists to remove
//!
//! Production checkpoints were taken with `RunMemory::default()`. Literally:
//! `recording::remember_outcome` ran after every tool result and committed an
//! empty set of notes, so the one record a recovery reads said the run had no
//! goal, no plan position, no evidence, no artifacts and — the dangerous one —
//! no completed effects.
//!
//! The loop's real notes existed, in the child process, and reached Rust only
//! in the `RunResult` at the very end. A run that died in the middle therefore
//! left a checkpoint describing a run that had done nothing, and
//! `notes_to_resume_from` filtered it out as empty, so the resumption started
//! over. The approval note gets written twice, and neither record says why.
//!
//! ## Why the proposal is not simply believed
//!
//! Because the notes are assembled in the process that runs the model, and one
//! of the things they claim is *which side effects have already happened*. A
//! resumed run reads `completed` to decide what not to do again. So a note
//! saying `create_docx` is done is, functionally, an instruction not to write
//! the document — and if the model can put it there, the model can decide the
//! document does not need writing.
//!
//! `note-taking.ts` is careful about this today: entries are taken from tool
//! *results*, not from the model's text. That is the right design and it is not
//! a guarantee, because it is enforced on the far side of a JSON-RPC channel by
//! the process being constrained. The guarantee has to be here.
//!
//! So every claim that could excuse work is checked against something Rust
//! itself recorded:
//!
//! | Claim | Checked against |
//! |---|---|
//! | a completed side effect | `ToolSucceeded` / `ArtifactProduced` in the durable event log |
//! | an evidence marker `[En]` | the run's own passage table, which issued the markers |
//! | an artifact id | the run's produced-file table and the log |
//! | a calculation id | the run's calculation table |
//! | a milestone | never proposable — Rust writes those itself |
//!
//! Everything else in the notes — goal, stage, next action, decisions, open
//! questions — is the model's account of its own work. It cannot excuse an
//! action, it is already bounded by the runtime's caps, and it is what makes a
//! resumption legible. It is kept as sent.
//!
//! ## Monotonic, so a late commit cannot lose an effect
//!
//! A proposal is merged with what was already committed rather than replacing
//! it. An effect Rust has accepted once stays accepted: a loop that compacted
//! its notes, or a proposal that arrived shorter because a cap dropped the
//! oldest entry, must not be able to make a completed write look un-done.
//! Losing an entry here is the same failure as never recording it.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::events::machine::RunState;
use super::memory::{CompletedEffect, MilestoneRecord, RunMemory};

/// The layout of a state commit.
///
/// Bumped when a field changes meaning. A proposal under a version this build
/// does not know is refused rather than partly applied.
pub const STATE_COMMIT_VERSION: u32 = 1;

/// What the loop asks Rust to write down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateProposal {
    pub commit_version: u32,
    pub run_id: String,
    /// The attempt this proposal belongs to.
    ///
    /// Checked against the attempt Rust started, so a commit from a worker that
    /// belongs to a previous attempt — a straggler that outlived a restart —
    /// cannot move the resume point of the attempt now running.
    pub attempt_id: String,
    /// Where the run is. Named by the loop because the loop is what knows; used
    /// by the recovery path to tell a run that died mid-tool from one that died
    /// mid-compaction.
    pub state: RunState,
    /// The loop's account of the run.
    pub notes: RunMemory,
}

/// Why a claim in a proposal was not written down.
///
/// Each variant names something specific. A correction that said only
/// "rejected" would leave an operator with a run that silently disagrees with
/// its own notes and no way to find out where.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "correction", rename_all = "camelCase")]
pub enum Correction {
    /// A side effect was claimed that the durable log does not corroborate.
    /// The most important one: it is the claim that would stop a resumed run
    /// doing work that never happened.
    UnbackedEffect { tool: String, target: String },
    /// An evidence marker no search on this run handed out.
    UnbackedEvidence { marker: String },
    /// An artifact id this run did not produce.
    UnbackedArtifact { id: String },
    /// A calculation id the engine did not issue.
    UnbackedCalculation { id: String },
    /// The proposal carried milestones. Rust writes those when a person signs
    /// off, and a loop cannot propose one.
    MilestonesAreNotProposable { offered: u32 },
    /// The proposal omitted an effect Rust had already accepted. Kept anyway.
    EffectWouldHaveBeenLost { tool: String, target: String },
}

impl Correction {
    /// The sentence written to the log, in an operator's terms.
    pub fn explain(&self) -> String {
        match self {
            Self::UnbackedEffect { tool, target } => format!(
                "the notes claimed {tool} had already been done to {target:?}, and no tool \
                 receipt in this run's history corroborates it; the claim was dropped so a \
                 resumption does not skip work that never happened"
            ),
            Self::UnbackedEvidence { marker } => format!(
                "the notes cited {marker}, which no search on this run handed out; dropped"
            ),
            Self::UnbackedArtifact { id } => {
                format!("the notes named artifact {id:?}, which this run did not produce; dropped")
            }
            Self::UnbackedCalculation { id } => format!(
                "the notes named calculation {id:?}, which the engine did not issue; dropped"
            ),
            Self::MilestonesAreNotProposable { offered } => format!(
                "the proposal carried {offered} milestone(s); milestones are written when a \
                 person signs one off and were replaced with the recorded ones"
            ),
            Self::EffectWouldHaveBeenLost { tool, target } => format!(
                "the proposal no longer listed {tool} on {target:?}, which was already accepted; \
                 it was kept, because forgetting a completed effect is how one happens twice"
            ),
        }
    }

    /// Whether this correction means the run's own account was wrong about
    /// something that could have caused duplicate work.
    pub fn is_serious(&self) -> bool {
        matches!(
            self,
            Self::UnbackedEffect { .. } | Self::EffectWouldHaveBeenLost { .. }
        )
    }
}

/// What Rust itself recorded about this run.
///
/// Assembled by the caller from the durable event log and the run's own tables,
/// and compared here. The split is the same one [`super::events::checkpoint`]
/// makes between `WorldNow` and `resumable_against`: gathering needs an
/// application, and the rule needs to be testable exhaustively without one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RustFacts {
    /// How many times each tool succeeded, from `ToolSucceeded` events.
    pub succeeded_tools: HashMap<String, u32>,
    /// `(artifact name, tool)` pairs from `ArtifactProduced` events.
    pub produced_artifacts: Vec<(String, String)>,
    /// How many evidence markers this run's searches handed out. Markers are
    /// numbered from one, so `E{n}` is backed when `n <= evidence_count`.
    pub evidence_count: u32,
    /// How many calculations the engine ran for this run. Ids are `C{n}`.
    pub calculation_count: u32,
    /// Artifact names the run's produced-file table holds.
    pub artifact_names: Vec<String>,
    /// The milestones a person actually signed off. Authoritative.
    pub milestones: Vec<MilestoneRecord>,
}

/// A marker of the form `E12` or `C3`, as a number.
fn numbered(marker: &str, prefix: char) -> Option<u32> {
    let mut chars = marker.chars();
    if chars.next()? != prefix {
        return None;
    }
    let rest: String = chars.collect();
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// Whether the log corroborates one claimed effect.
///
/// Two ways, because tools leave two different kinds of receipt. A tool that
/// produces a file is matched on the *file*, which is the strong check: the
/// event names it. A tool that has an effect and leaves no file is matched on
/// the tool having succeeded at least as many times as it is claimed to have
/// been done, which is the strongest check the log supports for it.
fn corroborates(
    effect: &CompletedEffect,
    facts: &RustFacts,
    claimed_so_far: &HashMap<String, u32>,
) -> bool {
    let named = facts
        .produced_artifacts
        .iter()
        .any(|(name, tool)| name == &effect.target && tool == &effect.tool);
    if named {
        return true;
    }
    // The produced-file table is the same fact from the run's own side.
    if facts.artifact_names.iter().any(|name| name == &effect.target) {
        return true;
    }
    let succeeded = facts.succeeded_tools.get(&effect.tool).copied().unwrap_or(0);
    let already = claimed_so_far.get(&effect.tool).copied().unwrap_or(0);
    already < succeeded
}

/// Checks a proposal against what Rust recorded, and returns what will be
/// written.
///
/// `previous` is the last set of notes Rust accepted for this run. The result
/// is never smaller than it in the ways that matter: an effect already accepted
/// stays accepted.
pub fn validate(
    proposal: &StateProposal,
    facts: &RustFacts,
    previous: &RunMemory,
) -> (RunMemory, Vec<Correction>) {
    let mut corrections = Vec::new();
    let mut notes = proposal.notes.clone();

    // ── Completed effects ────────────────────────────────────────────────
    // The claim that can excuse work, so the one checked hardest.
    let mut admitted: Vec<CompletedEffect> = Vec::new();
    let mut claimed_so_far: HashMap<String, u32> = HashMap::new();
    for effect in &proposal.notes.completed {
        // Something Rust already accepted needs no second corroboration: it was
        // checked when it was first written, and the receipt it was checked
        // against may since have aged out of the window this build reads.
        let already_accepted = previous.has_done(&effect.tool, &effect.target);
        if already_accepted || corroborates(effect, facts, &claimed_so_far) {
            *claimed_so_far.entry(effect.tool.clone()).or_default() += 1;
            admitted.push(effect.clone());
        } else {
            corrections.push(Correction::UnbackedEffect {
                tool: effect.tool.clone(),
                target: effect.target.clone(),
            });
        }
    }

    // Anything Rust had accepted and this proposal dropped is put back. A cap
    // that shifted the oldest entry out, or a compaction that rebuilt the
    // notes, must not be able to un-complete a write.
    for effect in &previous.completed {
        if !admitted
            .iter()
            .any(|kept| kept.tool == effect.tool && kept.target == effect.target)
        {
            corrections.push(Correction::EffectWouldHaveBeenLost {
                tool: effect.tool.clone(),
                target: effect.target.clone(),
            });
            admitted.push(effect.clone());
        }
    }
    notes.completed = admitted;

    // ── Evidence markers ─────────────────────────────────────────────────
    notes.evidence_ids = proposal
        .notes
        .evidence_ids
        .iter()
        .filter(|marker| {
            let backed = numbered(marker, 'E')
                .map(|n| n >= 1 && n <= facts.evidence_count)
                .unwrap_or(false);
            if !backed {
                corrections.push(Correction::UnbackedEvidence {
                    marker: (*marker).clone(),
                });
            }
            backed
        })
        .cloned()
        .collect();

    // ── Calculation ids ──────────────────────────────────────────────────
    notes.calculation_ids = proposal
        .notes
        .calculation_ids
        .iter()
        .filter(|id| {
            let backed = numbered(id, 'C')
                .map(|n| n >= 1 && n <= facts.calculation_count)
                .unwrap_or(false);
            if !backed {
                corrections.push(Correction::UnbackedCalculation { id: (*id).clone() });
            }
            backed
        })
        .cloned()
        .collect();

    // ── Artifact ids ─────────────────────────────────────────────────────
    notes.artifact_ids = proposal
        .notes
        .artifact_ids
        .iter()
        .filter(|id| {
            let backed = facts.artifact_names.iter().any(|name| name == *id)
                || facts.produced_artifacts.iter().any(|(name, _)| name == *id)
                || previous.artifact_ids.iter().any(|kept| kept == *id);
            if !backed {
                corrections.push(Correction::UnbackedArtifact { id: (*id).clone() });
            }
            backed
        })
        .cloned()
        .collect();

    // ── Milestones ───────────────────────────────────────────────────────
    // Never proposable. A milestone is a person's signature, written by
    // `agent_acknowledge_milestone` against the plan; a loop offering one is
    // offering to sign on their behalf.
    if !proposal.notes.milestones.is_empty() && proposal.notes.milestones != facts.milestones {
        corrections.push(Correction::MilestonesAreNotProposable {
            offered: proposal.notes.milestones.len() as u32,
        });
    }
    notes.milestones = facts.milestones.clone();

    // ── Narration ────────────────────────────────────────────────────────
    // Goal, stage, next action, decisions and open questions are the model's
    // account of its own work. They cannot excuse an action and are already
    // bounded by the runtime's caps. Kept as sent — except that a proposal
    // which has *forgotten* the goal does not get to blank one Rust holds,
    // because a resumption with no goal reads as a run that never had one.
    if notes.goal.trim().is_empty() && !previous.goal.trim().is_empty() {
        notes.goal = previous.goal.clone();
    }
    if notes.stage.ordinal < previous.stage.ordinal {
        notes.stage = previous.stage.clone();
    }

    (notes, corrections)
}

/// What Rust decided about a proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitOutcome {
    /// False only when the proposal could not be considered at all — an unknown
    /// version, or an attempt that is not the one running.
    pub accepted: bool,
    /// Why not, when `accepted` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refused_because: Option<String>,
    /// The durable event this commit was taken after.
    pub last_event_seq: i64,
    /// What Rust changed about the proposal before writing it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<Correction>,
    /// The notes as written. Sent back so the loop's copy converges on Rust's
    /// rather than drifting from it — a loop that went on believing a dropped
    /// claim would propose it again on every commit.
    pub notes: RunMemory,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::memory::{RunMemory, RunStage};

    fn effect(tool: &str, target: &str) -> CompletedEffect {
        CompletedEffect {
            tool: tool.to_string(),
            target: target.to_string(),
            at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn proposal(notes: RunMemory) -> StateProposal {
        StateProposal {
            commit_version: STATE_COMMIT_VERSION,
            run_id: "run-1".into(),
            attempt_id: "attempt-1".into(),
            state: RunState::ToolResultRecorded,
            notes,
        }
    }

    fn facts() -> RustFacts {
        RustFacts::default()
    }

    /// The headline rule. A loop that says it wrote the approval note, with
    /// nothing in the history saying so, does not get to stop a resumption
    /// writing it.
    #[test]
    fn notes_cannot_mark_an_unexecuted_tool_done() {
        let notes = RunMemory {
            completed: vec![effect("artifact.create_approval_note", "note.docx")],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts(), &RunMemory::default());

        assert!(written.completed.is_empty(), "an unbacked effect was written");
        assert!(!written.has_done("artifact.create_approval_note", "note.docx"));
        assert_eq!(
            corrections,
            vec![Correction::UnbackedEffect {
                tool: "artifact.create_approval_note".into(),
                target: "note.docx".into(),
            }]
        );
        assert!(corrections[0].is_serious());
    }

    #[test]
    fn an_effect_the_log_names_is_written() {
        let mut facts = facts();
        facts
            .produced_artifacts
            .push(("note.docx".into(), "artifact.create_approval_note".into()));
        let notes = RunMemory {
            completed: vec![effect("artifact.create_approval_note", "note.docx")],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());
        assert!(written.has_done("artifact.create_approval_note", "note.docx"));
        assert!(corrections.is_empty());
    }

    /// A file-less side effect is corroborated by the tool having succeeded,
    /// and only as many times as it actually did.
    #[test]
    fn a_tool_claimed_more_often_than_it_succeeded_is_trimmed() {
        let mut facts = facts();
        facts.succeeded_tools.insert("sandbox.run_code".into(), 1);
        let notes = RunMemory {
            completed: vec![
                effect("sandbox.run_code", "first"),
                effect("sandbox.run_code", "second"),
            ],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());
        assert_eq!(written.completed.len(), 1);
        assert_eq!(written.completed[0].target, "first");
        assert_eq!(
            corrections,
            vec![Correction::UnbackedEffect {
                tool: "sandbox.run_code".into(),
                target: "second".into(),
            }]
        );
    }

    /// The monotonic rule. Forgetting a completed effect is how one happens
    /// twice, so a proposal that drops one does not get to.
    #[test]
    fn an_already_accepted_effect_cannot_be_dropped_by_a_later_proposal() {
        let previous = RunMemory {
            completed: vec![effect("artifact.create_approval_note", "note.docx")],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(RunMemory::default()), &facts(), &previous);

        assert!(
            written.has_done("artifact.create_approval_note", "note.docx"),
            "a completed write was forgotten"
        );
        assert_eq!(
            corrections,
            vec![Correction::EffectWouldHaveBeenLost {
                tool: "artifact.create_approval_note".into(),
                target: "note.docx".into(),
            }]
        );
        assert!(corrections[0].is_serious());
    }

    /// And it does not need re-corroborating: the receipt it was checked
    /// against may have aged out of the window this build reads.
    #[test]
    fn an_already_accepted_effect_is_not_rechecked() {
        let previous = RunMemory {
            completed: vec![effect("sandbox.run_code", "once")],
            ..RunMemory::default()
        };
        let notes = RunMemory {
            completed: vec![effect("sandbox.run_code", "once")],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts(), &previous);
        assert!(written.has_done("sandbox.run_code", "once"));
        assert!(corrections.is_empty());
    }

    #[test]
    fn evidence_markers_beyond_what_the_run_issued_are_dropped() {
        let mut facts = facts();
        facts.evidence_count = 2;
        let notes = RunMemory {
            evidence_ids: vec!["E1".into(), "E2".into(), "E7".into(), "banana".into()],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());
        assert_eq!(written.evidence_ids, vec!["E1", "E2"]);
        assert_eq!(
            corrections,
            vec![
                Correction::UnbackedEvidence { marker: "E7".into() },
                Correction::UnbackedEvidence {
                    marker: "banana".into()
                },
            ]
        );
    }

    #[test]
    fn calculation_ids_beyond_what_the_engine_ran_are_dropped() {
        let mut facts = facts();
        facts.calculation_count = 1;
        let notes = RunMemory {
            calculation_ids: vec!["C1".into(), "C2".into()],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());
        assert_eq!(written.calculation_ids, vec!["C1"]);
        assert_eq!(
            corrections,
            vec![Correction::UnbackedCalculation { id: "C2".into() }]
        );
    }

    #[test]
    fn artifact_ids_this_run_did_not_produce_are_dropped() {
        let mut facts = facts();
        facts.artifact_names.push("report.xlsx".into());
        let notes = RunMemory {
            artifact_ids: vec!["report.xlsx".into(), "someone-elses.docx".into()],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());
        assert_eq!(written.artifact_ids, vec!["report.xlsx"]);
        assert_eq!(
            corrections,
            vec![Correction::UnbackedArtifact {
                id: "someone-elses.docx".into()
            }]
        );
    }

    /// A milestone is a person's signature. The loop cannot offer one, and
    /// cannot delete one either.
    #[test]
    fn milestones_come_from_rust_and_never_from_the_proposal() {
        let signed = MilestoneRecord {
            checkpoint_id: "m1".into(),
            ordinal: 1,
            intent: "Approve the scope".into(),
            acknowledged_by: "user-1".into(),
            at: "2026-01-01T00:00:00Z".into(),
            decision: "approved".into(),
        };
        let mut facts = facts();
        facts.milestones.push(signed.clone());

        let forged = MilestoneRecord {
            checkpoint_id: "m2".into(),
            ordinal: 2,
            intent: "Approve the spend".into(),
            acknowledged_by: "user-1".into(),
            at: "2026-01-01T00:00:00Z".into(),
            decision: "approved".into(),
        };
        let notes = RunMemory {
            milestones: vec![signed.clone(), forged],
            ..RunMemory::default()
        };
        let (written, corrections) = validate(&proposal(notes), &facts, &RunMemory::default());

        assert_eq!(written.milestones, vec![signed]);
        assert_eq!(
            corrections,
            vec![Correction::MilestonesAreNotProposable { offered: 2 }]
        );
    }

    /// Narration is kept. It cannot excuse work, and it is what makes a
    /// resumption legible to the person reading it.
    #[test]
    fn the_models_account_of_its_own_work_is_kept() {
        let notes = RunMemory {
            goal: "Produce the inspection note".into(),
            next_action: "Draft section 3".into(),
            stage: RunStage {
                ordinal: 3,
                intent: "Drafting".into(),
            },
            open_questions: vec!["Which revision of the P&ID applies?".into()],
            ..RunMemory::default()
        };
        let (written, corrections) =
            validate(&proposal(notes.clone()), &facts(), &RunMemory::default());
        assert_eq!(written.goal, notes.goal);
        assert_eq!(written.next_action, notes.next_action);
        assert_eq!(written.stage, notes.stage);
        assert_eq!(written.open_questions, notes.open_questions);
        assert!(corrections.is_empty());
    }

    /// A proposal that has forgotten the goal does not get to blank one Rust
    /// holds: a resumption with no goal reads as a run that never had one.
    #[test]
    fn a_blank_proposal_cannot_erase_the_goal_or_rewind_the_stage() {
        let previous = RunMemory {
            goal: "Produce the inspection note".into(),
            stage: RunStage {
                ordinal: 4,
                intent: "Reviewing".into(),
            },
            ..RunMemory::default()
        };
        let (written, _) = validate(&proposal(RunMemory::default()), &facts(), &previous);
        assert_eq!(written.goal, "Produce the inspection note");
        assert_eq!(written.stage.ordinal, 4);
    }

    /// The whole point, as one assertion: what comes out is never
    /// `RunMemory::default()` when anything was known.
    #[test]
    fn a_validated_commit_is_never_empty_when_something_was_known() {
        let previous = RunMemory {
            goal: "Produce the inspection note".into(),
            completed: vec![effect("artifact.create_approval_note", "note.docx")],
            ..RunMemory::default()
        };
        let (written, _) = validate(&proposal(RunMemory::default()), &facts(), &previous);
        assert!(
            !written.is_empty(),
            "a checkpoint would have been written empty"
        );
    }

    #[test]
    fn markers_are_read_strictly() {
        assert_eq!(numbered("E12", 'E'), Some(12));
        assert_eq!(numbered("C3", 'C'), Some(3));
        assert_eq!(numbered("E", 'E'), None);
        assert_eq!(numbered("E1a", 'E'), None);
        assert_eq!(numbered("e1", 'E'), None);
        assert_eq!(numbered("X1", 'E'), None);
        assert_eq!(numbered("", 'E'), None);
    }
}

//! Interrupting a run where it hurts, and continuing it.
//!
//! ## What these test, and what they deliberately do not
//!
//! They drive the production recovery machinery: the real `TaskEventLog` on a
//! real SQLite file, the real `RuntimeDeps::commit_state` with its validation,
//! the real `save_checkpoint` with its guards, and the real
//! `RunCheckpoint::resumable_against`. The fault is injected the way a crash
//! injects one — by stopping at a particular line and reading back only what
//! reached disk.
//!
//! They do **not** drive `drive_run`, which needs a Tauri `AppHandle`, a model
//! server and a child process. That is the native gate, and calling these an
//! end-to-end proof would be exactly the claim this repository has a standing
//! rule against. What is proved here is that the durable state a recovery reads
//! is correct and complete; what is not proved here is that the application
//! around it behaves, which is Phase 11's job.
//!
//! ## Why the assertions are about *absence* as often as presence
//!
//! The failures being fixed were all silent. A checkpoint that said the run had
//! done nothing looked exactly like a run that had done nothing. So several of
//! these assert that a thing is *not* empty, *not* forgotten and *not*
//! overwritten — because every one of those was true in production and nothing
//! failed.

use std::sync::Arc;

use serde_json::json;

use super::events::{self, EventDraft, TaskEventType};
use super::memory::RunMemory;
use super::state_commit::{Correction, StateProposal, STATE_COMMIT_VERSION};
use super::tests::deps_with;
use super::RuntimeDeps;

const RUN: &str = "r";
const ATTEMPT: &str = "attempt-1";

/// The signed-in operator these tests run as.
fn operator() -> Arc<std::sync::RwLock<Option<crate::identity::Session>>> {
    Arc::new(std::sync::RwLock::new(Some(crate::identity::Session::open(
        crate::identity::User::new(
            "priya",
            "Priya Sharma",
            vec![crate::identity::Role::Employee],
        ),
    ))))
}

fn manifest() -> crate::agent_runtime::context_manifest::ContextManifest {
    use crate::agent_runtime::context_manifest::{
        ContextManifest, DocumentBinding, HistoryBinding, ResearchBinding,
    };
    ContextManifest::new(
        RUN,
        ATTEMPT,
        "conv-1",
        "a-r",
        "model-a",
        32_768,
        vec![DocumentBinding {
            sha256: "ab12".into(),
            name: "vessel.pdf".into(),
            pages: 12,
        }],
        Some(ResearchBinding {
            notebook_id: "nb-1".into(),
            manifest_run_id: RUN.into(),
            selection: crate::knowledge::graph::SourceSelection::subset(vec!["ab12".into()]),
            node_ids: Vec::new(),
            assertion_ids: Vec::new(),
            graph_revision: Some("2026-09-16T11:02:03Z".into()),
        }),
        HistoryBinding {
            carried: 4,
            dropped: 1,
            tokens: 900,
            pinned: vec!["msg:u1".into()],
            omitted_pins: Vec::new(),
        },
    )
}

fn seed(attempt: &str, notes: RunMemory) -> crate::agent_runtime::resume::CheckpointSeed {
    crate::agent_runtime::resume::CheckpointSeed {
        attempt_id: attempt.to_string(),
        committed_notes: notes,
        manifest: Some(manifest()),
        plan_hash: "plan-hash".to_string(),
        policy_hash: "policy-hash".to_string(),
        workspace_hash: "workspace-hash".to_string(),
        model_id: "model-a".to_string(),
    }
}

/// A run that has started: a seed, so checkpoints have a world to be taken
/// against, and an opening event so the log is not empty.
fn started_run() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let (deps, dir) = deps_with(operator());
    deps.checkpoints
        .lock()
        .expect("fresh lock")
        .insert(RUN.to_string(), seed(ATTEMPT, RunMemory::default()));
    deps.events
        .record(EventDraft::new(RUN, TaskEventType::RunCreated, "priya"))
        .expect("the run is recorded");
    (deps, dir)
}

/// Records the receipt a real tool call leaves behind.
fn tool_succeeded(deps: &Arc<RuntimeDeps>, tool: &str, artifact: Option<&str>) {
    deps.events
        .record(
            EventDraft::new(RUN, TaskEventType::ToolSucceeded, "priya")
                .with(json!({ "tool": tool, "toolCallId": "tc-1", "detail": "done" })),
        )
        .expect("the receipt is recorded");
    if let Some(name) = artifact {
        deps.events
            .record(
                EventDraft::new(RUN, TaskEventType::ArtifactProduced, "priya")
                    .with(json!({ "name": name, "tool": tool })),
            )
            .expect("the artifact is recorded");
    }
}

fn proposal(state: events::RunState, notes: RunMemory) -> StateProposal {
    StateProposal {
        commit_version: STATE_COMMIT_VERSION,
        run_id: RUN.to_string(),
        attempt_id: ATTEMPT.to_string(),
        state,
        notes,
    }
}

fn notes_after_the_note_was_written() -> RunMemory {
    RunMemory {
        goal: "Produce the inspection note".into(),
        next_action: "Send it for approval".into(),
        stage: crate::agent_runtime::memory::RunStage {
            ordinal: 2,
            intent: "Drafting".into(),
        },
        artifact_ids: vec!["note.docx".into()],
        completed: vec![crate::agent_runtime::memory::CompletedEffect {
            tool: "artifact.create_approval_note".into(),
            target: "note.docx".into(),
            at: "2026-01-01T00:00:00Z".into(),
        }],
        ..RunMemory::default()
    }
}

fn world_now() -> crate::agent_runtime::events::WorldNow {
    crate::agent_runtime::events::WorldNow {
        policy_hash: "policy-hash".to_string(),
        plan_hash: "plan-hash".to_string(),
        workspace_hash: Some("workspace-hash".to_string()),
        model_available: true,
        same_operator: true,
        ended: false,
        state: events::RunState::Running,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fault 1: after a completed tool, before the final response.
// ─────────────────────────────────────────────────────────────────────────────

/// The headline case, and the one production got wrong.
///
/// The document was written, the receipt is in the log, and the process dies
/// before the model says anything. What a recovery reads has to know the
/// document exists — otherwise the resumed run writes it again.
#[test]
fn a_crash_after_a_tool_leaves_a_checkpoint_that_knows_about_it() {
    let (deps, _dir) = started_run();

    tool_succeeded(&deps, "artifact.create_approval_note", Some("note.docx"));
    let outcome = deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));
    assert!(outcome.accepted, "{:?}", outcome.refused_because);
    assert!(outcome.corrections.is_empty());

    // — the process dies here —

    let recovered = deps
        .events
        .checkpoint(RUN)
        .expect("the checkpoint is readable")
        .expect("there is one");

    assert!(
        !recovered.notes.is_empty(),
        "the checkpoint was written empty, which is the production bug this replaces"
    );
    assert!(
        recovered
            .notes
            .has_done("artifact.create_approval_note", "note.docx"),
        "a resumption would write the approval note a second time"
    );
    assert_eq!(recovered.notes.goal, "Produce the inspection note");
    assert_eq!(recovered.notes.stage.ordinal, 2);
    assert_eq!(recovered.notes.artifact_ids, vec!["note.docx"]);
    assert_eq!(recovered.state, events::RunState::ToolResultRecorded);
    assert!(recovered.is_intact());
}

/// The same crash, and the record of what the turn was working from survives
/// it: which conversation, which cell, which documents, which sources.
#[test]
fn the_context_the_turn_was_built_from_survives_the_crash() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "knowledge.search_authorized", None);
    deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        RunMemory {
            goal: "Answer from the notebook".into(),
            ..RunMemory::default()
        },
    ));

    let recovered = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    let manifest = recovered
        .manifest
        .expect("a resumption cannot rebuild the turn without this");

    assert!(manifest.is_intact());
    assert!(manifest.is_known_version());
    // The original transcript identity — not a new conversation.
    assert_eq!(manifest.conversation_id, "conv-1");
    assert_eq!(manifest.message_id, "a-r");
    // The documents at the versions the turn read.
    assert_eq!(manifest.document_hashes(), vec!["ab12"]);
    // The frozen selection, still a subset and not widened to everything.
    let research = manifest.research.expect("the scope was frozen");
    assert_eq!(
        research.selection,
        crate::knowledge::graph::SourceSelection::subset(vec!["ab12".into()])
    );
    assert_eq!(
        research.graph_revision.as_deref(),
        Some("2026-09-16T11:02:03Z")
    );
    // The model it ran under, so a resumption does not silently re-route.
    assert_eq!(manifest.model_id, "model-a");
}

// ─────────────────────────────────────────────────────────────────────────────
// Fault 2: after the checkpoint commits, before the loop is told.
// ─────────────────────────────────────────────────────────────────────────────

/// The loop never hears the answer, retries the commit, and exactly one effect
/// survives. A second entry would be a resumed run believing the document was
/// written twice — and a `completed` list that no longer matches the receipts it
/// was checked against.
#[test]
fn a_commit_retried_after_a_lost_acknowledgement_does_not_double_the_effect() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "artifact.create_approval_note", Some("note.docx"));

    let first = deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));
    assert!(first.accepted);

    // — the acknowledgement is lost; the loop sends the same commit again —
    let second = deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));
    assert!(second.accepted);
    assert!(
        second.corrections.is_empty(),
        "a replay of the same commit is not a disagreement: {:?}",
        second.corrections
    );

    let recovered = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert_eq!(
        recovered.notes.completed.len(),
        1,
        "the effect was recorded twice"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fault 3: while an approval is pending.
// ─────────────────────────────────────────────────────────────────────────────

/// A run stopped waiting for a person resumes still waiting for them, and the
/// state it was in is on the record rather than inferred.
#[test]
fn a_crash_while_an_approval_is_pending_records_that_it_was_pending() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "knowledge.search_authorized", None);
    deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        RunMemory {
            goal: "Produce the inspection note".into(),
            open_questions: vec!["Waiting for sign-off on the scope".into()],
            ..RunMemory::default()
        },
    ));

    // The approval is raised and the process dies before it is answered.
    deps.commit_state(&proposal(
        events::RunState::AwaitingApproval,
        RunMemory {
            goal: "Produce the inspection note".into(),
            next_action: "Wait for the approval".into(),
            open_questions: vec!["Waiting for sign-off on the scope".into()],
            ..RunMemory::default()
        },
    ));

    let recovered = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert_eq!(recovered.state, events::RunState::AwaitingApproval);
    assert_eq!(
        recovered.notes.open_questions,
        vec!["Waiting for sign-off on the scope"]
    );
    assert_eq!(recovered.notes.next_action, "Wait for the approval");
}

// ─────────────────────────────────────────────────────────────────────────────
// The guards.
// ─────────────────────────────────────────────────────────────────────────────

/// The exact shape of the production bug, asserted as a refusal.
///
/// An empty checkpoint taken *after* a complete one has a higher sequence, so
/// the sequence guard let it through and the resume point was destroyed. This
/// now cannot happen even if a caller tries.
#[test]
fn an_empty_checkpoint_cannot_replace_a_complete_one() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "artifact.create_approval_note", Some("note.docx"));
    deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));

    let complete = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert!(!complete.notes.is_empty());

    // A later checkpoint, at a higher sequence, carrying nothing.
    let held = deps
        .checkpoints
        .lock()
        .expect("lock")
        .get(RUN)
        .cloned()
        .expect("seeded");
    let emptied = held.checkpoint(
        RUN,
        events::RunState::Running,
        complete.last_event_seq + 10,
        RunMemory::default(),
        None,
        Some(manifest()),
        Vec::new(),
    );
    let written = deps
        .events
        .save_checkpoint(&emptied)
        .expect("the write is attempted");
    assert!(!written, "an empty checkpoint overwrote a complete one");

    let after = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert!(
        after
            .notes
            .has_done("artifact.create_approval_note", "note.docx"),
        "the resume point was emptied"
    );
}

/// And the same for the manifest: losing it leaves a resumption unable to
/// rebuild the turn, which is a refusal rather than a smaller answer.
#[test]
fn a_checkpoint_without_a_manifest_cannot_replace_one_that_has_it() {
    let (deps, _dir) = started_run();
    deps.commit_state(&proposal(
        events::RunState::Running,
        RunMemory {
            goal: "Produce the inspection note".into(),
            ..RunMemory::default()
        },
    ));
    let stored = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");

    let held = deps
        .checkpoints
        .lock()
        .expect("lock")
        .get(RUN)
        .cloned()
        .expect("seeded");
    let blind = held.checkpoint(
        RUN,
        events::RunState::Running,
        stored.last_event_seq + 5,
        stored.notes.clone(),
        None,
        None,
        Vec::new(),
    );
    assert!(
        !deps.events.save_checkpoint(&blind).expect("attempted"),
        "the record of what the turn was built from was overwritten with nothing"
    );
}

/// A worker that outlived a restart cannot move the live attempt's resume point
/// back to a state a dead process believed.
#[test]
fn a_commit_from_a_previous_attempt_is_refused() {
    let (deps, _dir) = started_run();
    let mut stale = proposal(
        events::RunState::ToolResultRecorded,
        RunMemory {
            goal: "a state the dead worker believed".into(),
            ..RunMemory::default()
        },
    );
    stale.attempt_id = "attempt-0".to_string();

    let outcome = deps.commit_state(&stale);
    assert!(!outcome.accepted);
    let because = outcome.refused_because.expect("a reason");
    assert!(because.contains("attempt-0"), "{because}");
    assert!(because.contains(ATTEMPT), "{because}");
}

/// A build that does not understand the proposal refuses it rather than
/// applying the half it recognises.
#[test]
fn a_commit_at_an_unknown_version_is_refused() {
    let (deps, _dir) = started_run();
    let mut future = proposal(events::RunState::Running, RunMemory::default());
    future.commit_version = STATE_COMMIT_VERSION + 7;

    let outcome = deps.commit_state(&future);
    assert!(!outcome.accepted);
    assert!(outcome
        .refused_because
        .expect("a reason")
        .contains("version"));
}

/// The rule the whole module exists for, exercised against a real event log
/// rather than a hand-built fact table.
#[test]
fn a_claimed_effect_with_no_receipt_in_the_log_is_dropped() {
    let (deps, _dir) = started_run();
    // No `tool_succeeded` call: nothing in the log corroborates the claim.
    let outcome = deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));

    assert!(outcome.accepted, "the commit applies, with the claim removed");
    assert!(outcome.corrections.iter().any(|correction| matches!(
        correction,
        Correction::UnbackedEffect { tool, target }
            if tool == "artifact.create_approval_note" && target == "note.docx"
    )));
    assert!(
        !outcome
            .notes
            .has_done("artifact.create_approval_note", "note.docx"),
        "a resumption would have skipped writing a document that was never written"
    );

    let recovered = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert!(recovered.notes.completed.is_empty());
    // The narration is still kept: it cannot excuse work, and it is what makes
    // the resumption legible.
    assert_eq!(recovered.notes.goal, "Produce the inspection note");
}

/// A checkpoint somebody edited is refused, not read.
#[test]
fn an_unreadable_checkpoint_is_a_refusal_rather_than_an_absence() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "artifact.create_approval_note", Some("note.docx"));
    deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));

    let mut tampered = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    // Someone edits the record so the run looks as though it never wrote the
    // document — the edit a resumption must not act on.
    tampered.notes.completed.clear();
    assert!(
        !tampered.is_intact(),
        "the seal did not cover the completed effects"
    );

    // And it is refused rather than reported as no checkpoint at all: the two
    // are different, and only one of them means somebody should look.
    let refusal = tampered
        .resumable_against(&world_now())
        .expect_err("a tampered checkpoint is not resumable");
    assert!(matches!(
        refusal,
        crate::agent_runtime::events::NotResumable::CorruptCheckpoint
    ));
}

/// An effect nobody settled stops a resumption, whatever else is true.
#[test]
fn an_unsettled_effect_stops_the_resumption() {
    let (deps, _dir) = started_run();
    let held = deps
        .checkpoints
        .lock()
        .expect("lock")
        .get(RUN)
        .cloned()
        .expect("seeded");
    let point = held.checkpoint(
        RUN,
        events::RunState::ExecutingTool,
        1,
        notes_after_the_note_was_written(),
        None,
        Some(manifest()),
        vec!["create_docx:abc123".to_string()],
    );
    let refusal = point
        .resumable_against(&world_now())
        .expect_err("an unsettled effect is not resumable");
    assert!(matches!(
        refusal,
        crate::agent_runtime::events::NotResumable::UnknownEffects { .. }
    ));
    assert!(refusal.needs_human_reconciliation());
}

/// Across a restart the run keeps its identity and gains an attempt, rather
/// than the two being separately minted and never agreeing.
#[test]
fn a_resumed_attempt_starts_from_what_the_previous_one_established() {
    let (deps, _dir) = started_run();
    tool_succeeded(&deps, "artifact.create_approval_note", Some("note.docx"));
    deps.commit_state(&proposal(
        events::RunState::ToolResultRecorded,
        notes_after_the_note_was_written(),
    ));

    let carried = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one")
        .notes;

    // The restart: a new attempt over the same run, seeded from the record —
    // which is what `drive_run` now does with `resumed_notes`.
    const NEXT: &str = "attempt-2";
    deps.checkpoints
        .lock()
        .expect("lock")
        .insert(RUN.to_string(), seed(NEXT, carried.clone()));

    // The resumed loop starts with empty notes and commits before it has
    // rebuilt them — the moment at which the old code lost everything.
    let mut fresh = proposal(events::RunState::Running, RunMemory::default());
    fresh.attempt_id = NEXT.to_string();
    let outcome = deps.commit_state(&fresh);

    assert!(outcome.accepted);
    assert!(
        outcome
            .notes
            .has_done("artifact.create_approval_note", "note.docx"),
        "the resumed attempt forgot the document the previous one wrote"
    );
    assert!(outcome
        .corrections
        .iter()
        .any(|correction| matches!(correction, Correction::EffectWouldHaveBeenLost { .. })));
    assert_eq!(outcome.notes.goal, "Produce the inspection note");

    // And the run id is the same run id throughout: one task, two attempts.
    let after = deps
        .events
        .checkpoint(RUN)
        .expect("readable")
        .expect("there is one");
    assert_eq!(after.run_id, RUN);
    assert_eq!(after.attempt_id, NEXT);
}

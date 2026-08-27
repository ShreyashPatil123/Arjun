//! Storing and reloading the point a run can be continued from.
//!
//! [`super::checkpoint`]'s own tests check the decision — given a checkpoint and
//! a world, may this run continue. These check the storage underneath it: that
//! a checkpoint survives a restart, that a late writer cannot move the resume
//! point backwards, that one run's checkpoint is never another's, and that a
//! damaged row is refused rather than acted on.

use super::checkpoint::{NotResumable, RunCheckpoint, CHECKPOINT_SCHEMA_VERSION};
use super::machine::RunState;
use super::store::TaskEventLog;
use crate::agent_runtime::memory::{CompletedEffect, RunMemory};

fn on_disk(dir: &std::path::Path) -> TaskEventLog {
    TaskEventLog::open(dir).expect("a log on disk")
}

fn at_seq(run_id: &str, attempt: &str, seq: i64) -> RunCheckpoint {
    RunCheckpoint::new(
        run_id,
        attempt,
        RunState::ToolResultRecorded,
        seq,
        RunMemory::default(),
        None,
        "plan-hash",
        "policy-hash",
        "workspace-hash",
        "qwen2.5-7b",
        Vec::new(),
    )
}

#[test]
fn a_checkpoint_written_during_a_run_is_readable_after_a_restart() {
    // The whole point. A resume path that only works while the process that
    // wrote the checkpoint is alive is a resume path for the case that does not
    // need one.
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let log = on_disk(dir.path());
        assert!(log
            .save_checkpoint(&at_seq("run-1", "attempt-1", 12))
            .expect("saved"));
    }

    let reopened = on_disk(dir.path());
    let held = reopened
        .checkpoint("run-1")
        .expect("readable")
        .expect("present");

    assert_eq!(held.run_id, "run-1");
    assert_eq!(held.attempt_id, "attempt-1");
    assert_eq!(held.last_event_seq, 12);
    assert!(held.is_intact());
}

#[test]
fn a_later_checkpoint_replaces_an_earlier_one() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());

    log.save_checkpoint(&at_seq("run-1", "attempt-1", 4))
        .expect("saved");
    log.save_checkpoint(&at_seq("run-1", "attempt-1", 9))
        .expect("saved");

    let held = log.checkpoint("run-1").expect("readable").expect("present");
    assert_eq!(held.last_event_seq, 9);
}

#[test]
fn a_late_write_cannot_move_the_resume_point_backwards() {
    // Two writers race at the end of a run: the loop settling a tool call, and
    // the shutdown path recording the ending. Whichever finishes last is not
    // always the further-along one, and letting it win would offer a resume
    // point behind work that had already been done.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());

    log.save_checkpoint(&at_seq("run-1", "attempt-1", 20))
        .expect("saved");
    let moved = log
        .save_checkpoint(&at_seq("run-1", "attempt-1", 5))
        .expect("no error");

    assert!(!moved, "a stale checkpoint was written");
    let held = log.checkpoint("run-1").expect("readable").expect("present");
    assert_eq!(held.last_event_seq, 20);
}

#[test]
fn a_checkpoint_at_the_same_sequence_is_allowed_to_refresh_the_row() {
    // Re-checkpointing at the same event is ordinary: a side-effecting tool
    // writes one before and one after, and the notes differ even when nothing
    // new has been recorded to the history.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.save_checkpoint(&at_seq("run-1", "attempt-1", 7))
        .expect("saved");

    let mut refreshed = at_seq("run-1", "attempt-1", 7);
    refreshed.notes.goal = "Draft the approval note.".to_string();
    refreshed.checkpoint_hash = refreshed.compute_hash();

    assert!(log.save_checkpoint(&refreshed).expect("no error"));
    let held = log.checkpoint("run-1").expect("readable").expect("present");
    assert_eq!(held.notes.goal, "Draft the approval note.");
}

#[test]
fn one_runs_checkpoint_is_never_anothers() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.save_checkpoint(&at_seq("run-1", "attempt-1", 12))
        .expect("saved");

    assert!(log.checkpoint("run-2").expect("readable").is_none());

    // And a second run keeps its own point, rather than overwriting the first.
    log.save_checkpoint(&at_seq("run-2", "attempt-1", 3))
        .expect("saved");
    assert_eq!(
        log.checkpoint("run-1")
            .expect("readable")
            .expect("present")
            .last_event_seq,
        12
    );
    assert_eq!(
        log.checkpoint("run-2")
            .expect("readable")
            .expect("present")
            .last_event_seq,
        3
    );
}

#[test]
fn a_run_that_was_never_checkpointed_reads_as_absent_rather_than_broken() {
    // Absence and damage are different answers with different remedies, and a
    // screen that shows one for the other sends somebody to the wrong place.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    assert!(log.checkpoint("never-ran").expect("readable").is_none());
}

#[test]
fn a_checkpoint_whose_hash_was_not_computed_here_is_refused_at_the_door() {
    // The store seals nothing it did not verify. A caller that hands over a
    // body with a hash computed elsewhere is handing over a hash that proves
    // nothing about the body.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());

    let mut forged = at_seq("run-1", "attempt-1", 12);
    forged.unknown_effects = vec!["create_docx:abc".to_string()];
    // Deliberately not re-sealed: the hash still describes the version without
    // the unsettled effect, which is exactly the edit that would make an unsafe
    // run look safe.

    assert!(log.save_checkpoint(&forged).is_err());
    assert!(log.checkpoint("run-1").expect("readable").is_none());
}

#[test]
fn a_damaged_row_is_reported_as_corrupt_rather_than_returned() {
    // A corrupted or hand-edited row must surface as damage. Returned as though
    // it were sound, it would offer a resume point derived from a record nobody
    // can vouch for; returned as absent, it would hide that the record was
    // damaged at all.
    let dir = tempfile::tempdir().expect("temp dir");
    {
        let log = on_disk(dir.path());
        log.save_checkpoint(&at_seq("run-1", "attempt-1", 12))
            .expect("saved");
    }

    // Edited behind the back of the store, the way a corrupted file or a
    // curious operator would.
    {
        let conn =
            rusqlite::Connection::open(dir.path().join("sarathi.db")).expect("the events database");
        let body: String = conn
            .query_row(
                "SELECT body FROM run_checkpoints WHERE run_id = 'run-1'",
                [],
                |row| row.get(0),
            )
            .expect("a row");
        let tampered = body.replace("\"lastEventSeq\":12", "\"lastEventSeq\":9999");
        assert_ne!(tampered, body, "the edit did not apply");
        conn.execute(
            "UPDATE run_checkpoints SET body = ?1 WHERE run_id = 'run-1'",
            [tampered],
        )
        .expect("updated");
    }

    let reopened = on_disk(dir.path());
    assert_eq!(
        reopened.checkpoint("run-1"),
        Err(NotResumable::CorruptCheckpoint)
    );
}

#[test]
fn clearing_a_checkpoint_leaves_the_run_view_only() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.save_checkpoint(&at_seq("run-1", "attempt-1", 12))
        .expect("saved");

    log.clear_checkpoint("run-1").expect("cleared");
    assert!(log.checkpoint("run-1").expect("readable").is_none());
}

#[test]
fn a_stored_checkpoint_carries_completed_effects_but_no_content() {
    // What makes a resumption safe is the list of effects already performed.
    // What makes the record shareable is that the list holds identities, not
    // the documents those effects produced.
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());

    let mut point = at_seq("run-1", "attempt-1", 12);
    point.notes.goal = "Draft the approval note.".to_string();
    point.notes.evidence_ids = vec!["E1".to_string(), "E2".to_string()];
    point.notes.completed = vec![CompletedEffect {
        tool: "create_docx".to_string(),
        target: "approval-note.docx".to_string(),
        at: "2026-08-28T09:15:00+00:00".to_string(),
    }];
    point.checkpoint_hash = point.compute_hash();
    log.save_checkpoint(&point).expect("saved");

    let held = log.checkpoint("run-1").expect("readable").expect("present");
    assert!(held.notes.has_done("create_docx", "approval-note.docx"));
    assert_eq!(held.notes.evidence_ids, vec!["E1", "E2"]);

    // Markers and filenames, not passages. A checkpoint that grew large enough
    // to hold a document would be one that had started carrying content.
    let serialised = serde_json::to_string(&held).expect("serialises");
    assert!(
        serialised.len() < 2_000,
        "the checkpoint grew big enough to hold content: {} bytes",
        serialised.len()
    );
}

#[test]
fn the_stored_schema_version_is_the_one_this_build_writes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = on_disk(dir.path());
    log.save_checkpoint(&at_seq("run-1", "attempt-1", 1))
        .expect("saved");

    let held = log.checkpoint("run-1").expect("readable").expect("present");
    assert_eq!(held.schema_version, CHECKPOINT_SCHEMA_VERSION);
}

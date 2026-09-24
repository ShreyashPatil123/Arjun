//! Plan P02, at the store: versions, staleness, revocation, the outbox and the
//! migration, each through the real [`MemoryGraph`] and, where a second store is
//! involved, the real [`TaskEventLog`].
//!
//! The worker-path tests -- a real observation published by a real worker on
//! its own receipt, and read by a sibling at its version -- are in
//! `subagents::worker_tests`, which drives the production delegation tool.

use std::sync::Arc;

use crate::agent_runtime::events::{TaskEventLog, TaskEventType};
use crate::agent_runtime::memory::Acl;
use crate::identity::Role;
use crate::knowledge::graph::receipts::{
    record_tool_receipt, OutboxConsumer, ReceiptLedger, EVENTS_TARGET,
};
use crate::knowledge::graph::runtime_feed::FeedChange;
use crate::knowledge::graph::runtime_memory::tests::{item, operator, session};
use crate::knowledge::graph::runtime_memory::{
    edge_id, item_id, Authority, Dependency, EdgeKind, ItemStatus, MemoryEdge, MemoryItem,
    MemoryKind, MemoryScope,
};
use crate::knowledge::graph::runtime_store::{MemoryError, MemoryGraph, OutboxRow};
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;

const AT: &str = "2026-02-01T00:00:00Z";

fn task() -> MemoryScope {
    MemoryScope::Task {
        task_id: "task-1".into(),
    }
}

fn reader() -> crate::identity::Session {
    session("priya", vec![Role::Employee])
}

fn colleague() -> crate::identity::Session {
    session("ravi", vec![Role::Employee])
}

fn committed(graph: &MemoryGraph, item: MemoryItem) -> MemoryItem {
    let done = graph.commit(item.clone(), None, &[]).expect("commits");
    let mut item = item;
    item.revision = done.revision;
    item.status = done.status;
    item
}

fn derived_from(source: &MemoryItem, kind: MemoryKind, content: &str) -> MemoryItem {
    let mut derived = item(kind, operator());
    derived.content = content.into();
    derived.depends_on = vec![Dependency {
        item_id: source.item_id.clone(),
        revision: source.revision,
    }];
    derived
}

// ── Correction invalidates an artifact ─────────────────────────────────────

/// **Correction invalidates an artifact.** A fact is corrected; the artifact
/// derived from it goes stale, and so does the review derived from the
/// artifact. Nothing is deleted: every earlier version is still in the history,
/// and a subscriber is told each of the three changed.
#[test]
fn a_correction_makes_everything_derived_from_the_corrected_fact_stale() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let fact = committed(&graph, {
        let mut fact = item(MemoryKind::Fact, operator());
        fact.content = "design pressure: 150 psi".into();
        fact
    });
    let note = committed(&graph, {
        let mut note = derived_from(&fact, MemoryKind::ArtifactRef, "approval note: rev 2");
        note.artifacts = vec![crate::knowledge::graph::runtime_memory::ArtifactRef {
            artifact_id: "art-note".into(),
            revision: 2,
            sha256: "e".repeat(64),
        }];
        note
    });
    let review = committed(&graph, derived_from(&note, MemoryKind::Fact, "approval note: reviewed"));
    // And one that rests on nothing, as a control.
    let control = committed(&graph, item(MemoryKind::Constraint, operator()));
    let before = graph.graph_revision().expect("a head");

    let mut correction = item(MemoryKind::Correction, operator());
    correction.content = "design pressure: 10 bar".into();
    graph.correct(correction, &fact.item_id).expect("corrected");

    let now = graph.snapshot(&reader(), &task(), None).expect("reads");
    let status = |id: &str| now.iter().find(|i| i.item_id == id).map(|i| i.status);
    assert_eq!(status(&fact.item_id), Some(ItemStatus::Superseded));
    assert_eq!(status(&note.item_id), Some(ItemStatus::Stale), "the artifact was not invalidated");
    assert_eq!(status(&review.item_id), Some(ItemStatus::Stale), "staleness stopped one step short");
    assert_eq!(status(&control.item_id), Some(ItemStatus::Admitted));

    // A new revision each, and the old ones kept.
    let history = graph.versions_of(&reader(), &note.item_id, None).expect("history");
    assert_eq!(
        history.iter().map(|v| (v.revision, v.item.status)).collect::<Vec<_>>(),
        vec![(1, ItemStatus::Admitted), (2, ItemStatus::Stale)]
    );

    // A subscriber at the pre-correction cursor is told about all three.
    let batch = graph
        .changes_since(&reader(), &task(), None, before, 100)
        .expect("the feed reads");
    let changed: Vec<String> = batch
        .entries
        .iter()
        .filter_map(|entry| match &entry.change {
            FeedChange::ItemChanged { item } => Some(item.item_id.clone()),
            _ => None,
        })
        .collect();
    for id in [&fact.item_id, &note.item_id, &review.item_id] {
        assert!(changed.contains(id), "{id} did not reach the feed");
    }
    // A stale item is readable, and not evidence.
    assert!(!ItemStatus::Stale.usable_as_evidence());
}

/// A result computed from an input that moved on while it was being worked out
/// is published as stale, not as current. Plan §6: revalidate dependencies
/// immediately before publication.
#[test]
fn a_result_whose_input_moved_during_the_work_is_published_stale() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let input = committed(&graph, item(MemoryKind::Fact, operator()));
    let result = derived_from(&input, MemoryKind::Fact, "margin: 0.8 mm");

    // The input is corrected while the result is in flight.
    let mut correction = item(MemoryKind::Correction, operator());
    correction.content = "a later reading".into();
    graph.correct(correction, &input.item_id).expect("corrected");

    let published = graph.commit(result, None, &[]).expect("commits");
    assert_eq!(published.status, ItemStatus::Stale, "{}", published.because);
    assert!(published.because.contains(&input.item_id), "{}", published.because);
}

// ── Conflicting updates ────────────────────────────────────────────────────

/// **Conflicting updates do not overwrite silently.** A writer holding the
/// revision it read is refused once anything -- including a supersede, which
/// used to change the row without moving its revision -- has written after it.
#[test]
fn a_write_against_a_revision_that_moved_is_a_named_conflict() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let fact = committed(&graph, item(MemoryKind::Fact, operator()));
    let held_revision = fact.revision;

    let mut correction = item(MemoryKind::Correction, operator());
    correction.content = "corrected".into();
    graph.correct(correction, &fact.item_id).expect("corrected");

    // A second writer, still holding revision 1, edits the fact.
    let mut edit = fact.clone();
    edit.content = "an edit made against the old revision".into();
    match graph.commit(edit, Some(held_revision), &[]) {
        Err(MemoryError::RevisionConflict { expected, actual, .. }) => {
            assert_eq!(expected, held_revision);
            assert!(actual > held_revision, "a supersede did not move the revision");
        }
        other => panic!("a stale write was applied: {other:?}"),
    }

    // A correction against a stale revision is refused the same way.
    let mut late = item(MemoryKind::Correction, operator());
    late.content = "another correction".into();
    assert!(matches!(
        graph.correct_at(late, &fact.item_id, Some(held_revision)),
        Err(MemoryError::RevisionConflict { .. })
    ));
}

// ── Denied users see nothing ───────────────────────────────────────────────

/// **Denied users see no hidden data or metadata.** One reader's access to one
/// item is withdrawn. Every read path agrees: the snapshot, the historical
/// read, the history, the neighbour list, the count and the changefeed -- whose
/// drop carries the id and nothing else. What was derived from the item is
/// withdrawn from them too. Another reader is unaffected.
#[test]
fn a_revoked_reader_sees_nothing_of_the_item_on_any_path() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let secret = committed(&graph, {
        let mut secret = item(MemoryKind::Fact, operator());
        secret.content = "unit price: 4.2 crore".into();
        secret
    });
    let derived = committed(&graph, derived_from(&secret, MemoryKind::Fact, "budget headroom: low"));
    let other = committed(&graph, item(MemoryKind::Constraint, operator()));
    graph
        .link(MemoryEdge {
            edge_id: edge_id(&other.item_id, EdgeKind::Supports, &secret.item_id),
            from_item: other.item_id.clone(),
            to_item: secret.item_id.clone(),
            kind: EdgeKind::Supports,
            agent_id: "ag-1".into(),
            scope: task(),
            created_at: AT.into(),
        })
        .expect("linked");
    let before = graph.graph_revision().expect("a head");

    let withdrawn = graph
        .revoke_reader(&secret.item_id, "priya", AT)
        .expect("revoked");
    assert!(withdrawn.contains(&derived.item_id), "the derived item kept its reader");

    let visible = graph.snapshot(&reader(), &task(), None).expect("reads");
    let ids: Vec<&str> = visible.iter().map(|i| i.item_id.as_str()).collect();
    assert_eq!(ids, vec![other.item_id.as_str()], "{ids:?}");
    // No neighbour, and no count, reveals what is behind the edge.
    assert!(graph.neighbours(&other.item_id, &visible).expect("reads").is_empty());
    assert_eq!(MemoryGraph::count(&visible).get("fact"), None);
    // No history, and no historical read, either.
    assert!(graph.versions_of(&reader(), &secret.item_id, None).expect("reads").is_empty());
    let historical = graph
        .snapshot_as_of(&reader(), &task(), None, before)
        .expect("reads");
    assert!(historical.items.iter().all(|i| i.item_id == other.item_id));
    assert!(historical.edges.is_empty(), "an edge into a revoked item was returned");
    // The feed drops it, carrying nothing but the id.
    let batch = graph
        .changes_since(&reader(), &task(), None, before, 100)
        .expect("reads");
    let mut dropped = 0;
    for entry in &batch.entries {
        match &entry.change {
            FeedChange::ItemDropped { item_id } => {
                assert!(item_id == &secret.item_id || item_id == &derived.item_id);
                dropped += 1;
            }
            FeedChange::ItemChanged { item } => {
                assert!(!item.content.contains("4.2"), "the revoked content was delivered")
            }
            _ => {}
        }
    }
    assert_eq!(dropped, 2, "the reader was not told to drop both");

    // Somebody else still reads all three.
    assert_eq!(graph.snapshot(&colleague(), &task(), None).expect("reads").len(), 3);
}

/// A derived record is never less restricted than what it came from.
#[test]
fn a_derived_record_inherits_the_restrictions_of_its_inputs() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let tender = committed(&graph, {
        let mut tender = item(MemoryKind::Fact, operator());
        tender.classification = Classification::VendorNegotiation;
        tender.acl = Acl::for_classification(Classification::VendorNegotiation, Some("unit-four"));
        tender
    });
    let summary = committed(&graph, derived_from(&tender, MemoryKind::Fact, "summary: cheaper"));
    let held = graph
        .versions_of(
            &session("ada", vec![Role::Administrator]),
            &summary.item_id,
            Some("unit-four"),
        )
        .expect("reads");
    let stored = &held.last().expect("a version").item;
    assert_eq!(stored.classification, Classification::VendorNegotiation);
    assert_eq!(stored.acl.project_id.as_deref(), Some("unit-four"));
    // A reader on no project reads neither the tender nor what was derived.
    assert!(graph.snapshot(&reader(), &task(), None).expect("reads").is_empty());
}

// ── Historical reads ───────────────────────────────────────────────────────

/// A read as of an earlier cursor returns what that cursor held -- the earlier
/// version, and not the items written after it.
#[test]
fn a_read_as_of_a_cursor_is_that_cursor_and_not_the_latest_rows() {
    let graph = MemoryGraph::in_memory().expect("a graph");
    let first = committed(&graph, item(MemoryKind::Fact, operator()));
    let at_first = graph.graph_revision().expect("a head");
    let mut correction = item(MemoryKind::Correction, operator());
    correction.content = "later".into();
    graph.correct(correction, &first.item_id).expect("corrected");

    let then = graph.snapshot_as_of(&reader(), &task(), None, at_first).expect("reads");
    assert_eq!(then.cursor, at_first);
    assert_eq!(then.items.len(), 1, "an item written after the cursor was returned");
    assert_eq!(then.items[0].status, ItemStatus::Admitted, "the latest version was returned");

    let head = graph.graph_revision().expect("a head");
    assert!(graph.snapshot_as_of(&reader(), &task(), None, head + 1).is_err());
}

// ── The outbox ─────────────────────────────────────────────────────────────

struct Failing;
impl OutboxConsumer for Failing {
    fn target(&self) -> &'static str {
        EVENTS_TARGET
    }
    fn deliver(&self, _row: &OutboxRow) -> Result<(), String> {
        Err("the event store was not reachable".into())
    }
}

/// **Failed outbox delivery recovers once after restart.** A commit writes its
/// outbox row; delivery fails; the process goes away. On the next start the row
/// is delivered -- once. A further pass does nothing, and a redelivery of the
/// same row (a crash after delivering and before marking it) does not write a
/// second event.
#[test]
fn a_failed_delivery_is_recovered_exactly_once_after_a_restart() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let key = "memory-published:k-1".to_string();
    let payload =
        serde_json::json!({ "runId": "run-1", "event": { "itemId": "mi-1" } }).to_string();
    {
        let graph = MemoryGraph::open(dir.path()).expect("a graph");
        let mut published = item(MemoryKind::Fact, operator());
        published.idempotency_key = Some("k-1".into());
        graph
            .commit(published, None, &[(EVENTS_TARGET.into(), key.clone(), payload.clone())])
            .expect("commits");
        let report = graph.deliver_pending(&[&Failing], AT).expect("tries");
        assert_eq!(report.failed, 1);
        let pending = graph.pending_effects().expect("reads");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempts, 1);
        assert!(pending[0].last_error.as_deref().is_some_and(|e| e.contains("not reachable")));
    } // The process goes away.

    let graph = MemoryGraph::open(dir.path()).expect("the graph reopens");
    let events = TaskEventLog::open(dir.path()).expect("the event log opens");
    let report = graph.deliver_pending(&[&events], AT).expect("delivers");
    assert_eq!(report.delivered, 1);
    let again = graph.deliver_pending(&[&events], AT).expect("delivers");
    assert_eq!(again.delivered, 0, "a delivered row was delivered again");

    // A crash between delivering and marking, replayed by hand.
    let row = OutboxRow {
        outbox_id: 1,
        target: EVENTS_TARGET.into(),
        idempotency_key: key,
        payload,
        created_at: AT.into(),
        attempts: 0,
        last_error: None,
    };
    events.deliver(&row).expect("a redelivery is accepted");

    let published = events
        .events_since("run-1", 0)
        .expect("reads")
        .events
        .into_iter()
        .filter(|event| event.event_type == TaskEventType::MemoryPublished)
        .count();
    assert_eq!(published, 1, "the publication was recorded {published} times");
}

// ── Receipts at the store ──────────────────────────────────────────────────

/// A receipt the log backs admits; the same shape the log does not back, and
/// the same receipt on a graph with no log in reach, stay proposals.
#[test]
fn only_a_receipt_resolved_in_the_event_log_admits() {
    let events = Arc::new(TaskEventLog::in_memory().expect("a log"));
    let backed = record_tool_receipt(
        &events,
        "child-1",
        ToolName::RunCalculation,
        "child-1:calc",
        "priya",
        "2 + 2 = 4",
    )
    .expect("recorded");

    let with_log = MemoryGraph::in_memory()
        .expect("a graph")
        .with_receipts(events.clone() as Arc<dyn ReceiptLedger>);
    let admitted = with_log
        .commit(item(MemoryKind::ToolObservation, backed.provenance()), None, &[])
        .expect("commits");
    assert_eq!(admitted.status, ItemStatus::Admitted, "{}", admitted.because);

    let mut tampered = backed.clone();
    tampered.output_sha256 = "0".repeat(64);
    let refused = with_log
        .commit(item(MemoryKind::ToolObservation, tampered.provenance()), None, &[])
        .expect("kept");
    assert_eq!(refused.status, ItemStatus::Proposed, "{}", refused.because);

    let without_log = MemoryGraph::in_memory().expect("a graph");
    let unchecked = without_log
        .commit(item(MemoryKind::ToolObservation, backed.provenance()), None, &[])
        .expect("kept");
    assert_eq!(unchecked.status, ItemStatus::Proposed, "{}", unchecked.because);
    assert!(unchecked.because.contains("no event log"), "{}", unchecked.because);
}

// ── The migration ──────────────────────────────────────────────────────────

fn remember(store: &crate::agent_runtime::memory::MemoryStore, key: &str, value: &str) {
    use crate::agent_runtime::memory::{MemoryKind as K, MemoryScope as S, MemorySource, Remember};
    store
        .remember(Remember {
            scope: S::Workspace {
                project_id: "unit-four".into(),
            },
            kind: K::ProjectFact,
            key: key.into(),
            value: value.into(),
            classification: Classification::Internal,
            source: MemorySource::Operator {
                user_id: "priya".into(),
            },
            approval: None,
            expires_at: None,
        })
        .expect("remembered");
}

fn legacy_store(dir: &std::path::Path) -> crate::agent_runtime::memory::MemoryStore {
    let store = crate::agent_runtime::memory::MemoryStore::open(dir);
    remember(&store, "units", "bar");
    remember(&store, "site", "Unit Four");
    store
}

/// Migrated records, and the versions of them a cleared reader can see.
fn row_counts(graph: &MemoryGraph) -> (usize, usize) {
    let items = graph
        .migrated_from(super::migration::LegacySource::RuntimeScopedMemory.key())
        .expect("reads");
    let versions: usize = items
        .values()
        .map(|item| {
            graph
                .versions_of(
                    &session("priya", vec![Role::Employee, Role::Administrator]),
                    &item.item_id,
                    Some("unit-four"),
                )
                .map(|v| v.len())
                .unwrap_or(0)
        })
        .sum();
    (items.len(), versions)
}

/// **Repeat migration does not duplicate rows**, and the migration is
/// restartable, verifiable and reversible without deleting anything.
#[test]
fn the_migration_is_repeatable_verifiable_and_reversible() {
    use super::migration::{
        migrate_runtime_scoped_memory, rollback_source, verify_runtime_scoped_memory, LegacySource,
    };

    let dir = tempfile::tempdir().expect("a temporary directory");
    let legacy = legacy_store(dir.path());
    let graph = MemoryGraph::in_memory().expect("a graph");

    let first = migrate_runtime_scoped_memory(&legacy, &graph, AT).expect("migrates");
    assert_eq!(first.migrated, 2, "{first:?}");
    let after_first = row_counts(&graph);
    assert_eq!(after_first, (2, 2));
    assert!(verify_runtime_scoped_memory(&legacy, &graph).expect("verifies").clean());

    // Again, and again: nothing new.
    for _ in 0..2 {
        let again = migrate_runtime_scoped_memory(&legacy, &graph, AT).expect("migrates");
        assert_eq!((again.migrated, again.already_present), (0, 2), "{again:?}");
        assert_eq!(row_counts(&graph), after_first, "a repeat pass duplicated rows");
    }

    // Records the legacy store still owns cannot be rewritten from the graph.
    let migrated = graph.migrated_from(LegacySource::RuntimeScopedMemory.key()).expect("reads");
    let one = migrated.values().next().expect("one").clone();
    assert!(matches!(one.authority, Authority::Legacy { .. }));
    let mut correction = item(MemoryKind::Correction, operator());
    correction.scope = one.scope.clone();
    assert!(matches!(
        graph.correct(correction, &one.item_id),
        Err(MemoryError::NotPermitted { .. })
    ));

    // Rolled back: tombstoned as new revisions, nothing deleted.
    let retired = rollback_source(LegacySource::RuntimeScopedMemory, &graph, AT).expect("rolls back");
    assert_eq!(retired.len(), 2);
    let (items, versions) = row_counts(&graph);
    assert_eq!(items, 2, "a rollback deleted rows");
    assert_eq!(versions, 0, "a tombstoned item's history is hidden from readers");
    assert!(!verify_runtime_scoped_memory(&legacy, &graph).expect("verifies").clean());

    // And restored by running the migration again.
    let restored = migrate_runtime_scoped_memory(&legacy, &graph, AT).expect("migrates");
    assert_eq!(restored.restored, 2, "{restored:?}");
    assert!(verify_runtime_scoped_memory(&legacy, &graph).expect("verifies").clean());
    assert!(graph.unfinished_migration_runs().expect("reads").is_empty());
}

/// A legacy record that changed is found by verification, and the next pass
/// follows it as a new revision of the same record; a legacy record that is
/// gone is retired -- tombstoned, never deleted.
#[test]
fn a_changed_or_removed_legacy_record_is_found_and_followed_by_the_next_pass() {
    use super::migration::{
        migrate_runtime_scoped_memory, verify_runtime_scoped_memory, LegacySource,
    };

    let dir = tempfile::tempdir().expect("a temporary directory");
    let legacy = legacy_store(dir.path());
    let graph = MemoryGraph::in_memory().expect("a graph");
    migrate_runtime_scoped_memory(&legacy, &graph, AT).expect("migrates");

    // Changed in the legacy store, which updates the record in place.
    remember(&legacy, "units", "kPa");
    let found = verify_runtime_scoped_memory(&legacy, &graph).expect("verifies");
    assert_eq!(found.differing.len(), 1, "the changed record was not found: {found:?}");

    let pass = migrate_runtime_scoped_memory(&legacy, &graph, AT).expect("migrates");
    assert_eq!((pass.migrated, pass.updated, pass.retired), (0, 1, 0), "{pass:?}");
    assert!(verify_runtime_scoped_memory(&legacy, &graph).expect("verifies").clean());
    // Still two records; the changed one now has two revisions.
    assert_eq!(row_counts(&graph), (2, 3));

    // Gone from the legacy store altogether.
    std::fs::remove_file(dir.path().join("memory").join("workspace-unit-four.json"))
        .expect("the legacy file is removed");
    let reloaded = crate::agent_runtime::memory::MemoryStore::open(dir.path());
    let pass = migrate_runtime_scoped_memory(&reloaded, &graph, AT).expect("migrates");
    assert_eq!(pass.retired, 2, "{pass:?}");
    assert!(verify_runtime_scoped_memory(&reloaded, &graph).expect("verifies").clean());
    // Retired, not deleted: the rows are still there, tombstoned.
    let kept = graph.migrated_from(LegacySource::RuntimeScopedMemory.key()).expect("reads");
    assert_eq!(kept.len(), 2);
    assert!(kept.values().all(|item| item.status == ItemStatus::Tombstoned));
}

#[test]
fn a_migrated_item_has_a_stable_derived_id() {
    // The id is what makes a re-run find its own earlier write.
    assert_eq!(
        super::migration::migrated_item_id("agent_runtime/memory", "m-1"),
        super::migration::migrated_item_id("agent_runtime/memory", "m-1")
    );
    assert_ne!(
        super::migration::migrated_item_id("agent_runtime/memory", "m-1"),
        item_id()
    );
}

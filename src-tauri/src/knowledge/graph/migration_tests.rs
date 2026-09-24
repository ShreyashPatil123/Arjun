//! The migration rehearsal, over real stores in a throwaway directory.
//!
//! "Rehearse in an isolated copy" is the requirement these exist to meet. Every
//! test here builds its own `tempfile::TempDir`, opens the **real**
//! `AgentRegistry` and the **real** `ConversationStore` over it, and migrates
//! into a real (in-memory) [`MemoryGraph`]. Nothing touches
//! `%APPDATA%\com.arjun.workbench`, which is the whole point: a migration
//! rehearsal that writes to the live profile is not a rehearsal.
//!
//! What is deliberately *not* asserted here: a count of items migrated from
//! this developer's machine. There are zero pins in the live conversation store
//! and a number taken from one profile proves nothing about the migration. The
//! properties below hold for one record or a thousand.

use tempfile::TempDir;

use super::migration::{
    migrate_agent_profiles, migrate_all, migrate_conversation_pins, migrated_item_id,
    LegacySource, LegacyStores,
};
use super::store::NotebookStore;
use crate::artifacts::conversation_store::ConversationArtifacts;
use crate::memory_engine::persistence::PersistenceManager;
use super::runtime_memory::{MemoryScope, Provenance};
use super::runtime_store::MemoryGraph;
use crate::agent_runtime::conversations::ConversationStore;
use crate::agents::store::AgentRegistry;
use crate::identity::{Role, Session, User};
use crate::policy::Classification;

const AT: &str = "2026-01-01T00:00:00Z";
const PROJECT: &str = "unit-four";
const OWNER: &str = "ada";

fn admin() -> Session {
    Session::open(User::new("ada", "Ada", vec![Role::Administrator]))
}

/// An isolated copy: its own directory, its own stores, nothing shared.
fn rehearsal() -> (TempDir, AgentRegistry, ConversationStore, MemoryGraph) {
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = AgentRegistry::open(dir.path()).expect("registry opens");
    let conversations = ConversationStore::open(dir.path()).expect("conversations open");
    let graph = MemoryGraph::in_memory().expect("graph opens");
    (dir, registry, conversations, graph)
}

fn seed_agents(registry: &AgentRegistry, how_many: usize) {
    let session = admin();
    for index in 0..how_many {
        let mut definition = crate::agents::tests::definition(&format!("ignored-{index}"));
        definition.display_name = format!("Agent {index}");
        definition.classification_ceiling = if index % 2 == 0 {
            Classification::Internal
        } else {
            Classification::VendorNegotiation
        };
        registry
            .create(&session, definition)
            .expect("the registry accepts a valid definition");
    }
}

// ---------------------------------------------------------------- stable ids

/// The same legacy record always lands on the same id.
///
/// This is what makes a re-run recognise its own earlier work, so it is worth
/// pinning against an exact shape rather than against itself.
#[test]
fn a_migrated_id_is_derived_and_therefore_reproducible() {
    let once = migrated_item_id("agents/registry.json", "ag-0001@1");
    let twice = migrated_item_id("agents/registry.json", "ag-0001@1");

    assert_eq!(once, twice, "the derivation must not vary between calls");
    assert!(
        once.starts_with("mi-mig-"),
        "a migrated id must be recognisable as one, got {once}"
    );
    assert_eq!(once.len(), "mi-mig-".len() + 64, "sha-256 hex, got {once}");
}

/// Two different stores cannot collide, and neither can two ids inside one.
///
/// The `\x1f` separator is what buys the first half: without it, store `a` with
/// id `bc` and store `ab` with id `c` would hash the same bytes.
#[test]
fn ids_from_different_records_do_not_collide() {
    let a = migrated_item_id("a", "bc");
    let b = migrated_item_id("ab", "c");
    assert_ne!(a, b, "the field separator must keep these apart");

    let first = migrated_item_id(LegacySource::AgentProfiles.key(), "ag-1@1");
    let second = migrated_item_id(LegacySource::AgentProfiles.key(), "ag-1@2");
    assert_ne!(
        first, second,
        "two versions of one agent are two records, not one"
    );
}

// ------------------------------------------------------------- idempotency

/// Running the whole migration twice writes nothing the second time.
#[test]
fn a_second_run_migrates_nothing_and_says_so() {
    let (_dir, registry, conversations, graph) = rehearsal();
    seed_agents(&registry, 3);

    let first = migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("first pass");
    assert_eq!(first.examined, 3);
    assert_eq!(first.migrated, 3);
    assert_eq!(first.already_present, 0);
    assert!(first.clean(), "nothing should be unexplained: {first:?}");

    let second = migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("second pass");
    assert_eq!(second.examined, 3);
    assert_eq!(
        second.migrated, 0,
        "a second pass must not write the same records again"
    );
    assert_eq!(second.already_present, 3);
    assert!(second.clean());

    // And the store agrees: three items, not six.
    let items = graph
        .snapshot(
            &admin(),
            &MemoryScope::Workspace {
                project_id: PROJECT.into(),
            },
            Some(PROJECT),
        )
        .expect("snapshot");
    assert_eq!(items.len(), 3, "the graph must hold one item per agent");

    let _ = conversations;
}

/// An interruption part-way through is recovered by simply running again.
///
/// Simulated the only way that is honest without a fault-injection hook: the
/// first pass is given *some* of the records, the run is abandoned, then the
/// full set is migrated. The already-written half must be recognised rather
/// than duplicated — which is exactly the state a crash would leave behind.
#[test]
fn an_interrupted_migration_is_finished_by_rerunning_it() {
    let (_dir, registry, _conversations, graph) = rehearsal();
    seed_agents(&registry, 2);

    // Pass one: migrate, then pretend the process died before recording that
    // it had finished. Nothing is rolled back, because nothing needs to be.
    let interrupted = migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("first pass");
    assert_eq!(interrupted.migrated, 2);

    // Two more agents arrive before the retry.
    seed_agents(&registry, 2);

    let resumed = migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("retry");
    assert_eq!(resumed.examined, 4, "the retry sees every record");
    assert_eq!(
        resumed.already_present, 2,
        "the half written before the interruption is recognised"
    );
    assert_eq!(resumed.migrated, 2, "only the new records are written");

    let items = graph
        .snapshot(
            &admin(),
            &MemoryScope::Workspace {
                project_id: PROJECT.into(),
            },
            Some(PROJECT),
        )
        .expect("snapshot");
    assert_eq!(items.len(), 4, "no record was written twice");
}

// --------------------------------------------------------- what must survive

/// Provenance survives, naming the store and the record it came from.
#[test]
fn every_migrated_item_names_where_it_came_from() {
    let (_dir, registry, _conversations, graph) = rehearsal();
    seed_agents(&registry, 1);
    migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("migrates");

    let items = graph
        .snapshot(
            &admin(),
            &MemoryScope::Workspace {
                project_id: PROJECT.into(),
            },
            Some(PROJECT),
        )
        .expect("snapshot");
    let item = items.first().expect("one item");

    match &item.provenance {
        Provenance::Migrated {
            legacy_store,
            legacy_id,
        } => {
            assert_eq!(legacy_store, LegacySource::AgentProfiles.key());
            assert!(
                legacy_id.contains('@'),
                "the legacy id must carry the definition version, got {legacy_id}"
            );
            // The id is reproducible from the provenance alone. That is what
            // lets a later pass, or a human, find the row again.
            assert_eq!(item.item_id, migrated_item_id(legacy_store, legacy_id));
        }
        other => panic!("expected migrated provenance, got {other:?}"),
    }
}

/// An agent's classification ceiling becomes the item's clearance.
///
/// A profile must not be readable by somebody who is not cleared for the
/// material that agent is allowed to handle, so the ceiling is the only
/// defensible source for this and it must not be widened in transit.
#[test]
fn a_classification_ceiling_survives_onto_the_acl() {
    let (_dir, registry, _conversations, graph) = rehearsal();
    seed_agents(&registry, 2); // index 0 Internal, index 1 VendorNegotiation
    migrate_agent_profiles(&registry, &graph, PROJECT, AT).expect("migrates");

    let items = graph
        .snapshot(
            &admin(),
            &MemoryScope::Workspace {
                project_id: PROJECT.into(),
            },
            Some(PROJECT),
        )
        .expect("snapshot");
    assert_eq!(items.len(), 2);

    for item in &items {
        assert_eq!(
            item.acl.cleared_roles,
            item.classification.cleared_roles().to_vec(),
            "the ACL must be the one this classification confers, not a default"
        );
        assert_eq!(
            item.acl.project_id.as_deref(),
            Some(PROJECT),
            "a workspace item must stay confined to its project"
        );
    }

    assert!(
        items
            .iter()
            .any(|i| i.classification == Classification::VendorNegotiation),
        "the stricter ceiling must not have been flattened to Internal"
    );
}

/// A pin stays confined to the person whose conversation it was.
#[test]
fn a_migrated_pin_is_confined_to_its_owner() {
    let (dir, _registry, conversations, graph) = rehearsal();

    let thread = conversations
        .create("Pump commissioning".into(), "welcome".into(), OWNER)
        .expect("create");
    conversations
        .set_pinned_context(&thread.id, &["drawing PV-2201 rev C".into()], OWNER)
        .expect("pin");

    let outcome = migrate_conversation_pins(&conversations, &graph, OWNER, AT).expect("migrates");
    assert_eq!(outcome.examined, 1);
    assert_eq!(outcome.migrated, 1);
    assert!(outcome.clean(), "{outcome:?}");

    let items = graph
        .snapshot(
            &Session::open(User::new(OWNER, "Ada", vec![Role::Administrator])),
            &MemoryScope::User {
                user_id: OWNER.into(),
            },
            None,
        )
        .expect("snapshot");
    let item = items.first().expect("one pin");
    assert_eq!(
        item.acl.owner.as_deref(),
        Some(OWNER),
        "a user-scoped item without an owner would be readable by the wrong person"
    );
    assert!(
        item.content.contains("drawing PV-2201 rev C"),
        "the pinned text must survive, got {:?}",
        item.content
    );

    drop(dir);
}

// ------------------------------------------------------- honesty properties

/// A conversation file that will not parse is named, not silently dropped.
///
/// This is the case that made the conversation-store fix worth writing: the
/// live store has two files damaged this exact way. A migration that skipped
/// them quietly would report success over data it never read.
#[test]
fn a_damaged_conversation_is_reported_rather_than_skipped_in_silence() {
    let (dir, _registry, conversations, graph) = rehearsal();

    let thread = conversations
        .create("Healthy".into(), "welcome".into(), OWNER)
        .expect("create");
    conversations
        .set_pinned_context(&thread.id, &["a pin that must still move".into()], OWNER)
        .expect("pin");

    // Valid JSON followed by the tail of a longer earlier version of itself.
    let root = dir.path().join("conversations");
    let good = std::fs::read_to_string(root.join(format!("{}.json", thread.id))).expect("read");
    std::fs::write(
        root.join("damaged.json"),
        format!("{good} \"compactions\": 0\n  }}\n}}"),
    )
    .expect("write damaged");

    let outcome = migrate_conversation_pins(&conversations, &graph, OWNER, AT).expect("migrates");

    assert_eq!(
        outcome.migrated, 1,
        "the healthy conversation's pin must still be migrated"
    );
    assert_eq!(
        outcome.unreadable.len(),
        1,
        "the damaged file must be reported: {outcome:?}"
    );
    assert!(
        outcome.unreadable[0].contains("damaged.json"),
        "the report must name the file, got {:?}",
        outcome.unreadable
    );
    assert!(
        !outcome.clean(),
        "a pass with an unread file is not a clean pass"
    );
}

/// Every declared source is actually attempted, and the report says so.
///
/// The `uncovered` list is empty today because all five legacy stores are
/// moved. That makes this test's job the opposite of what it was: instead of
/// checking the gap is *declared*, it checks the gap is genuinely *absent* —
/// that `migrate_all` ran every variant of [`LegacySource`] rather than
/// reporting complete coverage while quietly skipping three of them.
///
/// Adding a sixth variant without wiring it in fails here, and fails the
/// `debug_assert!` inside `migrate_all` first.
#[test]
fn every_declared_source_is_attempted_by_a_full_run() {
    let (dir, registry, conversations, graph) = rehearsal();
    seed_agents(&registry, 1);

    let notebooks = NotebookStore::open(dir.path()).expect("a notebook store");
    let memories = PersistenceManager::new(dir.path()).expect("a memory store");
    let artifacts = ConversationArtifacts::open(dir.path()).expect("an artifact store");
    let runtime_memory = crate::agent_runtime::memory::MemoryStore::open(dir.path());

    let stores = LegacyStores {
        agents: &registry,
        conversations: &conversations,
        notebooks: &notebooks,
        memories: &memories,
        runtime_memory: &runtime_memory,
        artifacts: &artifacts,
        conversation_ids: &[],
    };

    let report = migrate_all(stores, &graph, PROJECT, OWNER, AT).expect("migrates");

    assert_eq!(
        report.schema_version,
        super::runtime_memory::MEMORY_SCHEMA_VERSION,
        "the report must state the schema it wrote against"
    );
    for source in LegacySource::ALL {
        assert!(
            report.per_source.contains_key(source.key()),
            "{} was declared but never attempted",
            source.key()
        );
    }
    assert!(
        report.uncovered.is_empty(),
        "nothing should be declared uncovered now, got {:?}",
        report.uncovered
    );
    assert_eq!(
        report.total_migrated(),
        1,
        "only the one seeded agent had anything to move"
    );

    // The summary still enumerates every source, including the empty ones — a
    // source that moved nothing and a source that was never run look identical
    // in a summary that omits both.
    let explained = report.explain();
    for source in LegacySource::ALL {
        assert!(
            explained.contains(source.key()),
            "the summary must name {}, got:
{explained}",
            source.key()
        );
    }
}

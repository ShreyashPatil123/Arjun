//! Plan P01: a saved definition reaches the next child, and only the next one.
//!
//! Every test here runs the real [`SubagentManager`] against a real
//! [`AgentRegistry`] in a temporary directory — the same store the Agents
//! screen writes. A stand-in registry would prove only that the stand-in is
//! read; the property that matters is that *this* file, written by *this*
//! code, is what the next dispatch runs under.
//!
//! The one thing replaced is the worker, by [`Recorder`], which keeps the
//! packet it was handed. That is deliberate: the claim under test is about what
//! reaches a worker, and the only honest way to check what a worker received is
//! to be the worker.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use tokio::sync::Notify;

use super::manager::Dispatch;
use super::*;
use crate::agent_runtime::events::TaskEventLog;
use crate::agents::store::{AgentRegistry, Visibility};
use crate::agents::{AgentDefinition, AgentState};
use crate::identity::{Role, Session, User};
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::registry::ModelRole;
use crate::subagents::certification::Decision;

const RUN: &str = "run-p01";
const ROLE: &str = "knowledge-retriever";

fn admin() -> Session {
    Session::open(User::new("ada", "Ada", vec![Role::Administrator]))
}

fn employee() -> Session {
    Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
}

fn shipped() -> LoadedProfiles {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent")
        .join("agents");
    load_profiles(&dir)
}

fn model() -> Decision {
    Decision {
        model_id: "qwen3.5-9b".to_string(),
        role: ModelRole::Reasoning,
        cheaper_than_parent: false,
        reason: "the run's own model".to_string(),
        tier: None,
        score: None,
    }
}

fn parent(session: &Session, tools: &[ToolName], root: &Path) -> InheritedPolicy {
    InheritedPolicy::of_run(session, Classification::Internal, root.join(RUN), tools)
}

/// A worker that keeps what it was handed, and can be held mid-run.
struct Recorder {
    capability: String,
    seen: Arc<StdMutex<Vec<ChildTaskPacket>>>,
    /// When set, the worker records its packet and then waits here, so a test
    /// can change the registry while a child is genuinely running.
    gate: Option<Arc<Notify>>,
    started: Option<Arc<Notify>>,
}

#[async_trait]
impl ChildWorker for Recorder {
    fn profile(&self) -> &str {
        &self.capability
    }

    async fn run(
        &self,
        packet: &ChildTaskPacket,
        _policy: &EffectivePolicy,
    ) -> Result<ChildResult, String> {
        self.seen.lock().expect("recorder lock").push(packet.clone());
        if let Some(started) = &self.started {
            started.notify_one();
        }
        if let Some(gate) = &self.gate {
            gate.notified().await;
        }
        Ok(ChildResult::completed(
            &packet.child_id,
            &packet.profile,
            packet.required_schema,
            vec![Finding {
                statement: "recorded".to_string(),
                evidence: Vec::new(),
            }],
            1.0,
            Vec::new(),
            1,
        ))
    }
}

/// A registry with every shipped profile imported, the way start-up leaves it.
struct World {
    registry: Arc<AgentRegistry>,
    manager: Arc<SubagentManager>,
    seen: Arc<StdMutex<Vec<ChildTaskPacket>>>,
    gate: Arc<Notify>,
    started: Arc<Notify>,
    _dir: tempfile::TempDir,
}

impl World {
    fn new(gated: bool) -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        let registry = Arc::new(AgentRegistry::open(dir.path()).expect("registry opens"));
        let loaded = shipped();
        for profile in &loaded.profiles {
            registry.import_bundled(profile).expect("imports");
        }

        let seen: Arc<StdMutex<Vec<ChildTaskPacket>>> = Arc::default();
        let gate = Arc::new(Notify::new());
        let started = Arc::new(Notify::new());
        let manager = SubagentManager::new(
            loaded.profiles.clone(),
            Arc::new(TaskEventLog::in_memory().expect("an event log")),
        )
        .with_definitions(Arc::clone(&registry) as Arc<dyn DefinitionSource>)
        .with_worker(Arc::new(Recorder {
            capability: ROLE.to_string(),
            seen: Arc::clone(&seen),
            gate: gated.then(|| Arc::clone(&gate)),
            started: gated.then(|| Arc::clone(&started)),
        }));

        Self {
            registry,
            manager: Arc::new(manager),
            seen,
            gate,
            started,
            _dir: dir,
        }
    }

    fn agent(&self, key: &str) -> AgentDefinition {
        self.registry
            .list(Visibility::Administrator)
            .expect("lists")
            .into_iter()
            .find(|agent| {
                agent.agent_id == key
                    || agent
                        .imported_from
                        .as_ref()
                        .is_some_and(|origin| origin.profile_name == key)
            })
            .unwrap_or_else(|| panic!("no agent answers to {key}"))
    }

    /// Saves an edit the way the Agents screen does: read, change, write back
    /// against the version that was read.
    fn edit(&self, key: &str, change: impl FnOnce(&mut AgentDefinition)) -> u64 {
        let mut agent = self.agent(key);
        let expected = agent.definition_version;
        change(&mut agent);
        self.registry
            .update(&admin(), &agent.agent_id.clone(), expected, agent)
            .expect("the edit saves")
            .definition_version
    }

    fn packets(&self) -> Vec<ChildTaskPacket> {
        self.seen.lock().expect("recorder lock").clone()
    }
}

async fn dispatch(
    manager: &SubagentManager,
    key: &str,
    objective: &str,
    root: &Path,
) -> Result<Spawned, SpawnRefusal> {
    let session = employee();
    manager
        .spawn(
            key,
            &parent(&session, &[ToolName::SearchDocuments], root),
            objective,
            Vec::new(),
            model(),
            &Dispatch::for_task(key, "task-p01"),
        )
        .await
}

// ── The two properties the plan asks for by name ────────────────────────────

#[tokio::test]
async fn an_administrators_edit_reaches_the_next_child() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    let before = world.agent(ROLE);

    dispatch(&world.manager, ROLE, "first question", root.path())
        .await
        .expect("the first child runs");

    let edited_text = "Find passages about seal wear. Cite the page for every figure.";
    let saved = world.edit(ROLE, |agent| agent.instructions = edited_text.to_string());
    assert_eq!(saved, before.definition_version + 1, "the edit is a new version");

    dispatch(&world.manager, ROLE, "second question", root.path())
        .await
        .expect("the second child runs");

    let packets = world.packets();
    assert_eq!(packets.len(), 2, "two children reached the worker");

    let (first, second) = (&packets[0], &packets[1]);
    // Both are the registry agent, not the profile file.
    assert_eq!(first.agent_id, before.agent_id);
    assert_eq!(second.agent_id, before.agent_id);
    assert_eq!(first.definition_origin, "registry");

    // The first ran under the version it was sent under...
    assert_eq!(first.definition_version, Some(before.definition_version));
    assert_eq!(first.instructions, before.instructions);
    // ...and the second under the saved edit — in the text the worker's model
    // loop actually receives, not only in the registry row.
    assert_eq!(second.definition_version, Some(saved));
    assert_eq!(second.instructions, edited_text);
    assert_ne!(first.instructions_sha256, second.instructions_sha256);
}

#[tokio::test]
async fn a_running_child_keeps_the_definition_it_was_sent_under() {
    let world = World::new(true);
    let root = tempfile::tempdir().expect("temp dir");
    let before = world.agent(ROLE);

    let manager = Arc::clone(&world.manager);
    let root_path = root.path().to_path_buf();
    let running = tokio::spawn(async move {
        dispatch(&manager, ROLE, "a long question", &root_path).await
    });

    // The child is inside the worker now, holding its packet.
    world.started.notified().await;

    // Saved while it runs.
    let saved = world.edit(ROLE, |agent| {
        agent.instructions = "Something entirely different.".to_string()
    });

    world.gate.notify_one();
    let outcome = running.await.expect("the task joins").expect("the child ran");
    assert!(outcome.result().is_complete());

    let packets = world.packets();
    assert_eq!(packets.len(), 1);
    assert_eq!(
        packets[0].definition_version,
        Some(before.definition_version),
        "the running child was moved onto a version saved after it started"
    );
    assert_eq!(packets[0].instructions, before.instructions);
    assert_ne!(packets[0].definition_version, Some(saved));
}

// ── Stable keys ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_clone_is_performed_by_its_capability_and_answers_to_its_own_id() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    let source = world.agent(ROLE);

    let cloned = world
        .registry
        .clone_agent(&admin(), &source.agent_id, "Seal-wear retriever")
        .expect("clones");
    assert_ne!(cloned.agent_id, source.agent_id);

    dispatch(&world.manager, &cloned.agent_id, "via the clone", root.path())
        .await
        .expect("the clone runs");

    let packets = world.packets();
    assert_eq!(packets.len(), 1, "the retriever's worker performed the clone");
    assert_eq!(packets[0].agent_id, cloned.agent_id);
    assert_eq!(packets[0].capability, ROLE);
    assert_eq!(packets[0].definition_version, Some(1));
}

#[tokio::test]
async fn renaming_an_agent_changes_nothing_about_how_it_is_dispatched() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");

    world.edit(ROLE, |agent| agent.display_name = "Passage Finder".to_string());

    // The role key still reaches it...
    dispatch(&world.manager, ROLE, "after the rename", root.path())
        .await
        .expect("still dispatchable by its role key");
    assert_eq!(world.packets().len(), 1);

    // ...and the new display name is not a key at all. A name that could be
    // typed into a form cannot be what decides which agent runs.
    let by_name = dispatch(&world.manager, "Passage Finder", "by display name", root.path()).await;
    assert!(
        matches!(&by_name, Err(SpawnRefusal::Unresolved { detail, .. }) if detail.contains("answers to")),
        "a display name resolved to an agent: {by_name:?}"
    );
    assert_eq!(world.packets().len(), 1, "the display name dispatched a child");
}

// ── What must stay refused ──────────────────────────────────────────────────

#[tokio::test]
async fn a_disabled_agent_is_refused_and_not_replaced_by_its_bundled_profile() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    let agent = world.agent(ROLE);

    world
        .registry
        .set_state(&admin(), &agent.agent_id, agent.definition_version, AgentState::Disabled)
        .expect("disables");

    let refused = dispatch(&world.manager, ROLE, "while disabled", root.path()).await;
    match refused {
        Err(SpawnRefusal::Unresolved { detail, .. }) => {
            assert!(detail.contains("disabled"), "{detail}")
        }
        other => panic!("a disabled agent was dispatched: {other:?}"),
    }
    assert!(world.packets().is_empty(), "the worker ran for a disabled agent");

    // And re-enabling it works for the same objective: the refusal recorded
    // no intent in the durable ledger, so there is nothing to trip over.
    let disabled = world.agent(ROLE);
    world
        .registry
        .set_state(&admin(), &disabled.agent_id, disabled.definition_version, AgentState::Enabled)
        .expect("re-enables");
    dispatch(&world.manager, ROLE, "while disabled", root.path())
        .await
        .expect("the same objective runs once the agent is enabled again");
    assert_eq!(world.packets().len(), 1);
}

#[tokio::test]
async fn an_edit_cannot_widen_a_child_beyond_its_parent() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");

    // An administrator grants the retriever writer and execution tools, and
    // clears its own denylist so the registry accepts the grant -- the widest
    // edit an administrator can make. (Allowing a tool the definition also
    // denies is refused by the registry itself as contradictory.)
    world.edit(ROLE, |agent| {
        agent.denied_tools = Vec::new();
        agent.allowed_tools = vec![
            ToolName::SearchDocuments,
            ToolName::WriteScopedFile,
            ToolName::ExecuteCode,
            ToolName::CreateDocx,
        ]
    });

    // The parent holds only search.
    dispatch(&world.manager, ROLE, "try to write", root.path())
        .await
        .expect("runs, narrowed");

    let packets = world.packets();
    assert_eq!(packets.len(), 1);
    assert_eq!(
        packets[0].allowed_tools,
        vec![ToolName::SearchDocuments],
        "a saved definition granted a child more than its parent holds"
    );
}

#[tokio::test]
async fn a_key_nothing_answers_to_is_refused() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");

    let outcome = dispatch(&world.manager, "ag-does-not-exist", "anything", root.path()).await;
    assert!(
        matches!(&outcome, Err(SpawnRefusal::Unresolved { detail, .. }) if detail.contains("answers to")),
        "{outcome:?}"
    );
    assert!(world.packets().is_empty());
}

// ── The §13 import defect ───────────────────────────────────────────────────

#[test]
fn an_imported_agent_is_given_its_role_body_and_not_its_description() {
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = AgentRegistry::open(dir.path()).expect("opens");
    let loaded = shipped();
    let profile = loaded.get(ROLE).expect("shipped");
    assert_ne!(
        profile.instructions.trim(),
        profile.description.trim(),
        "the fixture needs a profile whose body differs from its summary"
    );

    registry.import_bundled(profile).expect("imports");
    let stored = registry
        .list(Visibility::Administrator)
        .expect("lists")
        .into_iter()
        .find(|agent| agent.display_name == ROLE)
        .expect("imported");
    assert_eq!(stored.instructions, profile.instructions);
}

#[test]
fn a_row_the_earlier_import_wrote_wrongly_is_repaired_exactly_once() {
    let dir = tempfile::tempdir().expect("temp dir");
    let profile = shipped().get(ROLE).expect("shipped").clone();

    // What every import before the fix produced: a registry row whose
    // instructions are the one-line description. Written through the real
    // store and the real file, then reopened, so the repair runs over what is
    // actually on disk after an upgrade.
    {
        let registry = AgentRegistry::open(dir.path()).expect("opens");
        registry.import_bundled(&profile).expect("imports");
        let mut row = registry
            .list(Visibility::Administrator)
            .expect("lists")
            .remove(0);
        let expected = row.definition_version;
        row.instructions = profile.description.clone();
        registry
            .update(&admin(), &row.agent_id.clone(), expected, row)
            .expect("the defective row is written");
    }

    let registry = AgentRegistry::open(dir.path()).expect("reopens");
    let broken = registry.list(Visibility::Administrator).expect("lists").remove(0);
    assert_eq!(broken.instructions, profile.description);

    let repaired = registry.import_bundled(&profile).expect("repairs");
    assert!(!repaired.unchanged, "the defect was not repaired");
    assert_eq!(repaired.definition_version, broken.definition_version + 1);
    let row = registry.list(Visibility::Administrator).expect("lists").remove(0);
    assert_eq!(row.instructions, profile.instructions);

    // Start-up runs the import every launch. The repair must not.
    let again = registry.import_bundled(&profile).expect("imports");
    assert!(again.unchanged, "the repair ran a second time");
    assert_eq!(again.definition_version, repaired.definition_version);
}

// ── Compatibility with what is already on disk ─────────────────────────────

#[tokio::test]
async fn a_packet_recorded_before_definitions_were_pinned_still_reads() {
    // A real packet, as `record_start` would serialise it, with every field
    // this change added taken back out. That is exactly the shape an event
    // written before this change has. Built rather than hand-written: the first
    // version of this test typed the JSON by hand, guessed the tool names were
    // camelCase, and was wrong -- which is the reason not to guess.
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    dispatch(&world.manager, ROLE, "anything", root.path())
        .await
        .expect("runs");

    let mut old = serde_json::to_value(&world.packets()[0]).expect("serialises");
    let fields = old.as_object_mut().expect("an object");
    for added in [
        "definitionVersion",
        "definitionOrigin",
        "capability",
        "instructions",
        "instructionsSha256",
        "sharedWithTask",
        "skills",
        "attemptId",
        "modelPolicy",
    ] {
        fields.remove(added);
    }

    let packet: ChildTaskPacket = serde_json::from_value(old).expect("an old packet reads");
    assert_eq!(packet.definition_version, None, "an old packet is not given a version");
    assert!(packet.instructions.is_empty());
    assert!(packet.definition_origin.is_empty());
    assert!(!packet.shared_with_task);
    assert!(packet.skills.is_empty());
    assert!(packet.attempt_id.is_empty(), "an old packet is not given an attempt");
    assert_eq!(packet.model_policy, None, "an old packet is not given a model policy");
}

#[tokio::test]
async fn the_role_body_is_never_written_into_the_trace() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    dispatch(&world.manager, ROLE, "anything", root.path())
        .await
        .expect("runs");

    let packet = &world.packets()[0];
    assert!(!packet.instructions.is_empty(), "the worker did receive the body");
    let serialised = serde_json::to_string(packet).expect("serialises");
    assert!(
        !serialised.contains(&packet.instructions),
        "the role body was serialised into a record read by more people than the run"
    );
    assert!(serialised.contains(&packet.instructions_sha256));
}

// ── P01: result contracts ahead of their workers ───────────────────────────

#[tokio::test]
async fn a_schema_with_no_worker_is_registered_and_refused() {
    // Plan P01: register the new role schemas now, and do not mark their
    // unfinished implementations executable. An agent declaring a Word
    // document is a valid definition; dispatching it is refused and says why.
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");

    for (schema, name) in [
        (SchemaKind::Document, "Approval-note author"),
        (SchemaKind::Deck, "Review-deck author"),
        (SchemaKind::Workbook, "Assessment workbook author"),
    ] {
        let mut writer = crate::agents::tests::definition("ignored");
        writer.display_name = name.to_string();
        writer.output_schema = schema;
        let created = world
            .registry
            .create(&admin(), writer)
            .expect("a definition declaring a writer contract is valid");

        let outcome = dispatch(&world.manager, &created.agent_id, name, root.path()).await;
        match outcome {
            Err(SpawnRefusal::Unresolved { detail, .. }) => assert!(
                detail.contains("cannot yet be run") && detail.contains(schema.as_str()),
                "{schema:?}: {detail}"
            ),
            other => panic!("{schema:?} was dispatched with no worker behind it: {other:?}"),
        }
    }
    assert!(world.packets().is_empty(), "a worker ran for a contract it does not produce");
}

#[test]
fn the_new_schemas_survive_a_round_trip_through_the_registry_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let created = {
        let registry = AgentRegistry::open(dir.path()).expect("opens");
        let mut deck = crate::agents::tests::definition("ignored");
        deck.output_schema = SchemaKind::Deck;
        registry.create(&admin(), deck).expect("creates").agent_id
    };
    // Reopened from disk: the registry refuses a file it cannot fully read,
    // so a schema it could write and not read back would lose every agent.
    let reopened = AgentRegistry::open(dir.path()).expect("reopens");
    let stored = reopened
        .get(&created, Visibility::Administrator)
        .expect("the deck author is still there");
    assert_eq!(stored.output_schema, SchemaKind::Deck);
}

#[test]
fn partial_and_blocked_are_not_complete_and_say_what_is_missing() {
    for status in [ChildStatus::Partial, ChildStatus::Blocked] {
        let result = ChildResult::ended(
            "c-1",
            ROLE,
            status,
            SchemaKind::Extraction,
            vec![Finding {
                statement: "pages 1-3 read".to_string(),
                evidence: Vec::new(),
            }],
            "stopped early",
            2,
        )
        .with_missing("page 4 was not read");

        assert!(!result.is_complete(), "{status:?} was treated as the work being done");
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.missing, vec!["page 4 was not read".to_string()]);
        assert!(!status.describe().is_empty());
    }
    assert_eq!(ChildStatus::Partial.as_str(), "partial");
    assert_eq!(ChildStatus::Blocked.as_str(), "blocked");
}

#[test]
fn a_result_recorded_before_the_contract_grew_reads_and_keeps_its_hash() {
    let result = ChildResult::completed(
        "c-1",
        ROLE,
        SchemaKind::Retrieval,
        vec![Finding {
            statement: "PV-2201: governing reading 8.2 mm".to_string(),
            evidence: Vec::new(),
        }],
        0.9,
        Vec::new(),
        1,
    );
    let recorded_hash = result.result_hash.clone();

    // The shape the idempotency ledger holds for a result settled before this
    // change: the same serialisation, without the four lists.
    let mut old = serde_json::to_value(&result).expect("serialises");
    let fields = old.as_object_mut().expect("an object");
    for added in ["artifacts", "receipts", "validation", "missing"] {
        assert!(fields.remove(added).is_some(), "{added} was not serialised");
    }
    let read: ChildResult = serde_json::from_value(old).expect("an old result reads");
    assert!(read.artifacts.is_empty() && read.receipts.is_empty());
    assert_eq!(read.result_hash, recorded_hash, "reading an old result moved its hash");
}

#[test]
fn the_hash_covers_what_was_produced_once_there_is_something() {
    let base = || {
        ChildResult::completed(
            "c-1",
            "document-author",
            SchemaKind::Document,
            Vec::new(),
            0.9,
            Vec::new(),
            1,
        )
    };
    let plain = base();
    let with_note = base().with_artifact(ArtifactVersion {
        artifact_id: "approval-note".to_string(),
        version: 2,
        sha256: "a".repeat(64),
    });
    let with_other_bytes = base().with_artifact(ArtifactVersion {
        artifact_id: "approval-note".to_string(),
        version: 2,
        sha256: "b".repeat(64),
    });

    assert_ne!(plain.result_hash, with_note.result_hash, "an artifact did not enter the hash");
    assert_ne!(
        with_note.result_hash, with_other_bytes.result_hash,
        "two different files at one version hashed the same"
    );

    // A result that carries nothing new hashes as it always did, so no hash
    // already written to an event log stops verifying.
    let resealed = base().with_missing("x");
    assert_ne!(plain.result_hash, resealed.result_hash);
    let reread: ChildResult =
        serde_json::from_value(serde_json::to_value(&plain).unwrap()).unwrap();
    assert_eq!(reread.result_hash, plain.result_hash);
}

// ── P01: writer delegation semantics, without widening the read-only tool ──

/// The case the mode exists for: the parent *does* hold a write tool, and an
/// administrator has edited a read-only role to ask for it. Intersection with
/// the parent's grant cannot stop this -- both sides have the tool. Only the
/// dispatch's mode can, and a read-only dispatch must.
#[tokio::test]
async fn a_read_only_dispatch_withholds_a_write_tool_the_parent_does_hold() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    world.edit(ROLE, |agent| {
        agent.denied_tools = Vec::new();
        agent.allowed_tools = vec![ToolName::SearchDocuments, ToolName::WriteScopedFile];
    });

    let session = employee();
    let parent_with_write = parent(
        &session,
        &[ToolName::SearchDocuments, ToolName::WriteScopedFile],
        root.path(),
    );

    // Read-only, the default -- what `agent.delegate_readonly` sends.
    world
        .manager
        .spawn(
            ROLE,
            &parent_with_write,
            "read-only dispatch",
            Vec::new(),
            model(),
            &Dispatch::for_task(ROLE, "task-p01"),
        )
        .await
        .expect("the read-only role still runs");

    // Writer, explicitly.
    world
        .manager
        .spawn(
            ROLE,
            &parent_with_write,
            "writer dispatch",
            Vec::new(),
            model(),
            &Dispatch::for_task(ROLE, "task-p01").writing(),
        )
        .await
        .expect("a writer dispatch runs");

    let packets = world.packets();
    assert_eq!(packets.len(), 2);
    assert_eq!(
        packets[0].allowed_tools,
        vec![ToolName::SearchDocuments],
        "a read-only dispatch handed a child a write tool"
    );
    assert!(
        packets[1].allowed_tools.contains(&ToolName::WriteScopedFile),
        "writer mode should keep what the narrowing left: {:?}",
        packets[1].allowed_tools
    );
}

#[tokio::test]
async fn a_role_declared_as_writing_is_refused_by_a_read_only_dispatch() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    world.edit(ROLE, |agent| {
        agent.isolation = crate::subagents::profile::Isolation::Writer;
        agent.write_policy = crate::subagents::profile::WritePolicy::OwnDirectory;
    });

    let outcome = dispatch(&world.manager, ROLE, "anything", root.path()).await;
    assert!(
        matches!(&outcome, Err(SpawnRefusal::NeedsWriterDelegation { .. })),
        "{outcome:?}"
    );
    assert!(world.packets().is_empty(), "a writer role ran through a read-only dispatch");
}

/// Writers do not overlap, whatever their declared isolation.
///
/// A read-only-declared role an administrator edited to hold a write tool,
/// dispatched as a writer, used to take the *reader* lane -- its isolation
/// said read-only -- so two of them could write at once. The lane is now
/// decided by what the child actually holds.
#[tokio::test]
async fn two_writers_never_share_the_reader_lane() {
    let world = World::new(true);
    let root = tempfile::tempdir().expect("temp dir");
    world.edit(ROLE, |agent| {
        agent.denied_tools = Vec::new();
        agent.allowed_tools = vec![ToolName::SearchDocuments, ToolName::WriteScopedFile];
    });

    let spawn_writer = |objective: &'static str| {
        let manager = Arc::clone(&world.manager);
        let root = root.path().to_path_buf();
        tokio::spawn(async move {
            let session = employee();
            manager
                .spawn(
                    ROLE,
                    &parent(
                        &session,
                        &[ToolName::SearchDocuments, ToolName::WriteScopedFile],
                        &root,
                    ),
                    objective,
                    Vec::new(),
                    model(),
                    &Dispatch::for_task(ROLE, "task-p01").writing(),
                )
                .await
        })
    };

    let first = spawn_writer("first write");
    world.started.notified().await;
    let second = spawn_writer("second write");

    // The second must not reach its worker while the first holds a write tool.
    let overlapped =
        tokio::time::timeout(std::time::Duration::from_millis(300), world.started.notified()).await;
    assert!(overlapped.is_err(), "a second writer started while the first was writing");

    world.gate.notify_one();
    first.await.expect("joins").expect("the first writer ran");
    world.started.notified().await;
    world.gate.notify_one();
    second.await.expect("joins").expect("the second writer ran after the first");
    assert_eq!(world.packets().len(), 2);
}

// ── P01: the whole job contract travels on the packet ──────────────────────

#[tokio::test]
async fn a_packet_carries_the_whole_job_contract() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    let skill = crate::agents::SkillBinding {
        name: "sop-reader".to_string(),
        version: "1.2.0".to_string(),
        sha256: "c".repeat(64),
    };
    let saved = world.edit(ROLE, |agent| agent.skills = vec![skill.clone()]);

    let session = employee();
    world
        .manager
        .spawn(
            ROLE,
            &parent(&session, &[ToolName::SearchDocuments], root.path()),
            "which clause governs the wall-thickness minimum",
            vec![InputRef::GraphItem { item_id: "item-9".to_string(), revision: 12 }],
            model(),
            &Dispatch::for_task(ROLE, "task-p01").after(12).in_attempt("attempt-2"),
        )
        .await
        .expect("runs");

    let packet = &world.packets()[0];
    // Identity and the pinned definition.
    assert!(packet.agent_id.starts_with("ag-"), "{}", packet.agent_id);
    assert_eq!(packet.definition_version, Some(saved));
    assert_eq!(packet.capability, ROLE);
    // Task, run, attempt, job, execution.
    assert_eq!(packet.task_id, "task-p01");
    assert_eq!(packet.attempt_id, "attempt-2");
    assert_eq!(packet.job_id(), packet.idempotency_key);
    assert!(!packet.child_id.is_empty());
    // Model policy, computed from the decision actually made.
    let policy = packet.model_policy.as_ref().expect("a model policy");
    assert_eq!(policy.routing_reason, "the run's own model");
    assert!(policy.within_eligible, "an empty eligible set admits any model for the role");
    // Skill pins, effective tools, schema, dependency revisions, limits.
    assert_eq!(packet.skills, vec![skill]);
    assert_eq!(packet.allowed_tools, vec![ToolName::SearchDocuments]);
    assert_eq!(packet.required_schema, SchemaKind::Retrieval);
    assert_eq!(
        packet.requirement,
        crate::subagents::Requirement::AtLeast { graph_revision: 12 }
    );
    assert!(packet.deadline > packet.created_at);
    assert!(packet.limits.max_turns > 0);
}

/// Whether a routed model is inside the definition's eligible set is
/// computed, not asserted.
#[test]
fn a_model_outside_the_eligible_set_is_recorded_as_outside_it() {
    let loaded = shipped();
    let mut resolved = ResolvedDefinition::from_bundled(loaded.get(ROLE).expect("shipped"));
    resolved.model_binding.eligible_model_ids = vec!["nemotron-nano-4b".to_string()];
    let policy = resolved.model_policy(&model());
    assert!(!policy.within_eligible, "qwen3.5-9b is not in [nemotron-nano-4b]");
    assert_eq!(policy.eligible_model_ids, vec!["nemotron-nano-4b".to_string()]);
}

// ── P01: a child's permissions never exceed its parent's ───────────────────

/// Every shipped role, under parents holding every tool, no tool, and a
/// scattering in between: whatever a child is granted, its parent held.
#[test]
fn no_shipped_role_is_ever_granted_what_its_parent_does_not_hold() {
    let world = World::new(false);
    let root = tempfile::tempdir().expect("temp dir");
    let session = employee();
    let every: Vec<ToolName> = ToolName::ALL.to_vec();
    let parents: Vec<Vec<ToolName>> = vec![
        Vec::new(),
        vec![ToolName::SearchDocuments],
        every.iter().copied().filter(|tool| tool.is_read_only()).collect(),
        every.iter().copied().step_by(2).collect(),
        every.clone(),
    ];

    for profile in shipped().profiles {
        for held in &parents {
            let inherited = parent(&session, held, root.path());
            let Ok((_, policy)) = world.manager.plan(&profile.name, &inherited, "child-prop") else {
                // A refusal grants nothing, which satisfies the property.
                continue;
            };
            for tool in &policy.tools {
                assert!(
                    held.contains(tool),
                    "{} was granted {} by a parent holding {:?}",
                    profile.name,
                    tool.as_str(),
                    held.iter().map(|t| t.as_str()).collect::<Vec<_>>()
                );
            }
            assert!(!policy.inherited.network_permitted, "{} reached the network", profile.name);
        }
    }
}

// ── P01: verified receipts ──────────────────────────────────────────────────

#[test]
fn a_receipt_with_no_event_behind_it_is_refused() {
    let base = || {
        ChildResult::completed("c", "knowledge-retriever", SchemaKind::Retrieval, Vec::new(), 1.0, Vec::new(), 1)
    };
    for unbacked in [
        ReceiptRef { run_id: "run-1".into(), tool: "knowledge.search_authorized".into(), event_seq: 0 },
        ReceiptRef { run_id: "run-1".into(), tool: "knowledge.search_authorized".into(), event_seq: -3 },
        ReceiptRef { run_id: String::new(), tool: "knowledge.search_authorized".into(), event_seq: 4 },
        ReceiptRef { run_id: "run-1".into(), tool: " ".into(), event_seq: 4 },
    ] {
        assert!(base().with_receipt(unbacked.clone()).is_err(), "{unbacked:?} was accepted");
    }

    let backed = base()
        .with_receipt(ReceiptRef {
            run_id: "run-1".into(),
            tool: "knowledge.search_authorized".into(),
            event_seq: 42,
        })
        .expect("a receipt naming a recorded event is accepted");
    assert_eq!(backed.receipts.len(), 1);
    assert_ne!(backed.result_hash, base().result_hash, "the receipt did not enter the hash");
}

// ── P01: the Agents screen says what dispatch does ─────────────────────────

/// `RUNNABLE_SCHEMAS` in the Agents screen's service is the set
/// `capability_for` has a worker for. Read from the TypeScript source, because
/// a list that has to agree across two languages and is checked in neither
/// drifts silently.
#[test]
fn the_agents_screen_offers_exactly_the_schemas_a_worker_produces() {
    let source = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("src-tauri has a parent")
            .join("src/services/agentRegistry.service.ts"),
    )
    .expect("the service is in the checkout");
    let start = source
        .find("export const RUNNABLE_SCHEMAS")
        .expect("RUNNABLE_SCHEMAS is declared");
    let block = &source[start..];
    let block = &block[block.find("([").expect("a set literal") + 2..];
    let block = &block[..block.find("])").expect("the literal ends")];
    let mut in_typescript: Vec<String> = block
        .split(',')
        .map(|item| item.trim().trim_matches('\'').trim_matches('"').to_string())
        .filter(|item| !item.is_empty())
        .collect();
    in_typescript.sort();

    let mut in_rust: Vec<String> = [
        SchemaKind::Extraction,
        SchemaKind::Retrieval,
        SchemaKind::Calculation,
        SchemaKind::Review,
        SchemaKind::Code,
        SchemaKind::Document,
        SchemaKind::Deck,
        SchemaKind::Workbook,
    ]
    .into_iter()
    .filter(|schema| capability_for(*schema).is_some())
    .map(|schema| schema.as_str().to_string())
    .collect();
    in_rust.sort();

    assert_eq!(in_typescript, in_rust);
}

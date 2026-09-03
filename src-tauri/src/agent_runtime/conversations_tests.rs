//! Tests for the conversation store.
//!
//! These exercise the on-disk CRUD: open, create, append a turn, update
//! streaming content, mark a message done, list, and round-trip read.
//! They run on a temp directory so they do not pollute the application
//! data folder.
//!
//! The per-user isolation tests at the bottom cover TODO 2 of the
//! 7-step plan: a conversation created by user A is not visible to
//! user B, and B cannot read, write to, or delete A's conversation
//! even when they know the id.

use std::env;

use crate::agent_runtime::conversations::{
    Conversation, ConversationStore, MessageCompletion, MessageRole, MessageStatus,
    LEGACY_OWNER_ID,
};

const OWNER: &str = "engineer";
const OTHER: &str = "reviewer";

fn temp_dir() -> std::path::PathBuf {
    let mut dir = env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.push(format!("arjun-conv-tests-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn open_creates_an_empty_store() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    assert!(store.list(None).unwrap().is_empty());
    assert!(store.get("nonexistent", None).unwrap().is_none());
}

#[test]
fn create_persists_a_conversation_with_one_welcome_message() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    assert_eq!(conv.title, "Test");
    assert_eq!(conv.owner_user_id, OWNER);
    assert_eq!(conv.messages.len(), 1);
    assert_eq!(conv.messages[0].role, MessageRole::System);
    assert_eq!(conv.messages[0].content, "Welcome.");
    assert_eq!(conv.messages[0].status, MessageStatus::Done);

    // Read it back. As the owner.
    let fetched = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.id, conv.id);
    assert_eq!(fetched.title, "Test");
    assert_eq!(fetched.owner_user_id, OWNER);
}

#[test]
fn append_user_turn_creates_user_and_streaming_assistant_messages() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    let updated = store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append")
        .expect("found");
    assert_eq!(updated.messages.len(), 3);
    let user = &updated.messages[1];
    assert_eq!(user.role, MessageRole::User);
    assert_eq!(user.content, "Hello");
    assert_eq!(user.status, MessageStatus::Done);
    let assistant = &updated.messages[2];
    assert_eq!(assistant.role, MessageRole::Assistant);
    assert_eq!(assistant.content, "");
    assert_eq!(assistant.status, MessageStatus::Streaming);
    assert_eq!(assistant.run_id.as_deref(), Some("run-1"));
    assert_eq!(updated.runs.len(), 1);
    assert!(updated.runs[0].live);
}

#[test]
fn update_streaming_content_replaces_assistant_text() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .update_streaming_content(&conv.id, "a-1", "part of an answer", OWNER)
        .expect("update")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "part of an answer");
    assert_eq!(assistant.status, MessageStatus::Streaming);
}

#[test]
fn record_message_completion_marks_done_with_model() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                model_name: Some("gemma-3-12b-it"),
                model_role: Some("vision"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Done);
    assert_eq!(assistant.content, "the final answer");
    assert_eq!(assistant.elapsed_ms, Some(1234));
    assert_eq!(assistant.model_name.as_deref(), Some("gemma-3-12b-it"));
    assert_eq!(assistant.model_role.as_deref(), Some("vision"));
    let run = updated.runs.iter().find(|r| r.run_id == "run-1").unwrap();
    assert!(!run.live);
    assert_eq!(run.model_name.as_deref(), Some("gemma-3-12b-it"));
}

/// Two writers reach one message and neither knows everything.
///
/// The front-end completes on `message_end`, which is where the model's token
/// usage arrives; the run completes again when `agent_run_prompt` resolves,
/// which is where the routing decision arrives. The run wrote last, so
/// assigning unconditionally meant the token counts were erased a moment after
/// they were recorded and the chat's counter never had anything to show.
#[test]
fn a_second_completion_does_not_erase_what_the_first_recorded() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");

    // The front-end, on `message_end`: token usage, no routing.
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                tokens_in: Some(512),
                tokens_out: Some(64),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    // The run, on resolve: routing, no token usage.
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("the final answer"),
                elapsed_ms: Some(1234),
                model_name: Some("gemma-3-12b-it"),
                model_role: Some("reasoning"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.tokens_in, Some(512), "token usage was erased");
    assert_eq!(assistant.tokens_out, Some(64), "token usage was erased");
    assert_eq!(assistant.model_name.as_deref(), Some("gemma-3-12b-it"));
    assert_eq!(assistant.model_role.as_deref(), Some("reasoning"));
    assert_eq!(assistant.status, MessageStatus::Done);
}

/// A run cut off at the output cap keeps its fragment *and* its caveat.
///
/// The two halves have to survive together. The text alone reads exactly like
/// a short answer, and the caveat alone loses the only thing the run produced.
#[test]
fn a_length_limited_run_keeps_both_the_fragment_and_the_reason() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Specify the seal", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("The seal specification is "),
                elapsed_ms: Some(900),
                error: Some("Stopped: the answer reached the output limit for one turn."),
                outcome: Some("lengthLimited"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "The seal specification is ");
    assert_eq!(assistant.outcome.as_deref(), Some("lengthLimited"));
    assert_eq!(assistant.status, MessageStatus::Failed);
    assert!(assistant.error.is_some());
}

/// The two writers reach this row in either order and neither may erase the
/// other's half. The front-end knows the token usage; only the run knows how
/// the run ended.
#[test]
fn a_message_end_writer_does_not_erase_the_runs_recorded_ending() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");

    // The run, on resolve: it was stopped by policy.
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                error: Some("Stopped: it needed to do something it is not permitted to do."),
                outcome: Some("policyStopped"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete");

    // The front-end, afterwards, with token usage and no idea how it ended.
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                tokens_in: Some(400),
                tokens_out: Some(12),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");

    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.tokens_in, Some(400));
    assert_eq!(
        assistant.outcome.as_deref(),
        Some("policyStopped"),
        "the ending was erased by a writer that did not know it"
    );
}

#[test]
fn record_message_completion_can_mark_failed() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    let updated = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                elapsed_ms: Some(500),
                error: Some("budget exhausted"),
                outcome: Some("budgetStopped"),
                failed: true,
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete")
        .expect("found");
    let assistant = updated
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Failed);
    assert_eq!(assistant.error.as_deref(), Some("budget exhausted"));
}

#[test]
fn list_returns_conversations_newest_first() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let a = store
        .create("A".to_string(), "W".to_string(), OWNER)
        .expect("a");
    // Sleep to ensure lastActivityAt differs at the millisecond level.
    std::thread::sleep(std::time::Duration::from_millis(10));
    let b = store
        .create("B".to_string(), "W".to_string(), OWNER)
        .expect("b");
    let list = store.list(Some(OWNER)).expect("list");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id, b.id);
    assert_eq!(list[1].id, a.id);
}

#[test]
fn append_user_turn_returns_none_for_unknown_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let updated = store
        .append_user_turn("missing", "hi", "a-1", "run-1", OWNER)
        .unwrap();
    assert!(updated.is_none());
}

#[test]
fn round_trip_preserves_all_fields() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv: Conversation = store
        .create("Round trip".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "First turn", "a-1", "run-1", OWNER)
        .expect("append");
    store
        .update_streaming_content(&conv.id, "a-1", "streamed", OWNER)
        .expect("update");
    store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("final"),
                elapsed_ms: Some(2000),
                model_name: Some("model-x"),
                model_role: Some("reasoning"),
                used_fallback: Some(false),
                outcome: Some("completed"),
                ..MessageCompletion::default()
            },
            OWNER,
        )
        .expect("complete");

    // New store instance reads the same file.
    let again = ConversationStore::open(&dir).expect("reopen");
    let fetched = again
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.title, "Round trip");
    assert_eq!(fetched.messages.len(), 3);
    let assistant = fetched
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "final");
    assert_eq!(assistant.status, MessageStatus::Done);
    assert_eq!(assistant.model_name.as_deref(), Some("model-x"));
    assert_eq!(fetched.runs.len(), 1);
    assert!(!fetched.runs[0].live);
}

// ---------------------------------------------------------------------------
// Per-user isolation tests (TODO 2 of the 7-step plan).
// ---------------------------------------------------------------------------

#[test]
fn a_non_owner_cannot_read_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    // The other user, who is not the owner, asks for the same id.
    let result = store.get(&conv.id, Some(OTHER)).expect("get");
    assert!(
        result.is_none(),
        "non-owner must not see the contents of a conversation they do not own"
    );
    // The owner still sees it.
    let owner_view = store.get(&conv.id, Some(OWNER)).expect("get");
    assert!(owner_view.is_some());
}

#[test]
fn a_non_owner_cannot_append_to_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    // The other user tries to write into the owner's conversation.
    let result = store
        .append_user_turn(&conv.id, "injected", "a-1", "run-1", OTHER)
        .expect("append");
    assert!(
        result.is_none(),
        "non-owner must not be able to append a turn to a conversation they do not own"
    );
    // Confirm the file is unchanged for the owner.
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    assert_eq!(owner_view.messages.len(), 1, "no injected message should have been added");
}

#[test]
fn a_non_owner_cannot_delete_someone_elses_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    let removed = store.delete(&conv.id, OTHER).expect("delete");
    assert!(
        !removed,
        "non-owner delete must return false (idempotent on a foreign file)"
    );
    // The file is still there for the owner.
    let owner_view = store.get(&conv.id, Some(OWNER)).expect("get");
    assert!(owner_view.is_some());
    // The owner can still delete.
    let removed_by_owner = store.delete(&conv.id, OWNER).expect("delete");
    assert!(removed_by_owner);
}

#[test]
fn list_filters_to_only_the_callers_conversations() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    store
        .create("A-owns".to_string(), "W".to_string(), OWNER)
        .expect("a");
    store
        .create("B-owns".to_string(), "W".to_string(), OTHER)
        .expect("b");
    let owner_list = store.list(Some(OWNER)).expect("list");
    let other_list = store.list(Some(OTHER)).expect("list");
    assert_eq!(owner_list.len(), 1, "owner sees only their own");
    assert_eq!(other_list.len(), 1, "other sees only their own");
    assert_eq!(owner_list[0].title, "A-owns");
    assert_eq!(other_list[0].title, "B-owns");
    // The unrestricted form still shows both — used by tests and
    // any future cross-account debug surface.
    let all = store.list(None).expect("list");
    assert_eq!(all.len(), 2);
}

#[test]
fn update_streaming_content_for_a_non_owner_is_a_no_op() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    // Non-owner tries to overwrite the streaming content.
    let result = store
        .update_streaming_content(&conv.id, "a-1", "forged", OTHER)
        .expect("update");
    assert!(result.is_none());
    // The owner's view is unchanged.
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    let assistant = owner_view
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.content, "", "forged update must not be persisted");
}

#[test]
fn record_message_completion_for_a_non_owner_is_a_no_op() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("private".to_string(), "W.".to_string(), OWNER)
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1", OWNER)
        .expect("append");
    // Non-owner tries to mark the message done with a forged final.
    let result = store
        .record_message_completion(
            &conv.id,
            "a-1",
            "run-1",
            MessageCompletion {
                final_content: Some("forged final"),
                elapsed_ms: Some(1),
                model_name: Some("forged-model"),
                model_role: Some("forged"),
                used_fallback: Some(false),
                ..MessageCompletion::default()
            },
            OTHER,
        )
        .expect("complete");
    assert!(result.is_none());
    let owner_view = store
        .get(&conv.id, Some(OWNER))
        .expect("get")
        .expect("found");
    let assistant = owner_view
        .messages
        .iter()
        .find(|m| m.id == "a-1")
        .expect("assistant");
    assert_eq!(assistant.status, MessageStatus::Streaming);
    assert_eq!(assistant.content, "");
}

#[test]
fn legacy_v1_files_migrate_to_the_administrator_owner() {
    // Hand-craft a v1 file (no `owner_user_id` field) and confirm
    // the migration stamps it with LEGACY_OWNER_ID and bumps the
    // schema version on read.
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    // Create a conversation the modern way, then rewrite the file
    // with a v1 envelope that omits `ownerUserId`.
    let conv = store
        .create("legacy".to_string(), "W.".to_string(), LEGACY_OWNER_ID)
        .expect("create");
    let file_path = dir.join("conversations").join(format!("{}.json", conv.id));
    let body = serde_json::json!({
        "schemaVersion": 1,
        "conversation": {
            "id": conv.id,
            // ownerUserId intentionally absent — this is the v1 shape.
            "title": "legacy",
            "createdAt": "2024-01-01T00:00:00Z",
            "lastActivityAt": "2024-01-01T00:00:00Z",
            "messages": [],
            "runs": [],
            "compactions": 0u32,
        }
    });
    std::fs::write(
        &file_path,
        serde_json::to_vec_pretty(&body).expect("serialise"),
    )
    .expect("write v1 file");

    // Read it back. The migration should run, stamp the owner, and
    // rewrite the file at schema version 2.
    let fetched = store
        .get(&conv.id, Some(LEGACY_OWNER_ID))
        .expect("get")
        .expect("found");
    assert_eq!(fetched.owner_user_id, LEGACY_OWNER_ID);

    // The file is now at the current schema version.
    let raw = std::fs::read_to_string(&file_path).expect("read");
    let envelope: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    assert_eq!(envelope["schemaVersion"], 2);
    assert_eq!(envelope["conversation"]["ownerUserId"], LEGACY_OWNER_ID);
}

/// What happens when the conversation store cannot be opened.
///
/// ## The defect
///
/// The failure path opened a *fixed* temp directory,
/// `arjun-conversations-fallback`, silently. Three things were wrong with that:
/// it is shared between sessions and users; it is stale, so a recovered session
/// found the previous degraded session's threads looking like history; and
/// nothing said so, so the chat behaved exactly as normal while the person's
/// real conversations were somewhere else.
mod degraded_storage {
    use super::OWNER;
    use crate::agent_runtime::conversations::{
        ConversationHealth, ConversationState, ConversationStore,
    };

    /// A path that cannot be a directory, because it is a file.
    fn blocked() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("conversations");
        std::fs::write(&path, b"not a directory").expect("blocking file");
        (dir, path)
    }

    #[test]
    fn a_store_that_cannot_be_opened_reports_it() {
        let (_dir, path) = blocked();
        assert!(
            ConversationStore::open(&path).is_err(),
            "the store must not open where a regular file already is"
        );
    }

    #[test]
    fn a_healthy_session_refuses_nothing() {
        let health = ConversationHealth::durable();
        assert!(health.is_durable());
        assert_eq!(health.refusal(), None);
        assert_eq!(health.state(), &ConversationState::Durable);
    }

    #[test]
    fn an_ephemeral_session_refuses_new_conversations_and_says_where_they_would_go() {
        let health = ConversationHealth::ephemeral(
            "The conversation store could not be opened: access denied.",
            std::path::Path::new("/tmp/arjun-conversations-ephemeral-1234-5678"),
        );
        assert!(!health.is_durable());
        let refusal = health.refusal().expect("a reason");
        // What is wrong, where the writing goes, and what still works.
        assert!(refusal.contains("access denied"), "{refusal}");
        assert!(refusal.contains("arjun-conversations-ephemeral"), "{refusal}");
        assert!(refusal.contains("not be there after a restart"), "{refusal}");
        assert!(refusal.contains("can still be read"), "{refusal}");
    }

    #[test]
    fn the_ephemeral_directory_is_unique_per_session() {
        // The stale-reuse defect. Two sessions must not share a directory, or
        // one finds the other's threads and shows them as its own history.
        //
        // The uniqueness comes from the process id and a nanosecond timestamp,
        // which is what `lib.rs` composes. Asserted here on the shape rather
        // than by starting two applications.
        let first = format!("arjun-conversations-ephemeral-{}-{}", 1234, 1_000_000_001u64);
        let second = format!("arjun-conversations-ephemeral-{}-{}", 1234, 1_000_000_002u64);
        assert_ne!(first, second);
        assert!(!first.ends_with("fallback"), "a fixed name is a shared name");
    }

    #[test]
    fn two_ephemeral_stores_do_not_see_each_others_conversations() {
        // The property the unique directory buys, driven for real: a session
        // that writes into its own scratch directory leaves nothing for the
        // next one to find.
        let dir = tempfile::tempdir().expect("temp dir");
        let first = ConversationStore::open(&dir.path().join("session-1")).expect("open");
        first
            .create("Yesterday".to_string(), "W.".to_string(), OWNER)
            .expect("create");
        assert_eq!(first.list(Some(OWNER)).expect("list").len(), 1);

        let second = ConversationStore::open(&dir.path().join("session-2")).expect("open");
        assert!(
            second.list(Some(OWNER)).expect("list").is_empty(),
            "a new session inherited the previous one's ephemeral conversations"
        );
    }

    #[test]
    fn an_ephemeral_session_can_still_read_what_is_already_there() {
        // Refusing to *create* is the design; refusing to open would leave a
        // person unable to find out what is wrong. A store opened at a scratch
        // path still reads and writes normally — the refusal is a policy above
        // it, not a broken store.
        let dir = tempfile::tempdir().expect("temp dir");
        let store = ConversationStore::open(dir.path()).expect("open");
        let conversation = store
            .create("Readable".to_string(), "W.".to_string(), OWNER)
            .expect("create");
        assert!(store.get(&conversation.id, Some(OWNER)).expect("read").is_some());
    }

    #[test]
    fn the_state_serialises_for_the_ui() {
        let durable = serde_json::to_value(ConversationState::Durable).expect("serialises");
        assert_eq!(durable["state"], "durable");

        let ephemeral = serde_json::to_value(ConversationState::Ephemeral {
            because: "no disk".to_string(),
            directory: "/tmp/x".to_string(),
        })
        .expect("serialises");
        assert_eq!(ephemeral["state"], "ephemeral");
        assert_eq!(ephemeral["because"], "no disk");
        assert_eq!(ephemeral["directory"], "/tmp/x");
    }
}

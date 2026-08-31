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
    Conversation, ConversationStore, MessageRole, MessageStatus, LEGACY_OWNER_ID,
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
            Some("the final answer"),
            Some(1234),
            Some("gemma-3-12b-it"),
            Some("vision"),
            Some(false),
            None,
            false,
            None,
            None,
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
            None,
            Some(500),
            None,
            None,
            None,
            Some("budget exhausted"),
            true,
            None,
            None,
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
            Some("final"),
            Some(2000),
            Some("model-x"),
            Some("reasoning"),
            Some(false),
            None,
            false,
            None,
            None,
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
            Some("forged final"),
            Some(1),
            Some("forged-model"),
            Some("forged"),
            Some(false),
            None,
            false,
            None,
            None,
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

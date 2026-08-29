//! Tests for the conversation store.
//!
//! These exercise the on-disk CRUD: open, create, append a turn, update
//! streaming content, mark a message done, list, and round-trip read.
//! They run on a temp directory so they do not pollute the application
//! data folder.

use std::env;

use crate::agent_runtime::conversations::{
    Conversation, ConversationStore, MessageRole, MessageStatus,
};

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
    assert!(store.list().unwrap().is_empty());
    assert!(store.get("nonexistent").unwrap().is_none());
}

#[test]
fn create_persists_a_conversation_with_one_welcome_message() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string())
        .expect("create");
    assert_eq!(conv.title, "Test");
    assert_eq!(conv.messages.len(), 1);
    assert_eq!(conv.messages[0].role, MessageRole::System);
    assert_eq!(conv.messages[0].content, "Welcome.");
    assert_eq!(conv.messages[0].status, MessageStatus::Done);

    // Read it back.
    let fetched = store.get(&conv.id).expect("get").expect("found");
    assert_eq!(fetched.id, conv.id);
    assert_eq!(fetched.title, "Test");
}

#[test]
fn append_user_turn_creates_user_and_streaming_assistant_messages() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv = store
        .create("Test".to_string(), "Welcome.".to_string())
        .expect("create");
    let updated = store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1")
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
        .create("Test".to_string(), "Welcome.".to_string())
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1")
        .expect("append");
    let updated = store
        .update_streaming_content(&conv.id, "a-1", "part of an answer")
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
        .create("Test".to_string(), "Welcome.".to_string())
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1")
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
        .create("Test".to_string(), "Welcome.".to_string())
        .expect("create");
    store
        .append_user_turn(&conv.id, "Hello", "a-1", "run-1")
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
    let a = store.create("A".to_string(), "W".to_string()).expect("a");
    // Sleep to ensure lastActivityAt differs at the millisecond level.
    std::thread::sleep(std::time::Duration::from_millis(10));
    let b = store.create("B".to_string(), "W".to_string()).expect("b");
    let list = store.list().expect("list");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id, b.id);
    assert_eq!(list[1].id, a.id);
}

#[test]
fn append_user_turn_returns_none_for_unknown_conversation() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let updated = store.append_user_turn("missing", "hi", "a-1", "run-1").unwrap();
    assert!(updated.is_none());
}

#[test]
fn round_trip_preserves_all_fields() {
    let dir = temp_dir();
    let store = ConversationStore::open(&dir).expect("open");
    let conv: Conversation = store
        .create("Round trip".to_string(), "W.".to_string())
        .expect("create");
    store
        .append_user_turn(&conv.id, "First turn", "a-1", "run-1")
        .expect("append");
    store
        .update_streaming_content(&conv.id, "a-1", "streamed")
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
        )
        .expect("complete");

    // New store instance reads the same file.
    let again = ConversationStore::open(&dir).expect("reopen");
    let fetched = again.get(&conv.id).expect("get").expect("found");
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

//! Tauri commands for chat conversations.
//!
//! These commands are the *front* of the conversation layer: the chat
//! surface calls `agent_create_conversation` once on first open, then
//! `agent_append_turn` for every user message, and `agent_get_conversation` /
//! `agent_list_conversations` to render the sidebar.
//!
//! The relationship between a conversation and the existing run machinery is
//! this: `agent_start_run` continues to start a run, and from now on it
//! returns the new `runId` paired with a freshly-created `conversationId`.
//! The chat surface treats that first turn the same as any later one — the
//! commands below are the only difference between "first message" and
//! "follow-up", and the chat surface does not need to care which it is.

use std::sync::Arc;

use tauri::State;

use crate::agent_runtime::conversations::{
    Conversation, ConversationStore, Message, MessageStatus, RunToConversation,
};

/// Tauri-managed wrapper around the conversation store.
pub struct ConversationsState(pub Arc<ConversationStore>);

/// Tauri-managed wrapper around the run→conversation index.
pub struct RunToConversationState(pub Arc<RunToConversation>);

/// Create a new conversation with one system welcome message.
#[tauri::command]
pub fn agent_create_conversation(
    title: String,
    welcome: Option<String>,
    state: State<'_, ConversationsState>,
) -> Result<Conversation, String> {
    let welcome = welcome.unwrap_or_else(|| {
        "Arjun is ready. Ask anything; nothing leaves this machine.".to_string()
    });
    state
        .0
        .create(title, welcome)
        .map_err(|error| format!("the conversation could not be created: {error}"))
}

/// Read one conversation by id.
///
/// Returns `Ok(None)` for a conversation that does not exist, which the
/// front-end treats as a fresh empty conversation rather than an error.
#[tauri::command]
pub fn agent_get_conversation(
    id: String,
    state: State<'_, ConversationsState>,
) -> Result<Option<Conversation>, String> {
    state.0.get(&id).map_err(|e| e.to_string())
}

/// All conversations, newest first by `lastActivityAt`.
#[tauri::command]
pub fn agent_list_conversations(
    state: State<'_, ConversationsState>,
) -> Result<Vec<Conversation>, String> {
    state.0.list().map_err(|e| e.to_string())
}

/// The user has sent a new message in an existing conversation.
///
/// This command:
///  1. Persists a new user `Message` and a fresh streaming assistant
///     `Message` in the conversation, both stamped with the supplied
///     `runId`.
///  2. Records the `(runId → conversationId)` mapping in the in-memory
///     index so subsequent `agent://event` messages for the run can be
///     routed back to the right conversation's assistant cell.
///
/// The actual `agent_start_run` is *not* triggered here: the front-end
/// already has the user's prompt and calls `agent_start_run` itself so the
/// existing event-subscription machinery is unchanged. This command only
/// persists the user message and reserves the assistant cell.
/// Reserve the user message and the streaming assistant cell for a new turn.
///
/// The assistant `message_id` is supplied by the front-end (which
/// already chose a stable id it can correlate `message_end` with) and
/// is named `message_id` here so the wire form is `messageId` — the
/// same name the rest of the conversation API uses. The role
/// (user or assistant) is decided by the store, not by the argument
/// name.
#[tauri::command]
pub fn agent_append_turn(
    conversation_id: String,
    run_id: String,
    message_id: String,
    user_prompt: String,
    conversations: State<'_, ConversationsState>,
    run_to_conversation: State<'_, RunToConversationState>,
) -> Result<Option<Conversation>, String> {
    let updated = conversations
        .0
        .append_user_turn(&conversation_id, &user_prompt, &message_id, &run_id)
        .map_err(|e| e.to_string())?;
    if updated.is_some() {
        run_to_conversation.0.bind(&run_id, &conversation_id);
    }
    Ok(updated)
}

/// Streaming-content update for an in-flight assistant message.
///
/// Called by the front-end as `message_update` events arrive. The front-end
/// keeps the live `content` in component state and writes the snapshot
/// through this command so a remount can pick up the latest in-progress
/// text from disk rather than from the (best-effort) event channel.
#[tauri::command]
pub fn agent_update_streaming_content(
    conversation_id: String,
    message_id: String,
    content: String,
    state: State<'_, ConversationsState>,
) -> Result<Option<Conversation>, String> {
    state
        .0
        .update_streaming_content(&conversation_id, &message_id, &content)
        .map_err(|e| e.to_string())
}

/// Mark an assistant message as finished (clean or failed).
///
/// Called by the front-end on `message_end` or run completion. The
/// `(runId → conversationId)` mapping is dropped on success: the run is
/// over and the index entry would just leak.
#[tauri::command]
pub fn agent_complete_message(
    conversation_id: String,
    message_id: String,
    run_id: String,
    final_content: Option<String>,
    elapsed_ms: Option<u64>,
    model_name: Option<String>,
    model_role: Option<String>,
    used_fallback: Option<bool>,
    error: Option<String>,
    failed: bool,
    conversations: State<'_, ConversationsState>,
    run_to_conversation: State<'_, RunToConversationState>,
) -> Result<Option<Conversation>, String> {
    let updated = conversations
        .0
        .record_message_completion(
            &conversation_id,
            &message_id,
            &run_id,
            final_content.as_deref(),
            elapsed_ms,
            model_name.as_deref(),
            model_role.as_deref(),
            used_fallback,
            error.as_deref(),
            failed,
        )
        .map_err(|e| e.to_string())?;
    run_to_conversation.0.unbind(&run_id);
    Ok(updated)
}

/// Reverse-lookup: which conversation does this run belong to?
///
/// Used by the front-end when an event arrives on `agent://event` to figure
/// out which `Conversation` (and therefore which `Message`) to update. The
/// index is in-memory; on a remount the index is rebuilt lazily as
/// `agent_append_turn` is called again.
#[tauri::command]
pub fn agent_run_conversation(
    run_id: String,
    run_to_conversation: State<'_, RunToConversationState>,
) -> Option<String> {
    run_to_conversation.0.lookup(&run_id)
}

/// Re-export for callers that need the types.
pub use crate::agent_runtime::conversations::{MessageRole, RunMeta};

/// Helper to construct a fresh assistant message id from the front-end.
pub fn new_assistant_message_id() -> String {
    format!("a-{}", uuid::Uuid::new_v4())
}

/// Helper to read the final status of a message without having to plumb a
/// full Conversation through every callsite that needs only the status.
pub fn is_terminal(status: MessageStatus) -> bool {
    matches!(status, MessageStatus::Done | MessageStatus::Failed)
}

/// Read-only access to a single message by id.
///
/// Used by the front-end on remount: the durable event channel has the
/// `runId` of an in-flight run, the in-memory index can be empty, and
/// reading from disk is the honest answer.
#[tauri::command]
pub fn agent_get_message(
    conversation_id: String,
    message_id: String,
    state: State<'_, ConversationsState>,
) -> Result<Option<Message>, String> {
    let conv = state.0.get(&conversation_id).map_err(|e| e.to_string())?;
    Ok(conv.and_then(|c| c.messages.into_iter().find(|m| m.id == message_id)))
}

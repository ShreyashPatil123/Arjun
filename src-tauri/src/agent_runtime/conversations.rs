//! Persistent chat threads.
//!
//! A [`Conversation`] owns an ordered list of [`Message`]s and the list of
//! [`RunMeta`] entries that produced the assistant messages inside it. The
//! current `agent_runtime` machinery is still keyed by `runId`; the
//! conversation layer sits on top of that and gives the UI a place to keep
//! the user-visible transcript between runs.
//!
//! ## Why this is a file-per-conversation store and not a SQL table
//!
//! The audit log is append-only and hash-chained, and intentionally not the
//! place for chat-shaped content (a 50-turn conversation is a 50 KB JSON
//! blob; the audit chain's value is that entries are small and tamper-evident
//! rather than that they hold everything). SQLite would be the right
//! destination for a multi-user product, but here the working set is one
//! signed-in user on one machine, and a JSON file per conversation is
//! honest, simple and easy to copy out for a bug report.
//!
//! The store is append-and-rewrite: every save rewrites the file under a
//! `.tmp` sibling, fsyncs, then renames over the old one. A reader that
//! opens during a write either sees the old version or the new one, never a
//! half-written file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a participant said in a conversation.
///
/// User messages are produced by the human, assistant messages by an
/// `agent_runtime` Run, system messages are seeded by the UI (welcome,
/// refusals) and never appear in the user-editable transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: MessageRole,
    pub content: String,
    pub status: MessageStatus,
    /// Set on an assistant message, naming the run that produced it.
    /// Absent on user and system messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    /// Wall-clock ms the assistant took, set on `done` assistant messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    /// Where the run that produced the message stopped, when it did not
    /// finish cleanly. Verbatim from the runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Model name taken from the run's `RoutingDecision`, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Model role (vision, reasoning, etc.) when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_role: Option<String>,
    /// True if the routed model was a fallback.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_fallback: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageStatus {
    /// An assistant message that has not yet received its `message_end`.
    Streaming,
    /// Final, clean ending.
    Done,
    /// Run finished but did not produce an answer.
    Failed,
}

/// One row in the conversation's `runs[]` list. Carries enough to render
/// the assistant message and to open the per-run inspector, but not the
/// full run record — that lives in the existing `tasks` store.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunMeta {
    pub run_id: String,
    pub message_id: String,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    /// The model that actually answered, as the run's `RoutingDecision`
    /// named it. `None` while the run is still starting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Live while the run is in flight.
    pub live: bool,
}

/// One chat thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    /// What the conversation is about, in a few words. The UI shows it in the
    /// sidebar; a future build can have the model suggest a title.
    pub title: String,
    pub created_at: String,
    pub last_activity_at: String,
    pub messages: Vec<Message>,
    pub runs: Vec<RunMeta>,
    /// Number of compactions across all runs in this conversation, summed.
    /// Surfaced in the context chip.
    pub compactions: u32,
}

/// The on-disk envelope. Carries a schema version so an older client can
/// read a newer file (and a newer one can migrate, when we ever need to).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationFile {
    schema_version: u32,
    conversation: Conversation,
}

const SCHEMA_VERSION: u32 = 1;

/// Where the conversations live on disk.
pub struct ConversationStore {
    root: PathBuf,
}

impl ConversationStore {
    /// Open the store at `<app_data_dir>/conversations/`. The directory is
    /// created on first use; an empty store is a directory without files.
    pub fn open(app_data_dir: &Path) -> std::io::Result<Self> {
        let root = app_data_dir.join("conversations");
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn file_path(&self, id: &str) -> PathBuf {
        // The id is a uuid, so the directory does not need to defend against
        // path traversal; assert it just in case a future caller changes that.
        debug_assert!(
            id.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "conversation id is not a safe filename: {id}",
        );
        self.root.join(format!("{id}.json"))
    }

    /// Read one conversation. Returns `Ok(None)` if the id is unknown,
    /// which is the honest answer rather than an error.
    pub fn get(&self, id: &str) -> std::io::Result<Option<Conversation>> {
        let path = self.file_path(id);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let parsed: ConversationFile = serde_json::from_slice(&bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(Some(parsed.conversation))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Every conversation in the store, newest first by `lastActivityAt`.
    /// Conversations that fail to parse are skipped, not surfaced, on the
    /// principle that a corrupt one entry should not hide the others.
    pub fn list(&self) -> std::io::Result<Vec<Conversation>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let parsed: ConversationFile = match serde_json::from_slice(&bytes) {
                Ok(f) => f,
                Err(_) => continue,
            };
            out.push(parsed.conversation);
        }
        out.sort_by(|a, b| b.last_activity_at.cmp(&a.last_activity_at));
        Ok(out)
    }

    /// Write a conversation atomically. The file is written to `<id>.json.tmp`,
    /// fsynced, and renamed over `<id>.json`. A reader that opens during the
    /// write sees either the old or the new file, never a half-written one.
    pub fn save(&self, conversation: &Conversation) -> std::io::Result<()> {
        let envelope = ConversationFile {
            schema_version: SCHEMA_VERSION,
            conversation: conversation.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&envelope)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let final_path = self.file_path(&conversation.id);
        let tmp_path = final_path.with_extension("json.tmp");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Create a new conversation with one system welcome message.
    pub fn create(&self, title: String, system_welcome: String) -> std::io::Result<Conversation> {
        let now = chrono::Utc::now().to_rfc3339();
        let id = uuid::Uuid::new_v4().to_string();
        let welcome_id = format!("sys-{}", uuid::Uuid::new_v4());
        let welcome = Message {
            id: welcome_id,
            conversation_id: id.clone(),
            role: MessageRole::System,
            content: system_welcome,
            status: MessageStatus::Done,
            run_id: None,
            created_at: now.clone(),
            completed_at: Some(now.clone()),
            elapsed_ms: None,
            error: None,
            model_name: None,
            model_role: None,
            used_fallback: None,
        };
        let conversation = Conversation {
            id: id.clone(),
            title,
            created_at: now.clone(),
            last_activity_at: now,
            messages: vec![welcome],
            runs: Vec::new(),
            compactions: 0,
        };
        self.save(&conversation)?;
        Ok(conversation)
    }

    /// Append a user message and an in-progress assistant message to a
    /// conversation. Returns the new `Conversation` if the id is known.
    ///
    /// The assistant message starts as `Streaming`; the chat surface flips
    /// it to `Done` on `message_end` via `record_message_completion`.
    pub fn append_user_turn(
        &self,
        id: &str,
        user_prompt: &str,
        assistant_message_id: &str,
        run_id: &str,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id)? else {
            return Ok(None);
        };
        let now = chrono::Utc::now().to_rfc3339();
        let user_id = format!("u-{}", uuid::Uuid::new_v4());
        let user_msg = Message {
            id: user_id,
            conversation_id: id.to_string(),
            role: MessageRole::User,
            content: user_prompt.to_string(),
            status: MessageStatus::Done,
            run_id: None,
            created_at: now.clone(),
            completed_at: Some(now.clone()),
            elapsed_ms: None,
            error: None,
            model_name: None,
            model_role: None,
            used_fallback: None,
        };
        let assistant_msg = Message {
            id: assistant_message_id.to_string(),
            conversation_id: id.to_string(),
            role: MessageRole::Assistant,
            content: String::new(),
            status: MessageStatus::Streaming,
            run_id: Some(run_id.to_string()),
            created_at: now.clone(),
            completed_at: None,
            elapsed_ms: None,
            error: None,
            model_name: None,
            model_role: None,
            used_fallback: None,
        };
        conversation.messages.push(user_msg);
        conversation.messages.push(assistant_msg);
        conversation.runs.push(RunMeta {
            run_id: run_id.to_string(),
            message_id: assistant_message_id.to_string(),
            started_at: now.clone(),
            finished_at: None,
            model_name: None,
            live: true,
        });
        conversation.last_activity_at = now;
        self.save(&conversation)?;
        Ok(Some(conversation))
    }

    /// Replace the streaming content of an in-flight assistant message.
    /// Idempotent: the message is identified by `message_id`.
    pub fn update_streaming_content(
        &self,
        id: &str,
        message_id: &str,
        content: &str,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id)? else {
            return Ok(None);
        };
        if let Some(msg) = conversation
            .messages
            .iter_mut()
            .find(|m| m.id == message_id && m.role == MessageRole::Assistant)
        {
            msg.content = content.to_string();
        }
        conversation.last_activity_at = chrono::Utc::now().to_rfc3339();
        self.save(&conversation)?;
        Ok(Some(conversation))
    }

    /// Mark an assistant message as done (or failed) and set the elapsed
    /// time and final error, if any. Returns the updated conversation.
    pub fn record_message_completion(
        &self,
        id: &str,
        message_id: &str,
        run_id: &str,
        final_content: Option<&str>,
        elapsed_ms: Option<u64>,
        model_name: Option<&str>,
        model_role: Option<&str>,
        used_fallback: Option<bool>,
        error: Option<&str>,
        failed: bool,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id)? else {
            return Ok(None);
        };
        let now = chrono::Utc::now().to_rfc3339();
        if let Some(msg) = conversation
            .messages
            .iter_mut()
            .find(|m| m.id == message_id && m.role == MessageRole::Assistant)
        {
            if let Some(c) = final_content {
                msg.content = c.to_string();
            }
            msg.status = if failed {
                MessageStatus::Failed
            } else {
                MessageStatus::Done
            };
            msg.completed_at = Some(now.clone());
            msg.elapsed_ms = elapsed_ms;
            msg.model_name = model_name.map(str::to_string);
            msg.model_role = model_role.map(str::to_string);
            msg.used_fallback = used_fallback;
            msg.error = error.map(str::to_string);
        }
        if let Some(run) = conversation.runs.iter_mut().find(|r| r.run_id == run_id) {
            run.finished_at = Some(now.clone());
            run.live = false;
            run.model_name = model_name.map(str::to_string);
        }
        conversation.last_activity_at = now;
        self.save(&conversation)?;
        Ok(Some(conversation))
    }
}

/// In-memory index from `runId` to `conversationId`. The runtime has
/// `runId`s; the conversation layer has `conversationId`s. When a Run is
/// in flight, the chat surface needs to know which conversation the run
/// belongs to so it can update the right assistant message.
///
/// Kept in a `Mutex<HashMap>` because writes are rare (one entry per
/// `agent_append_turn` or `agent_start_run`) and reads are common (every
/// `message_update` event).
pub struct RunToConversation {
    inner: std::sync::Mutex<HashMap<String, String>>,
}

impl Default for RunToConversation {
    fn default() -> Self {
        Self::new()
    }
}

impl RunToConversation {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn bind(&self, run_id: &str, conversation_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            map.insert(run_id.to_string(), conversation_id.to_string());
        }
    }

    pub fn unbind(&self, run_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            map.remove(run_id);
        }
    }

    pub fn lookup(&self, run_id: &str) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|map| map.get(run_id).cloned())
    }
}

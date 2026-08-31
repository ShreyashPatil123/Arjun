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
    /// Token counts from the model (assistant messages only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_in: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_out: Option<u64>,
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
    /// The account id (`User::id`) of the user who created the
    /// conversation. Every read, list, and write is filtered by
    /// this id — a different user asking for this conversation
    /// gets `None`, not the contents. See TODO 2 of the
    /// 7-step plan: per-user data/history isolation.
    pub owner_user_id: String,
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

/// The v1 envelope, kept for migration only. A v1 file has no
/// `ownerUserId` on the conversation. The migration reads as
/// `ConversationFileV1`, sets the owner to `LEGACY_OWNER_ID`, and
/// re-serialises at the current schema version.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationFileV1 {
    schema_version: u32,
    conversation: ConversationV1,
}

/// The v1 conversation: same as v2 minus `owner_user_id`. The
/// struct lives only to make the v1 → v2 migration parseable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversationV1 {
    id: String,
    title: String,
    created_at: String,
    last_activity_at: String,
    messages: Vec<Message>,
    runs: Vec<RunMeta>,
    compactions: u32,
}

/// Bumped from 1 to 2 when `owner_user_id` was added in TODO 2.
/// Files at version 1 are migrated on first read — the owner is
/// treated as the seed administrator (S. Kulkarni, id
/// "modeladmin"), which matches the pre-TODO-2 reality where
/// every conversation lived in a single global store. After the
/// migration the file is written back at the current schema
/// version, so the migration runs once per file.
const SCHEMA_VERSION: u32 = 2;

/// The id of the account that owned conversations before
/// per-user isolation. Used only as the migration target for
/// pre-TODO-2 files; new conversations always get the real
/// session user id.
pub const LEGACY_OWNER_ID: &str = "modeladmin";

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
    ///
    /// The `owner_user_id` filter, when supplied, is the per-user
    /// isolation boundary from TODO 2 of the 7-step plan. A request
    /// from a different user returns `Ok(None)` rather than the
    /// conversation — the file is on disk, but it is not for the
    /// caller. Passing `None` is the unrestricted form, used by
    /// internal callers that have already authorised the read.
    pub fn get(
        &self,
        id: &str,
        owner_user_id: Option<&str>,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.read_raw(id)? else {
            return Ok(None);
        };
        if let Some(owner) = owner_user_id {
            if conversation.owner_user_id != owner {
                return Ok(None);
            }
        }
        // Persist a migrated v1 file back to disk at the new schema
        // version so the migration is one-shot per file. v1 files
        // have no `owner_user_id`; we set it to the legacy owner id
        // before the next write.
        Ok(Some(conversation))
    }

    /// The migration step, separated so `list` can reuse it.
    /// v1 → v2 stamps the legacy owner id and rewrites the file
    /// atomically. Idempotent: a v2 file is left alone.
    fn migrate_in_place(&self, id: &str, file: &mut ConversationFile) {
        if file.schema_version >= SCHEMA_VERSION {
            return;
        }
        if file.schema_version == 1 {
            file.conversation.owner_user_id = LEGACY_OWNER_ID.to_string();
        }
        file.schema_version = SCHEMA_VERSION;
        // Best-effort write; if it fails, the in-memory state is
        // still correct and the next save will retry.
        let _ = self.save(&file.conversation);
    }

    /// Read a conversation file without applying any owner filter.
    /// Used by `list` (which filters after reading) and by internal
    /// callers (admins, audits). v1 files are migrated in place.
    fn read_raw(&self, id: &str) -> std::io::Result<Option<Conversation>> {
        let path = self.file_path(id);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let mut parsed = self.parse_envelope(&bytes)?;
                if parsed.schema_version < SCHEMA_VERSION {
                    let id_owned = parsed.conversation.id.clone();
                    self.migrate_in_place(&id_owned, &mut parsed);
                }
                Ok(Some(parsed.conversation))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Parse a file as either a v1 or v2 envelope. A v1 file has no
    /// `ownerUserId` on the conversation; this is a v1-only shape
    /// that the migration stamps. The struct is otherwise the
    /// same — version 2 added one optional field.
    fn parse_envelope(&self, bytes: &[u8]) -> std::io::Result<ConversationFile> {
        // First try the current shape; on a missing-field error,
        // fall back to a v1 envelope that allows absent
        // `ownerUserId`. The fallback is the migration path; in
        // steady state (v2-only) only the first branch runs.
        match serde_json::from_slice::<ConversationFile>(bytes) {
            Ok(env) => Ok(env),
            Err(serde_err) => {
                let v1: ConversationFileV1 = serde_json::from_slice(bytes).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, serde_err)
                })?;
                Ok(ConversationFile {
                    schema_version: v1.schema_version,
                    conversation: Conversation {
                        id: v1.conversation.id,
                        owner_user_id: String::new(),
                        title: v1.conversation.title,
                        created_at: v1.conversation.created_at,
                        last_activity_at: v1.conversation.last_activity_at,
                        messages: v1.conversation.messages,
                        runs: v1.conversation.runs,
                        compactions: v1.conversation.compactions,
                    },
                })
            }
        }
    }

    /// Delete a conversation by id.
    ///
    /// Returns `true` when a file was removed, `false` when the id was
    /// not present in the store *or* is owned by a different user.
    /// The store does not raise on "not found" — the front-end treats
    /// a successful delete of a missing entry as idempotent, the
    /// same way an `rm` of a missing path is a no-op.
    ///
    /// The owner check is the per-user isolation boundary from
    /// TODO 2: a non-owner can neither read nor delete the file.
    ///
    /// The file is removed with `remove_file`; there is no tmp/rename
    /// dance because the deletion is not a partial-write hazard. The
    /// directory is left in place.
    pub fn delete(&self, id: &str, owner_user_id: &str) -> std::io::Result<bool> {
        let Some(conversation) = self.get(id, Some(owner_user_id))? else {
            return Ok(false);
        };
        let path = self.file_path(&conversation.id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Every conversation in the store, newest first by `lastActivityAt`.
    /// Conversations that fail to parse are skipped, not surfaced, on the
    /// principle that a corrupt one entry should not hide the others.
    ///
    /// The `owner_user_id` filter is the per-user isolation boundary
    /// from TODO 2: a non-Administrator sees only their own
    /// conversations. `None` is the unrestricted form, used by the
    /// administrator's "all conversations" view and by tests.
    pub fn list(&self, owner_user_id: Option<&str>) -> std::io::Result<Vec<Conversation>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(mut conversation) = self
                .read_path(&path)
                .ok()
                .flatten()
            else {
                continue;
            };
            if let Some(owner) = owner_user_id {
                if conversation.owner_user_id != owner {
                    continue;
                }
            }
            out.push(conversation);
        }
        out.sort_by(|a, b| b.last_activity_at.cmp(&a.last_activity_at));
        Ok(out)
    }

    /// Read a single file by full path, applying the v1→v2 migration
    /// in place. Split out from `read_raw` so `list` does not have
    /// to know the file naming convention.
    fn read_path(&self, path: &Path) -> std::io::Result<Option<Conversation>> {
        let bytes = std::fs::read(path)?;
        let mut parsed = self.parse_envelope(&bytes)?;
        if parsed.schema_version < SCHEMA_VERSION {
            let id_owned = parsed.conversation.id.clone();
            self.migrate_in_place(&id_owned, &mut parsed);
        }
        Ok(Some(parsed.conversation))
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
    ///
    /// `owner_user_id` is required: every conversation belongs to
    /// exactly one user, and the create call is the moment that
    /// ownership is decided. The caller is the Tauri command, which
    /// always has a session, so this is never `None` at the IPC
    /// boundary — but the helper still takes an `Option` so the
    /// internal tests can call it without a session.
    pub fn create(
        &self,
        title: String,
        system_welcome: String,
        owner_user_id: &str,
    ) -> std::io::Result<Conversation> {
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
            tokens_in: None,
            tokens_out: None,
        };
        let conversation = Conversation {
            id: id.clone(),
            owner_user_id: owner_user_id.to_string(),
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
    /// conversation. Returns the new `Conversation` if the id is known
    /// and is owned by `owner_user_id`. A request from a different
    /// user returns `Ok(None)`, the same shape as an unknown id —
    /// the surface cannot tell the difference.
    ///
    /// The assistant message starts as `Streaming`; the chat surface flips
    /// it to `Done` on `message_end` via `record_message_completion`.
    pub fn append_user_turn(
        &self,
        id: &str,
        user_prompt: &str,
        assistant_message_id: &str,
        run_id: &str,
        owner_user_id: &str,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id, Some(owner_user_id))? else {
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
            tokens_in: None,
            tokens_out: None,
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
            tokens_in: None,
            tokens_out: None,
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
    /// Idempotent: the message is identified by `message_id`. A request
    /// from a non-owner returns `Ok(None)`, the same shape as an
    /// unknown id.
    pub fn update_streaming_content(
        &self,
        id: &str,
        message_id: &str,
        content: &str,
        owner_user_id: &str,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id, Some(owner_user_id))? else {
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
    /// A request from a non-owner returns `Ok(None)`, the same shape
    /// as an unknown id.
    #[allow(clippy::too_many_arguments)]
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
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        owner_user_id: &str,
    ) -> std::io::Result<Option<Conversation>> {
        let Some(mut conversation) = self.get(id, Some(owner_user_id))? else {
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
            msg.tokens_in = tokens_in;
            msg.tokens_out = tokens_out;
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

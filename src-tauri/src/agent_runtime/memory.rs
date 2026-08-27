//! What the workbench is allowed to remember, and for whom.
//!
//! ## Why memory is a policy problem, not a storage problem
//!
//! Every agent product eventually adds memory, and the usual shape is a single
//! store that everything reads and everything writes. That shape is wrong here
//! for a reason that has nothing to do with retrieval quality: this workbench
//! reads vendor negotiations, unreleased designs and internal correspondence,
//! and the people who may read one of those are not the people who may read the
//! next. A memory that a run for one department writes and a run for another
//! reads has moved confidential material across a boundary somebody agreed to,
//! and it has done so without a refusal, a prompt, or an audit line — because
//! from the store's point of view nothing unusual happened.
//!
//! So the store here is not a cache with a key. Every item carries the
//! classification of what it came from and an access list, and the reader is
//! checked against both. Items are also *scoped*, and a scope is not a
//! namespacing convenience:
//!
//! - [`MemoryScope::Run`] — one task's own state. Dies with the task.
//! - [`MemoryScope::Workspace`] — terminology, templates and stable facts for
//!   one project. Read by every run *on that project* and by no other.
//! - [`MemoryScope::User`] — a person's preferences. Theirs alone.
//!
//! ## The promotion rule
//!
//! The dangerous operation is not writing, it is *promoting*: taking something
//! a run learned from a document and putting it somewhere later runs will read.
//! A run that reads a confidential tender and writes "the unit price is ₹4.2
//! crore" into workspace memory has published it, quietly and permanently.
//!
//! So a promotion out of run scope is refused whenever the value came from a
//! document that is not ordinary internal material, unless a person explicitly
//! approved that specific promotion. Never automatically, never because the
//! model judged it useful, and never because the value "looked like a fact
//! rather than a quote" — a paraphrase of a confidential figure is the
//! confidential figure.
//!
//! ## Why no vector store
//!
//! There is none, here or anywhere else in this crate. Recall is by scope and
//! key over a small local table. An embedding index would be a second copy of
//! the material in a form no reviewer can read, sitting outside the
//! classification checks above, and the volume of memory a workbench accumulates
//! does not need one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::identity::{Role, Session};
use crate::policy::Classification;

/// Who a memory item belongs to.
///
/// Untagged variants would let a workspace item deserialise as a run item on a
/// field-name collision, which is precisely the boundary this type exists to
/// hold, so the tag is explicit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MemoryScope {
    /// One task's own state. Never read by another run.
    Run { run_id: String },
    /// One project's shared, approved knowledge.
    Workspace { project_id: String },
    /// One person's preferences.
    User { user_id: String },
}

impl MemoryScope {
    /// True for scopes that outlive the run that wrote them.
    ///
    /// The test the promotion rule turns on: writing into run scope is private
    /// to that task and reversible by ending it, and writing anywhere else is
    /// publication.
    pub fn is_durable(&self) -> bool {
        !matches!(self, MemoryScope::Run { .. })
    }

    /// The project this scope belongs to, if any.
    pub fn project(&self) -> Option<&str> {
        match self {
            MemoryScope::Workspace { project_id } => Some(project_id.as_str()),
            _ => None,
        }
    }

    /// The filename this scope's items are stored under.
    fn file_name(&self) -> String {
        match self {
            MemoryScope::Run { run_id } => format!("run-{}.json", sanitise(run_id)),
            MemoryScope::Workspace { project_id } => {
                format!("workspace-{}.json", sanitise(project_id))
            }
            MemoryScope::User { user_id } => format!("user-{}.json", sanitise(user_id)),
        }
    }
}

/// Keeps an identifier to one safe path component.
///
/// Project and user ids come from configuration and from the UI, so neither is
/// a value this process generated. Anything that is not a letter, digit, dash or
/// underscore becomes an underscore, which cannot name a parent directory.
fn sanitise(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// What kind of thing is being remembered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryKind {
    /// A run's own goal, stage and next action.
    RunState,
    /// Something the run decided and is bound by.
    Decision,
    /// An approved term and what it means on this project.
    Terminology,
    /// A document or artifact template this project uses.
    Template,
    /// A stable fact about the project — a unit convention, a site name.
    ProjectFact,
    /// How one person likes their work presented.
    Preference,
}

impl MemoryKind {
    /// Which scopes this kind may legitimately live in.
    ///
    /// A `Preference` in workspace scope would be one person's taste imposed on
    /// a project; a `Terminology` in user scope would be a shared definition
    /// only one person sees. Both are the kind of mistake that is invisible
    /// afterwards, so they are refused at the point of writing.
    fn permitted_in(self, scope: &MemoryScope) -> bool {
        match self {
            MemoryKind::RunState | MemoryKind::Decision => !scope.is_durable(),
            MemoryKind::Terminology | MemoryKind::Template | MemoryKind::ProjectFact => {
                matches!(scope, MemoryScope::Workspace { .. })
            }
            MemoryKind::Preference => matches!(scope, MemoryScope::User { .. }),
        }
    }
}

/// Where a remembered value came from.
///
/// The field the promotion rule reads. A value with no traceable origin is
/// treated as though it came from the operator, because that is the only source
/// that could have supplied something the system did not read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "camelCase")]
pub enum MemorySource {
    /// A person typed it.
    Operator { user_id: String },
    /// A run produced it from its own reasoning, not by quoting a document.
    Run { run_id: String },
    /// It came out of an indexed document. Carries what that document was
    /// classified as, which is what decides whether it may be promoted.
    Document {
        document_sha256: String,
        page: u32,
        classification: Classification,
    },
}

impl MemorySource {
    /// The classification the *source* carried, which may exceed the item's own.
    fn source_classification(&self) -> Option<Classification> {
        match self {
            MemorySource::Document { classification, .. } => Some(*classification),
            _ => None,
        }
    }
}

/// Who may read an item.
///
/// Held on the item rather than derived at read time. Deriving it would mean a
/// later change to the clearance table silently widening what has already been
/// stored, and the whole point of writing the list down is that a reviewer can
/// see what it was when the decision was taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Acl {
    /// Roles cleared to read this. Empty means nobody, which is the correct
    /// reading of an item whose clearance was never established.
    pub cleared_roles: Vec<Role>,
    /// The project this is confined to. `None` only for user-scope items.
    pub project_id: Option<String>,
    /// The person this belongs to, for user-scope items.
    pub owner: Option<String>,
}

impl Acl {
    /// The list a value of this classification gets by default.
    pub fn for_classification(classification: Classification, project_id: Option<&str>) -> Self {
        Self {
            cleared_roles: classification.cleared_roles().to_vec(),
            project_id: project_id.map(str::to_string),
            owner: None,
        }
    }

    /// Whether this reader, working on this project, may see the item.
    ///
    /// Both halves are required and neither implies the other: holding the role
    /// for a vendor negotiation does not entitle somebody to *another project's*
    /// vendor negotiation, and being on the project does not confer the role.
    pub fn admits(&self, session: &Session, project_id: Option<&str>) -> bool {
        if let Some(owner) = &self.owner {
            if owner != &session.user.id {
                return false;
            }
        }
        if let Some(confined_to) = &self.project_id {
            // A reader who named no project is not thereby cleared for every
            // project. Absence of a project is not a wildcard.
            if project_id != Some(confined_to.as_str()) {
                return false;
            }
        }
        self.cleared_roles
            .iter()
            .any(|role| session.user.roles.contains(role))
    }
}

/// One thing the workbench remembers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryItem {
    pub id: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    /// What this is about, in a form a later run can ask for by name.
    pub key: String,
    pub value: String,
    /// How sensitive the value is. Every item has one; there is no default that
    /// means "not classified", because that is how unclassified material ends up
    /// being treated as public.
    pub classification: Classification,
    pub acl: Acl,
    pub source: MemorySource,
    /// RFC 3339, UTC.
    pub created_at: String,
    pub updated_at: String,
}

/// Why a write was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    #[error(
        "{value_kind:?} cannot be remembered in this scope: it belongs to a different kind of memory"
    )]
    WrongScope { value_kind: MemoryKind },

    #[error(
        "this came from {document_name}, which is classified {classification}. Material of that \
         classification is not promoted into shared memory automatically; a person has to approve \
         this specific entry."
    )]
    PromotionNeedsApproval {
        document_name: String,
        classification: String,
    },

    #[error(
        "{key:?} is a sensitive preference and is not remembered unless the person it belongs to \
         explicitly approves it."
    )]
    SensitivePreference { key: String },

    #[error("the memory for this scope could not be read or written: {detail}")]
    Storage { detail: String },
}

/// Preference keys that are never stored on a shrug.
///
/// Not a filter over the *value* — a filter over values is a guess, and a wrong
/// guess here stores a credential. These are the keys whose whole purpose is to
/// hold something personal or secret, and storing one requires the person to say
/// so for that entry.
const SENSITIVE_PREFERENCE_KEYS: &[&str] = &[
    "password",
    "passphrase",
    "token",
    "api_key",
    "apikey",
    "secret",
    "credential",
    "pin",
    "salary",
    "compensation",
    "health",
    "medical",
    "home_address",
    "personal_phone",
    "personal_email",
    "national_id",
    "aadhaar",
    "pan",
];

fn is_sensitive_preference(key: &str) -> bool {
    let normalised = key.to_ascii_lowercase().replace(['-', ' '], "_");
    SENSITIVE_PREFERENCE_KEYS
        .iter()
        .any(|sensitive| normalised.contains(sensitive))
}

/// A person's explicit go-ahead for one specific entry.
///
/// A struct rather than a `bool` so a caller cannot pass `true` without saying
/// who, and so the approver ends up in the stored record where a reviewer can
/// see whose decision it was.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Approval {
    pub approved_by: String,
    /// RFC 3339, UTC.
    pub at: String,
}

impl Approval {
    pub fn by(user_id: impl Into<String>) -> Self {
        Self {
            approved_by: user_id.into(),
            at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// What a caller asks to have remembered.
#[derive(Debug, Clone)]
pub struct Remember {
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub key: String,
    pub value: String,
    pub classification: Classification,
    pub source: MemorySource,
    /// Present only when a person approved this specific entry.
    pub approval: Option<Approval>,
}

/// The store.
///
/// Held in memory and mirrored to one JSON file per durable scope. Run-scope
/// items are never written to disk: they belong to the task record, which is
/// already written atomically and already under access control, and a second
/// copy would be a second place to leak them from.
#[derive(Debug, Default)]
pub struct MemoryStore {
    items: Mutex<HashMap<MemoryScope, Vec<MemoryItem>>>,
    root: Option<PathBuf>,
}

/// Shared handle, as the runtime holds it.
pub type SharedMemory = Arc<MemoryStore>;

impl MemoryStore {
    /// A store with no disk behind it. Used by tests and by a run that has no
    /// data directory yet.
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Opens the store under the application's data directory.
    ///
    /// Existing files are read lazily, on first access to their scope, rather
    /// than all at start-up: a deployment with two hundred projects should not
    /// pay for two hundred file reads to answer a question about one.
    pub fn open(app_data_dir: &Path) -> Self {
        Self {
            items: Mutex::new(HashMap::new()),
            root: Some(app_data_dir.join("memory")),
        }
    }

    /// Records something, or explains why it will not be.
    ///
    /// The single write path. Every rule this module exists to hold is enforced
    /// here, so there is no second entry point that could be added later without
    /// noticing what it skipped.
    pub fn remember(&self, request: Remember) -> Result<MemoryItem, MemoryError> {
        if !request.kind.permitted_in(&request.scope) {
            return Err(MemoryError::WrongScope {
                value_kind: request.kind,
            });
        }

        // The promotion rule. Checked against where the value *came from*, not
        // against how the caller classified it: a run that read a vendor
        // negotiation and labelled its summary `Internal` is exactly the case
        // this must catch, and trusting `request.classification` here would let
        // one mislabelling defeat the whole mechanism.
        if request.scope.is_durable() && request.approval.is_none() {
            if let Some(source_classification) = request.source.source_classification() {
                if source_classification != Classification::Internal {
                    let document_name = match &request.source {
                        MemorySource::Document {
                            document_sha256, ..
                        } => document_sha256.clone(),
                        _ => "a document".to_string(),
                    };
                    return Err(MemoryError::PromotionNeedsApproval {
                        document_name,
                        classification: source_classification.label().to_string(),
                    });
                }
            }
        }

        if matches!(request.scope, MemoryScope::User { .. })
            && request.approval.is_none()
            && is_sensitive_preference(&request.key)
        {
            return Err(MemoryError::SensitivePreference {
                key: request.key.clone(),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut acl = Acl::for_classification(request.classification, request.scope.project());
        if let MemoryScope::User { user_id } = &request.scope {
            acl.owner = Some(user_id.clone());
        }
        // An item promoted on somebody's approval is not thereby widened: the
        // approval says this entry may exist, not that everyone may read it.

        let item = MemoryItem {
            id: format!("{}::{}", scope_key(&request.scope), request.key),
            scope: request.scope.clone(),
            kind: request.kind,
            key: request.key,
            value: request.value,
            classification: request.classification,
            acl,
            source: request.source,
            created_at: now.clone(),
            updated_at: now,
        };

        {
            let mut table = self.lock()?;
            let bucket = table.entry(request.scope.clone()).or_default();
            match bucket.iter_mut().find(|held| held.key == item.key) {
                // Updated in place rather than appended, so a key means one
                // value and recall does not have to decide between two.
                Some(existing) => {
                    existing.value = item.value.clone();
                    existing.classification = item.classification;
                    existing.acl = item.acl.clone();
                    existing.source = item.source.clone();
                    existing.updated_at = item.updated_at.clone();
                }
                None => bucket.push(item.clone()),
            }
        }

        if request.scope.is_durable() {
            self.persist(&request.scope)?;
        }
        Ok(item)
    }

    /// Everything in one scope this reader may see.
    ///
    /// `project_id` is the project the *reader* is working on. Passing `None`
    /// does not widen the result — it narrows it to items confined to no
    /// project, which is what a reader outside every project should get.
    pub fn recall(
        &self,
        scope: &MemoryScope,
        session: &Session,
        project_id: Option<&str>,
    ) -> Vec<MemoryItem> {
        self.load_if_needed(scope);
        let Ok(table) = self.items.lock() else {
            // A poisoned lock means a previous writer panicked. Returning
            // nothing is the safe reading: an empty recall makes a run do the
            // work again, and a wrong one makes it act on somebody else's facts.
            return Vec::new();
        };
        table
            .get(scope)
            .map(|items| {
                items
                    .iter()
                    .filter(|item| item.acl.admits(session, project_id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One item by key, if this reader may see it.
    pub fn recall_one(
        &self,
        scope: &MemoryScope,
        key: &str,
        session: &Session,
        project_id: Option<&str>,
    ) -> Option<MemoryItem> {
        self.recall(scope, session, project_id)
            .into_iter()
            .find(|item| item.key == key)
    }

    /// Drops a finished run's memory.
    ///
    /// Called when the run ends and its record has been written. The record is
    /// the durable copy; holding the run's items for the life of the process
    /// would grow without bound for no reader.
    pub fn forget_run(&self, run_id: &str) {
        if let Ok(mut table) = self.items.lock() {
            table.remove(&MemoryScope::Run {
                run_id: run_id.to_string(),
            });
        }
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<MemoryScope, Vec<MemoryItem>>>, MemoryError> {
        self.items.lock().map_err(|_| MemoryError::Storage {
            detail: "the memory table was left locked by a failed write".to_string(),
        })
    }

    /// Reads a durable scope's file the first time it is asked for.
    fn load_if_needed(&self, scope: &MemoryScope) {
        if !scope.is_durable() {
            return;
        }
        let Some(root) = &self.root else { return };
        {
            let Ok(table) = self.items.lock() else { return };
            if table.contains_key(scope) {
                return;
            }
        }
        let path = root.join(scope.file_name());
        let loaded: Vec<MemoryItem> = std::fs::read(&path)
            .ok()
            .and_then(|body| serde_json::from_slice(&body).ok())
            .unwrap_or_default();
        if let Ok(mut table) = self.items.lock() {
            table.entry(scope.clone()).or_insert(loaded);
        }
    }

    /// Writes one durable scope to disk, atomically.
    fn persist(&self, scope: &MemoryScope) -> Result<(), MemoryError> {
        let Some(root) = &self.root else {
            return Ok(());
        };
        std::fs::create_dir_all(root).map_err(|error| MemoryError::Storage {
            detail: error.to_string(),
        })?;

        let body = {
            let table = self.lock()?;
            let items = table.get(scope).cloned().unwrap_or_default();
            serde_json::to_vec_pretty(&items).map_err(|error| MemoryError::Storage {
                detail: error.to_string(),
            })?
        };

        // Written to a temporary name and renamed into place, so a crash midway
        // leaves the previous file rather than half of a new one.
        let path = root.join(scope.file_name());
        let temporary = path.with_extension("json.writing");
        std::fs::write(&temporary, body).map_err(|error| MemoryError::Storage {
            detail: error.to_string(),
        })?;
        std::fs::rename(&temporary, &path).map_err(|error| MemoryError::Storage {
            detail: error.to_string(),
        })
    }
}

fn scope_key(scope: &MemoryScope) -> String {
    match scope {
        MemoryScope::Run { run_id } => format!("run:{run_id}"),
        MemoryScope::Workspace { project_id } => format!("workspace:{project_id}"),
        MemoryScope::User { user_id } => format!("user:{user_id}"),
    }
}

/// A run's own memory, in the shape the runtime keeps it.
///
/// The Rust mirror of `working-notes.ts`. It exists on this side so a run's
/// state can be persisted with the task record and handed back to a resumed
/// run — the runtime process does not survive a restart, and this does.
///
/// The caps are not repeated here: the runtime enforces them on the way in and
/// on the way out, and a second set of numbers in a second language would be
/// two things to keep in step. What this holds is whatever the runtime last
/// reported, which is by construction already bounded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunMemory {
    pub goal: String,
    pub stage: RunStage,
    pub decisions: Vec<RunDecision>,
    /// Citation markers, not passages. The passages are in the evidence table.
    pub evidence_ids: Vec<String>,
    pub calculation_ids: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub open_questions: Vec<String>,
    pub next_action: String,
    /// Side effects that already happened. Read before a resumed run acts.
    pub completed: Vec<CompletedEffect>,
    /// How many entries the runtime's caps dropped, per list.
    #[serde(default)]
    pub dropped: HashMap<String, u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunStage {
    pub ordinal: u32,
    pub intent: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDecision {
    pub what: String,
    pub because: String,
    /// RFC 3339, UTC.
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletedEffect {
    pub tool: String,
    pub target: String,
    /// RFC 3339, UTC.
    pub at: String,
}

impl RunMemory {
    /// Whether this side effect is already known to have happened.
    ///
    /// What makes a resumption safe rather than merely faster. A run that
    /// resumes without asking this writes the approval note twice.
    pub fn has_done(&self, tool: &str, target: &str) -> bool {
        self.completed
            .iter()
            .any(|effect| effect.tool == tool && effect.target == target)
    }

    /// True when nothing has been recorded, so a resumption has nothing to read.
    pub fn is_empty(&self) -> bool {
        self.goal.is_empty()
            && self.next_action.is_empty()
            && self.stage.ordinal == 0
            && self.decisions.is_empty()
            && self.evidence_ids.is_empty()
            && self.calculation_ids.is_empty()
            && self.artifact_ids.is_empty()
            && self.open_questions.is_empty()
            && self.completed.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::User;

    fn session(id: &str, roles: Vec<Role>) -> Session {
        Session::open(User::new(id, id, roles))
    }

    fn workspace(project: &str) -> MemoryScope {
        MemoryScope::Workspace {
            project_id: project.to_string(),
        }
    }

    fn term(project: &str, key: &str, value: &str) -> Remember {
        Remember {
            scope: workspace(project),
            kind: MemoryKind::Terminology,
            key: key.to_string(),
            value: value.to_string(),
            classification: Classification::Internal,
            source: MemorySource::Operator {
                user_id: "kiran".to_string(),
            },
            approval: None,
        }
    }

    #[test]
    fn memory_from_one_project_is_not_visible_from_another() {
        // The boundary this module exists to hold. A run on Project B that can
        // read Project A's terminology has crossed it, and nothing in the audit
        // record would say so.
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::User]);

        store
            .remember(term("project-a", "hot-tap", "A tap made on a live line."))
            .expect("stored");

        assert_eq!(
            store
                .recall(&workspace("project-a"), &reader, Some("project-a"))
                .len(),
            1
        );
        assert!(store
            .recall(&workspace("project-b"), &reader, Some("project-b"))
            .is_empty());
    }

    #[test]
    fn one_projects_item_is_not_returned_to_a_reader_working_on_another() {
        // The same boundary from the other direction: asking Project A's scope
        // while working on Project B must not succeed either, or the confinement
        // would only be a matter of which key the caller happened to use.
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::User]);
        store
            .remember(term("project-a", "hot-tap", "…"))
            .expect("stored");

        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-b"))
            .is_empty());
        // And naming no project is not a wildcard.
        assert!(store
            .recall(&workspace("project-a"), &reader, None)
            .is_empty());
    }

    #[test]
    fn one_runs_memory_is_not_anothers() {
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::User]);
        store
            .remember(Remember {
                scope: MemoryScope::Run {
                    run_id: "run-1".to_string(),
                },
                kind: MemoryKind::Decision,
                key: "revision".to_string(),
                value: "Use the 2019 revision.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Run {
                    run_id: "run-1".to_string(),
                },
                approval: None,
            })
            .expect("stored");

        let other = MemoryScope::Run {
            run_id: "run-2".to_string(),
        };
        assert!(store.recall(&other, &reader, None).is_empty());
    }

    #[test]
    fn restricted_document_text_is_not_promoted_into_shared_memory() {
        // The publication failure. A run reads a vendor negotiation, decides the
        // price is a useful fact, and writes it where every later run reads.
        let store = MemoryStore::in_memory();
        let refusal = store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::ProjectFact,
                key: "unit-price".to_string(),
                value: "The tendered unit price is ₹4.2 crore.".to_string(),
                // Mislabelled as ordinary on purpose: the rule must read the
                // source, not the label the caller chose.
                classification: Classification::Internal,
                source: MemorySource::Document {
                    document_sha256: "tender-2026".to_string(),
                    page: 12,
                    classification: Classification::VendorNegotiation,
                },
                approval: None,
            })
            .expect_err("must be refused");

        assert!(matches!(refusal, MemoryError::PromotionNeedsApproval { .. }));
        // And nothing was stored on the way to being refused.
        let reader = session("kiran", vec![Role::User]);
        assert!(store
            .recall(&workspace("project-a"), &reader, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn the_same_promotion_is_allowed_once_a_person_approves_it() {
        // The rule is "not automatically", not "never". A refusal with no way
        // through would make people work around the mechanism entirely.
        let store = MemoryStore::in_memory();
        let stored = store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::ProjectFact,
                key: "unit-price".to_string(),
                value: "The tendered unit price is ₹4.2 crore.".to_string(),
                classification: Classification::VendorNegotiation,
                source: MemorySource::Document {
                    document_sha256: "tender-2026".to_string(),
                    page: 12,
                    classification: Classification::VendorNegotiation,
                },
                approval: Some(Approval::by("asha")),
            })
            .expect("approved promotion is stored");

        // Approval permits the entry; it does not widen who may read it.
        assert_eq!(stored.classification, Classification::VendorNegotiation);
        assert!(!stored
            .acl
            .cleared_roles
            .contains(&Role::KnowledgeAdministrator));
    }

    #[test]
    fn ordinary_internal_material_promotes_without_ceremony() {
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::Terminology,
                key: "hot-tap".to_string(),
                value: "A tap made on a live line.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Document {
                    document_sha256: "sop".to_string(),
                    page: 4,
                    classification: Classification::Internal,
                },
                approval: None,
            })
            .is_ok());
    }

    #[test]
    fn a_run_may_hold_what_it_may_not_publish() {
        // Reading a confidential document into the run's own state is the work.
        // Only promotion out of run scope is publication.
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: MemoryScope::Run {
                    run_id: "run-1".to_string(),
                },
                kind: MemoryKind::Decision,
                key: "price".to_string(),
                value: "The tendered unit price is ₹4.2 crore.".to_string(),
                classification: Classification::VendorNegotiation,
                source: MemorySource::Document {
                    document_sha256: "tender-2026".to_string(),
                    page: 12,
                    classification: Classification::VendorNegotiation,
                },
                approval: None,
            })
            .is_ok());
    }

    #[test]
    fn a_sensitive_preference_is_not_remembered_on_a_shrug() {
        let store = MemoryStore::in_memory();
        for key in ["password", "personal-phone", "API_KEY"] {
            let refusal = store
                .remember(Remember {
                    scope: MemoryScope::User {
                        user_id: "kiran".to_string(),
                    },
                    kind: MemoryKind::Preference,
                    key: key.to_string(),
                    value: "something".to_string(),
                    classification: Classification::Internal,
                    source: MemorySource::Operator {
                        user_id: "kiran".to_string(),
                    },
                    approval: None,
                })
                .expect_err("must be refused");
            assert!(
                matches!(refusal, MemoryError::SensitivePreference { .. }),
                "{key} was not treated as sensitive"
            );
        }
    }

    #[test]
    fn an_ordinary_preference_is_remembered_without_asking() {
        // The mechanism has to stay usable for what it is actually for.
        let store = MemoryStore::in_memory();
        assert!(store
            .remember(Remember {
                scope: MemoryScope::User {
                    user_id: "kiran".to_string(),
                },
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "Prefers SI units in drafted notes.".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
            })
            .is_ok());
    }

    #[test]
    fn one_persons_preferences_are_not_anothers() {
        let store = MemoryStore::in_memory();
        store
            .remember(Remember {
                scope: MemoryScope::User {
                    user_id: "kiran".to_string(),
                },
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "SI".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
            })
            .expect("stored");

        let someone_else = session("asha", vec![Role::User]);
        assert!(store
            .recall(
                &MemoryScope::User {
                    user_id: "kiran".to_string()
                },
                &someone_else,
                None
            )
            .is_empty());
    }

    #[test]
    fn every_item_carries_a_classification_and_an_access_list() {
        let store = MemoryStore::in_memory();
        let item = store
            .remember(term("project-a", "hot-tap", "…"))
            .expect("stored");

        assert_eq!(item.classification, Classification::Internal);
        assert!(!item.acl.cleared_roles.is_empty());
        assert_eq!(item.acl.project_id.as_deref(), Some("project-a"));
    }

    #[test]
    fn a_reader_without_the_role_sees_nothing() {
        let store = MemoryStore::in_memory();
        store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::ProjectFact,
                key: "terms".to_string(),
                value: "Payment is net 60.".to_string(),
                classification: Classification::VendorNegotiation,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
            })
            .expect("stored");

        // An auditor reads the record, and nothing else.
        let auditor = session("ravi", vec![Role::Auditor]);
        assert!(store
            .recall(&workspace("project-a"), &auditor, Some("project-a"))
            .is_empty());
    }

    #[test]
    fn a_kind_cannot_be_filed_in_a_scope_it_does_not_belong_to() {
        let store = MemoryStore::in_memory();
        let refusal = store
            .remember(Remember {
                scope: workspace("project-a"),
                kind: MemoryKind::Preference,
                key: "units".to_string(),
                value: "SI".to_string(),
                classification: Classification::Internal,
                source: MemorySource::Operator {
                    user_id: "kiran".to_string(),
                },
                approval: None,
            })
            .expect_err("must be refused");

        assert!(matches!(refusal, MemoryError::WrongScope { .. }));
    }

    #[test]
    fn writing_a_key_twice_updates_it_rather_than_storing_two_answers() {
        let store = MemoryStore::in_memory();
        let reader = session("kiran", vec![Role::User]);
        store
            .remember(term("project-a", "hot-tap", "first"))
            .expect("stored");
        store
            .remember(term("project-a", "hot-tap", "second"))
            .expect("stored");

        let recalled = store.recall(&workspace("project-a"), &reader, Some("project-a"));
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].value, "second");
    }

    #[test]
    fn durable_memory_survives_a_restart_and_run_memory_does_not() {
        let dir = tempfile::tempdir().expect("temp dir");
        let reader = session("kiran", vec![Role::User]);

        {
            let store = MemoryStore::open(dir.path());
            store
                .remember(term("project-a", "hot-tap", "A tap made on a live line."))
                .expect("stored");
            store
                .remember(Remember {
                    scope: MemoryScope::Run {
                        run_id: "run-1".to_string(),
                    },
                    kind: MemoryKind::Decision,
                    key: "revision".to_string(),
                    value: "2019".to_string(),
                    classification: Classification::Internal,
                    source: MemorySource::Run {
                        run_id: "run-1".to_string(),
                    },
                    approval: None,
                })
                .expect("stored");
        }

        let reopened = MemoryStore::open(dir.path());
        assert_eq!(
            reopened
                .recall(&workspace("project-a"), &reader, Some("project-a"))
                .len(),
            1
        );
        // The run's own state is in the task record, not here.
        assert!(reopened
            .recall(
                &MemoryScope::Run {
                    run_id: "run-1".to_string()
                },
                &reader,
                None
            )
            .is_empty());
    }

    #[test]
    fn a_project_id_cannot_name_a_path_outside_the_memory_directory() {
        let scope = workspace("../../etc/passwd");
        assert_eq!(
            Path::new(&scope.file_name()).components().count(),
            1,
            "a project id became more than one path component"
        );
    }

    #[test]
    fn a_resumed_run_can_tell_what_it_already_did() {
        let memory = RunMemory {
            completed: vec![CompletedEffect {
                tool: "create_docx".to_string(),
                target: "approval-note.docx".to_string(),
                at: "2026-08-28T09:15:00+00:00".to_string(),
            }],
            ..RunMemory::default()
        };

        assert!(memory.has_done("create_docx", "approval-note.docx"));
        assert!(!memory.has_done("create_docx", "something-else.docx"));
        assert!(!memory.is_empty());
    }
}

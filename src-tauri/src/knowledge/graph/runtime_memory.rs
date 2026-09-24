//! What agents know, as a graph the deployment can audit.
//!
//! ## Why this is a new domain beside the notebook graph, not inside it
//!
//! [`super::assertions`] holds claims extracted from documents: a subject, a
//! predicate, an object, and the passages they were read out of. It is a good
//! model of *what the documents say* and the wrong model for *what an agent has
//! established while working*, for three reasons that matter:
//!
//! - Its nodes are terms, keyed by `notebook_id`. A goal, a plan step or an
//!   operator's correction is not a term and belongs to a task, not a notebook.
//! - Its claims are the output of an extractor run over a fixed corpus. Runtime
//!   memory accumulates from tool receipts, model proposals and people, which
//!   have completely different admission rules.
//! - Reinterpreting an inferred co-occurrence link as a verified agent fact
//!   would be the worst thing this module could do: those links are
//!   statistical, they were never reviewed, and the whole point of the status
//!   ladder below is that nothing counts as established until something
//!   established it.
//!
//! So the tables are separate and prefixed `agent_memory_*`. What *is* reused
//! is the machinery whose semantics already fit: [`Acl`] and [`Classification`]
//! decide who may read, and the proposed/accepted/rejected ladder is the same
//! idea `AssertionStatus` encodes, extended with the states a long-lived record
//! needs.
//!
//! ## The admission rule, which is the point of the module
//!
//! A model saying something is a *proposal*. It stays a proposal until
//! something outside the model corroborates it. A tool receipt Rust itself
//! wrote is corroboration; another model sentence is not.
//!
//! This is the rule [`crate::agent_runtime::state_commit`] applies to working
//! notes, for the same reason: a record that can be written by the thing being
//! recorded is not a record. What is new here is that a proposal is *kept*
//! rather than dropped — a claim nobody has corroborated is still worth showing
//! to a person, labelled as what it is.
//!
//! ## Corrections do not delete
//!
//! An operator correcting a fact supersedes it. The superseded item stays, with
//! a link saying what replaced it, because "the model said 150 PSI and a person
//! changed it to 10 bar" is the record somebody will need later — and deleting
//! the first half turns a correction into an assertion with no history.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent_runtime::memory::Acl;
use crate::identity::Session;
use crate::policy::Classification;

/// The layout of the stored runtime-memory domain.
pub const MEMORY_SCHEMA_VERSION: u32 = 1;

/// What a memory item is.
///
/// Typed rather than a free string, because the type decides the admission
/// rule, the retention and how it is drawn. A `ToolObservation` may be admitted
/// from a receipt; a `Fact` from a model may not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryKind {
    /// What the task is for.
    Goal,
    /// Something held to be true.
    Fact,
    /// A rule the work is bound by ("every pressure in bar").
    Constraint,
    /// A person changing something. Outranks everything it supersedes.
    Correction,
    /// Something decided, and therefore not to be re-litigated silently.
    Decision,
    /// A plan, or a step of one.
    Plan,
    /// Something not yet known, kept so it is not quietly forgotten.
    OpenQuestion,
    /// What a tool actually returned. The only kind with a receipt behind it.
    ToolObservation,
    /// A pointer to source bytes, at an exact version.
    SourceRef,
    /// A pointer to a produced artifact, at an exact revision.
    ArtifactRef,
}

impl MemoryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Goal => "goal",
            Self::Fact => "fact",
            Self::Constraint => "constraint",
            Self::Correction => "correction",
            Self::Decision => "decision",
            Self::Plan => "plan",
            Self::OpenQuestion => "openQuestion",
            Self::ToolObservation => "toolObservation",
            Self::SourceRef => "sourceRef",
            Self::ArtifactRef => "artifactRef",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "goal" => Self::Goal,
            "fact" => Self::Fact,
            "constraint" => Self::Constraint,
            "correction" => Self::Correction,
            "decision" => Self::Decision,
            "plan" => Self::Plan,
            "openQuestion" => Self::OpenQuestion,
            "toolObservation" => Self::ToolObservation,
            "sourceRef" => Self::SourceRef,
            "artifactRef" => Self::ArtifactRef,
            _ => return None,
        })
    }
}

/// How two items relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EdgeKind {
    /// The source corroborates the target.
    Supports,
    /// The source disagrees with the target. Kept, never resolved by deletion.
    Contradicts,
    /// The source replaces the target. The target stays, marked superseded.
    Supersedes,
    /// The source was derived from the target.
    DerivedFrom,
    /// The source cites the target as evidence.
    Cites,
    /// The source is part of the target (a plan step in a plan).
    PartOf,
    /// The source answers the target open question.
    Answers,
}

impl EdgeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
            Self::DerivedFrom => "derivedFrom",
            Self::Cites => "cites",
            Self::PartOf => "partOf",
            Self::Answers => "answers",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "supports" => Self::Supports,
            "contradicts" => Self::Contradicts,
            "supersedes" => Self::Supersedes,
            "derivedFrom" => Self::DerivedFrom,
            "cites" => Self::Cites,
            "partOf" => Self::PartOf,
            "answers" => Self::Answers,
            _ => return None,
        })
    }
}

/// Where an item stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ItemStatus {
    /// Something said it and nothing has corroborated it.
    ///
    /// Retrieved, and labelled. A proposal is worth showing a person — it is
    /// how they find out the model believes something wrong — and it must never
    /// be presented as established.
    Proposed,
    /// Corroborated under the rules in [`admit`].
    Admitted,
    /// Replaced by a later item, which names it.
    Superseded,
    /// A person looked and said no. Kept, because deleting it invites the same
    /// wrong thing to be proposed again with nothing to say it was refused.
    Rejected,
    /// Deleted, as far as reading is concerned.
    ///
    /// The row stays so that edges pointing at it still resolve and an export
    /// can say something was removed rather than silently omitting it.
    Tombstoned,
    /// Something it rests on changed after it was written.
    ///
    /// A fact it was derived from was corrected, a source it cites was
    /// withdrawn, or a dependency it pinned moved on before it was published.
    /// Readable -- a person needs to see what went stale -- and not usable as
    /// evidence until something revalidates it against its new inputs. Plan §6:
    /// "descendant artifacts and cached answers whose inputs changed become
    /// stale and require revalidation."
    Stale,
}

impl ItemStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Admitted => "admitted",
            Self::Superseded => "superseded",
            Self::Rejected => "rejected",
            Self::Tombstoned => "tombstoned",
            Self::Stale => "stale",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "proposed" => Self::Proposed,
            "admitted" => Self::Admitted,
            "superseded" => Self::Superseded,
            "rejected" => Self::Rejected,
            "tombstoned" => Self::Tombstoned,
            "stale" => Self::Stale,
            _ => return None,
        })
    }

    /// Whether a change of an input should make an item in this state stale.
    ///
    /// Only what is still standing can go stale. Something already superseded,
    /// rejected or tombstoned says so already, and overwriting that with
    /// "stale" would lose the more specific answer.
    pub fn can_go_stale(self) -> bool {
        matches!(self, Self::Admitted | Self::Proposed)
    }

    /// Whether an item in this state no longer stands, so anything resting on
    /// it has lost an input.
    pub fn withdrawn(self) -> bool {
        matches!(
            self,
            Self::Superseded | Self::Rejected | Self::Tombstoned | Self::Stale
        )
    }

    /// Whether an answer may be built on this.
    ///
    /// Stated once, here, rather than re-decided at each call site — the same
    /// reason `AssertionStatus::usable_as_evidence` exists. A proposal *is*
    /// retrievable and must be labelled; a rejected or tombstoned item is not.
    pub fn usable_as_evidence(self) -> bool {
        matches!(self, Self::Admitted | Self::Proposed)
    }

    /// Whether this item is established rather than merely offered.
    pub fn is_established(self) -> bool {
        matches!(self, Self::Admitted)
    }
}

/// Who or what put this in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Provenance {
    /// A tool receipt Rust itself wrote into the durable event log.
    ///
    /// `event_seq` names the row. Carrying the sequence rather than a boolean
    /// is what makes the claim checkable later: somebody auditing can go and
    /// read the event this was admitted from.
    ///
    /// `run_id` is the run that *made the call* -- for a worker, the child's
    /// own run, never its parent's -- and `output_sha256` is the hash of what
    /// the tool returned, as the event recorded it. Admission resolves all four
    /// against the event log ([`super::receipts`]); a positive sequence number
    /// on its own proves nothing, because anyone can write one.
    ToolReceipt {
        run_id: String,
        tool: String,
        event_seq: i64,
        /// Absent on a receipt written before outputs were hashed, and then it
        /// cannot be verified and is kept as a proposal.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_sha256: Option<String>,
    },
    /// A model said it. Stays [`ItemStatus::Proposed`] until admitted.
    Model { model_id: String, run_id: String },
    /// A person. Outranks a model on the same subject; see [`may_supersede`].
    Operator { user_id: String },
    /// Brought across from a store that predates this one.
    ///
    /// The legacy id is kept so a record written against the old store can
    /// still be found, and so a migration that runs twice recognises what it
    /// already moved.
    Migrated {
        legacy_store: String,
        legacy_id: String,
    },
}

impl Provenance {
    pub fn origin(&self) -> &'static str {
        match self {
            Self::ToolReceipt { .. } => "toolReceipt",
            Self::Model { .. } => "model",
            Self::Operator { .. } => "operator",
            Self::Migrated { .. } => "migrated",
        }
    }
}

/// What kind of knowing an item is.
///
/// Plan §6: preserve the distinction between *the source says X*, *a tool
/// measured X*, *the model inferred X* and *a user supplied X*. Computed by the
/// store from the provenance and the receipt's tool -- never chosen by the
/// writer, because a writer that could label its own inference "measured"
/// could launder it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Basis {
    /// A retrieval tool returned passages that say this. The claim is what the
    /// source says, not that it is true.
    SourceText,
    /// A tool computed, read or re-opened something and reported it.
    Measured,
    /// A model said it.
    Inferred,
    /// A person said it.
    Supplied,
    /// Brought across from a store that predates this one.
    Carried,
}

impl Basis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceText => "sourceText",
            Self::Measured => "measured",
            Self::Inferred => "inferred",
            Self::Supplied => "supplied",
            Self::Carried => "carried",
        }
    }

    /// The basis a provenance implies.
    ///
    /// A receipt from a tool that returns evidence (`knowledge.search_authorized`
    /// and the other retrieval tools) is what a source says; a receipt from any
    /// other tool is a measurement. Decided by the tool's contract, so a new
    /// retrieval tool is classified by being registered as one.
    pub fn of(provenance: &Provenance) -> Self {
        match provenance {
            Provenance::ToolReceipt { tool, .. } => {
                match crate::orchestrator::tools::ToolName::from_str(tool) {
                    Some(name)
                        if crate::orchestrator::contract::output_of(name)
                            == crate::orchestrator::contract::OutputKind::Evidence =>
                    {
                        Self::SourceText
                    }
                    _ => Self::Measured,
                }
            }
            Provenance::Model { .. } => Self::Inferred,
            Provenance::Operator { .. } => Self::Supplied,
            Provenance::Migrated { .. } => Self::Carried,
        }
    }
}

/// An item this one rests on, at the revision it was read at.
///
/// The dependency set a result was computed from. Publication re-checks every
/// pin: an input that moved on while the work was in flight makes the result
/// stale rather than silently current. Plan §6: "Record the source-version
/// dependency set, then withhold publication or mark the result stale."
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Dependency {
    pub item_id: String,
    pub revision: u64,
}

/// Which store is the authority for a record.
///
/// Explicit, per record, because until every legacy writer is routed through
/// the graph some records here are *copies*: the legacy store is still written
/// directly and is what is true. A graph write to such a record would be a
/// second writer racing the first. See [`super::migration`] for the cutover
/// that moves authority here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Authority {
    /// The graph is the only writer.
    #[default]
    Graph,
    /// A mirror of a legacy record. Only the migration may write it; a
    /// correction belongs in the legacy store until the source is cut over.
    Legacy { store: String },
}

/// What the event log said about a receipt, when it was asked.
///
/// Passed into [`admit`] rather than decided inside it: the rule lives here,
/// the lookup lives with the storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptVerdict {
    /// The item does not claim a receipt.
    NotAReceipt,
    /// The event exists, succeeded, names the same tool and the same output.
    Verified,
    /// It does not, and this says which part failed.
    Refused { because: String },
}

/// Source bytes, at the version they were read at.
///
/// The hash *is* the version — the document store is content-addressed — and
/// the locator says where inside it. Both, because a claim citing "the vessel
/// drawing" is not checkable and a claim citing "page 12 of ab12…" is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceRef {
    pub sha256: String,
    /// Where inside it: a page, a chunk id, a cell reference.
    pub locator: String,
    /// The extraction revision this was read at, when the store had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_revision: Option<String>,
}

/// A produced artifact, at an exact revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRef {
    pub artifact_id: String,
    /// Which revision. An artifact corrected twice has three, and a claim about
    /// "the approval note" that does not say which is not round-trippable.
    pub revision: u32,
    /// SHA-256 of the bytes, so the exact file can be proven later.
    pub sha256: String,
}

/// Which slice of the world an item belongs to.
///
/// Three, and they are not interchangeable. A task's memory ends under an
/// explicit retention decision; a workspace's is shared by everyone on the
/// project; a user's is one person's. An item that could not say which would be
/// an item nobody can safely delete.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MemoryScope {
    Task { task_id: String },
    Workspace { project_id: String },
    User { user_id: String },
    /// One agent's private working notes on one task.
    ///
    /// Where a child publishes when its definition does not share with the
    /// task (P01's `shared_with_task`): committed, attributed and replayable
    /// like anything else, and invisible to the task's other agents, which read
    /// the `Task` scope. Sharing is a decision the definition makes, and
    /// promotion to a project is a further one that needs a person.
    Scratch { task_id: String, agent_id: String },
}

impl MemoryScope {
    pub fn project(&self) -> Option<&str> {
        match self {
            Self::Workspace { project_id } => Some(project_id),
            _ => None,
        }
    }

    pub fn key(&self) -> String {
        match self {
            Self::Task { task_id } => format!("task:{task_id}"),
            Self::Workspace { project_id } => format!("workspace:{project_id}"),
            Self::User { user_id } => format!("user:{user_id}"),
            // The agent first, and `@` between: agent ids and task ids are
            // generated or file-derived and hold neither `@` nor `:`.
            Self::Scratch { task_id, agent_id } => format!("scratch:{agent_id}@{task_id}"),
        }
    }

    /// The inverse of [`Self::key`].
    pub fn from_key(key: &str) -> Option<Self> {
        let (kind, value) = key.split_once(':')?;
        Some(match kind {
            "task" => Self::Task {
                task_id: value.to_string(),
            },
            "workspace" => Self::Workspace {
                project_id: value.to_string(),
            },
            "user" => Self::User {
                user_id: value.to_string(),
            },
            "scratch" => {
                let (agent_id, task_id) = value.split_once('@')?;
                Self::Scratch {
                    task_id: task_id.to_string(),
                    agent_id: agent_id.to_string(),
                }
            }
            _ => return None,
        })
    }
}

/// One thing an agent knows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryItem {
    /// Stable for the life of the item, across every revision of it.
    pub item_id: String,
    /// Which revision of this item. Monotonic per item.
    pub revision: u64,
    pub kind: MemoryKind,
    /// The agent this is attributed to. Never a model id: an agent's model
    /// changes and its memory does not move with it.
    pub agent_id: String,
    pub scope: MemoryScope,
    pub classification: Classification,
    pub acl: Acl,
    /// The model that was running when this was created, for the trace only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creator_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creator_run_id: Option<String>,
    pub provenance: Provenance,
    pub content: String,
    #[serde(default)]
    pub sources: Vec<SourceRef>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    /// What the creator claimed, 0.0–1.0. Never used to rank across
    /// provenances: a model's 0.99 does not outrank an operator's correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub status: ItemStatus,
    /// RFC 3339, UTC.
    pub valid_from: String,
    /// When this stops being true. `None` means it has no known end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    /// The item this replaced, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    /// Items this disagrees with. Recorded, never resolved by deletion.
    #[serde(default)]
    pub conflicts_with: Vec<String>,
    /// What this was written in reaction to. Lets a reader reconstruct order
    /// without trusting wall-clock timestamps from two processes.
    #[serde(default)]
    pub causal_parents: Vec<String>,
    /// Supplied by the writer so a retry after a lost acknowledgement is
    /// recognised as the same write rather than performed twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// What kind of knowing this is. Set by the store from the provenance --
    /// see [`Basis::of`]. `None` on an item written before it existed; read it
    /// through [`MemoryItem::basis`], which derives it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<Basis>,
    /// The items this was computed from, at the revisions they were read at.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<Dependency>,
    /// People whose access to this one item was withdrawn, whatever their
    /// roles. A per-reader revocation: narrower than an ACL change, and
    /// recorded as a new revision so a reader's cached copy is dropped.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub revoked_readers: Vec<String>,
    /// Which store is the authority for this record. See [`Authority`].
    #[serde(default)]
    pub authority: Authority,
}

impl MemoryItem {
    /// Whether this item should be returned by a read at all.
    pub fn is_readable(&self) -> bool {
        !matches!(self.status, ItemStatus::Tombstoned)
    }

    /// Whether this reader may see it.
    ///
    /// ## Why the caller passes a project rather than this inferring one
    ///
    /// Because absence of a project is not a wildcard, and the only place that
    /// can be decided is where the request came from. `Acl::admits` already
    /// holds that rule; this adds the scope check, which is the one an item
    /// carries and an ACL does not.
    pub fn readable_by(&self, session: &Session, project_id: Option<&str>) -> bool {
        if !self.is_readable() {
            return false;
        }
        // A person whose access to this item was withdrawn does not get it
        // back through a role, a project or an owner field. Checked first
        // because it is the most specific rule there is.
        if self.revoked_readers.iter().any(|user| user == &session.user.id) {
            return false;
        }
        // A user-scope item belongs to one person, whatever roles anybody else
        // holds. Checked before the ACL because it is the narrower rule and the
        // ACL's `owner` field is optional.
        if let MemoryScope::User { user_id } = &self.scope {
            if user_id != &session.user.id {
                return false;
            }
        }
        // A workspace item is confined to its project. A reader working on a
        // different project — or on none — is not thereby cleared for it.
        if let MemoryScope::Workspace { project_id: owner } = &self.scope {
            if project_id != Some(owner.as_str()) {
                return false;
            }
        }
        self.acl.admits(session, project_id)
    }

    /// What kind of knowing this is, derived for an item written before the
    /// field existed.
    pub fn basis(&self) -> Basis {
        self.basis.unwrap_or_else(|| Basis::of(&self.provenance))
    }

    /// Whether this item is expired at the given instant.
    pub fn is_expired_at(&self, now: &str) -> bool {
        self.valid_until
            .as_deref()
            .is_some_and(|until| until <= now)
    }
}

/// A typed link between two items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEdge {
    /// Stable, and derived from the triple so the same link written twice is
    /// one row. See [`edge_id`].
    pub edge_id: String,
    pub from_item: String,
    pub to_item: String,
    pub kind: EdgeKind,
    /// The agent that drew this link. Carried because the graph colours an edge
    /// by its author, and because "who said these two things are related" is a
    /// different question from "who said each of them".
    pub agent_id: String,
    pub scope: MemoryScope,
    pub created_at: String,
}

/// The stable id of a link.
///
/// Derived from everything that makes the link what it is, so drawing the same
/// conclusion twice updates one row instead of making a second identical one —
/// and so a *different* relation between the same two items is a different row,
/// which is what keeps a disagreement visible.
pub fn edge_id(from_item: &str, kind: EdgeKind, to_item: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(from_item.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(kind.as_str().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(to_item.as_bytes());
    format!("me-{:x}", hasher.finalize())
}

/// The stable id of an item.
///
/// Random rather than content-derived, deliberately. Two agents independently
/// establishing the same fact are two observations of it, and collapsing them
/// into one row would destroy the attribution that makes the graph worth
/// having. They are related by an edge, not by an id collision.
pub fn item_id() -> String {
    format!("mi-{}", uuid::Uuid::new_v4())
}

/// What a proposal was allowed to become.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdmissionOutcome {
    pub status: ItemStatus,
    /// Why, in an operator's terms. Present even on admission, because "this
    /// was admitted because a tool receipt says so" is the sentence that makes
    /// the graph auditable.
    pub because: String,
}

/// Decides whether a newly written item is established or merely offered.
///
/// ## The rule
///
/// | Provenance | Outcome |
/// |---|---|
/// | operator | admitted — a person is the authority this defers to |
/// | tool receipt | admitted **only if** the event log verified it: the event exists, is a success, names the same tool and the same output hash |
/// | migrated | admitted — it was established under the old store's rules |
/// | model | **proposed**, always |
///
/// A model's confidence does not enter into it. A model that is certain and
/// wrong is the ordinary failure this exists to contain, and letting a number
/// the model chose decide whether the model is believed would be circular.
///
/// ## Why the receipt is judged outside this function
///
/// It used to be judged here, by shape: a positive sequence number, a tool
/// name and a run id were enough. Any writer can produce those, and the
/// worker that published every finding did -- with `event_seq: 0` on the
/// parent's run id, which this rule correctly refused, and which a one-line
/// "fix" to a positive number would have turned into manufactured
/// corroboration. So the shape check stays, as the first gate, and the
/// verdict comes from the event log ([`super::receipts`]).
pub fn admit(item: &MemoryItem, receipt: &ReceiptVerdict) -> AdmissionOutcome {
    match &item.provenance {
        Provenance::Operator { user_id } => AdmissionOutcome {
            status: ItemStatus::Admitted,
            because: format!("{user_id} recorded this directly"),
        },
        Provenance::ToolReceipt {
            run_id,
            tool,
            event_seq,
            output_sha256,
        } => {
            // A receipt has to name something checkable. A sequence of zero is
            // what an unset field looks like, and admitting on it would let a
            // caller manufacture corroboration by leaving a field blank.
            let named = *event_seq > 0
                && !tool.is_empty()
                && !run_id.is_empty()
                && output_sha256.as_deref().is_some_and(|hash| !hash.is_empty());
            match (named, receipt) {
                (true, ReceiptVerdict::Verified) => AdmissionOutcome {
                    status: ItemStatus::Admitted,
                    because: format!(
                        "{tool} succeeded in run {run_id}, recorded at event {event_seq}, and the \
                         event log holds that exact output"
                    ),
                },
                (false, _) => AdmissionOutcome {
                    status: ItemStatus::Proposed,
                    because: "this claims a tool receipt and does not name the run, tool, event \
                              and output that would corroborate it, so it is kept as a proposal"
                        .to_string(),
                },
                (true, ReceiptVerdict::Refused { because }) => AdmissionOutcome {
                    status: ItemStatus::Proposed,
                    because: format!(
                        "this claims {tool} event {event_seq} in run {run_id}, and the event log \
                         does not back it ({because}), so it is kept as a proposal"
                    ),
                },
                (true, ReceiptVerdict::NotAReceipt) => AdmissionOutcome {
                    status: ItemStatus::Proposed,
                    because: "this claims a tool receipt that was never checked against the \
                              event log, so it is kept as a proposal"
                        .to_string(),
                },
            }
        }
        Provenance::Migrated {
            legacy_store,
            legacy_id,
        } => AdmissionOutcome {
            status: ItemStatus::Admitted,
            because: format!("migrated from {legacy_store} entry {legacy_id}"),
        },
        Provenance::Model { model_id, .. } => AdmissionOutcome {
            status: ItemStatus::Proposed,
            because: format!(
                "{model_id} asserted this and nothing outside the model corroborates it yet"
            ),
        },
    }
}

/// Whether `candidate` may supersede `existing`.
///
/// ## Operator precedence, without erasing anything
///
/// A person's correction supersedes whatever it corrects, including an admitted
/// fact. A model may supersede only its own proposals — letting a model
/// supersede an operator's correction would make the correction a suggestion,
/// which is the opposite of what a correction is.
///
/// Neither case deletes: superseding marks the old item
/// [`ItemStatus::Superseded`] and leaves it, with an edge saying what replaced
/// it.
pub fn may_supersede(candidate: &MemoryItem, existing: &MemoryItem) -> Result<(), String> {
    if candidate.item_id == existing.item_id {
        return Err("an item cannot supersede itself".to_string());
    }
    match (&candidate.provenance, &existing.provenance) {
        // A person outranks everything.
        (Provenance::Operator { .. }, _) => Ok(()),
        // Nothing outranks a person.
        (_, Provenance::Operator { .. }) => Err(
            "an operator's correction cannot be superseded by a model or a tool; record a \
             contradiction instead, so both survive and a person can decide"
                .to_string(),
        ),
        // A receipt may correct a model's guess about the same thing.
        (Provenance::ToolReceipt { .. }, Provenance::Model { .. }) => Ok(()),
        // A model may withdraw its own proposal, and nothing more.
        (Provenance::Model { .. }, Provenance::Model { .. }) => {
            if existing.status.is_established() {
                Err(
                    "a model may not supersede an established fact; record a contradiction instead"
                        .to_string(),
                )
            } else {
                Ok(())
            }
        }
        (Provenance::Model { .. }, Provenance::ToolReceipt { .. }) => Err(
            "a model may not supersede what a tool actually returned; record a contradiction \
             instead"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::identity::{Role, User};

    pub(crate) fn session(id: &str, roles: Vec<Role>) -> Session {
        Session::open(User::new(id, id, roles))
    }

    pub(crate) fn item(kind: MemoryKind, provenance: Provenance) -> MemoryItem {
        MemoryItem {
            item_id: item_id(),
            revision: 1,
            kind,
            agent_id: "ag-1".into(),
            scope: MemoryScope::Task {
                task_id: "task-1".into(),
            },
            classification: Classification::Internal,
            acl: Acl::for_classification(Classification::Internal, None),
            creator_model_id: Some("model-a".into()),
            creator_run_id: Some("run-1".into()),
            provenance,
            content: "The design pressure is 10 bar.".into(),
            sources: Vec::new(),
            artifacts: Vec::new(),
            confidence: None,
            status: ItemStatus::Proposed,
            valid_from: "2026-01-01T00:00:00Z".into(),
            valid_until: None,
            supersedes: None,
            conflicts_with: Vec::new(),
            causal_parents: Vec::new(),
            idempotency_key: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            basis: None,
            depends_on: Vec::new(),
            revoked_readers: Vec::new(),
            authority: Authority::Graph,
        }
    }

    pub(crate) fn model() -> Provenance {
        Provenance::Model {
            model_id: "model-a".into(),
            run_id: "run-1".into(),
        }
    }

    /// A receipt of the right shape. Whether it is *backed* is a separate
    /// question the event log answers; see `receipts`.
    pub(crate) fn receipt() -> Provenance {
        Provenance::ToolReceipt {
            run_id: "run-1".into(),
            tool: "knowledge.search_authorized".into(),
            event_seq: 42,
            output_sha256: Some("c0ffee".into()),
        }
    }

    pub(crate) fn operator() -> Provenance {
        Provenance::Operator {
            user_id: "priya".into(),
        }
    }

    /// The headline rule: a model asserting something does not make it so.
    #[test]
    fn a_model_assertion_stays_a_proposal() {
        let outcome = admit(&item(MemoryKind::Fact, model()), &ReceiptVerdict::NotAReceipt);
        assert_eq!(outcome.status, ItemStatus::Proposed);
        assert!(outcome.because.contains("corroborates"));
    }

    /// And a confident model is still only proposing.
    #[test]
    fn confidence_does_not_admit_anything() {
        let mut confident = item(MemoryKind::Fact, model());
        confident.confidence = Some(1.0);
        assert_eq!(
            admit(&confident, &ReceiptVerdict::Verified).status,
            ItemStatus::Proposed,
            "a model was admitted because something said 'verified'"
        );
    }

    #[test]
    fn a_verified_tool_receipt_is_admitted_and_names_the_event() {
        let outcome = admit(
            &item(MemoryKind::ToolObservation, receipt()),
            &ReceiptVerdict::Verified,
        );
        assert_eq!(outcome.status, ItemStatus::Admitted);
        assert!(outcome.because.contains("event 42"), "{}", outcome.because);
    }

    /// Plan P02: a positive integer alone is not proof. A well-formed receipt
    /// the event log did not back, or that nobody checked, stays a proposal.
    #[test]
    fn a_well_formed_receipt_the_log_does_not_back_is_a_proposal() {
        for verdict in [
            ReceiptVerdict::NotAReceipt,
            ReceiptVerdict::Refused {
                because: "no such event".into(),
            },
        ] {
            let outcome = admit(&item(MemoryKind::ToolObservation, receipt()), &verdict);
            assert_eq!(outcome.status, ItemStatus::Proposed, "{verdict:?}");
        }
    }

    /// A receipt that names nothing checkable is not a receipt, whatever the
    /// verdict says.
    #[test]
    fn a_receipt_with_no_event_behind_it_is_not_corroboration() {
        for provenance in [
            Provenance::ToolReceipt {
                run_id: "run-1".into(),
                tool: "create_docx".into(),
                event_seq: 0,
                output_sha256: Some("aa".into()),
            },
            Provenance::ToolReceipt {
                run_id: "run-1".into(),
                tool: String::new(),
                event_seq: 7,
                output_sha256: Some("aa".into()),
            },
            Provenance::ToolReceipt {
                run_id: String::new(),
                tool: "create_docx".into(),
                event_seq: 7,
                output_sha256: Some("aa".into()),
            },
            Provenance::ToolReceipt {
                run_id: "run-1".into(),
                tool: "create_docx".into(),
                event_seq: 7,
                output_sha256: None,
            },
        ] {
            assert_eq!(
                admit(
                    &item(MemoryKind::ToolObservation, provenance),
                    &ReceiptVerdict::Verified
                )
                .status,
                ItemStatus::Proposed,
                "an unverifiable receipt was admitted"
            );
        }
    }

    #[test]
    fn a_person_is_admitted_directly() {
        assert_eq!(
            admit(
                &item(MemoryKind::Correction, operator()),
                &ReceiptVerdict::NotAReceipt
            )
            .status,
            ItemStatus::Admitted
        );
    }

    /// Plan §6: what a source says, what a tool measured, what a model
    /// inferred and what a person supplied are four different things.
    #[test]
    fn the_basis_is_derived_from_what_happened_and_not_chosen() {
        assert_eq!(Basis::of(&receipt()), Basis::SourceText);
        assert_eq!(
            Basis::of(&Provenance::ToolReceipt {
                run_id: "r".into(),
                tool: "calculation.evaluate_with_units".into(),
                event_seq: 1,
                output_sha256: Some("a".into()),
            }),
            Basis::Measured
        );
        assert_eq!(Basis::of(&model()), Basis::Inferred);
        assert_eq!(Basis::of(&operator()), Basis::Supplied);
        // An item written before the field existed still has an answer.
        let old = item(MemoryKind::Fact, model());
        assert_eq!(old.basis, None);
        assert_eq!(old.basis(), Basis::Inferred);
    }

    /// A reader whose access to one item was withdrawn cannot read it, whatever
    /// roles they hold.
    #[test]
    fn a_revoked_reader_cannot_read_the_item_and_others_still_can() {
        let mut shared = item(MemoryKind::Fact, operator());
        shared.revoked_readers = vec!["priya".into()];
        assert!(!shared.readable_by(&session("priya", vec![Role::Administrator]), None));
        assert!(shared.readable_by(&session("ravi", vec![Role::Employee]), None));
    }

    #[test]
    fn a_scratch_scope_key_round_trips() {
        let scratch = MemoryScope::Scratch {
            task_id: "task-9".into(),
            agent_id: "ag-3".into(),
        };
        assert_eq!(scratch.key(), "scratch:ag-3@task-9");
        assert_eq!(MemoryScope::from_key(&scratch.key()), Some(scratch));
        for scope in [
            MemoryScope::Task { task_id: "t".into() },
            MemoryScope::Workspace { project_id: "p".into() },
            MemoryScope::User { user_id: "u".into() },
        ] {
            assert_eq!(MemoryScope::from_key(&scope.key()), Some(scope));
        }
    }

    /// A body written before P02's fields existed still reads.
    #[test]
    fn an_item_written_before_the_p02_fields_still_reads() {
        let mut body = serde_json::to_value(item(MemoryKind::Fact, receipt())).expect("serialises");
        let fields = body.as_object_mut().expect("object");
        for added in ["basis", "dependsOn", "revokedReaders", "authority"] {
            fields.remove(added);
        }
        // Snake case: `rename_all` on this enum renames its variants, not the
        // fields inside them, which is how `event_seq` was always stored too.
        let provenance = fields["provenance"].as_object_mut().expect("provenance");
        assert!(
            provenance.remove("output_sha256").is_some(),
            "the fixture did not carry an output hash to remove"
        );
        let read: MemoryItem = serde_json::from_value(body).expect("an old body reads");
        assert_eq!(read.authority, Authority::Graph);
        assert!(read.depends_on.is_empty() && read.revoked_readers.is_empty());
        assert!(matches!(
            read.provenance,
            Provenance::ToolReceipt { output_sha256: None, .. }
        ));
    }

    /// A proposal is retrievable — that is how somebody finds out the model
    /// believes something wrong — and is never established.
    #[test]
    fn a_proposal_is_retrievable_but_not_established() {
        assert!(ItemStatus::Proposed.usable_as_evidence());
        assert!(!ItemStatus::Proposed.is_established());
        assert!(ItemStatus::Admitted.is_established());
        assert!(!ItemStatus::Rejected.usable_as_evidence());
        assert!(!ItemStatus::Tombstoned.usable_as_evidence());
    }

    // ── Supersession ─────────────────────────────────────────────────────

    #[test]
    fn an_operator_correction_supersedes_anything() {
        let correction = item(MemoryKind::Correction, operator());
        for existing in [
            item(MemoryKind::Fact, model()),
            item(MemoryKind::ToolObservation, receipt()),
            item(MemoryKind::Fact, operator()),
        ] {
            assert!(may_supersede(&correction, &existing).is_ok());
        }
    }

    /// The rule that makes a correction a correction.
    #[test]
    fn nothing_may_supersede_an_operators_correction() {
        let correction = item(MemoryKind::Correction, operator());
        for candidate in [
            item(MemoryKind::Fact, model()),
            item(MemoryKind::ToolObservation, receipt()),
        ] {
            let refusal = may_supersede(&candidate, &correction)
                .expect_err("an operator's correction was overridden");
            assert!(refusal.contains("contradiction"), "{refusal}");
        }
    }

    #[test]
    fn a_model_may_withdraw_its_own_proposal_but_not_an_established_fact() {
        let candidate = item(MemoryKind::Fact, model());

        let proposal = item(MemoryKind::Fact, model());
        assert!(may_supersede(&candidate, &proposal).is_ok());

        let mut established = item(MemoryKind::Fact, model());
        established.status = ItemStatus::Admitted;
        assert!(may_supersede(&candidate, &established).is_err());

        let observation = item(MemoryKind::ToolObservation, receipt());
        assert!(
            may_supersede(&candidate, &observation).is_err(),
            "a model overrode what a tool actually returned"
        );
    }

    #[test]
    fn a_receipt_may_correct_a_models_guess() {
        let observation = item(MemoryKind::ToolObservation, receipt());
        let guess = item(MemoryKind::Fact, model());
        assert!(may_supersede(&observation, &guess).is_ok());
    }

    #[test]
    fn an_item_cannot_supersede_itself() {
        let one = item(MemoryKind::Fact, operator());
        assert!(may_supersede(&one, &one).is_err());
    }

    // ── Reading ──────────────────────────────────────────────────────────

    /// A user-scope item belongs to one person, whatever roles anybody holds.
    #[test]
    fn a_user_scope_item_is_not_readable_by_another_person() {
        let mut mine = item(MemoryKind::Fact, operator());
        mine.scope = MemoryScope::User {
            user_id: "priya".into(),
        };
        assert!(mine.readable_by(&session("priya", vec![Role::Employee]), None));
        assert!(
            !mine.readable_by(&session("ada", vec![Role::Administrator]), None),
            "an administrator read another person's user-scope memory"
        );
    }

    /// Absence of a project is not a wildcard.
    #[test]
    fn a_workspace_item_is_not_readable_from_another_project_or_from_none() {
        let mut confined = item(MemoryKind::Fact, operator());
        confined.scope = MemoryScope::Workspace {
            project_id: "unit-four".into(),
        };
        confined.acl = Acl::for_classification(Classification::Internal, Some("unit-four"));

        let reader = session("priya", vec![Role::Employee]);
        assert!(confined.readable_by(&reader, Some("unit-four")));
        assert!(
            !confined.readable_by(&reader, Some("unit-five")),
            "cross-project inference"
        );
        assert!(
            !confined.readable_by(&reader, None),
            "naming no project read a confined item"
        );
    }

    #[test]
    fn a_tombstone_is_not_readable_and_its_row_still_exists() {
        let mut gone = item(MemoryKind::Fact, operator());
        gone.status = ItemStatus::Tombstoned;
        assert!(!gone.is_readable());
        assert!(!gone.readable_by(&session("priya", vec![Role::Employee]), None));
        assert_eq!(gone.status, ItemStatus::Tombstoned);
    }

    #[test]
    fn expiry_is_read_against_an_instant() {
        let mut expiring = item(MemoryKind::Fact, operator());
        expiring.valid_until = Some("2026-06-01T00:00:00Z".into());
        assert!(!expiring.is_expired_at("2026-05-31T23:59:59Z"));
        assert!(expiring.is_expired_at("2026-06-01T00:00:00Z"));
        assert!(expiring.is_expired_at("2026-07-01T00:00:00Z"));

        let forever = item(MemoryKind::Fact, operator());
        assert!(!forever.is_expired_at("2999-01-01T00:00:00Z"));
    }

    // ── Identity ─────────────────────────────────────────────────────────

    /// The same link drawn twice is one row; a different relation is another.
    #[test]
    fn an_edge_id_is_the_triple_and_nothing_else() {
        let a = edge_id("mi-1", EdgeKind::Supports, "mi-2");
        assert_eq!(a, edge_id("mi-1", EdgeKind::Supports, "mi-2"));
        assert_ne!(a, edge_id("mi-1", EdgeKind::Contradicts, "mi-2"));
        assert_ne!(a, edge_id("mi-2", EdgeKind::Supports, "mi-1"));
        assert!(a.starts_with("me-"));
    }

    /// Two agents establishing the same fact are two observations of it.
    /// Collapsing them would destroy the attribution the graph exists for.
    #[test]
    fn two_items_with_identical_content_get_different_ids() {
        let first = item(MemoryKind::Fact, model());
        let second = item(MemoryKind::Fact, model());
        assert_eq!(first.content, second.content);
        assert_ne!(first.item_id, second.item_id);
    }

    #[test]
    fn every_kind_and_edge_round_trips_through_its_string() {
        for kind in [
            MemoryKind::Goal,
            MemoryKind::Fact,
            MemoryKind::Constraint,
            MemoryKind::Correction,
            MemoryKind::Decision,
            MemoryKind::Plan,
            MemoryKind::OpenQuestion,
            MemoryKind::ToolObservation,
            MemoryKind::SourceRef,
            MemoryKind::ArtifactRef,
        ] {
            assert_eq!(MemoryKind::parse(kind.as_str()), Some(kind));
        }
        for edge in [
            EdgeKind::Supports,
            EdgeKind::Contradicts,
            EdgeKind::Supersedes,
            EdgeKind::DerivedFrom,
            EdgeKind::Cites,
            EdgeKind::PartOf,
            EdgeKind::Answers,
        ] {
            assert_eq!(EdgeKind::parse(edge.as_str()), Some(edge));
        }
        for status in [
            ItemStatus::Proposed,
            ItemStatus::Admitted,
            ItemStatus::Superseded,
            ItemStatus::Rejected,
            ItemStatus::Tombstoned,
            ItemStatus::Stale,
        ] {
            assert_eq!(ItemStatus::parse(status.as_str()), Some(status));
        }
        assert!(!ItemStatus::Stale.usable_as_evidence(), "a stale item was offered as evidence");
        assert_eq!(MemoryKind::parse("nonsense"), None);
    }

    /// A source is pinned to bytes and a place inside them, so a claim is
    /// checkable by somebody who was not there.
    #[test]
    fn a_source_reference_round_trips_exactly() {
        let reference = SourceRef {
            sha256: "ab12cd34".into(),
            locator: "page 12".into(),
            extraction_revision: Some("rev-3".into()),
        };
        let text = serde_json::to_string(&reference).expect("serialises");
        assert_eq!(
            serde_json::from_str::<SourceRef>(&text).expect("parses"),
            reference
        );
    }

    #[test]
    fn an_artifact_reference_names_a_revision_not_just_a_file() {
        let reference = ArtifactRef {
            artifact_id: "art-9".into(),
            revision: 3,
            sha256: "ffee".into(),
        };
        let text = serde_json::to_string(&reference).expect("serialises");
        let back: ArtifactRef = serde_json::from_str(&text).expect("parses");
        assert_eq!(back.revision, 3);
        assert_eq!(back, reference);
    }

    #[test]
    fn a_scope_key_tells_the_three_apart() {
        assert_eq!(
            MemoryScope::Task {
                task_id: "t".into()
            }
            .key(),
            "task:t"
        );
        assert_eq!(
            MemoryScope::Workspace {
                project_id: "t".into()
            }
            .key(),
            "workspace:t"
        );
        assert_eq!(
            MemoryScope::User {
                user_id: "t".into()
            }
            .key(),
            "user:t"
        );
    }
}

//! The memory graph's changefeed: what a reader is told has changed, and how a
//! reader that was disconnected catches up without a hole in the middle.
//!
//! ## Why a snapshot alone is not enough, and a feed alone is not either
//!
//! A view that polls [`super::runtime_store::MemoryGraph::snapshot`] shows the
//! truth and shows it late; a view fed only incremental changes has no starting
//! point. The pair is the usual answer, and the usual bug with the pair is the
//! window between them: take a snapshot, then subscribe, and every change that
//! landed in between is lost — silently, because nothing in either half knows
//! it happened.
//!
//! This is why [`MemorySnapshot`] carries a `cursor`. The snapshot is read
//! inside the same transaction that reads the revision, so the cursor names
//! exactly the last change the snapshot already contains. A reader subscribes
//! *from that cursor*, and the first batch it gets is everything after it. There
//! is no window, because the two facts were never observed separately.
//!
//! ## Commit before publish
//!
//! Every row in `agent_memory_log` is written in the same transaction as the
//! change it describes. Nothing is announced that is not durable: a crash
//! between the write and the announcement leaves a reader behind, which the
//! cursor fixes on reconnect, rather than ahead, which nothing can fix.
//!
//! The push side carries no payload at all — see
//! [`crate::commands::memory_graph`]. A window is told only that the graph
//! moved to revision N; it then asks, as itself, what *it* may see. That is the
//! difference between a broadcast and a notification, and it is the reason a
//! second window signed in as somebody else cannot receive a fact it is not
//! cleared for.
//!
//! ## Revocation is a change, not an absence
//!
//! When an item stops being readable — its ACL narrowed, its source
//! invalidated, its row tombstoned — the feed emits [`FeedChange::ItemDropped`]
//! rather than simply ceasing to mention it. A viewer holding a cached copy has
//! no other way to find out: "I have not heard about this in a while" is not a
//! signal, and a canvas that kept drawing a node whose permission was withdrawn
//! is the exact failure this feed exists to prevent.
//!
//! `ItemDropped` carries an id and nothing else. It never carries the label,
//! the content, the kind or the count, because those are the things the reader
//! just lost the right to.

use serde::{Deserialize, Serialize};

use super::runtime_memory::{MemoryEdge, MemoryItem};

/// Most changes one read returns.
///
/// Batching is not an optimisation here, it is the backpressure. A reader that
/// has been away for an hour must not be handed sixty thousand rows in one
/// message — the deserialise alone would stall the window it is drawn in. It
/// takes `MAX_BATCH` at a time, applies them, and asks again from the cursor it
/// reached; the backend never pushes faster than the reader pulls because the
/// backend never pushes rows at all.
pub const MAX_BATCH: usize = 500;

/// What happened to one subject.
///
/// Four cases rather than a `change: String` and a nullable body, so that a
/// reader cannot receive a "deleted" carrying content or an "updated" carrying
/// none. The shape of the message is the guarantee.
//
// The per-variant `rename_all` below is not redundant with the container's. On
// an enum, `rename_all` renames the *variants*; the fields inside a struct
// variant keep their Rust names unless each variant says so itself. Without
// them the wire would carry `item_id` while every other type this frontend
// reads carries `itemId`, and the mismatch would surface as an undefined field
// at a call site far from here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "camelCase")]
pub enum FeedChange {
    /// The item exists and this reader may see it, at this revision.
    ///
    /// Sent for a creation and for an update alike: a reader applying changes
    /// by id does not need them distinguished, and a reader that joined late
    /// would get "updated" for something it has never seen. Upsert is the only
    /// semantics that is correct for both.
    #[serde(rename_all = "camelCase")]
    ItemChanged { item: Box<MemoryItem> },
    /// The reader must forget this id. Tombstoned, revoked, or never theirs.
    ///
    /// Deliberately indistinguishable between those three. "You may no longer
    /// see this" and "this was deleted" would otherwise be a channel for
    /// learning which of the two happened, which is a fact about the item.
    #[serde(rename_all = "camelCase")]
    ItemDropped { item_id: String },
    /// The link exists and both of its ends are visible to this reader.
    #[serde(rename_all = "camelCase")]
    EdgeChanged { edge: Box<MemoryEdge> },
    /// The link must be forgotten — removed, or one of its ends went away.
    #[serde(rename_all = "camelCase")]
    EdgeDropped { edge_id: String },
}

impl FeedChange {
    /// The id this change is about, whichever kind it is.
    pub fn subject(&self) -> &str {
        match self {
            Self::ItemChanged { item } => &item.item_id,
            Self::ItemDropped { item_id } => item_id,
            Self::EdgeChanged { edge } => &edge.edge_id,
            Self::EdgeDropped { edge_id } => edge_id,
        }
    }

    /// Whether this change removes something rather than establishing it.
    pub fn is_removal(&self) -> bool {
        matches!(self, Self::ItemDropped { .. } | Self::EdgeDropped { .. })
    }
}

/// One entry of the feed, at its position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedEntry {
    /// The changefeed position. Strictly increasing across the whole store, so
    /// a reader's cursor is one number rather than one per scope.
    pub revision: i64,
    #[serde(flatten)]
    pub change: FeedChange,
    /// When the underlying write happened. RFC 3339, UTC. For display; ordering
    /// is by `revision`, never by this — two processes' clocks disagree and the
    /// log's own sequence does not.
    pub at: String,
}

/// What one read of the feed returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangeBatch {
    pub entries: Vec<FeedEntry>,
    /// Where to ask from next. Always set, even for an empty batch, so a reader
    /// that is caught up still advances past changes it was not cleared for.
    pub cursor: i64,
    /// True when more is waiting. The reader asks again immediately rather than
    /// waiting for the next push, which is what stops a long catch-up from
    /// needing one round trip per write.
    pub has_more: bool,
    /// The reader asked from a position the log no longer covers, so this batch
    /// is not a continuation of anything.
    ///
    /// The reader must take a fresh snapshot. Reported rather than papered
    /// over: silently returning "the changes we still have" would leave the
    /// reader's cache holding rows that were deleted while it was away, with
    /// nothing anywhere saying so.
    pub reset: bool,
}

impl ChangeBatch {
    /// A batch saying "start again from a snapshot".
    pub fn reset_at(cursor: i64) -> Self {
        Self {
            entries: Vec::new(),
            cursor,
            has_more: false,
            reset: true,
        }
    }

    /// An empty batch: nothing new, and the cursor still advances.
    pub fn caught_up_at(cursor: i64) -> Self {
        Self {
            entries: Vec::new(),
            cursor,
            has_more: false,
            reset: false,
        }
    }
}

/// A consistent picture of the graph, and the cursor that continues it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySnapshot {
    /// The items this reader may see, in this scope.
    pub items: Vec<MemoryItem>,
    /// The links whose *both* ends are in `items`.
    ///
    /// An edge to something the reader may not see is not reported at all — not
    /// as a stub with a hidden end, which would still say "there is something
    /// over there, and it contradicts this".
    pub edges: Vec<MemoryEdge>,
    /// Subscribe from here. Read in the same transaction as `items`, so no
    /// change can fall between the two.
    pub cursor: i64,
    /// What the running turn actually carried, pinned to the revision it
    /// carried it at. Empty when no run was named or no context was compiled.
    ///
    /// Revision-pinned rather than by id: an item corrected after the turn was
    /// compiled is a different claim, and highlighting the new one as "in
    /// context" would be a lie about what the model read.
    #[serde(default)]
    pub in_context: Vec<InContextItem>,
    /// The graph revision the context above was compiled against, when a run
    /// was named and had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_revision: Option<i64>,
}

/// One item the running turn carried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InContextItem {
    pub item_id: String,
    /// The revision the model actually read.
    pub revision: u64,
    /// Why the compiler chose it: `mandatory`, `neighbour`, `recall`,
    /// `evidence`.
    pub reason: String,
    /// False when the item has moved on since the turn was compiled — the
    /// model read revision 3 and the graph now holds revision 4.
    ///
    /// Drawn differently, because "this is in context" and "a corrected version
    /// of this is in context" are different things to tell a person who is
    /// about to trust an answer.
    pub current: bool,
}

/// Which table a log row is about.
///
/// Stored as a column rather than inferred from the id prefix. Ids are opaque
/// and a reader that decoded them would be depending on
/// [`super::runtime_memory::edge_id`] never changing its format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subject {
    Item,
    Edge,
}

impl Subject {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Item => "item",
            Self::Edge => "edge",
        }
    }

    pub fn parse(raw: &str) -> Self {
        // An unrecognised value is read as an item, which is what every row
        // written before this column existed is. Defaulting rather than
        // dropping: a log row nobody can classify is still a change, and losing
        // it would put a hole in a feed whose whole contract is not having one.
        match raw {
            "edge" => Self::Edge,
            _ => Self::Item,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The disclosure rule, pinned as a shape rather than as a convention: a
    /// drop is an id and a tag, and there is nowhere in it for content to go.
    #[test]
    fn a_drop_carries_an_id_and_nothing_else() {
        let change = FeedChange::ItemDropped {
            item_id: "mi-1".into(),
        };
        let json = serde_json::to_value(&change).expect("serialises");
        let object = json.as_object().expect("an object");
        assert_eq!(object.len(), 2, "a drop grew a field: {json}");
        assert_eq!(object["change"], "itemDropped");
        assert_eq!(object["itemId"], "mi-1");
    }

    #[test]
    fn a_reset_batch_carries_no_entries() {
        let batch = ChangeBatch::reset_at(42);
        assert!(batch.reset);
        assert!(batch.entries.is_empty());
        assert!(!batch.has_more);
        assert_eq!(batch.cursor, 42);
    }

    /// Caught up and reset are different answers. A reader that confused them
    /// would re-snapshot on every idle poll.
    #[test]
    fn caught_up_is_not_a_reset() {
        let batch = ChangeBatch::caught_up_at(7);
        assert!(!batch.reset);
        assert_eq!(batch.cursor, 7);
    }

    #[test]
    fn an_unknown_subject_reads_as_an_item() {
        assert_eq!(Subject::parse("edge"), Subject::Edge);
        assert_eq!(Subject::parse("item"), Subject::Item);
        assert_eq!(Subject::parse(""), Subject::Item);
    }
}

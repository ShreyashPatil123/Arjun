//! The commands behind the Knowledge screen.
//!
//! Read-only, and deliberately thin. Everything that decides *what a person may
//! see* lives in [`crate::knowledge::index`], where the clearance is part of the
//! SQL rather than a filter applied to results already fetched. These commands
//! establish who is asking and hand that session down; they do not re-implement
//! the rule, because a second copy of an access check is a second thing that can
//! disagree with the first.
//!
//! There is no ingest command here. Documents enter the index through the
//! document sidecar's pipeline, which classifies and chunks them; an "add
//! document" button that bypassed that would put unclassified text in front of
//! the retrieval path, which is the thing the classification exists to stop.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::commands::governance::{require_session, CurrentSession};
use crate::knowledge::index::{IndexedDocument, KnowledgeIndex, SearchResult};

/// Most passages one search returns.
///
/// The screen shows retrieved passages so a person can see what the model would
/// be given, not so it can be used as a document viewer. Twenty is enough to
/// judge whether the index holds what a question needs.
const MAX_RESULTS: usize = 20;

/// What the Knowledge screen reports about the index as a whole.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeHealth {
    /// Distinct, non-superseded documents in the index.
    pub documents: usize,
    /// How many of those this person is cleared to see.
    ///
    /// Shown next to the total because the difference is the honest answer to
    /// "why is the list shorter than the count?" — the alternative is a screen
    /// that looks broken to anyone not cleared for everything.
    pub visible_documents: usize,
    /// Passages across the documents this person can see.
    pub visible_passages: usize,
    /// True when the index opened and answered.
    pub readable: bool,
}

/// Every document the signed-in person may retrieve from.
#[tauri::command]
pub async fn knowledge_documents(
    index: State<'_, Arc<KnowledgeIndex>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<IndexedDocument>, String> {
    let signed_in = require_session(&session)?;
    index
        .documents(&signed_in)
        .map_err(|error| format!("the knowledge index could not be read: {error}"))
}

/// Searches the index as the signed-in person, returning passages with their
/// source and page.
///
/// The same call the agent's `knowledge.search_authorized` tool makes, so what
/// this screen shows is what a run would actually be given rather than an
/// approximation of it.
#[tauri::command]
pub async fn knowledge_search(
    query: String,
    limit: Option<usize>,
    index: State<'_, Arc<KnowledgeIndex>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<SearchResult>, String> {
    let signed_in = require_session(&session)?;
    let limit = limit.unwrap_or(MAX_RESULTS).clamp(1, MAX_RESULTS);
    index
        .search(&signed_in, &query, limit)
        .map_err(|error| format!("the search could not be run: {error}"))
}

/// The index's own state, for the header of the Knowledge screen.
#[tauri::command]
pub async fn knowledge_health(
    index: State<'_, Arc<KnowledgeIndex>>,
    session: State<'_, CurrentSession>,
) -> Result<KnowledgeHealth, String> {
    let signed_in = require_session(&session)?;

    // A count that cannot be read is reported as unreadable rather than as
    // zero. An empty index and an index nobody could open are different
    // problems with different fixes, and a zero here would send somebody off to
    // re-ingest documents that are already there.
    let Ok(documents) = index.document_count() else {
        return Ok(KnowledgeHealth {
            documents: 0,
            visible_documents: 0,
            visible_passages: 0,
            readable: false,
        });
    };

    let visible = index.documents(&signed_in).unwrap_or_default();
    Ok(KnowledgeHealth {
        documents,
        visible_documents: visible.len(),
        visible_passages: visible.iter().map(|doc| doc.chunks).sum(),
        readable: true,
    })
}

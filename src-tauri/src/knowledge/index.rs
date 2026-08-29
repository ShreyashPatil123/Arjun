//! Finding the right passage, and never returning one the asker may not see.
//!
//! PS step 22 is precise about the ordering: *"The policy gateway filters results
//! by document and user permissions **before** the passages reach the model."*
//! That word does the work. A filter applied afterwards has already let the
//! model see the text, and even discarding it leaks — result counts, ranking
//! positions and "no results" versus "results you cannot see" all say something
//! about material the asker was not cleared for.
//!
//! So clearance is part of the SQL. A passage the asker cannot see is not
//! fetched, not ranked, and not counted.
//!
//! ## Keyword search, for now
//!
//! Retrieval here is full-text search over the chunks. The plan calls for a
//! hybrid of keyword and vector search with a reranker over the merged set, and
//! the two model-backed halves are not built because no embedding model is
//! installed. What exists is real and useful on its own — exact terms like
//! `PV-2201` or `9.0 mm` are precisely where keyword search beats embeddings —
//! and [`SearchResult::retrieval`] says which method found each passage, so a
//! deployment on keyword-only never looks like one running the full pipeline.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use super::Chunk;
use crate::identity::Session;
use crate::policy::Classification;

/// How a passage was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Retrieval {
    /// Full-text keyword match.
    Keyword,
    /// Vector similarity. Not yet available — no embedding model is installed.
    Vector,
}

/// One passage, with everything needed to cite and to trust it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub text: String,
    pub page: u32,
    pub section_path: Vec<String>,
    pub classification: Classification,
    /// Lower is a better match, following the underlying ranking.
    pub score: f64,
    pub retrieval: Retrieval,
}

impl SearchResult {
    /// How a citation to this passage reads.
    pub fn citation(&self) -> String {
        if self.section_path.is_empty() {
            format!("{}, page {}", self.document_name, self.page)
        } else {
            format!(
                "{} — {}, page {}",
                self.document_name,
                self.section_path.join(" › "),
                self.page
            )
        }
    }
}

/// Turns what a person typed into a query FTS5 will read as they meant it.
///
/// FTS5 has its own query language: `-` means NOT, `*` is a prefix wildcard,
/// `OR`/`NOT`/`NEAR` are operators, and an unbalanced quote is a syntax error.
/// A refinery user does not know that and should not have to — they type
/// `PV-2201`, which FTS5 reads as "PV, but not 2201" and matches nothing at all.
/// Silently returning no results for the single most likely query in the domain
/// would be the worst kind of bug: invisible, and indistinguishable from the tag
/// genuinely not appearing anywhere.
///
/// So every token is wrapped as a quoted phrase. `PV-2201` becomes `"PV-2201"`,
/// which FTS5 matches as the adjacent sequence its tokeniser produced — exactly
/// what the person meant. Internal quotes are doubled, which is how FTS5 escapes
/// them, so no input can break out of the phrase and become an operator.
fn to_match_expression(query: &str) -> Option<String> {
    sanitise_fts(query)
}

/// Wraps an FTS5 query so the user's tokens are matched as adjacent phrases.
///
/// Public so the multimodal index can share the same sanitisation — a
/// refinery user typing `PT-2201` should land on the right page whether
/// the page is a passage, an image region, or a table row, and the FTS5
/// expression that does the matching is the same in all three cases.
pub fn sanitise_fts(query: &str) -> Option<String> {
    let phrases: Vec<String> = query
        .split_whitespace()
        .filter(|token| !token.trim_matches(|c: char| !c.is_alphanumeric()).is_empty())
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect();

    (!phrases.is_empty()).then(|| phrases.join(" "))
}

pub struct KnowledgeIndex {
    conn: Arc<Mutex<Connection>>,
}

impl KnowledgeIndex {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the knowledge index")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id             TEXT PRIMARY KEY,
                document_sha256 TEXT NOT NULL,
                document_name  TEXT NOT NULL,
                ordinal        INTEGER NOT NULL,
                page           INTEGER NOT NULL,
                section_path   TEXT NOT NULL,
                kind           TEXT NOT NULL,
                classification TEXT NOT NULL,
                superseded     INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_sha256);

            -- The searchable text lives in its own FTS table, joined by id.
            -- Kept external rather than as a contentless index so a passage can
            -- be read back for citation without re-reading the source document.
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text USING fts5(
                id UNINDEXED,
                body
            );",
        )?;
        Ok(())
    }

    /// How many distinct documents are currently retrievable.
    ///
    /// Superseded documents are excluded: they remain traceable, but they are
    /// not current guidance, so counting them would overstate what an answer
    /// can actually be grounded in.
    pub fn document_count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("the index lock is never poisoned");
        let count: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT document_sha256) FROM chunks WHERE superseded = 0",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Adds a document's chunks to the index, replacing anything held for it.
    ///
    /// Replacing rather than appending means re-reading a document with a better
    /// engine does not leave the old, worse passages behind to be retrieved
    /// alongside the new ones.
    pub fn index_document(
        &self,
        document_name: &str,
        classification: Classification,
        chunks: &[Chunk],
    ) -> Result<usize> {
        let Some(first) = chunks.first() else {
            return Ok(0);
        };
        let sha = first.document_sha256.clone();

        let mut conn = self.conn.lock().expect("index lock poisoned");
        let tx = conn.transaction()?;

        // Clear the old copy first, inside the same transaction, so a failure
        // never leaves a document half-replaced.
        {
            let mut stale = tx.prepare("SELECT id FROM chunks WHERE document_sha256 = ?1")?;
            let ids: Vec<String> = stale
                .query_map([&sha], |row| row.get(0))?
                .filter_map(Result::ok)
                .collect();
            for id in ids {
                tx.execute("DELETE FROM chunk_text WHERE id = ?1", [&id])?;
            }
        }
        tx.execute("DELETE FROM chunks WHERE document_sha256 = ?1", [&sha])?;

        for chunk in chunks {
            tx.execute(
                "INSERT INTO chunks
                    (id, document_sha256, document_name, ordinal, page, section_path,
                     kind, classification, superseded)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)",
                params![
                    chunk.id,
                    chunk.document_sha256,
                    document_name,
                    chunk.ordinal,
                    chunk.page,
                    serde_json::to_string(&chunk.section_path)?,
                    serde_json::to_string(&chunk.kind)?,
                    serde_json::to_string(&classification)?,
                ],
            )?;

            // The heading trail is indexed alongside the body, so a search for
            // "wall thickness" finds a passage sitting under that heading even
            // when the passage itself only says "minimum 9.0 mm".
            let searchable = if chunk.section_path.is_empty() {
                chunk.text.clone()
            } else {
                format!("{}\n{}", chunk.section_path.join(" "), chunk.text)
            };

            tx.execute(
                "INSERT INTO chunk_text (id, body) VALUES (?1, ?2)",
                params![chunk.id, searchable],
            )?;
        }

        tx.commit()?;
        log::info!("[KNOWLEDGE] indexed {} chunk(s) from {document_name}", chunks.len());
        Ok(chunks.len())
    }

    /// Marks a document superseded.
    ///
    /// Its passages stay in the index and remain traceable — a citation made
    /// last year must still resolve — but they are not returned as current
    /// guidance, which is what PS step 11 asks for.
    pub fn supersede(&self, document_sha256: &str) -> Result<()> {
        let conn = self.conn.lock().expect("index lock poisoned");
        conn.execute(
            "UPDATE chunks SET superseded = 1 WHERE document_sha256 = ?1",
            [document_sha256],
        )?;
        Ok(())
    }

    /// Searches, returning only what this person is cleared to see.
    ///
    /// The clearance test is a bound parameter in the query. It is not a filter
    /// over results, because by then the passages exist and their number is
    /// already informative.
    pub fn search(&self, session: &Session, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // A query with nothing searchable in it matches nothing, rather than
        // reaching FTS5 as a syntax error the caller would have to interpret.
        let Some(expression) = to_match_expression(query) else {
            return Ok(Vec::new());
        };

        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();

        if cleared.is_empty() {
            // Cleared for nothing. An empty result is the correct and complete
            // answer — and identical to what they would see if nothing matched,
            // which is the point.
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification, bm25(chunk_text) AS score
             FROM chunk_text t
             JOIN chunks c ON c.id = t.id
             WHERE chunk_text MATCH ?1
               AND c.superseded = 0
               AND c.classification IN ({placeholders})
             ORDER BY score
             LIMIT ?{}",
            cleared.len() + 2
        );

        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(&sql)?;

        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&expression];
        for value in &cleared {
            bound.push(value);
        }
        let limit = limit as i64;
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                document_sha256: row.get(1)?,
                document_name: row.get(2)?,
                text: row.get(3)?,
                page: row.get(4)?,
                section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                score: row.get(7)?,
                retrieval: Retrieval::Keyword,
            })
        })?;

        Ok(rows.filter_map(Result::ok).collect())
    }

    /// The chunks of one document between two pages, in reading order.
    ///
    /// ## Why a region and not the document
    ///
    /// The obvious way to let a model "read more" is to hand it the document.
    /// On this workbench a document is a 200-page drawing set, and pasting one
    /// into an 8k window does not give the model more context — it ends the run,
    /// because the inference server refuses a prompt at or over its window.
    ///
    /// So the model asks for the part it wants. A search returns a passage and
    /// its page; when that passage is cut off mid-clause, the next thing the
    /// model needs is pages 11 to 13 of that document, not the whole of it.
    ///
    /// ## The checks are the same ones search uses
    ///
    /// Clearance is applied here rather than at the caller, and it is applied
    /// the same way: a reader sees only classifications their roles are cleared
    /// for, and superseded documents are excluded. That matters because this is
    /// a second door into the same shelf — a retrieval path that filtered less
    /// than search would be a way to read by page number what could not be read
    /// by searching for it.
    ///
    /// `to_page` is inclusive. A reader who names one page gets that page.
    pub fn region(
        &self,
        session: &Session,
        document_sha256: &str,
        from_page: u32,
        to_page: u32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // A reversed range is a caller mistake, not a reason to return nothing
        // silently: read the way it was plainly meant.
        let (from_page, to_page) = if from_page <= to_page {
            (from_page, to_page)
        } else {
            (to_page, from_page)
        };

        let cleared: Vec<String> = Classification::ALL
            .iter()
            .filter(|c| {
                c.cleared_roles()
                    .iter()
                    .any(|role| session.user.roles.contains(role))
            })
            .filter_map(|c| serde_json::to_string(c).ok())
            .collect();

        if cleared.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.id, c.document_sha256, c.document_name, t.body, c.page,
                    c.section_path, c.classification
             FROM chunks c
             JOIN chunk_text t ON c.id = t.id
             WHERE c.document_sha256 = ?1
               AND c.page >= ?2
               AND c.page <= ?3
               AND c.superseded = 0
               AND c.classification IN ({placeholders})
             ORDER BY c.page, c.ordinal
             LIMIT ?{}",
            cleared.len() + 4
        );

        let conn = self.conn.lock().expect("index lock poisoned");
        let mut stmt = conn.prepare(&sql)?;

        let document = document_sha256.to_string();
        let from = from_page as i64;
        let to = to_page as i64;
        let limit = limit as i64;
        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&document, &from, &to];
        for value in &cleared {
            bound.push(value);
        }
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
            let section_path: String = row.get(5)?;
            let classification: String = row.get(6)?;
            Ok(SearchResult {
                chunk_id: row.get(0)?,
                document_sha256: row.get(1)?,
                document_name: row.get(2)?,
                text: row.get(3)?,
                page: row.get(4)?,
                section_path: serde_json::from_str(&section_path).unwrap_or_default(),
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                // Not a match against a query, so there is no relevance to
                // report. Zero rather than an invented figure: a score the
                // caller might sort by would be a lie about ranking.
                score: 0.0,
                retrieval: Retrieval::Keyword,
            })
        })?;

        Ok(rows.filter_map(Result::ok).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::{Chunk, ChunkKind};

    fn chunk(id: &str, sha: &str, ordinal: u32, text: &str, section: Vec<&str>) -> Chunk {
        Chunk {
            id: id.into(),
            document_sha256: sha.into(),
            ordinal,
            char_count: text.len() as u32,
            text: text.into(),
            page: 1,
            section_path: section.into_iter().map(String::from).collect(),
            kind: ChunkKind::Prose,
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        index: KnowledgeIndex,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        Fixture { _dir: dir, index }
    }

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn index_sop(f: &Fixture) {
        f.index
            .index_document(
                "Maintenance SOP rev C",
                Classification::ProcessDiagram,
                &[chunk(
                    "c1",
                    "sop",
                    0,
                    "Minimum acceptable wall thickness is 9.0 mm.",
                    vec!["4 Inspection", "4.2 Wall Thickness"],
                )],
            )
            .unwrap();
    }

    fn page_chunk(id: &str, sha: &str, ordinal: u32, page: u32, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            document_sha256: sha.into(),
            ordinal,
            char_count: text.len() as u32,
            text: text.into(),
            page,
            section_path: Vec::new(),
            kind: ChunkKind::Prose,
        }
    }

    /// A document with one chunk on each of pages 10 to 14.
    fn index_manual(f: &Fixture, classification: Classification) {
        let chunks: Vec<Chunk> = (10..=14)
            .map(|page| {
                page_chunk(
                    &format!("m{page}"),
                    "manual",
                    page,
                    page,
                    &format!("Manual text on page {page}."),
                )
            })
            .collect();
        f.index
            .index_document("Pump Manual", classification, &chunks)
            .unwrap();
    }

    #[test]
    fn a_region_returns_the_pages_asked_for_and_no_others() {
        // The point of loading a region rather than a document: the model asks
        // for three pages of a 200-page manual and gets three pages.
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::User]), "manual", 11, 13, 50)
            .unwrap();

        assert_eq!(
            hits.iter().map(|hit| hit.page).collect::<Vec<_>>(),
            vec![11, 12, 13]
        );
    }

    #[test]
    fn a_region_of_one_page_is_that_page() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::User]), "manual", 12, 12, 50)
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].page, 12);
    }

    #[test]
    fn a_region_reads_the_same_clearance_as_a_search() {
        // The failure this guards: a second door into the same shelf that
        // filters less than the first. Reading by page number must not return
        // what searching for the same words would refuse.
        let f = fixture();
        index_manual(&f, Classification::VendorNegotiation);

        // A knowledge administrator curates manuals and is not cleared for deal
        // terms — the same rule `search` applies.
        let refused = f
            .index
            .region(
                &session(vec![Role::KnowledgeAdministrator]),
                "manual",
                11,
                13,
                50,
            )
            .unwrap();
        assert!(refused.is_empty());

        let permitted = f
            .index
            .region(&session(vec![Role::User]), "manual", 11, 13, 50)
            .unwrap();
        assert_eq!(permitted.len(), 3);
    }

    #[test]
    fn a_region_of_a_superseded_document_is_empty() {
        // Superseded material stays traceable and stops being current guidance.
        // Reading it by page would be a way to quote a withdrawn revision.
        let f = fixture();
        index_manual(&f, Classification::Internal);
        f.index.supersede("manual").unwrap();

        assert!(f
            .index
            .region(&session(vec![Role::User]), "manual", 11, 13, 50)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_reversed_range_is_read_the_way_it_was_plainly_meant() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::User]), "manual", 13, 11, 50)
            .unwrap();

        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn a_region_is_bounded_by_its_limit_however_dense_the_pages_are() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::User]), "manual", 10, 14, 2)
            .unwrap();

        // Two, not five. The ceiling is what actually keeps a dense range from
        // filling the window.
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn a_region_of_a_document_nobody_indexed_is_empty_rather_than_an_error() {
        // A model naming a document it half-remembers should read "nothing
        // there" and search instead, not see a failure it will try to work
        // around.
        let f = fixture();
        index_manual(&f, Classification::Internal);

        assert!(f
            .index
            .region(&session(vec![Role::User]), "not-a-document", 1, 5, 50)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_region_comes_back_in_reading_order() {
        let f = fixture();
        index_manual(&f, Classification::Internal);

        let hits = f
            .index
            .region(&session(vec![Role::User]), "manual", 10, 14, 50)
            .unwrap();
        let pages: Vec<u32> = hits.iter().map(|hit| hit.page).collect();
        let mut sorted = pages.clone();
        sorted.sort_unstable();

        assert_eq!(pages, sorted);
    }

    #[test]
    fn a_cleared_user_finds_the_passage() {
        let f = fixture();
        index_sop(&f);

        let hits = f.index.search(&session(vec![Role::User]), "wall thickness", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("9.0 mm"));
    }

    /// The passage is found through its heading even though the body never says
    /// those words — which is what the heading trail is indexed for.
    #[test]
    fn a_passage_is_findable_through_the_heading_above_it() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "Minimum 9.0 mm.", vec!["4.2 Wall Thickness"])],
            )
            .unwrap();

        let hits = f.index.search(&session(vec![Role::User]), "thickness", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    /// The requirement this module exists for: an uncleared passage is never
    /// fetched, so it cannot reach a model even to be discarded.
    #[test]
    fn an_uncleared_passage_is_never_returned() {
        let f = fixture();
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price is 4.2 lakh per valve.", vec![])],
            )
            .unwrap();

        // The knowledge administrator is not cleared for commercial material.
        let hits = f
            .index
            .search(&session(vec![Role::KnowledgeAdministrator]), "valve", 10)
            .unwrap();
        assert!(hits.is_empty());

        // An ordinary user is.
        let hits = f.index.search(&session(vec![Role::User]), "valve", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    /// An auditor reads the record of what happened, not the documents.
    #[test]
    fn an_auditor_searching_finds_nothing_at_all() {
        let f = fixture();
        index_sop(&f);
        let hits = f.index.search(&session(vec![Role::Auditor]), "thickness", 10).unwrap();
        assert!(hits.is_empty());
    }

    /// "No results" and "results you cannot see" must look identical.
    #[test]
    fn an_uncleared_hit_is_indistinguishable_from_no_hit() {
        let f = fixture();
        f.index
            .index_document(
                "Vendor terms",
                Classification::VendorNegotiation,
                &[chunk("c1", "deal", 0, "Unit price per valve.", vec![])],
            )
            .unwrap();

        let uncleared = f
            .index
            .search(&session(vec![Role::KnowledgeAdministrator]), "valve", 10)
            .unwrap();
        let absent = f
            .index
            .search(&session(vec![Role::KnowledgeAdministrator]), "sasquatch", 10)
            .unwrap();

        assert_eq!(uncleared.len(), absent.len());
    }

    #[test]
    fn a_superseded_document_stays_traceable_but_is_not_returned() {
        let f = fixture();
        index_sop(&f);
        assert_eq!(f.index.search(&session(vec![Role::User]), "thickness", 10).unwrap().len(), 1);

        f.index.supersede("sop").unwrap();
        assert!(f.index.search(&session(vec![Role::User]), "thickness", 10).unwrap().is_empty());

        // Still held, so a citation made before it was superseded still resolves.
        let conn = f.index.conn.lock().unwrap();
        let held: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks WHERE document_sha256 = 'sop'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(held, 1);
    }

    /// Re-reading a document must not leave its old passages behind.
    #[test]
    fn re_indexing_replaces_rather_than_accumulates() {
        let f = fixture();
        f.index
            .index_document("SOP", Classification::Internal,
                &[chunk("c1", "sop", 0, "Old text about valves.", vec![])])
            .unwrap();
        f.index
            .index_document("SOP", Classification::Internal,
                &[chunk("c2", "sop", 0, "New text about valves.", vec![])])
            .unwrap();

        let hits = f.index.search(&session(vec![Role::User]), "valves", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("New text"));
    }

    #[test]
    fn results_carry_everything_a_citation_needs() {
        let f = fixture();
        index_sop(&f);
        let hit = &f.index.search(&session(vec![Role::User]), "thickness", 10).unwrap()[0];

        assert_eq!(
            hit.citation(),
            "Maintenance SOP rev C — 4 Inspection › 4.2 Wall Thickness, page 1"
        );
        assert_eq!(hit.retrieval, Retrieval::Keyword);
        assert_eq!(hit.document_sha256, "sop");
    }

    #[test]
    fn the_limit_is_respected() {
        let f = fixture();
        let chunks: Vec<_> = (0..20)
            .map(|i| chunk(&format!("c{i}"), "sop", i, "valve inspection notes", vec![]))
            .collect();
        f.index.index_document("SOP", Classification::Internal, &chunks).unwrap();

        let hits = f.index.search(&session(vec![Role::User]), "valve", 5).unwrap();
        assert_eq!(hits.len(), 5);
    }

    // ── Queries a person actually types ─────────────────────────────────

    /// The query most likely to be typed in a refinery, and the one FTS5 would
    /// silently read as "PV, but not 2201".
    #[test]
    fn a_tag_number_with_a_hyphen_is_found() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "Vessel PV-2201 was inspected in March.", vec![])],
            )
            .unwrap();

        let hits = f.index.search(&session(vec![Role::User]), "PV-2201", 10).unwrap();
        assert_eq!(hits.len(), 1, "a hyphenated tag number must be findable");
    }

    /// FTS5 operators typed as ordinary words must not change the query's meaning.
    #[test]
    fn fts_operators_typed_by_a_person_are_treated_as_words() {
        let f = fixture();
        f.index
            .index_document(
                "SOP",
                Classification::Internal,
                &[chunk("c1", "sop", 0, "The valve OR the pump may be isolated.", vec![])],
            )
            .unwrap();

        // Reads as the literal words, not as a boolean operator.
        let hits = f.index.search(&session(vec![Role::User]), "valve OR pump", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    /// An unbalanced quote is a syntax error in FTS5. It must not surface as one.
    #[test]
    fn punctuation_never_produces_a_query_error() {
        let f = fixture();
        index_sop(&f);

        for awkward in [
            "wall \"thickness",
            "thickness*",
            "9.0 mm -- minimum",
            "NEAR(wall thickness)",
            "^thickness",
            "(unbalanced",
        ] {
            let outcome = f.index.search(&session(vec![Role::User]), awkward, 10);
            assert!(outcome.is_ok(), "{awkward:?} should not error: {outcome:?}");
        }
    }

    #[test]
    fn a_query_with_nothing_searchable_matches_nothing_quietly() {
        let f = fixture();
        index_sop(&f);

        for empty in ["", "   ", "!!!", "--"] {
            let hits = f.index.search(&session(vec![Role::User]), empty, 10).unwrap();
            assert!(hits.is_empty(), "{empty:?} should match nothing");
        }
    }

    #[test]
    fn a_measurement_with_a_decimal_point_is_found() {
        let f = fixture();
        index_sop(&f);
        let hits = f.index.search(&session(vec![Role::User]), "9.0 mm", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn indexing_an_empty_document_is_harmless() {
        let f = fixture();
        assert_eq!(
            f.index.index_document("Empty", Classification::Internal, &[]).unwrap(),
            0
        );
    }
}

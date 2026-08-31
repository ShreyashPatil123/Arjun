//! Multimodal retrieval index: image regions, tables, document-type metadata.
//!
//! PS 26117's "multimodal engine" requirement lands hardest on the retrieval
//! side. A model that has to answer a question about a P&ID needs to be able
//! to find a tag on the drawing; a model summarising a datasheet needs to
//! find the row that says "design pressure: 14 bar" without scanning the
//! whole document. Neither is served by the keyword search over chunk text
//! alone, which is what the rest of the knowledge layer does.
//!
//! This module is the answer. Three new tables, one per modality:
//!
//! - `image_regions` — bounding boxes with a caption and a kind (text,
//!   image, table, figure, symbol). The caption is what the FTS path
//!   indexes; the box is what the citation renders.
//! - `tables` — headers, rows, page, and a flat-text mirror of the same
//!   cells for FTS. The flat text is what the FTS path indexes; the
//!   structured form is what the retriever returns.
//! - `documents` — one row per indexed document, carrying the auto-detected
//!   document type, the engine that read it, and the original SHA so
//!   re-ingestion can detect "this is a better read of the same file".
//!
//! All three are scoped by the same document SHA the prose index uses, so a
//! retrieval that finds a passage can ask for the regions on the same page
//! without going through the document service a second time. Clearance is
//! applied the same way: a row the asker cannot see is not returned.
//!
//! ## What this module deliberately does not do
//!
//! It does not embed regions. Vector search over region captions is a
//! future addition and is gated on an embedding model being installed, the
//! same as the rest of the knowledge layer. The text path is real and
//! useful on its own, and a deployment on text-only never looks like one
//! running the full pipeline — [`Method::Keyword`] vs [`Method::Vector`]
//! tells the caller which.
//!
//! It does not store the image itself. The image is on disk; we store the
//! path and the bounding box, not the pixels. Storing the pixels would
//! double the disk footprint of every document and would not improve
//! retrieval, because the multimodal retriever never shows the raw image
//! to the model — it shows the caption and the citation.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::identity::Session;
use crate::policy::Classification;

/// How a multimodal result was found. Mirrors [`crate::knowledge::Retrieval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Method {
    Keyword,
    Vector,
}

/// The kind of region, matching the sidecar's [`crate::documents::Region`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RegionKind {
    Text,
    Table,
    Figure,
    Image,
    Formula,
    Heading,
    /// P&ID-specific: an instrument bubble, valve tag, or equipment tag.
    /// Distinct from `Image` because it carries a semantic identity
    /// (`label`) a generic image does not.
    Symbol,
}

impl RegionKind {
    /// Parses the sidecar's string form. Unknown values are stored as
    /// `Text` rather than refused — a future sidecar version may add new
    /// kinds, and the index should not reject what the sidecar accepted.
    pub fn from_sidecar(s: &str) -> Self {
        match s {
            "text" => Self::Text,
            "table" => Self::Table,
            "figure" => Self::Figure,
            "image" => Self::Image,
            "formula" => Self::Formula,
            "heading" => Self::Heading,
            "symbol" => Self::Symbol,
            _ => Self::Text,
        }
    }
}

/// A bounding box on a page. Fractions, not pixels — see the sidecar's
/// `Region` for the same property.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BBox {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

/// One image region, with a caption and a kind.
///
/// `label` is the P&ID-specific tag identifier (e.g. `"PT-2201"`); for
/// non-P&ID regions it is the same as `caption` minus the kind prefix.
/// The split is so a search for `PT-2201` lands on the bounding box, while
/// a search for "pump" lands on the descriptive caption.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRegion {
    pub id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub page: u32,
    pub kind: RegionKind,
    pub bbox: BBox,
    /// The caption. Indexed for FTS.
    pub caption: String,
    /// The semantic label, distinct from the caption (e.g. `"instrument"`,
    /// `"pump"`). Used as a facet for filtering.
    pub label: Option<String>,
    /// 0.0–1.0. The sidecar's confidence that the box is in the right
    /// place. Distinct from the page's `confidence`, which is about text
    /// fidelity. Reported to the model so it can weight the citation.
    pub box_confidence: f32,
    /// Lower is a better match.
    pub score: f64,
    pub retrieval: Method,
}

impl ImageRegion {
    /// A short, citable description of this region.
    pub fn citation(&self) -> String {
        match (&self.label, &self.caption) {
            (Some(label), caption) if label != caption => {
                format!("{}, page {} ({})", self.document_name, self.page, caption)
            }
            _ => format!("{}, page {}", self.document_name, self.page),
        }
    }
}

/// One row of a table, plus its position on the page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableChunk {
    pub id: String,
    pub document_sha256: String,
    pub document_name: String,
    pub page: u32,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// The flat-text mirror of `rows`. Indexed for FTS so a query for
    /// "design pressure" finds the row that says it.
    pub flat_text: String,
    /// Page-position. Defaults to the page rectangle when the sidecar did
    /// not report a precise box.
    pub bbox: BBox,
    /// Lower is a better match.
    pub score: f64,
    pub retrieval: Method,
}

impl TableChunk {
    /// A short, citable description.
    pub fn citation(&self) -> String {
        format!("{}, page {} (table)", self.document_name, self.page)
    }
}

/// One indexed document's metadata. Carries the auto-detected type and the
/// engine that read it, so a reviewer can see how a document was classified
/// at a glance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentMeta {
    pub sha256: String,
    pub name: String,
    /// The auto-detected document type, from the sidecar's `doc_type` verdict.
    pub document_type: String,
    /// 0.0–1.0. The detector's confidence, when it returned one.
    pub type_confidence: f32,
    /// Whether the detector abstained. An abstained verdict means the
    /// detector is "I don't know"; the document is still indexed, but
    /// `document_type` is `"unknown"`.
    pub type_abstained: bool,
    /// Which engine read this document — `docling`, `text-layer`, or `pid`.
    /// Distinct from the runtime that produced the sidecar process, which
    /// is the same for every document on a single deployment.
    pub extraction_engine: String,
    /// The classification the document was indexed under.
    pub classification: Classification,
    /// Total pages in the document. Mirrors the sidecar's count.
    pub page_count: u32,
}

/// The multimodal index. Lives in the same database as the prose index,
/// which is what makes a search that finds a passage and a search that
/// finds a region address the same shelf.
pub struct MultimodalIndex {
    conn: Arc<Mutex<Connection>>,
}

impl MultimodalIndex {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the multimodal index")?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                sha256          TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                document_type   TEXT NOT NULL,
                type_confidence REAL NOT NULL DEFAULT 0.0,
                type_abstained  INTEGER NOT NULL DEFAULT 0,
                extraction_engine TEXT NOT NULL DEFAULT 'unknown',
                classification  TEXT NOT NULL,
                page_count      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS image_regions (
                id              TEXT PRIMARY KEY,
                document_sha256 TEXT NOT NULL,
                page            INTEGER NOT NULL,
                kind            TEXT NOT NULL,
                left            REAL NOT NULL,
                top             REAL NOT NULL,
                right           REAL NOT NULL,
                bottom          REAL NOT NULL,
                caption         TEXT NOT NULL,
                label           TEXT,
                box_confidence  REAL NOT NULL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS image_regions_doc_idx
                ON image_regions(document_sha256);

            CREATE VIRTUAL TABLE IF NOT EXISTS image_region_text USING fts5(
                id UNINDEXED,
                body
            );

            CREATE TABLE IF NOT EXISTS tables (
                id              TEXT PRIMARY KEY,
                document_sha256 TEXT NOT NULL,
                page            INTEGER NOT NULL,
                headers         TEXT NOT NULL,
                rows            TEXT NOT NULL,
                flat_text       TEXT NOT NULL,
                left            REAL NOT NULL DEFAULT 0.0,
                top             REAL NOT NULL DEFAULT 0.0,
                right           REAL NOT NULL DEFAULT 1.0,
                bottom          REAL NOT NULL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS tables_doc_idx
                ON tables(document_sha256);

            CREATE VIRTUAL TABLE IF NOT EXISTS table_text USING fts5(
                id UNINDEXED,
                body
            );",
        )?;
        Ok(())
    }

    /// Replace all of a document's multimodal content. Replaces, not
    /// appends, for the same reason the prose index does: a re-read with a
    /// better engine should not leave the old region list behind.
    pub fn index_document(
        &self,
        meta: &DocumentMeta,
        regions: &[NewRegion<'_>],
        tables: &[NewTable<'_>],
    ) -> Result<()> {
        let sha = meta.sha256.clone();
        let mut conn = self.conn.lock().expect("multimodal lock poisoned");
        let tx = conn.transaction()?;

        // Document metadata: replace on the same SHA, so a re-ingestion
        // updates type/engine/page-count rather than accumulating them.
        tx.execute(
            "INSERT OR REPLACE INTO documents
                (sha256, name, document_type, type_confidence, type_abstained,
                 extraction_engine, classification, page_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                meta.sha256,
                meta.name,
                meta.document_type,
                meta.type_confidence,
                meta.type_abstained as i32,
                meta.extraction_engine,
                serde_json::to_string(&meta.classification)?,
                meta.page_count as i64,
            ],
        )?;

        // Clear the old regions and tables for this document, in their own
        // transaction, so a failure mid-way leaves a coherent state.
        for table in ["image_regions", "tables"] {
            let sql = format!(
                "SELECT id FROM {table} WHERE document_sha256 = ?1"
            );
            let mut stmt = tx.prepare(&sql)?;
            let ids: Vec<String> = stmt
                .query_map([&sha], |row| row.get(0))?
                .filter_map(Result::ok)
                .collect();
            for id in ids {
                let fts = if table == "image_regions" {
                    "image_region_text"
                } else {
                    "table_text"
                };
                tx.execute(
                    &format!("DELETE FROM {fts} WHERE id = ?1"),
                    [&id],
                )?;
            }
            tx.execute(
                &format!("DELETE FROM {table} WHERE document_sha256 = ?1"),
                [&sha],
            )?;
        }

        for region in regions {
            tx.execute(
                "INSERT INTO image_regions
                    (id, document_sha256, page, kind, left, top, right, bottom,
                     caption, label, box_confidence)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    region.id,
                    sha,
                    region.page as i64,
                    region.kind_str,
                    region.bbox.left,
                    region.bbox.top,
                    region.bbox.right,
                    region.bbox.bottom,
                    region.caption,
                    region.label,
                    region.box_confidence,
                ],
            )?;
            // The FTS body is the caption joined with the label, so a
            // query for `PT-2201` matches both the descriptive caption
            // and the semantic label.
            let body = match region.label {
                Some(label) if label != region.caption => {
                    format!("{}\n{}", region.caption, label)
                }
                _ => region.caption.to_string(),
            };
            tx.execute(
                "INSERT INTO image_region_text (id, body) VALUES (?1, ?2)",
                params![region.id, body],
            )?;
        }

        for table in tables {
            tx.execute(
                "INSERT INTO tables
                    (id, document_sha256, page, headers, rows, flat_text,
                     left, top, right, bottom)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    table.id,
                    sha,
                    table.page as i64,
                    serde_json::to_string(&table.headers)?,
                    serde_json::to_string(&table.rows)?,
                    table.flat_text,
                    table.bbox.left,
                    table.bbox.top,
                    table.bbox.right,
                    table.bbox.bottom,
                ],
            )?;
            tx.execute(
                "INSERT INTO table_text (id, body) VALUES (?1, ?2)",
                params![table.id, table.flat_text],
            )?;
        }

        tx.commit()?;
        log::info!(
            "[MULTIMODAL] indexed {} region(s) and {} table(s) for {}",
            regions.len(),
            tables.len(),
            meta.name,
        );
        Ok(())
    }

    /// Search the image region FTS, returning at most `limit` regions.
    pub fn search_regions(
        &self,
        session: &Session,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ImageRegion>> {
        let Some(expression) = crate::knowledge::index::sanitise_fts(query) else {
            return Ok(Vec::new());
        };

        let cleared = cleared_classifications(session);
        if cleared.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT r.id, r.document_sha256, d.name, r.page, r.kind,
                    r.left, r.top, r.right, r.bottom,
                    r.caption, r.label, r.box_confidence,
                    bm25(image_region_text) AS score
             FROM image_region_text t
             JOIN image_regions r ON r.id = t.id
             JOIN documents d ON d.sha256 = r.document_sha256
             WHERE image_region_text MATCH ?1
               AND d.classification IN ({placeholders})
             ORDER BY score
             LIMIT ?{}",
            cleared.len() + 2
        );

        let conn = self.conn.lock().expect("multimodal lock poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&expression];
        for value in &cleared {
            bound.push(value);
        }
        let limit = limit as i64;
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
            let id: String = row.get(0)?;
            let sha: String = row.get(1)?;
            let name: String = row.get(2)?;
            let page: i64 = row.get(3)?;
            let kind: String = row.get(4)?;
            let left: f32 = row.get(5)?;
            let top: f32 = row.get(6)?;
            let right: f32 = row.get(7)?;
            let bottom: f32 = row.get(8)?;
            let caption: String = row.get(9)?;
            let label: Option<String> = row.get(10)?;
            let box_confidence: f32 = row.get(11)?;
            let score: f64 = row.get(12)?;
            Ok(ImageRegion {
                id,
                document_sha256: sha,
                document_name: name,
                page: page as u32,
                kind: RegionKind::from_sidecar(&kind),
                bbox: BBox { left, top, right, bottom },
                caption,
                label,
                box_confidence,
                score,
                retrieval: Method::Keyword,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Search the table FTS, returning at most `limit` tables.
    pub fn search_tables(
        &self,
        session: &Session,
        query: &str,
        limit: usize,
    ) -> Result<Vec<TableChunk>> {
        let Some(expression) = crate::knowledge::index::sanitise_fts(query) else {
            return Ok(Vec::new());
        };

        let cleared = cleared_classifications(session);
        if cleared.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = cleared.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT ta.id, ta.document_sha256, d.name, ta.page,
                    ta.headers, ta.rows, ta.flat_text,
                    ta.left, ta.top, ta.right, ta.bottom,
                    bm25(table_text) AS score
             FROM table_text t
             JOIN tables ta ON ta.id = t.id
             JOIN documents d ON d.sha256 = ta.document_sha256
             WHERE table_text MATCH ?1
               AND d.classification IN ({placeholders})
             ORDER BY score
             LIMIT ?{}",
            cleared.len() + 2
        );

        let conn = self.conn.lock().expect("multimodal lock poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let mut bound: Vec<&dyn rusqlite::ToSql> = vec![&expression];
        for value in &cleared {
            bound.push(value);
        }
        let limit = limit as i64;
        bound.push(&limit);

        let rows = stmt.query_map(bound.as_slice(), |row| {
            let id: String = row.get(0)?;
            let sha: String = row.get(1)?;
            let name: String = row.get(2)?;
            let page: i64 = row.get(3)?;
            let headers: String = row.get(4)?;
            let rows_str: String = row.get(5)?;
            let flat_text: String = row.get(6)?;
            let left: f32 = row.get(7)?;
            let top: f32 = row.get(8)?;
            let right: f32 = row.get(9)?;
            let bottom: f32 = row.get(10)?;
            let score: f64 = row.get(11)?;
            Ok(TableChunk {
                id,
                document_sha256: sha,
                document_name: name,
                page: page as u32,
                headers: serde_json::from_str(&headers).unwrap_or_default(),
                rows: serde_json::from_str(&rows_str).unwrap_or_default(),
                flat_text,
                bbox: BBox { left, top, right, bottom },
                score,
                retrieval: Method::Keyword,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Returns metadata for a single document, or `None` if not indexed.
    pub fn document_meta(&self, sha256: &str) -> Result<Option<DocumentMeta>> {
        let conn = self.conn.lock().expect("multimodal lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT sha256, name, document_type, type_confidence, type_abstained,
                    extraction_engine, classification, page_count
             FROM documents WHERE sha256 = ?1",
        )?;
        let mut rows = stmt.query([sha256])?;
        if let Some(row) = rows.next()? {
            let sha: String = row.get(0)?;
            let name: String = row.get(1)?;
            let doc_type: String = row.get(2)?;
            let type_confidence: f32 = row.get(3)?;
            let type_abstained: i64 = row.get(4)?;
            let engine: String = row.get(5)?;
            let classification: String = row.get(6)?;
            let page_count: i64 = row.get(7)?;
            Ok(Some(DocumentMeta {
                sha256: sha,
                name,
                document_type: doc_type,
                type_confidence,
                type_abstained: type_abstained != 0,
                extraction_engine: engine,
                classification: serde_json::from_str(&classification)
                    .unwrap_or(Classification::Internal),
                page_count: page_count as u32,
            }))
        } else {
            Ok(None)
        }
    }

    /// Marks a document's multimodal content as retired. The rows are
    /// deleted, so a future search for the document's regions is empty
    /// — different from a search for a passage, where a retired document's
    /// chunks stay in the table for traceability.
    pub fn retire_document(&self, sha256: &str) -> Result<()> {
        let conn = self.conn.lock().expect("multimodal lock poisoned");
        let mut ids: Vec<String> = Vec::new();
        for table in ["image_regions", "tables"] {
            let sql = format!(
                "SELECT id FROM {table} WHERE document_sha256 = ?1"
            );
            let mut stmt = conn.prepare(&sql)?;
            ids.extend(
                stmt.query_map([sha256], |row| row.get(0))?
                    .filter_map(Result::ok),
            );
        }
        for id in &ids {
            conn.execute("DELETE FROM image_region_text WHERE id = ?1", [id])?;
            conn.execute("DELETE FROM table_text WHERE id = ?1", [id])?;
        }
        conn.execute("DELETE FROM image_regions WHERE document_sha256 = ?1", [sha256])?;
        conn.execute("DELETE FROM tables WHERE document_sha256 = ?1", [sha256])?;
        conn.execute("DELETE FROM documents WHERE sha256 = ?1", [sha256])?;
        Ok(())
    }
}

/// Builds an [`ImageRegion`] from a sidecar `Region`.
/// The function is a small adapter so the data flow from the sidecar to
/// the index is explicit at the call site and the sidecar's types do not
/// leak into the index.
pub fn region_from_sidecar(
    id: String,
    document_sha256: String,
    page: u32,
    region: &crate::documents::Region,
) -> (String, BBox, String, Option<String>, f32, RegionKind) {
    (
        id,
        BBox {
            left: region.left,
            top: region.top,
            right: region.right,
            bottom: region.bottom,
        },
        region.caption.clone().unwrap_or_default(),
        region.label.clone(),
        region.box_confidence,
        RegionKind::from_sidecar(&region.kind),
    )
}

/// A new region to be indexed. Fields are borrowed for the index call's
/// lifetime, which is the lifetime of the sidecar's extraction result.
pub struct NewRegion<'a> {
    pub id: &'a str,
    pub page: u32,
    pub kind_str: &'a str,
    pub bbox: BBox,
    pub caption: &'a str,
    pub label: Option<&'a str>,
    pub box_confidence: f32,
}

/// A new table to be indexed. As with [`NewRegion`], the strings are
/// borrowed from the sidecar's extraction result.
pub struct NewTable<'a> {
    pub id: &'a str,
    pub page: u32,
    pub headers: &'a [String],
    pub rows: &'a [Vec<String>],
    pub flat_text: &'a str,
    pub bbox: BBox,
}

/// What the asker is cleared to read. The set is computed once per call
/// and bound into the SQL rather than applied as a post-filter, for the
/// same reason the prose index does it: a filter applied afterwards has
/// already let the model see the row count, and the count is information.
fn cleared_classifications(session: &Session) -> Vec<String> {
    Classification::ALL
        .iter()
        .filter(|c| {
            c.cleared_roles()
                .iter()
                .any(|role| session.user.roles.contains(role))
        })
        .filter_map(|c| serde_json::to_string(c).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, Session, User};

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn fixture() -> (tempfile::TempDir, MultimodalIndex) {
        let dir = tempfile::tempdir().unwrap();
        let index = MultimodalIndex::open(dir.path()).unwrap();
        (dir, index)
    }

    fn meta(sha: &str, name: &str, doc_type: &str) -> DocumentMeta {
        DocumentMeta {
            sha256: sha.into(),
            name: name.into(),
            document_type: doc_type.into(),
            type_confidence: 0.95,
            type_abstained: false,
            extraction_engine: "docling".into(),
            classification: Classification::ProcessDiagram,
            page_count: 1,
        }
    }

    #[test]
    fn indexes_and_retrieves_image_regions() {
        let (_dir, idx) = fixture();
        let m = meta("doc-1", "P&ID", "pid");
        let bbox = BBox { left: 0.1, top: 0.2, right: 0.3, bottom: 0.4 };
        let regions = vec![NewRegion {
            id: "r1",
            page: 1,
            kind_str: "symbol",
            bbox,
            caption: "PT-2201",
            label: Some("instrument"),
            box_confidence: 0.7,
        }];
        idx.index_document(&m, &regions, &[]).unwrap();
        let results = idx.search_regions(&session(vec![Role::Employee]), "PT-2201", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].caption, "PT-2201");
        assert_eq!(results[0].label.as_deref(), Some("instrument"));
        assert_eq!(results[0].kind, RegionKind::Symbol);
    }

    #[test]
    fn indexes_and_retrieves_tables() {
        let (_dir, idx) = fixture();
        let m = meta("doc-2", "Datasheet", "datasheet");
        let headers = vec!["Parameter".into(), "Value".into()];
        let rows = vec![
            vec!["Design pressure".into(), "14 bar".into()],
            vec!["Material".into(), "SS 316".into()],
        ];
        let flat = "Parameter: Design pressure | Value: 14 bar\nParameter: Material | Value: SS 316";
        let tables = vec![NewTable {
            id: "t1",
            page: 1,
            headers: &headers,
            rows: &rows,
            flat_text: flat,
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
        }];
        idx.index_document(&m, &[], &tables).unwrap();
        let results = idx.search_tables(&session(vec![Role::Employee]), "design pressure", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].headers, headers);
        assert_eq!(results[0].rows, rows);
    }

    #[test]
    fn reindexing_replaces_old_regions() {
        let (_dir, idx) = fixture();
        let m = meta("doc-3", "P&ID", "pid");
        let first = vec![NewRegion {
            id: "r1", page: 1, kind_str: "symbol",
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
            caption: "old tag", label: None, box_confidence: 0.5,
        }];
        idx.index_document(&m, &first, &[]).unwrap();
        let second = vec![NewRegion {
            id: "r2", page: 1, kind_str: "symbol",
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
            caption: "new tag", label: None, box_confidence: 0.5,
        }];
        idx.index_document(&m, &second, &[]).unwrap();
        // The old region is gone; the new one is searchable.
        let by_old = idx.search_regions(&session(vec![Role::Employee]), "old", 5).unwrap();
        let by_new = idx.search_regions(&session(vec![Role::Employee]), "new", 5).unwrap();
        assert!(by_old.is_empty(), "old region should be replaced");
        assert_eq!(by_new.len(), 1);
    }

    #[test]
    fn clearance_filters_results() {
        let (_dir, idx) = fixture();
        let mut m = meta("doc-4", "Restricted", "datasheet");
        m.classification = Classification::VendorNegotiation;
        let regions = vec![NewRegion {
            id: "r1", page: 1, kind_str: "symbol",
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
            caption: "secret tag", label: None, box_confidence: 1.0,
        }];
        idx.index_document(&m, &regions, &[]).unwrap();
        // The legacy Auditor role is not cleared for any classification.
        // Pinned here so a regression that re-enables a legacy role is
        // caught at the search level.
        let s = session(vec![Role::Auditor]);
        let results = idx.search_regions(&s, "secret", 5).unwrap();
        assert!(results.is_empty(), "clearance must filter the row out");
    }

    #[test]
    fn document_meta_round_trips() {
        let (_dir, idx) = fixture();
        let m = meta("doc-5", "P&ID", "pid");
        idx.index_document(&m, &[], &[]).unwrap();
        let loaded = idx.document_meta("doc-5").unwrap().unwrap();
        assert_eq!(loaded.name, "P&ID");
        assert_eq!(loaded.document_type, "pid");
        assert_eq!(loaded.classification, Classification::ProcessDiagram);
    }

    #[test]
    fn retire_removes_everything() {
        let (_dir, idx) = fixture();
        let m = meta("doc-6", "P&ID", "pid");
        let regions = vec![NewRegion {
            id: "r1", page: 1, kind_str: "symbol",
            bbox: BBox { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 },
            caption: "tag", label: None, box_confidence: 1.0,
        }];
        idx.index_document(&m, &regions, &[]).unwrap();
        idx.retire_document("doc-6").unwrap();
        let results = idx.search_regions(&session(vec![Role::Employee]), "tag", 5).unwrap();
        assert!(results.is_empty());
        assert!(idx.document_meta("doc-6").unwrap().is_none());
    }

    #[test]
    fn region_kind_from_sidecar_handles_unknowns() {
        // A future sidecar may add new kinds; the index should not panic.
        assert_eq!(RegionKind::from_sidecar("text"), RegionKind::Text);
        assert_eq!(RegionKind::from_sidecar("symbol"), RegionKind::Symbol);
        assert_eq!(RegionKind::from_sidecar("something-new"), RegionKind::Text);
    }
}

//! The audit record — what happened, in an order nobody can quietly revise.
//!
//! PS 26117 asks the system to prove what it did. A log that can be edited after
//! the fact proves nothing, so this one is built so that tampering is *detectable*
//! rather than merely discouraged:
//!
//! 1. **Append-only in the database.** SQLite triggers abort any `UPDATE` or
//!    `DELETE` on the table. Not a convention the application agrees to follow —
//!    a rule the storage engine enforces on every writer, including a person with
//!    a SQLite shell.
//! 2. **Hash-chained.** Each row carries the hash of the row before it, so its
//!    contents and its position are both sealed. Changing one row breaks every
//!    hash after it, and [`AuditService::verify_chain`] reports where.
//!
//! The two work together. Triggers stop the easy edit; the chain catches the hard
//! one, where someone drops the triggers first. Neither makes tampering
//! impossible — a determined administrator owns the file — but both make it
//! impossible to do *silently*, which is the property an auditor actually needs.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod provenance_hmac;

/// The kind of thing that happened. Kept coarse — detail belongs in `detail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AuditKind {
    /// ARJUN moved between Provisioning and Work.
    ModeChanged,
    /// The broker permitted or refused an outbound call.
    EgressDecision,
    /// The policy gateway allowed or refused an operation.
    PolicyDecision,
    /// A user signed in or out.
    Session,
    /// A model was registered, enabled or removed.
    ModelRegistry,
    /// A document or collection entered or left the knowledge base.
    Knowledge,
    /// A task was planned, ran, or finished.
    Task,
    /// A human approved or rejected a proposed action.
    Approval,
}

impl AuditKind {
    /// Stable string used in the database and in the hash. Changing one of these
    /// would invalidate every existing chain, so they are written out explicitly
    /// rather than derived from the variant name.
    pub const fn as_str(self) -> &'static str {
        match self {
            AuditKind::ModeChanged => "mode_changed",
            AuditKind::EgressDecision => "egress_decision",
            AuditKind::PolicyDecision => "policy_decision",
            AuditKind::Session => "session",
            AuditKind::ModelRegistry => "model_registry",
            AuditKind::Knowledge => "knowledge",
            AuditKind::Task => "task",
            AuditKind::Approval => "approval",
        }
    }

    fn from_str(raw: &str) -> Option<Self> {
        Some(match raw {
            "mode_changed" => AuditKind::ModeChanged,
            "egress_decision" => AuditKind::EgressDecision,
            "policy_decision" => AuditKind::PolicyDecision,
            "session" => AuditKind::Session,
            "model_registry" => AuditKind::ModelRegistry,
            "knowledge" => AuditKind::Knowledge,
            "task" => AuditKind::Task,
            "approval" => AuditKind::Approval,
            _ => return None,
        })
    }
}

/// One sealed record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditEntry {
    pub seq: i64,
    pub at: DateTime<Utc>,
    /// Who caused it — a user id, or `system` for the application itself.
    pub actor: String,
    pub kind: AuditKind,
    /// One line, written for a person reading the log.
    pub summary: String,
    /// Structured context. Deliberately *not* the document text: PS step 14 is
    /// explicit that sensitive contents must not be copied into a log that more
    /// people can read than could read the original.
    pub detail: Option<serde_json::Value>,
    pub hash: String,
}

/// The result of walking the chain from the beginning.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChainVerification {
    pub entries_checked: usize,
    pub intact: bool,
    /// The first row whose recomputed hash disagrees with the stored one.
    /// Everything before it is sound; everything after is unverifiable.
    pub first_broken_seq: Option<i64>,
    pub detail: String,
}

/// Hash of the row before the first — the anchor the chain hangs from.
const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Computes the seal for one row.
///
/// Fields are joined with a separator that cannot occur inside them (they are
/// all either machine-generated or JSON), so no combination of contents can be
/// rearranged into a different row with the same hash.
fn seal(
    prev_hash: &str,
    seq: i64,
    at: &DateTime<Utc>,
    actor: &str,
    kind: AuditKind,
    summary: &str,
    detail: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(seq.to_string().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(at.to_rfc3339().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(actor.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(kind.as_str().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(summary.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(detail.unwrap_or("").as_bytes());
    format!("{:x}", hasher.finalize())
}

pub struct AuditService {
    conn: Arc<Mutex<Connection>>,
}

impl AuditService {
    /// Opens the audit log beside the rest of the application data.
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)
            .with_context(|| format!("could not create {}", app_data_dir.display()))?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the audit database")?;
        Self::from_connection(conn)
    }

    /// Used by the tests against an in-memory database.
    fn from_connection(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS audit_log (
                seq       INTEGER PRIMARY KEY AUTOINCREMENT,
                at        TEXT NOT NULL,
                actor     TEXT NOT NULL,
                kind      TEXT NOT NULL,
                summary   TEXT NOT NULL,
                detail    TEXT,
                prev_hash TEXT NOT NULL,
                hash      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS audit_log_kind_idx ON audit_log(kind);

            -- Append-only, enforced by the storage engine rather than by
            -- convention. These abort for any writer on this connection,
            -- including one that is not ARJUN.
            CREATE TRIGGER IF NOT EXISTS audit_log_is_append_only_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only: rows cannot be modified');
            END;

            CREATE TRIGGER IF NOT EXISTS audit_log_is_append_only_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only: rows cannot be deleted');
            END;
            ",
        )
        .context("could not prepare the audit schema")?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Appends one sealed record and returns it.
    pub fn record(
        &self,
        actor: &str,
        kind: AuditKind,
        summary: impl Into<String>,
        detail: Option<serde_json::Value>,
    ) -> Result<AuditEntry> {
        let summary = summary.into();
        let detail_text = match &detail {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };
        let at = Utc::now();

        let conn = self.conn.lock().expect("audit lock poisoned");

        // Reading the tail and writing the new row happen inside one immediate
        // transaction. Without it, two concurrent writers could read the same
        // tail and produce two rows claiming the same predecessor, which would
        // fail verification later for no good reason.
        conn.execute_batch("BEGIN IMMEDIATE")?;

        let result = (|| -> Result<AuditEntry> {
            let (prev_seq, prev_hash): (i64, String) = conn
                .query_row(
                    "SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap_or((0, GENESIS_HASH.to_string()));

            let seq = prev_seq + 1;
            let hash = seal(
                &prev_hash,
                seq,
                &at,
                actor,
                kind,
                &summary,
                detail_text.as_deref(),
            );

            conn.execute(
                "INSERT INTO audit_log (seq, at, actor, kind, summary, detail, prev_hash, hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    seq,
                    at.to_rfc3339(),
                    actor,
                    kind.as_str(),
                    summary,
                    detail_text,
                    prev_hash,
                    hash
                ],
            )?;

            Ok(AuditEntry {
                seq,
                at,
                actor: actor.to_string(),
                kind,
                summary,
                detail,
                hash,
            })
        })();

        match result {
            Ok(entry) => {
                conn.execute_batch("COMMIT")?;
                log::info!("[AUDIT] #{} {} — {}", entry.seq, kind.as_str(), entry.summary);
                Ok(entry)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Most recent entries, newest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<AuditEntry>> {
        let conn = self.conn.lock().expect("audit lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT seq, at, actor, kind, summary, detail, hash
             FROM audit_log ORDER BY seq DESC LIMIT ?1",
        )?;

        let rows = stmt.query_map([limit as i64], |row| {
            let at: String = row.get(1)?;
            let kind: String = row.get(3)?;
            let detail: Option<String> = row.get(5)?;
            Ok(AuditEntry {
                seq: row.get(0)?,
                at: DateTime::parse_from_rfc3339(&at)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                actor: row.get(2)?,
                kind: AuditKind::from_str(&kind).unwrap_or(AuditKind::Task),
                summary: row.get(4)?,
                detail: detail.and_then(|d| serde_json::from_str(&d).ok()),
                hash: row.get(6)?,
            })
        })?;

        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Walks the chain from the first row, recomputing every seal.
    ///
    /// Reports the *first* break rather than a count: once one link is broken
    /// every later hash is derived from a value that is already wrong, so a
    /// tally of mismatches would overstate how much was actually altered.
    pub fn verify_chain(&self) -> Result<ChainVerification> {
        let conn = self.conn.lock().expect("audit lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT seq, at, actor, kind, summary, detail, prev_hash, hash
             FROM audit_log ORDER BY seq ASC",
        )?;

        let mut expected_prev = GENESIS_HASH.to_string();
        let mut checked = 0usize;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let seq: i64 = row.get(0)?;
            let at_raw: String = row.get(1)?;
            let actor: String = row.get(2)?;
            let kind_raw: String = row.get(3)?;
            let summary: String = row.get(4)?;
            let detail: Option<String> = row.get(5)?;
            let stored_prev: String = row.get(6)?;
            let stored_hash: String = row.get(7)?;

            checked += 1;

            if stored_prev != expected_prev {
                return Ok(ChainVerification {
                    entries_checked: checked,
                    intact: false,
                    first_broken_seq: Some(seq),
                    detail: format!(
                        "Entry {seq} claims to follow a different record than the one before it. \
                         A row was inserted, removed or reordered."
                    ),
                });
            }

            let at = DateTime::parse_from_rfc3339(&at_raw)
                .map(|d| d.with_timezone(&Utc))
                .map_err(|_| anyhow::anyhow!("entry {seq} has an unreadable timestamp"))?;
            let kind = AuditKind::from_str(&kind_raw)
                .ok_or_else(|| anyhow::anyhow!("entry {seq} has an unknown kind {kind_raw:?}"))?;

            let recomputed = seal(
                &stored_prev,
                seq,
                &at,
                &actor,
                kind,
                &summary,
                detail.as_deref(),
            );

            if recomputed != stored_hash {
                return Ok(ChainVerification {
                    entries_checked: checked,
                    intact: false,
                    first_broken_seq: Some(seq),
                    detail: format!(
                        "Entry {seq} does not match its seal — its contents were altered after \
                         it was written. Entries 1 to {} remain verifiable.",
                        seq - 1
                    ),
                });
            }

            expected_prev = stored_hash;
        }

        Ok(ChainVerification {
            entries_checked: checked,
            intact: true,
            first_broken_seq: None,
            detail: if checked == 0 {
                "The audit log is empty. Nothing has been recorded yet.".to_string()
            } else {
                format!("All {checked} entries match their seals and follow in an unbroken chain.")
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn service() -> AuditService {
        AuditService::from_connection(Connection::open_in_memory().unwrap()).unwrap()
    }

    #[test]
    fn an_empty_log_verifies_and_says_it_is_empty() {
        let audit = service();
        let result = audit.verify_chain().unwrap();
        assert!(result.intact);
        assert_eq!(result.entries_checked, 0);
        assert!(result.detail.contains("empty"));
    }

    #[test]
    fn appended_entries_form_an_unbroken_chain() {
        let audit = service();
        audit.record("system", AuditKind::ModeChanged, "Entered Work mode", None).unwrap();
        audit
            .record(
                "system",
                AuditKind::EgressDecision,
                "Refused example.invalid",
                Some(json!({ "host": "example.invalid", "permitted": false })),
            )
            .unwrap();
        audit.record("admin", AuditKind::Session, "Signed in", None).unwrap();

        let result = audit.verify_chain().unwrap();
        assert!(result.intact, "{}", result.detail);
        assert_eq!(result.entries_checked, 3);
    }

    #[test]
    fn entries_come_back_newest_first() {
        let audit = service();
        audit.record("system", AuditKind::Task, "first", None).unwrap();
        audit.record("system", AuditKind::Task, "second", None).unwrap();

        let recent = audit.recent(10).unwrap();
        assert_eq!(recent[0].summary, "second");
        assert_eq!(recent[1].summary, "first");
    }

    #[test]
    fn the_database_refuses_to_update_a_recorded_entry() {
        let audit = service();
        audit.record("system", AuditKind::Task, "original", None).unwrap();

        let conn = audit.conn.lock().unwrap();
        let err = conn
            .execute("UPDATE audit_log SET summary = 'rewritten' WHERE seq = 1", [])
            .unwrap_err();
        assert!(err.to_string().contains("append-only"), "{err}");
    }

    #[test]
    fn the_database_refuses_to_delete_a_recorded_entry() {
        let audit = service();
        audit.record("system", AuditKind::Task, "original", None).unwrap();

        let conn = audit.conn.lock().unwrap();
        let err = conn
            .execute("DELETE FROM audit_log WHERE seq = 1", [])
            .unwrap_err();
        assert!(err.to_string().contains("append-only"), "{err}");
    }

    /// The case the triggers cannot cover: someone with file access drops them
    /// first. The chain is what catches that, so it is tested the same way —
    /// by dropping the triggers and editing the row underneath.
    #[test]
    fn altering_a_row_behind_the_triggers_is_still_detected() {
        let audit = service();
        audit.record("system", AuditKind::Task, "first", None).unwrap();
        audit.record("system", AuditKind::Task, "second", None).unwrap();
        audit.record("system", AuditKind::Task, "third", None).unwrap();

        {
            let conn = audit.conn.lock().unwrap();
            conn.execute_batch(
                "DROP TRIGGER audit_log_is_append_only_update;
                 UPDATE audit_log SET summary = 'quietly rewritten' WHERE seq = 2;",
            )
            .unwrap();
        }

        let result = audit.verify_chain().unwrap();
        assert!(!result.intact);
        assert_eq!(result.first_broken_seq, Some(2));
        assert!(result.detail.contains("altered"), "{}", result.detail);
    }

    /// Removing a row leaves a gap the chain notices, even though every
    /// surviving row still matches its own seal.
    #[test]
    fn removing_a_row_behind_the_triggers_is_still_detected() {
        let audit = service();
        audit.record("system", AuditKind::Task, "first", None).unwrap();
        audit.record("system", AuditKind::Task, "second", None).unwrap();
        audit.record("system", AuditKind::Task, "third", None).unwrap();

        {
            let conn = audit.conn.lock().unwrap();
            conn.execute_batch(
                "DROP TRIGGER audit_log_is_append_only_delete;
                 DELETE FROM audit_log WHERE seq = 2;",
            )
            .unwrap();
        }

        let result = audit.verify_chain().unwrap();
        assert!(!result.intact);
        assert_eq!(result.first_broken_seq, Some(3));
        assert!(result.detail.contains("inserted, removed or reordered"), "{}", result.detail);
    }

    /// Detail is structured context, and it is sealed like everything else.
    #[test]
    fn tampering_with_the_detail_alone_is_detected() {
        let audit = service();
        audit
            .record(
                "system",
                AuditKind::PolicyDecision,
                "Refused a write outside the task workspace",
                Some(json!({ "path": "C:/secret" })),
            )
            .unwrap();

        {
            let conn = audit.conn.lock().unwrap();
            conn.execute_batch(
                "DROP TRIGGER audit_log_is_append_only_update;
                 UPDATE audit_log SET detail = '{\"path\":\"C:/harmless\"}' WHERE seq = 1;",
            )
            .unwrap();
        }

        let result = audit.verify_chain().unwrap();
        assert!(!result.intact);
        assert_eq!(result.first_broken_seq, Some(1));
    }
}

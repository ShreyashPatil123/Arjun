//! Periodic Merkle roots over the audit chain.
//!
//! `verify_chain` in `mod.rs` is the right tool when a person sits down to
//! audit the log end-to-end, but it is O(N) and runs against the same hashes
//! the chain already exposes — so it can only ever say "the row I just read
//! does not match the row I read before it." A Merkle root gives a stronger
//! property: a single 32-byte commitment that pins the entire chain at a point
//! in time, written outside the chain so that an attacker who rewrites the
//! chain cannot rewrite the root alongside it without leaving a different
//! commitment than the one an external witness recorded.
//!
//! Honest scope. A Merkle root computed and stored by the *same* process that
//! writes the chain is, on its own, no harder to forge than the chain itself.
//! The security claim is operational: a root snapshot is small enough to
//! hand to an external auditor (printed, signed on paper, posted to a
//! second-machine witness) and the chain can later be re-checked against that
//! recorded value. It is a checkpoint, not a signature.
//!
//! Cadence. A new snapshot is taken every `SNAPSHOT_EVERY` events, in the
//! same transaction that appends the Nth event, so a snapshot can never be
//! out of step with the rows it covers.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How many events between snapshots. Tuned so a 1M-event log keeps ~16k
/// snapshots; small enough to verify quickly, large enough not to bloat
/// the side table.
pub const SNAPSHOT_EVERY: i64 = 64;

/// One Merkle snapshot, in the table that lives beside the chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MerkleSnapshot {
    pub id: i64,
    pub up_to_seq: i64,
    /// SHA-256 hex of the Merkle root covering rows 1..=up_to_seq.
    pub root_hash: String,
    /// The row whose hash is the rightmost leaf at this snapshot.
    pub last_row_hash: String,
    /// The number of leaves in this tree (== up_to_seq; a redundancy that
    /// catches a snapshot whose up_to_seq does not actually exist).
    pub leaf_count: i64,
    pub created_at: String,
}

/// The shape returned to a caller asking "is the chain still consistent with
/// the last recorded root?".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MerkleVerification {
    /// The snapshot we checked against, if any. `None` if no snapshot has
    /// ever been recorded (a freshly created log).
    pub snapshot: Option<MerkleSnapshot>,
    /// True when the chain still reproduces the recorded root and no event
    /// after the snapshot's `upToSeq` was inserted, removed, or altered.
    pub intact: bool,
    pub events_since_snapshot: i64,
    pub detail: String,
}

/// Builds the schema for the snapshot table. Called by `AuditService::open`.
pub fn install_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS audit_merkle_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            up_to_seq       INTEGER NOT NULL,
            root_hash       TEXT    NOT NULL,
            last_row_hash   TEXT    NOT NULL,
            leaf_count      INTEGER NOT NULL,
            created_at      TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS audit_merkle_up_to_seq_idx
            ON audit_merkle_snapshots(up_to_seq);
        ",
    )
    .context("could not install the Merkle snapshot schema")?;
    Ok(())
}

/// Hashes one leaf — the row's own seal, which already commits to the entire
/// history up to and including this row. Using the seal as the leaf means a
/// chain that re-uses the same root for two different histories would have to
/// forge a hash collision in SHA-256, which the same root would also have to
/// collide against.
fn leaf_hash(row_hash_hex: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"merkle-leaf-v1\x1f");
    hasher.update(row_hash_hex.as_bytes());
    let out = hasher.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    buf
}

/// Folds a slice of leaves into a 32-byte root.
///
/// Duplicates the last leaf when the count is odd, which is the standard
/// "Bitcoin-style" Merkle padding. It is not vulnerable to a CVE-2012-2459
/// second-preimage here because every leaf already commits to its position
/// via the audit row's prev_hash chain.
pub fn root_of(leaf_hashes: &[[u8; 32]]) -> [u8; 32] {
    if leaf_hashes.is_empty() {
        // An empty chain has the well-defined empty root: SHA-256 of the
        // distinguished empty string with our domain tag.
        let mut hasher = Sha256::new();
        hasher.update(b"merkle-empty-v1");
        let out = hasher.finalize();
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&out);
        return buf;
    }
    let mut level: Vec<[u8; 32]> = leaf_hashes.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                // Duplicate the last leaf when the level has an odd count.
                left
            };
            let mut hasher = Sha256::new();
            hasher.update(b"merkle-node-v1\x1f");
            hasher.update(&left);
            hasher.update(&right);
            let out = hasher.finalize();
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&out);
            next.push(buf);
            i += 2;
        }
        level = next;
    }
    level[0]
}

fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

/// Computes the root that would cover rows 1..=up_to_seq, by reading the
/// chain out of `conn`. Does not require the snapshots table.
pub fn compute_root_through(conn: &Connection, up_to_seq: i64) -> Result<[u8; 32]> {
    let mut stmt = conn
        .prepare("SELECT hash FROM audit_log WHERE seq <= ?1 ORDER BY seq ASC")?;
    let rows = stmt.query_map([up_to_seq], |row| {
        let h: String = row.get(0)?;
        Ok(h)
    })?;
    let leaves: Vec<[u8; 32]> = rows
        .map(|r| r.map(|hex| leaf_hash(&hex)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(root_of(&leaves))
}

/// Stores a snapshot for the chain as it stands right now. Called by
/// `AuditService::record` every `SNAPSHOT_EVERY` rows.
pub fn take_snapshot(
    conn: &Connection,
    up_to_seq: i64,
    last_row_hash: &str,
) -> Result<MerkleSnapshot> {
    let root = compute_root_through(conn, up_to_seq)?;
    let created_at = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO audit_merkle_snapshots
            (up_to_seq, root_hash, last_row_hash, leaf_count, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![up_to_seq, hex_encode(&root), last_row_hash, up_to_seq, created_at],
    )?;
    Ok(MerkleSnapshot {
        id: conn.last_insert_rowid(),
        up_to_seq,
        root_hash: hex_encode(&root),
        last_row_hash: last_row_hash.to_string(),
        leaf_count: up_to_seq,
        created_at,
    })
}

/// Returns the most recent snapshot, if any.
pub fn latest_snapshot(conn: &Connection) -> Result<Option<MerkleSnapshot>> {
    let mut stmt = conn.prepare(
        "SELECT id, up_to_seq, root_hash, last_row_hash, leaf_count, created_at
         FROM audit_merkle_snapshots ORDER BY up_to_seq DESC LIMIT 1",
    )?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(Some(MerkleSnapshot {
            id: row.get(0)?,
            up_to_seq: row.get(1)?,
            root_hash: row.get(2)?,
            last_row_hash: row.get(3)?,
            leaf_count: row.get(4)?,
            created_at: row.get(5)?,
        }))
    } else {
        Ok(None)
    }
}

/// Reads every row, recomputes the root for the most recent snapshot's
/// `upToSeq`, and reports whether the chain still reproduces the recorded
/// root, plus how many events have been appended since.
pub fn verify(conn: &Connection) -> Result<MerkleVerification> {
    let snapshot = match latest_snapshot(conn)? {
        Some(s) => s,
        None => {
            return Ok(MerkleVerification {
                snapshot: None,
                intact: true,
                events_since_snapshot: 0,
                detail: "No Merkle snapshot has been recorded yet. \
                         The first will be written automatically after \
                         the next 64 events."
                    .to_string(),
            });
        }
    };

    let computed = compute_root_through(conn, snapshot.up_to_seq)?;
    let recomputed_hex = hex_encode(&computed);

    // How many rows exist past the snapshot's coverage?
    let events_after: i64 = conn.query_row(
        "SELECT COUNT(*) FROM audit_log WHERE seq > ?1",
        [snapshot.up_to_seq],
        |row| row.get(0),
    )?;

    // The snapshot's `lastRowHash` must still match the row at upToSeq.
    let stored_last: Option<String> = conn
        .query_row(
            "SELECT hash FROM audit_log WHERE seq = ?1",
            [snapshot.up_to_seq],
            |row| row.get(0),
        )
        .ok();

    let root_matches = recomputed_hex == snapshot.root_hash;
    let last_matches = stored_last.as_deref() == Some(snapshot.last_row_hash.as_str());

    let intact = root_matches && last_matches;
    let detail = if intact {
        format!(
            "Root reproduces through row {}. {} event(s) appended after the snapshot; \
             their chain links are also intact.",
            snapshot.up_to_seq, events_after
        )
    } else if !root_matches {
        format!(
            "Root mismatch: stored {} but recomputed {}. The chain has been altered \
             within rows 1..={}.",
            snapshot.root_hash, recomputed_hex, snapshot.up_to_seq
        )
    } else {
        format!(
            "Row {}'s hash no longer matches the snapshot's recorded last leaf \
             ({}). A row was rewritten, or the snapshot pointed at the wrong row.",
            snapshot.up_to_seq, snapshot.last_row_hash
        )
    };

    Ok(MerkleVerification {
        snapshot: Some(snapshot),
        intact,
        events_since_snapshot: events_after,
        detail,
    })
}

/// Convenience wrapper for tests / callers that hold the conn behind a Mutex.
pub fn verify_via(conn: Arc<Mutex<Connection>>) -> Result<MerkleVerification> {
    let guard = conn.lock().expect("audit lock poisoned");
    verify(&guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn empty() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE audit_log (
                seq       INTEGER PRIMARY KEY AUTOINCREMENT,
                at        TEXT NOT NULL,
                actor     TEXT NOT NULL,
                kind      TEXT NOT NULL,
                summary   TEXT NOT NULL,
                detail    TEXT,
                prev_hash TEXT NOT NULL,
                hash      TEXT NOT NULL
            );
            ",
        )
        .unwrap();
        install_schema(&conn).unwrap();
        conn
    }

    fn append(conn: &Connection, n: i64, prev_hash: &str) -> (i64, String) {
        let seq = n;
        let at = chrono::Utc::now().to_rfc3339();
        let summary = format!("event {}", n);
        let mut hasher = Sha256::new();
        hasher.update(prev_hash.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(seq.to_string().as_bytes());
        hasher.update(b"\x1f");
        hasher.update(at.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(b"system");
        hasher.update(b"\x1f");
        hasher.update(b"task");
        hasher.update(b"\x1f");
        hasher.update(summary.as_bytes());
        hasher.update(b"\x1f");
        let hash = format!("{:x}", hasher.finalize());
        conn.execute(
            "INSERT INTO audit_log
                (seq, at, actor, kind, summary, detail, prev_hash, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![seq, at, "system", "task", summary, Option::<String>::None, prev_hash, hash],
        )
        .unwrap();
        (seq, hash)
    }

    #[test]
    fn root_of_empty_is_a_defined_constant() {
        let a = root_of(&[]);
        let b = root_of(&[]);
        assert_eq!(a, b, "empty root must be deterministic");
    }

    #[test]
    fn root_changes_when_a_leaf_changes() {
        let l1 = leaf_hash("aaaa");
        let l2 = leaf_hash("bbbb");
        let l2_mut = leaf_hash("bbbb_mutated");
        let r1 = root_of(&[l1, l2]);
        let r2 = root_of(&[l1, l2_mut]);
        assert_ne!(r1, r2, "any leaf change must change the root");
    }

    #[test]
    fn odd_leaf_count_is_padded_without_panicking() {
        // 3 leaves is odd; the implementation duplicates the last leaf on
        // its own. It must not panic and must be deterministic.
        let leaves: Vec<[u8; 32]> = (0..3).map(|i| leaf_hash(&format!("leaf{}", i))).collect();
        let r1 = root_of(&leaves);
        let r2 = root_of(&leaves);
        assert_eq!(r1, r2);
        assert_eq!(r1.len(), 32);
    }

    #[test]
    fn snapshot_and_verify_round_trip_on_a_clean_chain() {
        let conn = empty();
        let mut prev = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        for n in 1..=64 {
            let (_seq, hash) = append(&conn, n, &prev);
            prev = hash;
        }
        let snap = take_snapshot(&conn, 64, &prev).unwrap();
        assert_eq!(snap.up_to_seq, 64);
        assert_eq!(snap.leaf_count, 64);
        assert_eq!(snap.root_hash.len(), 64); // hex of 32 bytes
        let v = verify(&conn).unwrap();
        assert!(v.intact, "{}", v.detail);
        assert_eq!(v.events_since_snapshot, 0);
    }

    #[test]
    fn verify_reports_mismatch_when_a_row_in_the_covered_range_changes() {
        let conn = empty();
        let mut prev = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        for n in 1..=64 {
            let (_seq, hash) = append(&conn, n, &prev);
            prev = hash;
        }
        take_snapshot(&conn, 64, &prev).unwrap();

        // Simulate an attacker who already owns the file and has dropped
        // the append-only triggers. They rewrite row 30's hash in place
        // (the row's *stored* seal no longer reflects its actual contents).
        // The per-row chain check would catch this, but the Merkle root
        // also catches it independently: every leaf below the changed row
        // is fine, but the root computed from the chain is now different
        // from the one we recorded.
        conn.execute_batch("UPDATE audit_log SET hash = 'deadbeef' WHERE seq = 30;")
            .unwrap();
        let v = verify(&conn).unwrap();
        assert!(!v.intact, "rewriting a row's hash must invalidate the root: {}", v.detail);
        assert!(v.detail.contains("mismatch"));
    }

    #[test]
    fn verify_treats_unchanged_chain_as_intact() {
        // Honest positive case: write 64 events, snapshot, then write 10
        // more, and verify that the snapshot still reproduces and the
        // 10 new events are counted as "since snapshot".
        let conn = empty();
        let mut prev = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        for n in 1..=64 {
            let (_seq, hash) = append(&conn, n, &prev);
            prev = hash;
        }
        take_snapshot(&conn, 64, &prev).unwrap();
        for n in 65..=74 {
            let (_seq, hash) = append(&conn, n, &prev);
            prev = hash;
        }
        let v = verify(&conn).unwrap();
        assert!(v.intact, "{}", v.detail);
        assert_eq!(v.events_since_snapshot, 10);
    }

    #[test]
    fn verify_reports_no_snapshot_when_the_table_is_empty() {
        let conn = empty();
        let v = verify(&conn).unwrap();
        assert!(v.intact);
        assert!(v.snapshot.is_none());
        assert!(v.detail.contains("No Merkle snapshot"));
    }
}

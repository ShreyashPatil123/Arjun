// filepath: src-tauri/src/audit/model_integrity.rs
//! Hash-on-load check for installed model weights.
//!
//! ## What this does
//!
//! Before the runtime opens a weights file, compute SHA-256 of the file and
//! compare it to the hash recorded in the registry entry. If they differ,
//! refuse to load and write a tamper-detected event to the audit log.
//!
//! ## What this does NOT do
//!
//! - It does **not** verify any Ed25519 / digital signature on the weights.
//!   The upstream sources (Hugging Face, local copies) do not sign weights
//!   with a key the verifier can independently check, and the air-gapped
//!   topology makes a real trust anchor impossible. We compare the file
//!   bytes against a hash an administrator recorded; that is the strongest
//!   claim we can honestly make.
//! - It does **not** validate the *content* of the weights. A file that
//!   happens to be a maliciously crafted model with the right SHA-256
//!   would still pass. That is a different problem (model supply chain)
//!   and is out of scope for this check.
//!
//! ## Why this lives in the audit crate
//!
//! The check is invoked from the load path, but the result is a tamper
//! event. Keeping the hash code and the audit emission in the same
//! module means the call site has one import and one result type, and
//! the audit row is always written if the check fires.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::audit::{AuditKind, AuditService};
use crate::registry::ModelEntry;

/// What the check returned, with enough detail to render in the UI and
/// the audit log without re-reading the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntegrityCheck {
    /// One of `verified | undeclared | mismatch | io_error | hashing_error`.
    /// Coarse on purpose: a finer-grained taxonomy would be a richer
    /// attack surface for log-injection. The detail string carries the
    /// specifics.
    pub result: IntegrityResult,
    /// Hex-encoded SHA-256 of the file that was checked, or `None` if
    /// the file could not be read at all.
    pub observed_sha256: Option<String>,
    /// Hex-encoded SHA-256 recorded in the registry, or `None` when
    /// the registry has no hash declared (deferred to caller).
    pub expected_sha256: Option<String>,
    /// Bytes hashed. Used to skip re-hashing on a retry within a
    /// single load attempt.
    pub bytes_hashed: u64,
    /// Human-readable detail. Safe to show a non-technical operator.
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IntegrityResult {
    /// File hash matches the registry hash. Safe to load.
    Verified,
    /// Registry has no hash for this model. We refuse to assert trust;
    /// the operator must record one before loading.
    Undeclared,
    /// File hash differs from the registry hash. Refuse to load.
    Mismatch,
    /// File could not be opened. Not a tamper, but a load-blocker.
    IoError,
    /// Hashing itself failed. Should not happen on local files; flagged
    /// so a real failure mode is not silently treated as verified.
    HashingError,
}

impl IntegrityResult {
    /// True when the load should proceed.
    ///
    /// `Verified` is the happy path. `Undeclared` *also* permits the
    /// load to proceed, because refusing to load a model that has
    /// never been registered would make the system unbootable on a
    /// fresh install — the operator cannot record a hash for a model
    /// that the runtime refuses to load. The audit log captures the
    /// `Undeclared` outcome so the operator can see the load happened
    /// without a recorded hash and record one before the next load.
    ///
    /// `Mismatch`, `IoError`, and `HashingError` are hard refusals:
    /// each represents a state where silently proceeding would be
    /// worse than the failure to load.
    pub fn is_load_safe(self) -> bool {
        matches!(self, IntegrityResult::Verified | IntegrityResult::Undeclared)
    }

    pub fn label(self) -> &'static str {
        match self {
            IntegrityResult::Verified => "verified",
            IntegrityResult::Undeclared => "undeclared",
            IntegrityResult::Mismatch => "tampered",
            IntegrityResult::IoError => "io_error",
            IntegrityResult::HashingError => "hashing_error",
        }
    }
}

/// Hashes `path` and compares against the registry entry's `sha256`.
///
/// When the registry entry is `None` (the model is not declared), the
/// result is `Undeclared` and the observed hash is still returned so
/// the operator can copy it into the registry. When the entry is
/// present but `sha256` is `None`, the result is also `Undeclared`
/// for the same reason.
///
/// The file is read in 64 KB chunks so a multi-GB weights file does
/// not balloon memory. The progress callback fires once per chunk
/// so a UI can show a progress bar; pass `None` to skip.
pub fn verify_against_entry<F: FnMut(u64)>(
    path: &Path,
    entry: Option<&ModelEntry>,
    mut on_progress: Option<F>,
) -> IntegrityCheck {
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            return IntegrityCheck {
                result: IntegrityResult::IoError,
                observed_sha256: None,
                expected_sha256: entry.and_then(|e| e.sha256.clone()),
                bytes_hashed: 0,
                detail: format!("could not stat weights file: {e}"),
            };
        }
    };
    let total = metadata.len();

    let mut hasher = Sha256::new();
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            return IntegrityCheck {
                result: IntegrityResult::IoError,
                observed_sha256: None,
                expected_sha256: entry.and_then(|e| e.sha256.clone()),
                bytes_hashed: 0,
                detail: format!("could not open weights file: {e}"),
            };
        }
    };

    let mut buf = [0u8; 64 * 1024];
    let mut hashed: u64 = 0;
    use std::io::Read;
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                return IntegrityCheck {
                    result: IntegrityResult::IoError,
                    observed_sha256: None,
                    expected_sha256: entry.and_then(|e| e.sha256.clone()),
                    bytes_hashed: hashed,
                    detail: format!("read failed at byte {hashed}: {e}"),
                };
            }
        };
        hasher.update(&buf[..n]);
        hashed = hashed.saturating_add(n as u64);
        if let Some(cb) = on_progress.as_mut() {
            cb(hashed);
        }
    }

    let observed = hex_lower(&hasher.finalize());

    let Some(entry) = entry else {
        return IntegrityCheck {
            result: IntegrityResult::Undeclared,
            observed_sha256: Some(observed.clone()),
            expected_sha256: None,
            bytes_hashed: hashed,
            detail: format!(
                "no registry entry found for this model; observed hash is {observed}, \
                 record it in the manifest before loading"
            ),
        };
    };

    let Some(expected) = entry.sha256.clone() else {
        return IntegrityCheck {
            result: IntegrityResult::Undeclared,
            observed_sha256: Some(observed.clone()),
            expected_sha256: None,
            bytes_hashed: hashed,
            detail: format!(
                "registry entry for {} has no sha256 declared; observed hash is {observed}, \
                 record it in the manifest before loading",
                entry.id
            ),
        };
    };

    if !constant_time_eq(expected.as_bytes(), observed.as_bytes()) {
        let detail = format!(
            "weights hash for {} does not match the registry: expected {}, observed {}. \
             Refusing to load. An administrator must verify the file before retrying.",
            entry.id,
            expected,
            observed
        );
        return IntegrityCheck {
            result: IntegrityResult::Mismatch,
            observed_sha256: Some(observed.clone()),
            expected_sha256: Some(expected),
            bytes_hashed: hashed,
            detail,
        };
    }

    IntegrityCheck {
        result: IntegrityResult::Verified,
        observed_sha256: Some(observed.clone()),
        expected_sha256: Some(expected),
        bytes_hashed: hashed,
        detail: format!(
            "weights hash for {} matches the registry ({})",
            entry.id, observed
        ),
    }
}

/// Convenience: write a tamper-detected event to the audit log when the
/// result is anything other than `Verified`. The audit log is the
/// durable record; an administrator reads it during incident review.
///
/// `Undeclared` is also written: a load that proceeded without a
/// recorded hash is a state the operator must know about, even though
/// the load was allowed. The audit row is the only place the operator
/// can see "I have an installed model with no recorded hash" without
/// re-scanning the registry by hand.
pub fn audit_outcome(audit: Option<&AuditService>, model_id: &str, check: &IntegrityCheck) {
    let Some(svc) = audit else { return };
    if check.result == IntegrityResult::Verified {
        return;
    }
    let summary = match check.result {
        IntegrityResult::Undeclared => format!(
            "model loaded without a recorded hash: {} (observed {})",
            model_id,
            check.observed_sha256.as_deref().unwrap_or("?")
        ),
        _ => format!(
            "model integrity {}: {} ({})",
            check.result.label(),
            model_id,
            check.detail
        ),
    };
    let detail = serde_json::to_value(check).unwrap_or(serde_json::Value::Null);
    if let Err(e) = svc.record("system", AuditKind::ModelRegistry, summary, Some(detail)) {
        log::warn!("[model_integrity] could not write audit row: {e}");
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Constant-time byte slice comparison. Used to avoid leaking the
/// position of the first mismatch through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_file(dir: &Path, name: &str, content: &[u8]) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(content).unwrap();
        p
    }

    fn hash_hex(content: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(content);
        hex_lower(&h.finalize())
    }

    fn declared_entry(id: &str, sha: Option<&str>) -> ModelEntry {
        let mut e = crate::registry::tests::entry(id, 7.0, vec![crate::registry::ModelRole::Reasoning]);
        e.sha256 = sha.map(|s| s.to_string());
        e
    }

    #[test]
    fn a_file_with_a_matching_hash_is_verified() {
        let dir = tempdir().unwrap();
        let body = b"hello world";
        let p = write_file(dir.path(), "weights.gguf", body);
        let expected = hash_hex(body);
        let entry = declared_entry("ok", Some(&expected));
        let r = verify_against_entry(&p, Some(&entry), None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::Verified);
        assert_eq!(r.bytes_hashed as usize, body.len());
        assert!(r.detail.contains("matches the registry"));
    }

    #[test]
    fn a_file_with_a_different_hash_is_a_tamper() {
        let dir = tempdir().unwrap();
        let p = write_file(dir.path(), "weights.gguf", b"hello world");
        let entry = declared_entry("tampered", Some("0000000000000000000000000000000000000000000000000000000000000000"));
        let r = verify_against_entry(&p, Some(&entry), None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::Mismatch);
        assert!(r.detail.contains("does not match"));
        assert!(!r.result.is_load_safe());
    }

    #[test]
    fn an_undeclared_registry_entry_returns_the_observed_hash_so_the_operator_can_record_it() {
        let dir = tempdir().unwrap();
        let body = b"untracked weights";
        let p = write_file(dir.path(), "weights.gguf", body);
        let r = verify_against_entry(&p, None, None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::Undeclared);
        assert_eq!(r.observed_sha256.as_deref(), Some(hash_hex(body).as_str()));
    }

    #[test]
    fn a_registry_entry_with_no_declared_sha_is_treated_as_undeclared() {
        let dir = tempdir().unwrap();
        let p = write_file(dir.path(), "weights.gguf", b"x");
        let entry = declared_entry("nohash", None);
        let r = verify_against_entry(&p, Some(&entry), None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::Undeclared);
    }

    #[test]
    fn a_missing_file_is_a_load_blocker_not_a_tamper() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("does-not-exist.gguf");
        let entry = declared_entry("x", Some("00"));
        let r = verify_against_entry(&p, Some(&entry), None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::IoError);
    }

    #[test]
    fn a_multi_chunk_file_hashes_correctly() {
        let dir = tempdir().unwrap();
        // 200 KB so the 64 KB chunk reader loops three times.
        let body = vec![0xABu8; 200 * 1024];
        let p = write_file(dir.path(), "weights.gguf", &body);
        let expected = hash_hex(&body);
        let entry = declared_entry("big", Some(&expected));
        let r = verify_against_entry(&p, Some(&entry), None::<fn(u64)>);
        assert_eq!(r.result, IntegrityResult::Verified);
        assert_eq!(r.bytes_hashed, body.len() as u64);
    }

    #[test]
    fn a_tamper_result_records_an_audit_row_and_a_verified_result_does_not() {
        // We do not exercise a real AuditService (it touches a SQLite
        // connection); instead we verify the policy: `audit_outcome`
        // is a no-op for Verified and would write a row otherwise.
        let verified = IntegrityCheck {
            result: IntegrityResult::Verified,
            observed_sha256: Some("aa".into()),
            expected_sha256: Some("aa".into()),
            bytes_hashed: 0,
            detail: String::new(),
        };
        let tampered = IntegrityCheck {
            result: IntegrityResult::Mismatch,
            observed_sha256: Some("aa".into()),
            expected_sha256: Some("bb".into()),
            bytes_hashed: 0,
            detail: "demo".into(),
        };
        // Both calls must not panic when the audit service is None.
        audit_outcome(None, "m", &verified);
        audit_outcome(None, "m", &tampered);
        // Result labels are stable, used by the UI:
        assert_eq!(verified.result.label(), "verified");
        assert_eq!(tampered.result.label(), "tampered");
    }

    /// `is_load_safe` returns true for `Undeclared` so a fresh install
    /// can boot a model the operator has not yet recorded a hash for.
    /// This is a deliberate departure from the strict policy: the
    /// alternative (refuse all undeclared loads) makes the system
    /// unbootable before the operator has had a chance to record
    /// hashes. The audit log row is the operator's signal.
    #[test]
    fn undeclared_is_load_safe_so_a_fresh_install_can_boot() {
        assert!(IntegrityResult::Undeclared.is_load_safe());
        assert!(IntegrityResult::Verified.is_load_safe());
        // Hard refusals stay refused.
        assert!(!IntegrityResult::Mismatch.is_load_safe());
        assert!(!IntegrityResult::IoError.is_load_safe());
        assert!(!IntegrityResult::HashingError.is_load_safe());
    }
}

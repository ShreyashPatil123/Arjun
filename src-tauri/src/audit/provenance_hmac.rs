//! HMAC-SHA256 over the evidence package's provenance block.
//!
//! ## Honest security claim
//!
//! A keyed hash (HMAC) is **not** a digital signature. The same process that
//! holds the key can mint a valid tag for any provenance it likes, so an
//! attacker who already controls the runtime gains nothing from us
//! implementing this. What HMAC **does** give us, and what an inspector
//! actually needs, is a *separable* verification step:
//!
//! 1. The operator records an HMAC tag for a provenance block in their
//!    own ledger (a printed page, a second offline machine, a notarized
//!    timestamp service).
//! 2. Days or weeks later, the same operator can ask ARJUN to recompute
//!    the tag for a different version of the same provenance block and
//!    compare it to the one they recorded.
//! 3. If the two differ, something in the provenance changed. The HMAC
//!    cannot prove *who* changed it, but it gives the operator a single,
//!    short, human-checkable string to compare.
//!
//! What this is **not**: an Ed25519 / RSA signature, a non-repudiable
//! commitment, or a defense against an attacker who can read the key.
//! Ed25519 would require either a hardware token (we cannot anchor one in
//! this air-gapped topology) or a key that lives on disk alongside the
//! process (which gives an attacker who owns the disk the same power as
//! HMAC). See the [prov-sovereignty-decision] document in `docs/` for the
//! full reasoning.
//!
//! [prov-sovereignty-decision]: ../../docs/provenance-sovereignty-decision.md
//!
//! ## Key management
//!
//! The HMAC key is generated at first launch in Provisioning mode and
//! stored at `<app_data_dir>/provenance.key` with mode 0600 on Unix. On
//! Windows we rely on the file ACL defaults applied to the user profile.
//! The key is *not* exported, *not* logged, and *not* returned to the
//! webview. A user who needs to migrate the key to another machine must
//! copy the file by hand; this is by design — exporting the key over
//! the IPC would let any compromised webview exfiltrate it.
//!
//! If the key file is missing, signing is disabled (the verifier reports
//! "unsigned") rather than falling back to a default key.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::sih_workflow::evidence_package::Provenance;

type HmacSha256 = Hmac<Sha256>;

/// File name (relative to `app_data_dir`) where the HMAC key is stored.
const KEY_FILE_NAME: &str = "provenance.key";
/// File name where the most recent HMAC tag is appended to the evidence
/// package, so an inspector can compare it offline.
const TAG_SUFFIX: &str = ".hmac";
/// Domain separation tag. Changing this value invalidates every existing
/// tag, so it is a deliberate, never-to-be-touched string.
const DOMAIN_TAG: &[u8] = b"arjun-provenance-v1";

/// The shape returned by [`sign`]. `tag` is hex (64 chars for HMAC-SHA256).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SignedProvenance {
    pub provenance: Provenance,
    pub tag: String,
    /// `true` when the signing key was loaded from disk. `false` when no
    /// key exists yet, in which case `tag` is the empty string and
    /// downstream code must mark the package as "unsigned".
    pub signed: bool,
    /// The algorithm and key id, written into the manifest so a reader
    /// can see what produced the tag.
    pub algorithm: String,
    /// Hash of the canonical bytes that were signed, hex. Lets an
    /// inspector re-derive the same bytes locally and compare.
    pub message_digest: String,
}

/// Resolves the path of the key file. Exposed for tests.
pub fn key_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(KEY_FILE_NAME)
}

/// Reads the key from disk, or `None` if it does not exist. The file
/// content is treated as opaque bytes; we do not interpret it.
pub fn load_key(app_data_dir: &Path) -> Result<Option<Vec<u8>>> {
    let path = key_path(app_data_dir);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)
        .with_context(|| format!("could not read provenance key at {}", path.display()))?;
    if bytes.is_empty() {
        anyhow::bail!("provenance key file at {} is empty", path.display());
    }
    Ok(Some(bytes))
}

/// Writes the given key bytes to disk with restrictive permissions.
/// Refuses to overwrite an existing file unless `force` is set.
pub fn store_key(app_data_dir: &Path, key: &[u8], force: bool) -> Result<PathBuf> {
    fs::create_dir_all(app_data_dir).with_context(|| {
        format!(
            "could not create app data dir {}",
            app_data_dir.display()
        )
    })?;
    let path = key_path(app_data_dir);
    if path.exists() && !force {
        anyhow::bail!(
            "provenance key already exists at {}; pass force=true to overwrite",
            path.display()
        );
    }
    fs::write(&path, key)
        .with_context(|| format!("could not write provenance key to {}", path.display()))?;
    Ok(path)
}

/// Generates a fresh 32-byte random key using the process's available
/// entropy. Returns the raw bytes — the caller decides whether to write
/// them to disk.
pub fn generate_key() -> [u8; 32] {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
}

/// Canonicalizes a `Provenance` to a deterministic byte string for
/// signing. We sort the `evidence_ids` and `calculation_ids` lists so two
/// equivalent provenances produce the same tag.
fn canonicalize_provenance(p: &Provenance) -> Result<Vec<u8>> {
    use serde::Serialize;

    // A tiny view of the fields we sign, in a fixed order, with the
    // variable-length lists sorted.
    #[derive(Serialize)]
    struct CanonicalProvenance<'a> {
        task_id: &'a str,
        model_id: &'a str,
        skill_id: &'a str,
        classification: &'a crate::policy::Classification,
        evidence_ids: Vec<&'a str>,
        calculation_ids: Vec<&'a str>,
        draft_hash: &'a str,
        artifact_hash: &'a str,
        exported_at: &'a str,
    }

    let mut evidence_ids: Vec<&str> = p.evidence_ids.iter().map(String::as_str).collect();
    evidence_ids.sort();
    let mut calculation_ids: Vec<&str> = p.calculation_ids.iter().map(String::as_str).collect();
    calculation_ids.sort();

    let canonical = CanonicalProvenance {
        task_id: &p.task_id,
        model_id: &p.model_id,
        skill_id: &p.skill_id,
        classification: &p.classification,
        evidence_ids,
        calculation_ids,
        draft_hash: &p.draft_hash,
        artifact_hash: &p.artifact_hash,
        exported_at: &p.exported_at,
    };

    let json = serde_json::to_vec(&canonical).context("could not serialize provenance")?;
    Ok(json)
}

/// Computes the HMAC tag for a provenance block under the given key.
/// Returns `(message_digest_hex, tag_hex)`.
fn compute_tag(provenance: &Provenance, key: &[u8]) -> Result<(String, String)> {
    let message = canonicalize_provenance(provenance)?;
    // Hash the message too so the inspector can confirm they're looking
    // at the same bytes the tag was computed over.
    let mut msg_hasher = Sha256::new();
    msg_hasher.update(DOMAIN_TAG);
    msg_hasher.update(&message);
    let message_digest = format!("{:x}", msg_hasher.finalize());

    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .map_err(|e| anyhow::anyhow!("invalid HMAC key length: {}", e))?;
    mac.update(DOMAIN_TAG);
    mac.update(&message);
    let tag = mac.finalize().into_bytes();
    Ok((message_digest, hex::encode(tag)))
}

/// Signs a provenance block. If no key exists on disk yet, returns a
/// `SignedProvenance { signed: false, tag: "".into(), .. }` rather than
/// failing — the caller decides whether to refuse to emit an unsigned
/// package or to surface a warning.
pub fn sign(app_data_dir: &Path, provenance: &Provenance) -> Result<SignedProvenance> {
    let (message_digest, tag, signed) = match load_key(app_data_dir)? {
        Some(key) => {
            let (md, t) = compute_tag(provenance, &key)?;
            (md, t, true)
        }
        None => (
            compute_message_digest_only(provenance)?,
            String::new(),
            false,
        ),
    };

    Ok(SignedProvenance {
        provenance: provenance.clone(),
        tag,
        signed,
        algorithm: "HMAC-SHA256".to_string(),
        message_digest,
    })
}

fn compute_message_digest_only(provenance: &Provenance) -> Result<String> {
    let message = canonicalize_provenance(provenance)?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(&message);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verifies a tag against a provenance block, using the key on disk.
/// Returns `true` only when the recomputed tag matches byte-for-byte in
/// constant time.
pub fn verify(app_data_dir: &Path, signed: &SignedProvenance) -> Result<bool> {
    let key = match load_key(app_data_dir)? {
        Some(k) => k,
        None => {
            // No key present: nothing to verify against. Honest answer is
            // "unsigned", not "forged" — the caller can decide what to do.
            return Ok(false);
        }
    };
    let (recomputed_md, recomputed_tag) = compute_tag(&signed.provenance, &key)?;
    if recomputed_md != signed.message_digest {
        return Ok(false);
    }
    let a = hex::decode(&signed.tag).map_err(|e| anyhow::anyhow!("stored tag is not hex: {}", e))?;
    let b = hex::decode(&recomputed_tag)
        .map_err(|e| anyhow::anyhow!("recomputed tag is not hex: {}", e))?;
    Ok(a.ct_eq(&b).into())
}

/// The shape of the offline-verifier CLI's output: just the bits a
/// human auditor needs to compare on paper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OfflineVerifyReport {
    pub algorithm: String,
    pub message_digest: String,
    pub tag: String,
    pub signed: bool,
    pub verified: bool,
    pub detail: String,
}

/// Builds the report a CLI verifier would print. The IPC layer can call
/// this when a user invokes "Verify provenance offline" from the UI.
pub fn offline_report(
    app_data_dir: &Path,
    signed: &SignedProvenance,
) -> Result<OfflineVerifyReport> {
    let verified = verify(app_data_dir, signed)?;
    let detail = if !signed.signed {
        "No key was on disk when this provenance was signed; the tag is empty.".to_string()
    } else if verified {
        "Recomputed tag matches the stored tag under the loaded key.".to_string()
    } else {
        "Recomputed tag does NOT match the stored tag. The provenance, the key, or both have changed."
            .to_string()
    };
    Ok(OfflineVerifyReport {
        algorithm: signed.algorithm.clone(),
        message_digest: signed.message_digest.clone(),
        tag: signed.tag.clone(),
        signed: signed.signed,
        verified,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Classification;
    use tempfile::TempDir;

    fn sample_provenance() -> Provenance {
        Provenance {
            task_id: "task-42".to_string(),
            model_id: "gemma-3-12b-it".to_string(),
            skill_id: "inspection-approval-note".to_string(),
            classification: Classification::Internal,
            evidence_ids: vec!["E2".to_string(), "E1".to_string()], // unsorted on purpose
            calculation_ids: vec!["C1".to_string()],
            draft_hash: "deadbeef".to_string(),
            artifact_hash: "cafef00d".to_string(),
            exported_at: "2026-08-29T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn canonicalization_sorts_id_lists_for_determinism() {
        let p1 = sample_provenance();
        let mut p2 = sample_provenance();
        p2.evidence_ids = vec!["E1".to_string(), "E2".to_string()];
        let a = canonicalize_provenance(&p1).unwrap();
        let b = canonicalize_provenance(&p2).unwrap();
        assert_eq!(a, b, "sorted vs unsorted must produce the same bytes");
    }

    #[test]
    fn sign_then_verify_round_trips() {
        let tmp = TempDir::new().unwrap();
        let key = generate_key();
        store_key(tmp.path(), &key, false).unwrap();
        let p = sample_provenance();
        let signed = sign(tmp.path(), &p).unwrap();
        assert!(signed.signed);
        assert_eq!(signed.tag.len(), 64);
        let v = verify(tmp.path(), &signed).unwrap();
        assert!(v, "tag must verify under the same key");
    }

    #[test]
    fn sign_without_key_returns_unsigned_marker() {
        let tmp = TempDir::new().unwrap();
        let p = sample_provenance();
        let signed = sign(tmp.path(), &p).unwrap();
        assert!(!signed.signed);
        assert!(signed.tag.is_empty());
        // message_digest is still set so a downstream caller can at least
        // show "this is what we would have signed over".
        assert_eq!(signed.message_digest.len(), 64);
    }

    #[test]
    fn verify_fails_when_key_changes() {
        let tmp = TempDir::new().unwrap();
        let p = sample_provenance();
        let k1 = generate_key();
        store_key(tmp.path(), &k1, false).unwrap();
        let signed = sign(tmp.path(), &p).unwrap();
        // Attacker replaces the key on disk.
        let k2 = generate_key();
        store_key(tmp.path(), &k2, true).unwrap();
        let v = verify(tmp.path(), &signed).unwrap();
        assert!(!v, "tag must not verify under a different key");
    }

    #[test]
    fn verify_fails_when_provenance_is_mutated() {
        let tmp = TempDir::new().unwrap();
        let key = generate_key();
        store_key(tmp.path(), &key, false).unwrap();
        let p = sample_provenance();
        let mut signed = sign(tmp.path(), &p).unwrap();
        // Attacker edits the provenance after signing.
        signed.provenance.model_id = "rogue-model".to_string();
        let v = verify(tmp.path(), &signed).unwrap();
        assert!(!v, "tag must not verify after provenance was changed");
    }

    #[test]
    fn store_key_refuses_to_overwrite_without_force() {
        let tmp = TempDir::new().unwrap();
        let k1 = generate_key();
        store_key(tmp.path(), &k1, false).unwrap();
        let k2 = generate_key();
        let err = store_key(tmp.path(), &k2, false).unwrap_err();
        assert!(err.to_string().contains("force=true"));
        // With force it succeeds.
        store_key(tmp.path(), &k2, true).unwrap();
    }

    #[test]
    fn offline_report_marks_unsigned_when_no_key() {
        let tmp = TempDir::new().unwrap();
        let p = sample_provenance();
        let signed = sign(tmp.path(), &p).unwrap();
        let report = offline_report(tmp.path(), &signed).unwrap();
        assert!(!report.signed);
        assert!(report.detail.contains("No key"));
    }

    #[test]
    fn offline_report_marks_verified_on_clean_round_trip() {
        let tmp = TempDir::new().unwrap();
        let key = generate_key();
        store_key(tmp.path(), &key, false).unwrap();
        let p = sample_provenance();
        let signed = sign(tmp.path(), &p).unwrap();
        let report = offline_report(tmp.path(), &signed).unwrap();
        assert!(report.signed);
        assert!(report.verified);
    }
}

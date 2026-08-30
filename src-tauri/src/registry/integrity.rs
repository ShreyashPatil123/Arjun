//! Verifies model weights against their declared hashes before loading.
//!
//! A model whose SHA-256 does not match its manifest entry is refused, so a
//! tampered or partially-downloaded file can never be served. The check is
//! run when the model is first launched (see `serving::managed_endpoint`),
//! which is the moment a process is about to consume the weights — too
//! early and the cost of hashing a 7 GB file has been paid for a run that
//! never uses the model; too late and the model is already answering
//! requests.
//!
//! ## Threat model
//!
//! The threat is *not* the file being maliciously written to disk with a
//! new name — the path on disk is what the operator approved. The threat
//! is the *contents* under that path being different from what the
//! registry manifest claims. The two real-world sources of that:
//!
//! - A partial download (the file exists, has the right name, but is
//!   truncated or otherwise incomplete). Refusing to load these is
//!   correct, because a 4 GB model truncated to 2 GB will produce
//!   nonsense once loaded.
//! - A supply-chain compromise that swapped the file out from under the
//!   manifest. The manifest says model X with hash Y, but the file on
//!   disk is the attacker's model. A hash check refuses to load it.
//!
//! The model loader is the only place that *needs* to trust the bytes, so
//! the check belongs there, not at import time. (An import-time check
//! would catch the partial-download case earlier, but the cost — hashing
//! every model on every startup — is high enough that a check at the
//! point of use is the right shape.)
//!
//! ## Why SHA-256 and not a stronger hash
//!
//! SHA-256 is what the upstream manifest format already uses, and what
//! every operator-verification tool in the GGUF ecosystem produces. A
//! stronger hash would buy nothing here because the attacker is
//! constrained to second-preimage attacks on the model file itself, and
//! SHA-256 is not the weak link in that chain.

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Computes the SHA-256 of a file, streamed in 8 KiB chunks so multi-GB
/// weight files do not have to be loaded into memory.
///
/// The 8 KiB chunk size matches what the standard library's `BufReader`
/// uses for its default capacity, which is a fine trade-off between
/// syscall overhead and resident memory.
pub fn hash_file(path: &Path) -> Result<String> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file =
        File::open(path).with_context(|| format!("could not open {} for hashing", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8 * 1024];

    loop {
        let n = reader
            .read(&mut buffer)
            .with_context(|| format!("could not read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Verifies a file against an expected SHA-256 hex string.
///
/// The comparison is case-insensitive: manifests in the wild vary on
/// whether they uppercase the hex digits, and a strict equality would
/// produce a confusing failure on a perfectly valid file. The hash itself
/// is still 256 bits of entropy either way.
pub fn verify(path: &Path, expected: &str) -> Result<()> {
    let actual = hash_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        anyhow::bail!(
            "integrity check failed for {}: expected {expected}, got {actual}",
            path.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constant is part of the manifest contract. A change here is a
    /// breaking change for every administrator who has already approved a
    /// model under the old hash format.
    #[test]
    fn the_output_is_a_lowercase_hex_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        std::fs::write(&path, b"hello world").unwrap();

        let hex = hash_file(&path).unwrap();

        // SHA-256 is 32 bytes -> 64 hex digits.
        assert_eq!(hex.len(), 64, "expected 64 hex chars, got {hex:?}");
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "expected lowercase hex, got {hex:?}",
        );
        // Known answer for "hello world" — locks the algorithm too, not
        // just the format.
        assert_eq!(
            hex,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        );
    }

    /// Case-insensitive comparison so a manifest with uppercase hex still
    /// verifies.
    #[test]
    fn verify_accepts_uppercase_hex() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        std::fs::write(&path, b"hello world").unwrap();

        let uppercase = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9";
        assert!(verify(&path, uppercase).is_ok());
    }

    /// Mismatch is a hard failure with both hashes in the message so the
    /// operator can see at a glance which file drifted from its manifest.
    #[test]
    fn verify_rejects_mismatched_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        std::fs::write(&path, b"hello world").unwrap();

        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify(&path, wrong).unwrap_err().to_string();
        assert!(err.contains("integrity check failed"), "{err}");
        assert!(err.contains("expected 0000"), "{err}");
        assert!(err.contains("got b94d"), "{err}");
    }

    /// A missing file produces a clear open error rather than a hash
    /// mismatch — distinguishing "weights not downloaded yet" from
    /// "weights corrupted in place" matters for incident response.
    #[test]
    fn verify_reports_missing_files_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.bin");

        let err = verify(&path, "00").unwrap_err().to_string();
        assert!(err.contains("could not open"), "{err}");
    }
}
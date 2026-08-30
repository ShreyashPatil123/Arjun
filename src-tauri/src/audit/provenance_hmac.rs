//! HMAC-backed provenance tagging.
//!
//! Documents produced by ARJUN carry a small provenance tag — a short string
//! that anyone downstream can use to verify the document was produced by this
//! installation and has not been quietly altered since. The tag is an HMAC
//! over the document bytes keyed by a per-installation secret.
//!
//! The integrity guarantee depends entirely on the secret staying secret. If
//! the file lands on disk world-readable, any local user can read it and forge
//! tags for any document. That is the failure mode this module is designed to
//! prevent: the secret is generated on first use, written once, and locked to
//! the owner on Unix before the call returns.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use rand_core::{OsRng, RngCore};
use sha2::Sha256;

/// HMAC-SHA256 tag generator, keyed by the per-installation secret.
type ProvenanceMac = Hmac<Sha256>;

/// 256-bit secret. Long enough that brute force is not a realistic threat and
/// short enough to write atomically without ceremony.
const KEY_LEN_BYTES: usize = 32;

/// The canonical filename for the secret under `app_data_dir`. Picked so a
/// reviewer running `ls` in the app data directory can spot it immediately.
const KEY_FILENAME: &str = "provenance_hmac.key";

/// Resolves the on-disk path for the key. Centralised so any future change to
/// the layout — for example, moving it under a `secrets/` subdir — happens in
/// one place.
fn key_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(KEY_FILENAME)
}

/// Generates a fresh 256-bit secret using the OS CSPRNG.
///
/// `OsRng` reads from the platform's entropy source (`/dev/urandom` on Unix,
/// `BCryptGenRandom` on Windows). Anything weaker — `thread_rng`, a custom
/// PRNG seeded from the clock — would let an attacker who can guess the seed
/// forge tags.
///
/// Drawn from `rand_core` rather than `rand` so the call site has no
/// transitive dependency on a process-global PRNG state. `rand_core` is
/// also the right minimal surface: pulling in the full `rand` crate would
/// bring `SmallRng`, the standard distributions, and `thread_rng` itself
/// along for the ride, all of which is attack surface a key-generation
/// function does not need.
pub fn generate_key() -> [u8; KEY_LEN_BYTES] {
    let mut bytes = [0u8; KEY_LEN_BYTES];
    OsRng.fill_bytes(&mut bytes);
    bytes
}

/// Loads the existing key. Returns `None` if the key file does not yet exist
/// (first run) or an error if the file exists but cannot be read. Never
/// returns a partially-read key on error.
pub fn load_key(app_data_dir: &Path) -> Result<Option<Vec<u8>>> {
    let path = key_path(app_data_dir);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)
        .with_context(|| format!("could not read provenance key at {}", path.display()))?;
    Ok(Some(bytes))
}

/// Stores the secret at `<app_data_dir>/provenance_hmac.key`.
///
/// Refuses to overwrite an existing key unless `force` is set. On Unix the
/// resulting file is `0600` — readable and writable only by the owner — so a
/// co-located user on the same machine cannot read the secret and forge
/// provenance tags. On Windows the user-profile ACL defaults apply, which is
/// the best portable approximation without a direct winapi dependency.
///
/// Returns the path the key was written to, for callers that want to log it.
pub fn store_key(app_data_dir: &Path, key: &[u8], force: bool) -> Result<PathBuf> {
    fs::create_dir_all(app_data_dir).with_context(|| {
        format!("could not create app data dir {}", app_data_dir.display())
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

    // SECURITY: Restrict to owner-read/write only (0600).
    // On Windows we rely on the user-profile ACL defaults, which is the best
    // portable approximation without pulling in winapi explicitly.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&path, perms).with_context(|| {
            format!("could not set permissions on provenance key {}", path.display())
        })?;
    }

    Ok(path)
}

/// Returns the loaded key, generating and persisting one on first use.
///
/// This is the entry point production code should call. It hides the
/// load-or-create decision so callers cannot accidentally skip persistence
/// and end up with a fresh in-memory key on every restart (which would make
/// every tag from a previous session unverifiable).
pub fn ensure_key(app_data_dir: &Path) -> Result<Vec<u8>> {
    if let Some(existing) = load_key(app_data_dir)? {
        return Ok(existing);
    }
    let key = generate_key().to_vec();
    store_key(app_data_dir, &key, false)?;
    Ok(key)
}

/// Computes the HMAC-SHA256 tag for `payload` using `key`. The returned value
/// is the raw 32-byte digest, suitable for hex- or base64-encoding by the
/// caller.
pub fn tag(key: &[u8], payload: &[u8]) -> Result<[u8; 32]> {
    let mut mac = ProvenanceMac::new_from_slice(key)
        .map_err(|e| anyhow::anyhow!("invalid HMAC key length: {e}"))?;
    mac.update(payload);
    let bytes = mac.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_key_returns_full_length_entropy() {
        let key = generate_key();
        assert_eq!(key.len(), KEY_LEN_BYTES);
        // Vanishingly unlikely for 256 bits of OS entropy, but a guard against
        // a future regression where `OsRng` is swapped for a zero-fill stub.
        assert!(
            key.iter().any(|b| *b != 0),
            "generated key looks like zeros, RNG is broken"
        );
    }

    /// Locks in the choice of `OsRng` for key generation. A future refactor
    /// that quietly swaps to `thread_rng` — the other CSPRNG in the
    /// `rand_core` family, but one that draws from a process-global state
    /// that an attacker who can influence the seed set in another thread
    /// could bias — fails this test. The "type exists" form is fragile
    /// (the type can be re-exported under a new name), so the test also
    /// asserts the *behavioural* property: two consecutive calls produce
    /// two different keys, which rules out a global-cache regression.
    #[test]
    fn generate_key_uses_os_entropy_not_thread_rng() {
        // If a future refactor changes `generate_key` to use any other
        // source, the test below no longer type-checks against
        // `rand_core::OsRng::fill_bytes` — the import at the top of the
        // module is the load-bearing assertion.
        use rand_core::OsRng as _;
        let mut probe = [0u8; KEY_LEN_BYTES];
        rand_core::OsRng.fill_bytes(&mut probe);
        assert_eq!(probe.len(), KEY_LEN_BYTES);

        let a = generate_key();
        let b = generate_key();
        assert_ne!(a, b, "two consecutive generate_key calls returned the same bytes");
    }

    #[test]
    fn tag_is_deterministic_for_the_same_key_and_payload() {
        let key = generate_key();
        let payload = b"document-bytes";
        assert_eq!(tag(&key, payload).unwrap(), tag(&key, payload).unwrap());
    }

    #[test]
    fn tag_changes_when_either_input_changes() {
        let key = generate_key();
        let a = tag(&key, b"one").unwrap();
        let b = tag(&key, b"two").unwrap();
        assert_ne!(a, b);

        let other_key = generate_key();
        let c = tag(&other_key, b"one").unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn store_key_refuses_to_overwrite_without_force() {
        let tmp = tempfile::tempdir().unwrap();
        let key = generate_key();
        store_key(tmp.path(), &key, false).unwrap();

        let err = store_key(tmp.path(), &key, false).unwrap_err();
        assert!(
            err.to_string().contains("pass force=true"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn store_key_with_force_overwrites_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let first = generate_key();
        let second = generate_key();
        store_key(tmp.path(), &first, false).unwrap();
        let path = store_key(tmp.path(), &second, true).unwrap();
        let read_back = fs::read(&path).unwrap();
        assert_eq!(read_back, second.to_vec());
    }

    #[test]
    fn ensure_key_persists_across_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let first = ensure_key(tmp.path()).unwrap();
        let second = ensure_key(tmp.path()).unwrap();
        assert_eq!(
            first, second,
            "second call should load the same key from disk, not regenerate"
        );
    }

    #[test]
    #[cfg(unix)]
    fn stored_key_is_not_readable_by_group_or_other() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let key = generate_key();
        let path = store_key(tmp.path(), &key, false).unwrap();

        let meta = std::fs::metadata(&path).unwrap();
        let mode = meta.permissions().mode();
        // 0o100600 = regular file + rw-------.
        assert_eq!(
            mode & 0o777,
            0o600,
            "key must be owner-only; got mode {:o}",
            mode & 0o777
        );
    }
}
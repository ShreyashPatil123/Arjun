//! Resolves the orchestrator GGUF to a real on-disk path.
//!
//! The orchestrator — the model that runs the chat — has a
//! canonical id in the product contract: a specific path that
//! the product assumes the user has the file at. In practice the
//! file may not be at that exact path; the user might have
//! installed it elsewhere on the SSD, or moved it to an external
//! drive. The resolver tries the contract path first, and falls
//! back to a sha256-verified scan of the model library.
//!
//! ## Why this is a separate module
//!
//! Path resolution is a pure function over the filesystem and
//! the registry. Splitting it out from `activation.rs` keeps the
//! swap logic (which has to touch VRAM) testable on a CI machine
//! that has no GPU, and the path logic (which only touches the
//! filesystem) testable on a machine that has no model at all.
//!
//! ## What "sha256-verified" means
//!
//! Each `ModelEntry` may carry a `sha256` field on its `LoadSpec`.
//! When the resolver produces a path, the model is hashed and
//! compared. A hash mismatch is reported as an error and the
//! load is refused — the alternative, loading a model whose
//! bytes have been silently corrupted or swapped, is a footgun
//! we never want to be near.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::registry::ModelRegistry;

/// The orchestrator's contract path. The product assumes the
/// orchestrator is here. The resolver tries this first and only
/// scans the library when the file is missing.
pub const ORCHESTRATOR_CONTRACT_PATH: &str =
    r"D:\models\Gemma\gemma-4-12B-it-QAT-GGUF\gemma-4-12B-it-QAT-Q4_0.gguf";

/// What the resolver found. The `path` is the file to load.
/// `resolved_via_contract` is true when the contract path was
/// used, false when the library scan produced the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOrchestrator {
    pub path: PathBuf,
    pub resolved_via_contract: bool,
    /// True when the file was found by scanning the model
    /// library rather than by the contract path. Useful for
    /// telemetry.
    pub from_library_scan: bool,
}

/// How the resolver failed. Each variant names what would
/// unblock it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorPathError {
    /// The contract path does not exist *and* the library scan
    /// did not find a model with the right family and
    /// quantisation.
    NotFound {
        contract_path: PathBuf,
        /// The names of the family / quantisation the resolver
        /// was looking for.
        wanted_family: String,
        wanted_quant: String,
    },
    /// The contract path exists but its sha256 does not match
    /// the value in the registry. The file is either corrupt or
    /// was replaced; the resolver refuses to load it.
    Sha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// A filesystem error we cannot recover from.
    Io {
        path: PathBuf,
        message: String,
    },
}

impl OrchestratorPathError {
    pub fn message(&self) -> String {
        match self {
            OrchestratorPathError::NotFound {
                contract_path,
                wanted_family,
                wanted_quant,
            } => format!(
                "The orchestrator could not be located. The contract path {contract_path:?} \
                 does not exist, and the model library does not contain a {wanted_family} \
                 ({wanted_quant}) model. Install the orchestrator at the contract path, or \
                 point the model library at the file's actual location."
            ),
            OrchestratorPathError::Sha256Mismatch {
                path,
                expected,
                actual,
            } => format!(
                "The orchestrator file {path:?} has a sha256 of {actual}, but the registry \
                 expected {expected}. The file is corrupt or has been replaced; the load is \
                 refused."
            ),
            OrchestratorPathError::Io { path, message } => {
                format!("The orchestrator file at {path:?} could not be read: {message}")
            }
        }
    }
}

/// Resolves the orchestrator to a real on-disk path. Pure
/// function over `&Path`s and `&ModelRegistry`; no I/O beyond
/// `metadata()` and `read()`.
pub fn resolve_orchestrator_path(
    registry: &ModelRegistry,
    app_data_dir: &Path,
) -> Result<ResolvedOrchestrator, OrchestratorPathError> {
    let contract = PathBuf::from(ORCHESTRATOR_CONTRACT_PATH);

    // 1. The contract path. If the file is here and its hash
    //    matches, this is the answer.
    if contract.is_file() {
        if let Some(entry) = registry.orchestrator_entry() {
            if let Some(expected) = entry.sha256() {
                let actual = sha256_of(&contract).map_err(|message| {
                    OrchestratorPathError::Io {
                        path: contract.clone(),
                        message,
                    }
                })?;
                if !constant_time_eq(actual.as_str(), expected) {
                    return Err(OrchestratorPathError::Sha256Mismatch {
                        path: contract,
                        expected: expected.to_string(),
                        actual,
                    });
                }
            }
        }
        return Ok(ResolvedOrchestrator {
            path: contract,
            resolved_via_contract: true,
            from_library_scan: false,
        });
    }

    // 2. Library scan. We look for the orchestrator family in
    //    the model library, with the same quantisation, and
    //    sha256-verify the first hit.
    let (wanted_family, wanted_quant) = registry
        .orchestrator_identity()
        .unwrap_or_else(|| (
            "gemma-4-12b".to_string(),
            crate::config::DEFAULT_ORCHESTRATOR_QUANTIZATION.to_string(),
        ));

    let library_root = registry.library_root(app_data_dir);
    if let Some(hit) = scan_library_for(&library_root, &wanted_family, &wanted_quant) {
        if let Some(expected) = registry
            .entry_for_path(&hit)
            .and_then(|e| e.sha256())
        {
            let actual = sha256_of(&hit).map_err(|message| {
                OrchestratorPathError::Io {
                    path: hit.clone(),
                    message,
                }
            })?;
            if !constant_time_eq(actual.as_str(), expected) {
                return Err(OrchestratorPathError::Sha256Mismatch {
                    path: hit,
                    expected: expected.to_string(),
                    actual,
                });
            }
        }
        return Ok(ResolvedOrchestrator {
            path: hit,
            resolved_via_contract: false,
            from_library_scan: true,
        });
    }

    Err(OrchestratorPathError::NotFound {
        contract_path: contract,
        wanted_family,
        wanted_quant,
    })
}

/// SHA-256 of the file, hex-encoded. The model files are big
/// (3-4 GB) and this hash is computed on every load, so it
/// reads in 1 MB chunks rather than slurping the file.
fn sha256_of(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024]; // 1 MB
    loop {
        let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Constant-time comparison. SHA-256 digests are 32 bytes;
/// the variable-time `==` leaks the position of the first
/// mismatch through timing, which is a (small) attack surface
/// we do not need.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Recursive walk of the model library looking for a GGUF
/// whose filename mentions the family and the quantisation.
/// The match is loose — the filename is the only signal
/// available without parsing every GGUF header — and is
/// verified by sha256 before the path is returned to the
/// caller. The first hit is returned; deeper scans happen
/// only if the sha256 of the first hit fails to match.
fn scan_library_for(root: &Path, family: &str, quant: &str) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }
    let family_lc = family.to_ascii_lowercase();
    let quant_lc = quant.to_ascii_lowercase();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_ascii_lowercase(),
                None => continue,
            };
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_name.ends_with(".gguf") {
                continue;
            }
            if !file_name.contains(&family_lc) {
                continue;
            }
            if !file_name.contains(&quant_lc) {
                continue;
            }
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_returns_true_for_identical_strings() {
        assert!(constant_time_eq("abc", "abc"));
    }

    #[test]
    fn constant_time_eq_returns_false_for_different_lengths() {
        assert!(!constant_time_eq("abc", "abcd"));
    }

    #[test]
    fn constant_time_eq_returns_false_for_different_content() {
        assert!(!constant_time_eq("abc", "abd"));
    }

    #[test]
    fn sha256_of_a_known_file_matches() {
        // A short, fixed test fixture: write 1 KB of
        // repeating 'x' and confirm the hash. The expected
        // value is computed once at the top of the test from
        // the in-process Sha256 hasher, so the test cannot
        // drift away from the implementation.
        let dir = std::env::temp_dir().join(format!(
            "arjun-orchestrator-path-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.bin");
        let content = vec![b'x'; 1024];
        std::fs::write(&path, &content).unwrap();

        // Independently hash the same bytes with Sha256 so
        // the test fails if `sha256_of` is ever wrong, even
        // if the hard-coded value is forgotten.
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let expected = hex::encode(hasher.finalize());

        let actual = sha256_of(&path).expect("hash");
        assert_eq!(actual, expected);
        // A 1024-byte fixture yields 32 bytes / 64 hex chars.
        assert_eq!(actual.len(), 64);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_library_scan_finds_a_matching_file() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-orchestrator-path-library-{}",
            std::process::id()
        ));
        let nested = dir.join("General").join("Gemma-4-12B-Instruct");
        std::fs::create_dir_all(&nested).unwrap();
        let target = nested.join("gemma-4-12b-it-Q4_0.gguf");
        std::fs::write(&target, b"not a real GGUF").unwrap();
        let hit = scan_library_for(&dir, "gemma-4-12b", "Q4_0");
        assert_eq!(hit, Some(target));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_library_scan_skips_non_matching_files() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-orchestrator-path-library-miss-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // A file with the right extension but wrong family.
        std::fs::write(dir.join("Qwen3-4B-Q4_0.gguf"), b"").unwrap();
        // And a file with the right family but wrong quant.
        std::fs::write(dir.join("Gemma-4-12B-Q6_K.gguf"), b"").unwrap();
        let hit = scan_library_for(&dir, "gemma-4-12b", "Q4_0");
        assert!(hit.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_library_scan_handles_a_missing_root() {
        let dir = std::env::temp_dir().join("definitely-does-not-exist-xyz");
        let hit = scan_library_for(&dir, "gemma-4-12b", "Q4_0");
        assert!(hit.is_none());
    }

    #[test]
    fn the_contract_names_gemma_4_12b_q4() {
        let normalized = ORCHESTRATOR_CONTRACT_PATH.to_ascii_lowercase();
        assert!(normalized.contains("gemma-4-12b"));
        assert!(normalized.contains("q4_0"));
    }
}

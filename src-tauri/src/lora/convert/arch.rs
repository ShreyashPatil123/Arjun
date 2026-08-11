//! Learning the base model's architecture from the model already on disk.
//!
//! ## Why this is read locally
//!
//! llama.cpp's own converter takes the architecture from the base model's
//! HuggingFace `config.json`, which means fetching it from the network — and
//! Sarathi does not keep one, because it stores base models as GGUF.
//!
//! It does not need to. A conversion only ever happens for a base model the
//! user has already installed, and every GGUF states its own architecture in
//! `general.architecture`. Reading it from the local file makes conversion work
//! offline, removes a network failure mode from an automatic background step,
//! and rules out the mismatch where a fetched config describes a different
//! model than the one the adapter will actually attach to.
//!
//! The header parsing itself is [`crate::adapter_manager::gguf`]'s, not a second
//! implementation.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::adapter_manager::gguf::read_metadata_string;

/// The metadata key every GGUF uses to name its model family.
const ARCHITECTURE_KEY: &str = "general.architecture";

/// Reads `general.architecture` from a base model GGUF.
pub fn read_architecture(base_gguf: &Path) -> Result<String> {
    read_metadata_string(base_gguf, ARCHITECTURE_KEY)
        .with_context(|| format!("could not read {}", base_gguf.display()))?
        .ok_or_else(|| {
            anyhow!(
                "{} does not say which model family it is, so Sarathi cannot tell \
                 how to convert an adapter for it.",
                base_gguf.display()
            )
        })
}

/// Finds the base GGUF for an adapter package, looking at sibling packages when
/// the adapter's own package holds no weights.
///
/// ## Why a sibling search is needed
///
/// Adapters install under the repository the user was *browsing*
/// (`Qwen_Qwen2.5-Coder-1.5B-Instruct`), while GGUF weights come from whoever
/// published the quantisation (`bartowski_Qwen2.5-Coder-1.5B-Instruct-GGUF`).
/// They are the same model from two repositories, so the adapter's package
/// legitimately has an `adapters/` folder and no `base/` one.
///
/// This never mattered before: a downloaded GGUF adapter needs no base model.
/// Conversion does, which is what surfaced the split.
///
/// Matching is exact after normalisation, never fuzzy — attaching an adapter to
/// the wrong base produces a model that answers badly with nothing to say why.
pub fn resolve_base_gguf(package_dir: &Path) -> Result<PathBuf> {
    if let Ok(found) = find_base_gguf(package_dir) {
        return Ok(found);
    }

    let (models_root, package_name) = match (package_dir.parent(), package_dir.file_name()) {
        (Some(root), Some(name)) => (root, name.to_string_lossy().to_string()),
        _ => bail!("no base model is installed at {}", package_dir.join("base").display()),
    };

    let wanted = normalised_model_key(&package_name);

    let sibling = std::fs::read_dir(models_root)
        .with_context(|| format!("could not list {}", models_root.display()))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path != package_dir)
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| normalised_model_key(n) == wanted)
        })
        .find_map(|path| find_base_gguf(&path).ok());

    sibling.ok_or_else(|| {
        anyhow!(
            "No base model for '{package_name}' is installed. Download the model first — \
             an adapter is converted against the model it attaches to."
        )
    })
}

/// Reduces a package directory name to the model it describes.
///
/// Drops the publisher prefix and any `-GGUF` suffix, so the original
/// repository and a quantised republication of it compare equal:
///
/// ```text
/// Qwen_Qwen2.5-Coder-1.5B-Instruct            -> qwen2.5-coder-1.5b-instruct
/// bartowski_Qwen2.5-Coder-1.5B-Instruct-GGUF  -> qwen2.5-coder-1.5b-instruct
/// ```
fn normalised_model_key(package_name: &str) -> String {
    let after_publisher = package_name
        .split_once('_')
        .map(|(_, rest)| rest)
        .unwrap_or(package_name);

    let lowered = after_publisher.to_ascii_lowercase();
    lowered.strip_suffix("-gguf").unwrap_or(&lowered).to_string()
}

/// Finds the primary base GGUF inside an installed model package.
///
/// Mirrors the selection `AdapterRegistry::ensure_valid_manifest` makes: the
/// first shard when a model is split, otherwise the first file by name. Both
/// must agree, because the manifest is what the rest of the app trusts.
pub fn find_base_gguf(package_dir: &Path) -> Result<PathBuf> {
    let base_dir = package_dir.join("base");
    if !base_dir.is_dir() {
        bail!("no base model is installed at {}", base_dir.display());
    }

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base_dir)
        .with_context(|| format!("could not list {}", base_dir.display()))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .collect();

    if candidates.is_empty() {
        bail!("no .gguf file found in {}", base_dir.display());
    }

    candidates.sort();

    // A sharded model states its architecture in the first shard; later shards
    // carry tensor data only.
    let first_shard = candidates.iter().find(|path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("-00001-of-"))
    });

    Ok(first_shard.cloned().unwrap_or_else(|| candidates[0].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal GGUF header declaring an architecture.
    fn write_gguf_with_arch(dir: &Path, name: &str, arch: Option<&str>) -> PathBuf {
        let mut kvs = Vec::new();
        let mut count = 0u64;

        if let Some(arch) = arch {
            let key = ARCHITECTURE_KEY;
            kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
            kvs.extend_from_slice(key.as_bytes());
            kvs.extend_from_slice(&8u32.to_le_bytes()); // STRING
            kvs.extend_from_slice(&(arch.len() as u64).to_le_bytes());
            kvs.extend_from_slice(arch.as_bytes());
            count += 1;
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&kvs);

        let path = dir.join(name);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn temp_package(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_arch_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("base")).unwrap();
        dir
    }

    #[test]
    fn the_architecture_is_read_from_the_header() {
        let pkg = temp_package("plain");
        let gguf = write_gguf_with_arch(&pkg.join("base"), "model.gguf", Some("qwen3"));

        assert_eq!(read_architecture(&gguf).unwrap(), "qwen3");
    }

    #[test]
    fn a_header_without_an_architecture_is_reported_in_plain_language() {
        let pkg = temp_package("no_arch");
        let gguf = write_gguf_with_arch(&pkg.join("base"), "model.gguf", None);

        let err = read_architecture(&gguf).unwrap_err().to_string();
        assert!(err.contains("model family"), "got: {err}");
        assert!(!err.contains("general.architecture"), "should not name the key: {err}");
    }

    #[test]
    fn the_single_base_file_is_found() {
        let pkg = temp_package("single");
        write_gguf_with_arch(&pkg.join("base"), "qwen3-4b-q4.gguf", Some("qwen3"));

        let found = find_base_gguf(&pkg).unwrap();
        assert_eq!(found.file_name().unwrap(), "qwen3-4b-q4.gguf");
    }

    /// A sharded model must resolve to shard one — later shards hold no metadata.
    #[test]
    fn a_sharded_model_resolves_to_its_first_shard() {
        let base = temp_package("sharded").join("base");
        write_gguf_with_arch(&base, "model-00002-of-00003.gguf", None);
        write_gguf_with_arch(&base, "model-00001-of-00003.gguf", Some("llama"));
        write_gguf_with_arch(&base, "model-00003-of-00003.gguf", None);

        let pkg = base.parent().unwrap();
        let found = find_base_gguf(pkg).unwrap();
        assert_eq!(found.file_name().unwrap(), "model-00001-of-00003.gguf");
        assert_eq!(read_architecture(&found).unwrap(), "llama");
    }

    /// The real layout that broke conversion: the adapter's package holds only
    /// `adapters/`, while the weights sit in a `-GGUF` republication next door.
    #[test]
    fn the_base_model_is_found_in_a_gguf_sibling_package() {
        let root = std::env::temp_dir().join("sarathi_arch_sibling");
        let _ = std::fs::remove_dir_all(&root);

        let adapter_pkg = root.join("Qwen_Qwen2.5-Coder-1.5B-Instruct");
        std::fs::create_dir_all(adapter_pkg.join("adapters")).unwrap();

        let weights_pkg = root.join("bartowski_Qwen2.5-Coder-1.5B-Instruct-GGUF");
        write_gguf_with_arch(&weights_pkg.join("base"), "coder-q4.gguf", Some("qwen2"));

        // Its own package has no base/ at all.
        assert!(find_base_gguf(&adapter_pkg).is_err());

        let found = resolve_base_gguf(&adapter_pkg).unwrap();
        assert_eq!(found.file_name().unwrap(), "coder-q4.gguf");
        assert_eq!(read_architecture(&found).unwrap(), "qwen2");
    }

    /// A different model must never be borrowed as a base.
    #[test]
    fn an_unrelated_sibling_is_not_used_as_the_base() {
        let root = std::env::temp_dir().join("sarathi_arch_unrelated");
        let _ = std::fs::remove_dir_all(&root);

        let adapter_pkg = root.join("Qwen_Qwen2.5-Coder-1.5B-Instruct");
        std::fs::create_dir_all(adapter_pkg.join("adapters")).unwrap();

        // Same publisher family, genuinely different model.
        let other = root.join("Qwen_Qwen2.5-1.5B-Instruct-GGUF");
        write_gguf_with_arch(&other.join("base"), "plain.gguf", Some("qwen2"));

        let err = resolve_base_gguf(&adapter_pkg).unwrap_err().to_string();
        assert!(err.contains("No base model"), "got: {err}");
        assert!(err.contains("Download the model first"), "should say what to do: {err}");
    }

    #[test]
    fn normalisation_ignores_publisher_and_gguf_suffix() {
        assert_eq!(
            normalised_model_key("Qwen_Qwen2.5-Coder-1.5B-Instruct"),
            normalised_model_key("bartowski_Qwen2.5-Coder-1.5B-Instruct-GGUF")
        );
        assert_ne!(
            normalised_model_key("Qwen_Qwen2.5-Coder-1.5B-Instruct"),
            normalised_model_key("Qwen_Qwen2.5-1.5B-Instruct")
        );
    }

    /// A package that already holds its own weights must not consult siblings.
    #[test]
    fn a_package_with_its_own_base_is_preferred() {
        let root = std::env::temp_dir().join("sarathi_arch_own");
        let _ = std::fs::remove_dir_all(&root);

        let pkg = root.join("Qwen_Qwen2.5-Coder-1.5B-Instruct");
        write_gguf_with_arch(&pkg.join("base"), "own.gguf", Some("qwen2"));

        let sibling = root.join("bartowski_Qwen2.5-Coder-1.5B-Instruct-GGUF");
        write_gguf_with_arch(&sibling.join("base"), "sibling.gguf", Some("llama"));

        let found = resolve_base_gguf(&pkg).unwrap();
        assert_eq!(found.file_name().unwrap(), "own.gguf");
    }

    #[test]
    fn a_package_with_no_base_directory_is_reported() {
        let dir = std::env::temp_dir().join("sarathi_arch_empty_pkg");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let err = find_base_gguf(&dir).unwrap_err().to_string();
        assert!(err.contains("no base model"), "got: {err}");
    }

    #[test]
    fn a_base_directory_holding_no_gguf_is_reported() {
        let pkg = temp_package("no_gguf");
        std::fs::write(pkg.join("base").join("README.md"), b"not a model").unwrap();

        let err = find_base_gguf(&pkg).unwrap_err().to_string();
        assert!(err.contains("no .gguf"), "got: {err}");
    }
}

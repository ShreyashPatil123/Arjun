//! Downloading and managing LoRA adapters on disk.
//!
//! Adapters live beside the model they belong to:
//!
//! ```text
//! models/<provider>/<model>/adapters/<adapter-id>/adapter.gguf
//! ```
//!
//! Keeping them under the base model is deliberate — an adapter is only
//! meaningful with its base, and deleting a model should take its adapters with
//! it rather than leaving orphans behind.
//!
//! ## What is installable
//!
//! llama.cpp loads GGUF adapters only. Most published adapters are PEFT
//! safetensors, which [`crate::lora::convert_adapter`] turns into GGUF locally
//! during the install — so both are accepted.
//!
//! What is still refused is a repository that ships neither: merged model
//! weights wearing an adapter tag, or a `.bin` pickle checkpoint. Those are
//! rejected **before** any bytes are fetched, with an explanation — downloading
//! half a gigabyte that silently fails to load is worse than a clear refusal.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// An adapter installed on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledAdapter {
    /// Directory name, derived from the source repository.
    pub id: String,
    /// Repository it came from.
    pub repo_id: String,
    /// Human-readable name for the UI.
    pub name: String,
    /// The model this adapter belongs to.
    pub base_model_id: String,
    pub file_path: String,
    pub size_bytes: u64,
    /// Capability slot this adapter is bound to, when it has one.
    ///
    /// Read from the package manifest rather than from disk: the directory says
    /// what is installed, the manifest says what will actually be *used*. An
    /// adapter with `None` here is installed and inert until assigned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    /// `stated`, `suggested`, or `manual` — how that slot was arrived at, so the
    /// UI can show a guess as a guess.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment_confidence: Option<String>,
}

/// Turns a repository id into a directory name safe on every platform.
pub fn adapter_dir_name(repo_id: &str) -> String {
    repo_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
        .collect()
}

/// Where a model's adapters live.
pub fn adapters_root(package_dir: &Path) -> PathBuf {
    package_dir.join("adapters")
}

/// True when a file list looks like complete model weights rather than an
/// adapter.
///
/// Some repositories carry HuggingFace's `base_model:adapter:` tag while
/// actually shipping merged weights — `ericflo/Llama-3.1-8B-ContinuedTraining`
/// is one, with `config.json`, `model-00001-of-00007.safetensors` and
/// multi-gigabyte GGUFs. Taking the tag at its word would download an entire
/// 8B model into the adapters folder.
///
/// `config.json` alone is not enough to judge: adapters legitimately ship
/// `adapter_config.json`, and the two must not be confused.
fn looks_like_full_model(filenames: &[String]) -> bool {
    let base = |f: &String| f.rsplit('/').next().unwrap_or(f).to_ascii_lowercase();

    let has_model_config = filenames.iter().any(|f| base(f) == "config.json");
    let has_full_weights = filenames.iter().any(|f| {
        let b = base(f);
        b.starts_with("model-") && b.ends_with(".safetensors")
            || b.starts_with("model.safetensors")
            || b.starts_with("pytorch_model")
    });

    has_model_config && has_full_weights
}

/// Rejects adapters that could not be loaded, before anything is downloaded.
///
/// `filenames` is the repository's file list. This is a pre-filter to avoid
/// spending a large download on a file that cannot work; the authoritative
/// check is [`crate::adapter_manager::gguf::verify_is_lora_adapter`], which
/// reads what the downloaded file declares itself to be.
pub fn check_installable(filenames: &[String]) -> Result<String> {
    let sized: Vec<(String, u64)> = filenames.iter().map(|f| (f.clone(), 0)).collect();
    check_installable_sized(&sized)
}

/// Largest a GGUF LoRA adapter is plausibly allowed to be.
///
/// An adapter stores low-rank deltas for a subset of layers — tens to a few
/// hundred megabytes even for large bases. Anything of model scale is a merged
/// checkpoint that happens to live in an adapter repository.
const MAX_ADAPTER_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// As [`check_installable`], but able to reject files too large to be adapters.
///
/// Sizes come from the Hub's blob listing. A size of `0` means unknown and the
/// ceiling is not applied, so callers without sizes lose nothing.
///
/// Size matters because names and configs both lie in the same direction.
/// `sandepaAI/sandepaAI_gemma4_coder_12b` is a real LoRA — `adapter_config.json`
/// and `adapter_model.safetensors` — that also ships a 9.1 GB *merged* GGUF for
/// people who want the whole thing. Vouching for its GGUF on the strength of the
/// PEFT config offered that merged model as a "ready" 9.1 GB adapter. The right
/// answer is that its only real adapter is safetensors, which needs converting.
pub fn check_installable_sized(files: &[(String, u64)]) -> Result<String> {
    let filenames: Vec<String> = files.iter().map(|(n, _)| n.clone()).collect();
    let size_of = |name: &str| files.iter().find(|(n, _)| n == name).map(|(_, s)| *s).unwrap_or(0);
    let adapter_sized = |name: &str| {
        let s = size_of(name);
        s == 0 || s <= MAX_ADAPTER_BYTES
    };

    let ggufs: Vec<&String> = filenames.iter().filter(|f| f.ends_with(".gguf")).collect();

    if !ggufs.is_empty() {
        if looks_like_full_model(&filenames) {
            return Err(anyhow!(
                "This page is labelled as an adapter, but it contains a complete model \
                 rather than an add-on skill. Install it from the Discover page as a \
                 model instead."
            ));
        }

        // Repositories can hold several GGUFs. One named for LoRA is the
        // adapter; picking the first alphabetically could grab something else.
        if let Some(named) = ggufs.iter().find(|f| {
            let lower = f.to_ascii_lowercase();
            (lower.contains("lora") || lower.contains("adapter")) && adapter_sized(f)
        }) {
            return Ok((*named).clone());
        }

        // No GGUF names itself an adapter. A real one still declares itself
        // through its PEFT config, so that is the last thing worth accepting —
        // and only for a file small enough to be one.
        let has_peft_config = filenames
            .iter()
            .any(|f| f.rsplit('/').next().unwrap_or(f).eq_ignore_ascii_case("adapter_config.json"));

        if has_peft_config {
            if let Some(small) = ggufs.iter().find(|f| adapter_sized(f)) {
                return Ok((*small).clone());
            }

            // Its GGUFs are all model-sized. If the genuine adapter is here in
            // PEFT form, say so rather than reporting no adapter at all.
            if filenames.iter().any(|f| f.ends_with(".safetensors")) {
                return Err(anyhow!(
                    "This adapter is in PEFT safetensors format, which Sarathi converts to \
                     GGUF while installing it. \
                     (The GGUF on this page is a merged model, not the adapter.)"
                ));
            }

            return Err(anyhow!(
                "The GGUF on this page is a merged model rather than the adapter itself. \
                 Install it from the Discover page as a model instead."
            ));
        }

        // Nothing here claims to be an adapter.
        //
        // Falling back to the first GGUF is what listed a 9 GB quantized
        // fine-tune as a ready LoRA: repos that publish only GGUFs carry no
        // safetensors for `looks_like_full_model` to catch, so a whole model
        // slipped through as an "adapter" and its size was shown as the
        // adapter's. A fine-tune is a model, and belongs on the Discover page.
        return Err(anyhow!(
            "This page publishes model files, not an add-on skill — none of its GGUFs \
             is a LoRA adapter and it has no adapter_config.json. If it is a fine-tune, \
             install it from the Discover page as a model instead."
        ));
    }

    // PEFT safetensors are not a refusal any more: `download_adapter` fetches
    // them and converts to GGUF locally. This is still an `Err` because the
    // caller's contract is "name the ready-to-load GGUF", and there isn't one —
    // the caller reads it as "take the conversion path", not as a failure.
    let has_safetensors = filenames.iter().any(|f: &String| f.ends_with(".safetensors"));
    if has_safetensors {
        return Err(anyhow!(
            "This adapter is in PEFT safetensors format, which Sarathi converts to GGUF \
             while installing it."
        ));
    }

    Err(anyhow!("This repository contains no adapter file Sarathi recognises."))
}

/// Lists adapters installed for a model.
///
/// A directory with no readable adapter file is skipped rather than listed as
/// broken: it is usually a partial download, and showing it as installed would
/// be misleading.
pub fn list_installed(package_dir: &Path, base_model_id: &str) -> Vec<InstalledAdapter> {
    let root = adapters_root(package_dir);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    // Read once, outside the loop: every directory is looked up against the same
    // manifest, and re-reading it per adapter would be pure repetition.
    let manifest = crate::adapter_manager::AdapterRegistry::read_manifest(package_dir).ok();

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }

        let Some(id) = dir.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };

        // The first .gguf in the directory is the adapter.
        let Ok(files) = std::fs::read_dir(&dir) else { continue };
        let found = files.flatten().find(|f| {
            f.path().extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)
        });

        let Some(file) = found else { continue };
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if size == 0 {
            continue; // partial download
        }

        // The source repository is recorded alongside, so the card can link back.
        let repo_id = std::fs::read_to_string(dir.join("source.txt"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| id.clone());

        // Which slot, if any, this directory is wired to. Matching on the
        // directory prefix rather than the exact filename means a record written
        // against `adapter_model.safetensors` still resolves after conversion
        // replaced it with `adapter.gguf`.
        let prefix = format!("adapters/{}/", id);
        let assignment = manifest.as_ref().and_then(|m| {
            m.adapters.iter().find(|(_, a)| {
                a.adapter_file.as_deref().map(|f| f.starts_with(&prefix)).unwrap_or(false)
            })
        });

        out.push(InstalledAdapter {
            name: repo_id.split('/').next_back().unwrap_or(&id).replace(['-', '_'], " "),
            capability: assignment.map(|(key, _)| key.clone()),
            assignment_confidence: assignment.and_then(|(_, a)| a.assignment_confidence.clone()),
            id,
            repo_id,
            base_model_id: base_model_id.to_string(),
            file_path: file.path().to_string_lossy().to_string(),
            size_bytes: size,
        });
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Removes an installed adapter.
///
/// Refuses to touch anything outside the model's adapters directory — the id
/// arrives from the UI, and a crafted value must not be able to delete
/// elsewhere on disk.
pub fn remove(package_dir: &Path, adapter_id: &str) -> Result<()> {
    let root = adapters_root(package_dir);
    let target = root.join(adapter_id);

    let canonical_root = root.canonicalize().map_err(|e| anyhow!("no adapters folder: {e}"))?;
    let canonical_target = target
        .canonicalize()
        .map_err(|e| anyhow!("adapter '{adapter_id}' is not installed: {e}"))?;

    if !canonical_target.starts_with(&canonical_root) {
        return Err(anyhow!("refusing to delete outside the adapters folder"));
    }

    std::fs::remove_dir_all(&canonical_target)
        .map_err(|e| anyhow!("could not remove adapter '{adapter_id}': {e}"))
}

/// Total bytes used by a model's adapters.
pub fn total_size(package_dir: &Path, base_model_id: &str) -> u64 {
    list_installed(package_dir, base_model_id).iter().map(|a| a.size_bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_adapter_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn install_fake(package: &Path, id: &str, repo: &str, bytes: usize) {
        let dir = adapters_root(package).join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("adapter.gguf"), vec![0u8; bytes]).unwrap();
        std::fs::write(dir.join("source.txt"), repo).unwrap();
    }

    #[test]
    fn gguf_adapters_are_installable() {
        let files = vec!["README.md".to_string(), "adapter.gguf".to_string()];
        assert_eq!(check_installable(&files).unwrap(), "adapter.gguf");
    }

    #[test]
    fn safetensors_adapters_report_that_they_are_not_directly_loadable() {
        // This is *not* a user-facing refusal any more. `download_adapter` reads
        // the error as "no ready GGUF here" and switches to converting the PEFT
        // weights instead. The error still has to name the format and what is
        // needed, because it is also what the user sees when there is nothing to
        // convert either.
        let files = vec!["adapter_model.safetensors".to_string(), "adapter_config.json".to_string()];

        let err = check_installable(&files).unwrap_err().to_string();
        assert!(err.contains("safetensors"), "should name the format: {err}");
        assert!(err.contains("GGUF"), "should say what is needed: {err}");
    }

    #[test]
    fn a_repo_with_no_adapter_file_is_refused() {
        let files = vec!["README.md".to_string()];
        assert!(check_installable(&files).is_err());
    }

    #[test]
    fn a_full_model_wearing_an_adapter_tag_is_refused() {
        // The real file list of ericflo/Llama-3.1-8B-ContinuedTraining, which
        // HuggingFace lists under base_model:adapter: but which ships whole
        // 8B weights. Accepting it downloads gigabytes that cannot be bound
        // as an adapter.
        let files = vec![
            "README.md".to_string(),
            "config.json".to_string(),
            "generation_config.json".to_string(),
            "model-00001-of-00007.safetensors".to_string(),
            "Llama-3.1-8B-ContinuedTraining2.C650.Q8_0.gguf".to_string(),
        ];

        let err = check_installable(&files).unwrap_err().to_string();
        assert!(err.contains("complete model"), "should say what it really is: {err}");
    }

    #[test]
    fn a_genuine_adapter_repo_is_not_mistaken_for_a_model() {
        // adapter_config.json must not be read as a model's config.json.
        let files = vec![
            "adapter_config.json".to_string(),
            "adapter_model.safetensors".to_string(),
            "sql-lora-f16.gguf".to_string(),
        ];

        assert_eq!(check_installable(&files).unwrap(), "sql-lora-f16.gguf");
    }

    #[test]
    fn the_lora_file_is_chosen_when_a_repo_holds_several() {
        // Picking the first entry would grab the merged model here.
        let files = vec![
            "merged-model-Q4_K_M.gguf".to_string(),
            "my-adapter-lora-f16.gguf".to_string(),
        ];

        assert_eq!(check_installable(&files).unwrap(), "my-adapter-lora-f16.gguf");
    }

    #[test]
    fn a_gguf_only_finetune_is_not_offered_as_an_adapter() {
        // The shape that produced a "9.1 GB LoRA adapter" in the browser: a
        // fine-tune published only as quantized GGUFs. There are no safetensors
        // for the full-model check to notice, and nothing names itself an
        // adapter — so the old fallback to the first GGUF took the whole model.
        let files = vec![
            "README.md".to_string(),
            "gemma4-coder-12b-Q6_K.gguf".to_string(),
            "gemma4-coder-12b-Q4_K_M.gguf".to_string(),
        ];

        let err = check_installable(&files).unwrap_err().to_string();
        assert!(err.contains("not an add-on skill"), "should say what it really is: {err}");
        assert!(err.contains("Discover"), "should point at the right page: {err}");
    }

    #[test]
    fn a_merged_model_beside_a_real_adapter_is_not_offered_as_the_adapter() {
        // sandepaAI/sandepaAI_gemma4_coder_12b: a genuine LoRA that also ships
        // the merged result. Its PEFT config vouched for the 9.1 GB GGUF, which
        // the browser then listed as a "ready" adapter of that size.
        let files = vec![
            ("adapter_config.json".to_string(), 700),
            ("adapter_model.safetensors".to_string(), 400_000_000),
            ("sandepaAI_gemma4_coder_12b_Q6_K.gguf".to_string(), 9_100_000_000),
        ];

        let err = check_installable_sized(&files).unwrap_err().to_string();
        assert!(err.contains("safetensors"), "should name the real adapter's format: {err}");
        assert!(err.contains("merged model"), "should explain the GGUF: {err}");
    }

    #[test]
    fn an_adapter_sized_gguf_beside_a_peft_config_is_still_accepted() {
        let files = vec![
            ("adapter_config.json".to_string(), 700),
            ("gemma-heretic-f16.gguf".to_string(), 120_000_000),
        ];
        assert_eq!(check_installable_sized(&files).unwrap(), "gemma-heretic-f16.gguf");
    }

    #[test]
    fn a_lora_named_file_of_model_scale_is_refused() {
        // The name says adapter, the size says merged checkpoint.
        let files = vec![
            ("adapter_config.json".to_string(), 700),
            ("my-model-lora-merged.gguf".to_string(), 8_000_000_000),
        ];
        assert!(check_installable_sized(&files).is_err());
    }

    #[test]
    fn unknown_sizes_leave_the_name_rules_in_charge() {
        // Callers without a blob listing must behave exactly as before.
        let files = vec!["adapter_config.json".to_string(), "sql-lora-f16.gguf".to_string()];
        assert_eq!(check_installable(&files).unwrap(), "sql-lora-f16.gguf");
    }

    #[test]
    fn a_peft_config_still_vouches_for_an_oddly_named_gguf() {
        // Some genuine adapters name their weights after the task rather than
        // after LoRA. adapter_config.json is the author saying what it is.
        let files = vec![
            "adapter_config.json".to_string(),
            "gemma-heretic-f16.gguf".to_string(),
        ];

        assert_eq!(check_installable(&files).unwrap(), "gemma-heretic-f16.gguf");
    }

    #[test]
    fn repository_ids_become_safe_directory_names() {
        // Slashes would otherwise create unintended nesting.
        assert_eq!(adapter_dir_name("philschmid/code-lora"), "philschmid_code-lora");
        assert!(!adapter_dir_name("a/b\\c:d").contains(['/', '\\', ':']));
    }

    #[test]
    fn installed_adapters_are_listed_with_their_source() {
        let pkg = temp_dir("list");
        install_fake(&pkg, "philschmid_sql-lora", "philschmid/sql-lora", 2048);

        let found = list_installed(&pkg, "meta-llama/Llama-3.1-8B");

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].repo_id, "philschmid/sql-lora");
        assert_eq!(found[0].size_bytes, 2048);
        assert_eq!(found[0].base_model_id, "meta-llama/Llama-3.1-8B");
    }

    #[test]
    fn a_model_with_no_adapters_lists_nothing() {
        let pkg = temp_dir("empty");
        assert!(list_installed(&pkg, "x/y").is_empty());
    }

    #[test]
    fn partial_downloads_are_not_reported_as_installed() {
        // A zero-byte file is an interrupted download, not a usable adapter.
        let pkg = temp_dir("partial");
        let dir = adapters_root(&pkg).join("half");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("adapter.gguf"), b"").unwrap();

        assert!(list_installed(&pkg, "x/y").is_empty());
    }

    #[test]
    fn total_size_sums_every_adapter() {
        let pkg = temp_dir("size");
        install_fake(&pkg, "one", "a/one", 1000);
        install_fake(&pkg, "two", "a/two", 2500);

        assert_eq!(total_size(&pkg, "x/y"), 3500);
    }

    #[test]
    fn removing_an_adapter_deletes_only_that_folder() {
        let pkg = temp_dir("remove");
        install_fake(&pkg, "keep", "a/keep", 100);
        install_fake(&pkg, "drop", "a/drop", 100);

        remove(&pkg, "drop").expect("should remove");

        let left = list_installed(&pkg, "x/y");
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, "keep");
    }

    #[test]
    fn removal_cannot_escape_the_adapters_folder() {
        // The id comes from the UI; a crafted value must not reach other paths.
        let pkg = temp_dir("escape");
        install_fake(&pkg, "real", "a/real", 100);
        let sibling = pkg.join("important");
        std::fs::create_dir_all(&sibling).unwrap();

        assert!(remove(&pkg, "../important").is_err());
        assert!(sibling.exists(), "a path outside the adapters folder must survive");
    }

    #[test]
    fn removing_something_not_installed_is_an_error_not_a_panic() {
        let pkg = temp_dir("missing");
        std::fs::create_dir_all(adapters_root(&pkg)).unwrap();

        let err = remove(&pkg, "never-installed").unwrap_err().to_string();
        assert!(err.contains("not installed"), "got: {err}");
    }
}

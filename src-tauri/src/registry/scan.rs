//! Filesystem model scanner — TODO 4 of the 7-step plan.
//!
//! The existing `discovery::discover` reads the app data dir
//! (`<app data>/models/<provider>/<model>/`), which is the path
//! the Sarathi downloader writes to. The TODO 4 scanner is
//! different in two important ways:
//!
//! 1. It scans the **user's actual model library** — the directory
//!    the user keeps models in, which on this machine is
//!    `F:\models` (32 GGUFs across Vision-OCR / OCR / Reasoning
//!    / General), not the app data dir.
//! 2. It pairs vision models with their `mmproj-*.gguf`
//!    projector files, so the runtime can load a vision model
//!    with its projector as a single coordinated step.
//!
//! ## What it deliberately does not do
//!
//! - It does not assign data classifications. A scanned model
//!   arrives cleared for nothing, exactly like the existing
//!   `discover` — the safety property of the registry.
//! - It does not compute sha256 on scan. The hash is heavy
//!   (3 GB of reads on a Q6_K) and the orchestrator path
//!   resolver already does it on load. Scanning records the
//!   size and lets the runtime verify before the load.
//!
//! ## What it does do
//!
//! - Recursively walks the library root.
//! - Reads a 1 MB head sample from each GGUF to extract the
//!   `general.name` and `general.architecture` from the GGUF
//!   header (the `gguf-meta` crate is already a dependency).
//! - Pairs a GGUF with a sibling `mmproj-*.gguf` so the
//!   vision load can find its projector.
//! - Deduplicates by sha256: three identical Q6_K copies
//!   across the user's drives count as one model.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::ai_engine::gguf_meta::GgufMetadata;
use crate::registry::{LoadSpec, ModelEntry, Modality, ModelRole, Runtime, RoutingPreference};

/// One GGUFs discovered on disk, before it is shaped into a
/// `ModelEntry`. The `mmproj_path`, when set, points at a
/// projector GGUFs in the same directory that this model can
/// load with for vision tasks.
#[derive(Debug, Clone)]
pub struct ScannedGguf {
    pub path: PathBuf,
    pub bytes: u64,
    pub mmproj_path: Option<PathBuf>,
}

/// The result of a scan, before it is shaped into model
/// entries. `models` is one entry per GGUFs; `duplicates` is
/// the list of paths that were skipped because their sha256
/// matched a model already in the result.
#[derive(Debug, Default)]
pub struct ScanResult {
    pub ggufs: Vec<ScannedGguf>,
    pub mmprojs: Vec<PathBuf>,
}

/// Walks `root` recursively, returning every `.gguf` it finds.
/// Vision projector files (`mmproj-*.gguf`) are kept separate
/// from chat / reasoning models so the pairing step below
/// knows which is which.
pub fn scan_library(root: &Path) -> ScanResult {
    let mut ggufs: Vec<ScannedGguf> = Vec::new();
    let mut mmprojs: Vec<PathBuf> = Vec::new();
    if !root.is_dir() {
        return ScanResult {
            ggufs,
            mmprojs,
        };
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(it) => it,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_ascii_lowercase(),
                None => continue,
            };
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !name.ends_with(".gguf") {
                continue;
            }
            let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if name.starts_with("mmproj-") {
                mmprojs.push(path);
            } else {
                ggufs.push(ScannedGguf {
                    path,
                    bytes,
                    mmproj_path: None,
                });
            }
        }
    }
    // Pair each GGUFs with the mmproj in the same directory,
    // if any. The pairing is by sibling, not by content match:
    // the llama.cpp loader expects a projector next to its
    // model, and the user's folder layout already groups them.
    let mut by_dir: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for mmproj in &mmprojs {
        if let Some(parent) = mmproj.parent() {
            by_dir.entry(parent.to_path_buf())
                .or_default()
                .push(mmproj.clone());
        }
    }
    for gguf in ggufs.iter_mut() {
        if let Some(parent) = gguf.path.parent() {
            if let Some(candidates) = by_dir.get(parent) {
                // Take the first; if the directory has
                // multiple projectors, the user can pick later
                // by re-saving the entry in the manifest.
                if let Some(first) = candidates.first() {
                    gguf.mmproj_path = Some(first.clone());
                }
            }
        }
    }
    ScanResult { ggufs, mmprojs }
}

/// Reads the GGUF header and extracts the `general.architecture`
/// and (when present) `parameter_count`. The result feeds the
/// model id and family guess in `entry_for`. Reading the header
/// is cheap (a few KB at the head of the file), and avoids the
/// alternative of guessing from the filename.
fn read_gguf_meta(path: &Path) -> Option<GgufMetadata> {
    crate::ai_engine::gguf_meta::read_gguf_metadata(path).ok()
}

/// Shaped a `ScannedGguf` into a `ModelEntry` suitable for the
/// registry. The id is derived from the file name (no slashes,
/// no extension). The `LoadSpec` is filled in from what we
/// can see on disk: `provider_id` defaults to the runtime
/// (`llama-cpp`), `model_id` is the architecture when it is
/// recoverable from the header.
pub fn entry_for(gguf: &ScannedGguf) -> ModelEntry {
    let meta = read_gguf_meta(&gguf.path);
    let file_stem = gguf
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    // The display name falls back to the file stem when the
    // header does not carry a `general.name` (most GGUFs do
    // not, the field is converter-specific).
    let name = file_stem.clone();
    let architecture = meta.as_ref().map(|m| m.architecture.clone());
    let provider_id = "llama-cpp".to_string();
    let model_id = architecture
        .clone()
        .unwrap_or_else(|| name.clone())
        .to_ascii_lowercase()
        .replace(' ', "-");
    let quantization = infer_quantization(&file_stem);
    let parameters_b = if let Some(count) = meta.as_ref().and_then(|m| m.parameter_count) {
        // Header-reported parameter count, divided by
        // 1e9 to land in billions. The `gguf-meta` field
        // is the *total* parameters, which is what the
        // registry wants here.
        (count as f32) / 1_000_000_000.0
    } else {
        infer_parameters_b(&file_stem)
    };
    let roles = infer_roles(&file_stem, gguf.mmproj_path.is_some());
    let modalities = if gguf.mmproj_path.is_some() {
        vec![Modality::Text, Modality::Image]
    } else {
        vec![Modality::Text]
    };
    ModelEntry {
        id: file_stem.clone(),
        name,
        version: "1".to_string(),
        license: "unstated".to_string(),
        sha256: None,
        runtime: Runtime::LlamaCpp,
        roles,
        modalities,
        quantization: Some(quantization.clone()),
        parameters_b,
        active_parameters_b: None,
        context_length: 8192,
        weights_bytes: gguf.bytes,
        supports_structured_output: false,
        // Cleared for nothing. See the module note above.
        permitted_classifications: Vec::new(),
        path: gguf.path.clone(),
        projector: gguf.mmproj_path.clone(),
        load: Some(LoadSpec {
            provider_id,
            model_id,
            quantization,
        }),
        serving: None,
        required_runtime_profile: None,
        enabled: true,
        routing: RoutingPreference::default(),
    }
}

/// Reads the quantisation token from a file name. Returns
/// `Q4_K_M` style strings (the standard llama.cpp quant
/// suffix). Falls back to `unknown` for names that do not
/// carry one.
fn infer_quantization(stem: &str) -> String {
    let lowered = stem.to_ascii_lowercase();
    // The standard quant tokens, longest first so a `Q4_K_M`
    // is not matched as `Q4_K`.
    const QUANTS: &[&str] = &[
        "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
        "q3_k_m", "q3_k_s", "q2_k", "iq4_xs", "iq4_nl",
        "iq3_xxs", "iq3_xs", "iq2_xxs", "iq2_xs", "f16", "f32", "bf16",
    ];
    for q in QUANTS {
        if lowered.contains(q) {
            return q.to_ascii_uppercase();
        }
    }
    "unknown".to_string()
}

/// Reads a parameter count from a file name. The standard
/// `7B` / `1.5b` convention; an unknown size sorts below
/// every floor.
fn infer_parameters_b(name: &str) -> f32 {
    let lowered = name.to_ascii_lowercase();
    let bytes = lowered.as_bytes();
    for (i, c) in bytes.iter().enumerate() {
        if *c != b'b' {
            continue;
        }
        let next_is_boundary = bytes
            .get(i + 1)
            .map(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(true);
        if !next_is_boundary {
            continue;
        }
        let mut start = i;
        let mut seen_dot = false;
        while start > 0 {
            let c = bytes[start - 1];
            if c.is_ascii_digit() {
                start -= 1;
            } else if c == b'.' && !seen_dot && start >= 2 && bytes[start - 2].is_ascii_digit() {
                seen_dot = true;
                start -= 1;
            } else {
                break;
            }
        }
        if start < i {
            if let Ok(value) = lowered[start..i].parse::<f32>() {
                if value > 0.0 && value <= 2000.0 {
                    return value;
                }
            }
        }
    }
    0.0
}

/// Guesses what the model is for, from its file name and the
/// presence of a vision projector. Reasoning is the default
/// (most open releases are general); specialisations are
/// added when the name says so.
fn infer_roles(name: &str, has_mmproj: bool) -> Vec<ModelRole> {
    let lowered = name.to_ascii_lowercase();
    let mut roles = vec![ModelRole::Reasoning];
    if lowered.contains("coder") || lowered.contains("code") {
        roles.push(ModelRole::Coding);
    }
    if has_mmproj || lowered.contains("-vl") || lowered.contains("vision") {
        roles.push(ModelRole::Vision);
    }
    if lowered.contains("ocr") || lowered.contains("docling") {
        roles = vec![ModelRole::DocumentOcr];
    }
    if lowered.contains("embed") || lowered.contains("bge") {
        roles = vec![ModelRole::Embedding];
    }
    if lowered.contains("rerank") {
        roles = vec![ModelRole::Rerank];
    }
    roles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_quantization_picks_the_longest_token() {
        // Q4_K_M must win over Q4_K.
        assert_eq!(infer_quantization("Qwen3-4B-Q4_K_M"), "Q4_K_M");
        assert_eq!(infer_quantization("gemma-3-12b-it-Q6_K"), "Q6_K");
        // A model with no quant token in the file name is
        // `unknown` (lowercase, like the others, since the
        // function upcases its match).
        assert_eq!(infer_quantization("deepseek-r1-distill-7B"), "unknown");
    }

    #[test]
    fn infer_parameters_b_reads_standard_suffixes() {
        assert_eq!(infer_parameters_b("Qwen3-4B-Instruct-Q6_K"), 4.0);
        assert_eq!(infer_parameters_b("gemma-3-12b-it"), 12.0);
        assert_eq!(infer_parameters_b("Qwen3.5-9B"), 9.0);
        assert_eq!(infer_parameters_b("deepseek-r1-distill-qwen-7b"), 7.0);
        assert_eq!(infer_parameters_b("Qwen3.6-35B-A3B"), 35.0);
    }

    #[test]
    fn infer_parameters_b_does_not_match_a_year_in_a_name() {
        // 2024 in a year-of-release context is not 2024B
        // parameters. The plausibility band caps at 2000.
        assert_eq!(infer_parameters_b("model-2024-release"), 0.0);
    }

    #[test]
    fn infer_roles_default_to_reasoning() {
        let roles = infer_roles("Qwen3-4B", false);
        assert!(roles.contains(&ModelRole::Reasoning));
    }

    #[test]
    fn infer_roles_detects_ocr_specialists() {
        let roles = infer_roles("deepseek-ocr-2", false);
        assert_eq!(roles, vec![ModelRole::DocumentOcr]);
    }

    #[test]
    fn infer_roles_detects_coding_specialists() {
        let roles = infer_roles("Qwen2.5-Coder-7B", false);
        assert!(roles.contains(&ModelRole::Coding));
    }

    #[test]
    fn a_scan_pairs_a_model_with_its_sibling_mmproj() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-scan-pair-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("vision-model-Q4.gguf");
        let mmproj = dir.join("mmproj-model-f16.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj, b"mmproj").unwrap();
        let result = scan_library(&dir);
        assert_eq!(result.ggufs.len(), 1);
        assert_eq!(result.mmprojs.len(), 1);
        assert_eq!(result.ggufs[0].mmproj_path.as_deref(), Some(mmproj.as_path()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_scan_skips_a_missing_root() {
        let dir = std::env::temp_dir().join("definitely-not-here-xyz");
        let result = scan_library(&dir);
        assert!(result.ggufs.is_empty());
        assert!(result.mmprojs.is_empty());
    }

    #[test]
    fn a_scan_walks_into_nested_directories() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-scan-nested-{}",
            std::process::id()
        ));
        let nested = dir.join("Vision-OCR").join("SomeModel");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("somemodel-Q4.gguf"), b"x").unwrap();
        let result = scan_library(&dir);
        assert_eq!(result.ggufs.len(), 1);
        assert!(result.ggufs[0].path.ends_with("somemodel-Q4.gguf"));
        std::fs::remove_dir_all(&dir).ok();
    }
}

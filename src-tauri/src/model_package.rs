//! Installed base-model package manifests.
//!
//! Arjun stores one runnable GGUF model per package. Historical manifests may
//! contain additional unknown fields; serde deliberately ignores them so an
//! existing model remains loadable while the next write normalizes the file to
//! the base-model-only schema.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseManifestInfo {
    pub model_id: String,
    pub model_name: String,
    pub quantization: String,
    pub file_path: String,
    pub size_bytes: u64,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPackageManifest {
    pub package_id: String,
    pub provider_id: String,
    pub base_model: BaseManifestInfo,
    pub created_at: String,
    pub updated_at: String,
}

pub struct ModelPackageRegistry;

impl ModelPackageRegistry {
    pub fn resolve_package_dir(
        app_data_dir: &Path,
        provider_id: &str,
        model_id: &str,
    ) -> PathBuf {
        app_data_dir
            .join("models")
            .join(provider_id.to_lowercase())
            .join(model_id.replace('/', "_"))
    }

    pub fn read_manifest(package_dir: &Path) -> Result<ModelPackageManifest> {
        let manifest_path = package_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(anyhow!(
                "manifest.json does not exist at {:?}",
                manifest_path
            ));
        }
        Ok(serde_json::from_str(&fs::read_to_string(manifest_path)?)?)
    }

    pub fn ensure_valid_manifest(
        package_dir: &Path,
        provider_id: &str,
        model_id: &str,
    ) -> Result<ModelPackageManifest> {
        let existing = Self::read_manifest(package_dir).ok();
        let base_dir = package_dir.join("base");
        let mut gguf_files = Vec::new();

        if let Ok(entries) = fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
                    if let Ok(metadata) = fs::metadata(&path) {
                        gguf_files.push((
                            path.file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string(),
                            metadata.len(),
                        ));
                    }
                }
            }
        }

        gguf_files.sort_by(|left, right| left.0.cmp(&right.0));
        let primary = gguf_files
            .iter()
            .find(|(name, _)| name.contains("-00001-of-"))
            .or_else(|| gguf_files.first());
        let file_path = primary
            .map(|(name, _)| format!("base/{name}"))
            .unwrap_or_else(|| "base/".to_string());
        let total_size = gguf_files.iter().map(|(_, size)| size).sum::<u64>();

        if let Some(mut manifest) = existing {
            // A manifest written before this build could read the quantisation
            // out of a file name says "GGUF", which names the container and not
            // the weights. Left alone it stays wrong for the life of the
            // install, and an orchestrator chosen from it can never be matched
            // against the registry — so it is re-derived here, in place.
            //
            // Only the placeholder is replaced. A real label already on the
            // manifest is what an installer recorded at download time and is
            // better evidence than a file name.
            let healed = if is_placeholder(&manifest.base_model.quantization) {
                let derived = quantization_from_path(&file_path);
                if derived != manifest.base_model.quantization {
                    manifest.base_model.quantization = derived;
                    true
                } else {
                    false
                }
            } else {
                false
            };

            let current_file = package_dir.join(&manifest.base_model.file_path);
            if current_file.is_file() && manifest.base_model.size_bytes > 0 {
                if healed {
                    manifest.updated_at = chrono::Utc::now().to_rfc3339();
                    Self::write_manifest(package_dir, &manifest)?;
                }
                return Ok(manifest);
            }
            manifest.base_model.file_path = file_path;
            if total_size > 0 {
                manifest.base_model.size_bytes = total_size;
            }
            manifest.updated_at = chrono::Utc::now().to_rfc3339();
            Self::write_manifest(package_dir, &manifest)?;
            return Ok(manifest);
        }

        let quantization = quantization_from_path(&file_path);
        let timestamp = chrono::Utc::now().to_rfc3339();
        let manifest = ModelPackageManifest {
            package_id: format!("{model_id}::{quantization}::llama.cpp"),
            provider_id: provider_id.to_string(),
            base_model: BaseManifestInfo {
                model_id: model_id.to_string(),
                model_name: model_id.rsplit('/').next().unwrap_or(model_id).to_string(),
                quantization,
                file_path,
                size_bytes: total_size,
                checksum: None,
            },
            created_at: timestamp.clone(),
            updated_at: timestamp,
        };
        Self::write_manifest(package_dir, &manifest)?;
        Ok(manifest)
    }

    pub fn write_manifest(package_dir: &Path, manifest: &ModelPackageManifest) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        fs::write(
            package_dir.join("manifest.json"),
            serde_json::to_string_pretty(manifest)?,
        )?;
        Ok(())
    }
}

/// Whether a recorded quantisation names the container rather than the weights.
///
/// "GGUF" is the value this module writes when a file name declares nothing it
/// can parse. It identifies no variant, so it is safe to replace with a real
/// label the moment one can be read.
fn is_placeholder(quantization: &str) -> bool {
    let trimmed = quantization.trim();
    trimmed.is_empty() || trimmed.eq_ignore_ascii_case("gguf")
}

/// The quantisation a GGUF file name declares.
///
/// This is not cosmetic. The label recorded here is what
/// `set_orchestrator_model` persists, and the router matches an administrator's
/// choice against the registry on exactly these three coordinates. An earlier
/// version recognised three labels — `Q4_K_M`, `Q4_0`, `Q8_0` — and called
/// everything else "GGUF", so a machine holding `Q4_K_S` and `UD-Q4_K_XL` files
/// recorded every one of them as "GGUF", the choice never matched a registry
/// entry, and the chat kept answering from whichever model the capability sort
/// happened to reach first.
///
/// Recognised: the llama.cpp k-quant and legacy families, the i-quants, the
/// float formats, and Unsloth's `UD-` dynamic prefix. Anything genuinely
/// unrecognised is still "GGUF" — an honest "the name does not say" rather than
/// a guess at which quantisation a file holds.
fn quantization_from_path(path: &str) -> String {
    // The file name only. A directory called `Q4_K_M` further up the path
    // describes where a file was put, not what is inside it.
    let name = std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_uppercase())
        .unwrap_or_else(|| path.to_ascii_uppercase());

    let tokens: Vec<&str> = name
        .split(['-', '.'])
        .filter(|token| !token.is_empty())
        .collect();

    for (index, token) in tokens.iter().enumerate() {
        if !is_quantization_label(token) {
            continue;
        }
        // Unsloth publishes `…-UD-Q4_K_XL.gguf`, where the `UD-` is part of
        // the label rather than part of the model name. Two different files
        // whose only difference is that prefix are two different quantisations.
        if index > 0 && tokens[index - 1] == "UD" {
            return format!("UD-{token}");
        }
        return (*token).to_string();
    }

    "GGUF".to_string()
}

/// Whether one dash-separated token is a quantisation label.
///
/// `Q4_K_S`, `IQ3_XXS`, `Q8_0`, `F16`, `BF16` are; `GEMMA4`, `E4B`, `A3B` and
/// `35B` are not. The distinction is the leading `Q`/`IQ`/float family — a
/// parameter count ends in `B`, and a quantisation never does.
fn is_quantization_label(token: &str) -> bool {
    if matches!(token, "F16" | "F32" | "BF16") {
        return true;
    }
    let Some(digits) = token
        .strip_prefix("IQ")
        .or_else(|| token.strip_prefix("Q"))
        .filter(|rest| !rest.is_empty())
    else {
        return false;
    };
    // A quantisation is `Q<bits>` optionally followed by `_`-joined qualifiers:
    // `Q6_K`, `Q4_K_M`, `IQ2_XXS`. Nothing else may appear.
    let mut parts = digits.split('_');
    let bits = parts.next().unwrap_or("");
    if bits.is_empty() || !bits.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    parts.all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric()))
}

#[cfg(test)]
mod quantization_tests {
    use super::quantization_from_path;

    /// The file names that were on the machine where this was found. Every one
    /// of them recorded as "GGUF" before, which is why an administrator could
    /// choose an orchestrator and have the chat ignore the choice: the label
    /// stored here is matched against the registry's, and "GGUF" matches
    /// nothing.
    #[test]
    fn reads_the_quantisation_the_file_name_declares() {
        assert_eq!(quantization_from_path("base/Qwen3.5-9B-Q4_K_S.gguf"), "Q4_K_S");
        assert_eq!(
            quantization_from_path("base/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"),
            "Q4_K_M"
        );
        assert_eq!(quantization_from_path("base/gemma-3-12b-it-Q4_K_M.gguf"), "Q4_K_M");
    }

    /// Unsloth's dynamic quants carry the `UD-` in the label itself. Two files
    /// differing only by that prefix are two different quantisations, so
    /// dropping it would merge them.
    #[test]
    fn keeps_the_unsloth_dynamic_prefix() {
        assert_eq!(
            quantization_from_path("base/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
            "UD-Q4_K_XL"
        );
        assert_eq!(
            quantization_from_path("base/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
            "UD-Q4_K_M"
        );
    }

    /// A parameter count is not a quantisation. `E4B`, `A3B` and `35B` all put
    /// a letter after a digit, which a looser reader would take for one.
    #[test]
    fn does_not_mistake_a_parameter_count_for_a_quantisation() {
        assert_eq!(quantization_from_path("base/gemma-4-E4B-it.gguf"), "GGUF");
        assert_eq!(quantization_from_path("base/Qwen3.6-35B-A3B.gguf"), "GGUF");
    }

    #[test]
    fn reads_the_legacy_and_float_families() {
        assert_eq!(quantization_from_path("m-Q8_0.gguf"), "Q8_0");
        assert_eq!(quantization_from_path("m-Q4_0.gguf"), "Q4_0");
        assert_eq!(quantization_from_path("m-IQ3_XXS.gguf"), "IQ3_XXS");
        assert_eq!(quantization_from_path("m-F16.gguf"), "F16");
    }

    /// A directory named for a quantisation says where a file was put, not
    /// what is in it.
    #[test]
    fn reads_the_file_name_and_not_the_directory() {
        assert_eq!(quantization_from_path("Q4_K_M/base/model.gguf"), "GGUF");
    }

    /// Still honest when the name genuinely does not say.
    #[test]
    fn an_unlabelled_file_stays_unlabelled() {
        assert_eq!(quantization_from_path("base/model.gguf"), "GGUF");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_extra_fields_are_ignored() {
        let json = r#"{
          "packageId":"model::Q4_0::llama.cpp",
          "providerId":"huggingface",
          "baseModel":{
            "modelId":"owner/model",
            "modelName":"model",
            "quantization":"Q4_0",
            "filePath":"base/model.gguf",
            "sizeBytes":42,
            "checksum":null
          },
          "obsolete":{"status":"READY"},
          "createdAt":"2026-01-01T00:00:00Z",
          "updatedAt":"2026-01-01T00:00:00Z"
        }"#;

        let manifest: ModelPackageManifest = serde_json::from_str(json).unwrap();
        let normalized = serde_json::to_string(&manifest).unwrap();

        assert_eq!(manifest.base_model.model_id, "owner/model");
        assert!(!normalized.contains("obsolete"));
    }
}

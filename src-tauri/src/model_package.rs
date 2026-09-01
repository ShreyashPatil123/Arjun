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
            let current_file = package_dir.join(&manifest.base_model.file_path);
            if current_file.is_file() && manifest.base_model.size_bytes > 0 {
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

        let quantization = quantization_from_path(&file_path).to_string();
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

fn quantization_from_path(path: &str) -> &'static str {
    let normalized = path.to_ascii_lowercase();
    if normalized.contains("q4_k_m") {
        "Q4_K_M"
    } else if normalized.contains("q4_0") {
        "Q4_0"
    } else if normalized.contains("q8_0") {
        "Q8_0"
    } else {
        "GGUF"
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

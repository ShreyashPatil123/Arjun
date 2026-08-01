//! LoRA Adapter Format Validator
//!
//! Validates whether downloaded LoRA adapters are compatible with the
//! llama.cpp runtime (requires GGUF format) or need conversion from
//! HuggingFace PEFT SafeTensors format.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Runtime compatibility status of a downloaded LoRA adapter
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterRuntimeStatus {
    /// Adapter is in GGUF format and can be loaded directly via `--lora`
    Compatible,
    /// Adapter is in SafeTensors PEFT format and requires conversion
    /// via `convert_lora_to_gguf.py` before runtime loading
    RequiresConversion,
    /// Adapter format is unknown, corrupted, or incompatible
    Incompatible,
    /// No adapter files found in the directory
    NotPresent,
}

impl std::fmt::Display for AdapterRuntimeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterRuntimeStatus::Compatible => write!(f, "compatible"),
            AdapterRuntimeStatus::RequiresConversion => write!(f, "requires_conversion"),
            AdapterRuntimeStatus::Incompatible => write!(f, "incompatible"),
            AdapterRuntimeStatus::NotPresent => write!(f, "not_present"),
        }
    }
}

/// Result of validating an adapter directory
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterValidationResult {
    pub status: AdapterRuntimeStatus,
    pub adapter_format: Option<String>,
    pub file_size_bytes: Option<u64>,
    pub peft_type: Option<String>,
    pub reason: String,
}

/// Validates the format and runtime compatibility of a downloaded adapter.
///
/// Checks:
/// 1. If a `.gguf` adapter file exists → `Compatible`
/// 2. If `adapter_model.safetensors` + `adapter_config.json` exist:
///    a. Verify safetensors > 100KB (not a placeholder)
///    b. Parse adapter_config.json for `peft_type == "LORA"`
///    c. → `RequiresConversion`
/// 3. Otherwise → `Incompatible` or `NotPresent`
pub fn validate_adapter(adapter_dir: &Path) -> AdapterValidationResult {
    if !adapter_dir.exists() || !adapter_dir.is_dir() {
        return AdapterValidationResult {
            status: AdapterRuntimeStatus::NotPresent,
            adapter_format: None,
            file_size_bytes: None,
            peft_type: None,
            reason: format!("Adapter directory does not exist: {:?}", adapter_dir),
        };
    }

    // Check 1: Look for GGUF adapter file
    if let Some(gguf_result) = check_gguf_adapter(adapter_dir) {
        return gguf_result;
    }

    // Check 2: Look for SafeTensors PEFT adapter
    if let Some(safetensors_result) = check_safetensors_adapter(adapter_dir) {
        return safetensors_result;
    }

    // Check 3: Look for .bin adapter (older format)
    if let Some(bin_result) = check_bin_adapter(adapter_dir) {
        return bin_result;
    }

    // No recognized adapter files found
    AdapterValidationResult {
        status: AdapterRuntimeStatus::NotPresent,
        adapter_format: None,
        file_size_bytes: None,
        peft_type: None,
        reason: "No recognized adapter files found (expected .gguf, .safetensors, or .bin)".to_string(),
    }
}

/// Checks for a GGUF-format adapter file in the directory
fn check_gguf_adapter(adapter_dir: &Path) -> Option<AdapterValidationResult> {
    if let Ok(entries) = fs::read_dir(adapter_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "gguf" {
                        if let Ok(meta) = fs::metadata(&path) {
                            let size = meta.len();
                            if size > 100_000 {
                                return Some(AdapterValidationResult {
                                    status: AdapterRuntimeStatus::Compatible,
                                    adapter_format: Some("GGUF".to_string()),
                                    file_size_bytes: Some(size),
                                    peft_type: Some("LORA".to_string()),
                                    reason: format!(
                                        "GGUF adapter found: {} ({:.2} MB)",
                                        path.file_name().unwrap_or_default().to_string_lossy(),
                                        size as f64 / (1024.0 * 1024.0)
                                    ),
                                });
                            } else {
                                return Some(AdapterValidationResult {
                                    status: AdapterRuntimeStatus::Incompatible,
                                    adapter_format: Some("GGUF".to_string()),
                                    file_size_bytes: Some(size),
                                    peft_type: None,
                                    reason: format!(
                                        "GGUF adapter file is suspiciously small ({} bytes) — likely corrupted or placeholder",
                                        size
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Checks for SafeTensors PEFT adapter files
fn check_safetensors_adapter(adapter_dir: &Path) -> Option<AdapterValidationResult> {
    let safetensors_path = adapter_dir.join("adapter_model.safetensors");
    let config_path = adapter_dir.join("adapter_config.json");

    if !safetensors_path.exists() {
        return None;
    }

    let safetensors_size = fs::metadata(&safetensors_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Structural validation: file must be > 100KB to be a real weight file
    if safetensors_size < 100_000 {
        return Some(AdapterValidationResult {
            status: AdapterRuntimeStatus::Incompatible,
            adapter_format: Some("SafeTensors".to_string()),
            file_size_bytes: Some(safetensors_size),
            peft_type: None,
            reason: format!(
                "adapter_model.safetensors is only {} bytes — too small to be a valid LoRA weight file (minimum: 100 KB). Likely a placeholder or failed download.",
                safetensors_size
            ),
        });
    }

    // Parse adapter_config.json if present
    let peft_type = if config_path.exists() {
        parse_peft_type(&config_path)
    } else {
        None
    };

    // Valid SafeTensors PEFT adapter — but needs conversion for llama.cpp
    let is_lora = peft_type
        .as_ref()
        .map(|pt| pt.to_uppercase() == "LORA")
        .unwrap_or(false);

    if is_lora || peft_type.is_none() {
        Some(AdapterValidationResult {
            status: AdapterRuntimeStatus::RequiresConversion,
            adapter_format: Some("SafeTensors (PEFT)".to_string()),
            file_size_bytes: Some(safetensors_size),
            peft_type: peft_type.clone(),
            reason: format!(
                "SafeTensors PEFT adapter ({:.2} MB) — llama.cpp requires GGUF format. Conversion needed via convert_lora_to_gguf.py",
                safetensors_size as f64 / (1024.0 * 1024.0)
            ),
        })
    } else {
        Some(AdapterValidationResult {
            status: AdapterRuntimeStatus::Incompatible,
            adapter_format: Some("SafeTensors".to_string()),
            file_size_bytes: Some(safetensors_size),
            peft_type,
            reason: "Adapter is not a LoRA adapter (peft_type is not LORA)".to_string(),
        })
    }
}

/// Checks for .bin format adapter files (older PEFT format)
fn check_bin_adapter(adapter_dir: &Path) -> Option<AdapterValidationResult> {
    let bin_path = adapter_dir.join("adapter_model.bin");
    if !bin_path.exists() {
        return None;
    }

    let bin_size = fs::metadata(&bin_path).map(|m| m.len()).unwrap_or(0);

    if bin_size < 100_000 {
        return Some(AdapterValidationResult {
            status: AdapterRuntimeStatus::Incompatible,
            adapter_format: Some("PyTorch bin".to_string()),
            file_size_bytes: Some(bin_size),
            peft_type: None,
            reason: format!(
                "adapter_model.bin is only {} bytes — too small to be valid",
                bin_size
            ),
        });
    }

    Some(AdapterValidationResult {
        status: AdapterRuntimeStatus::RequiresConversion,
        adapter_format: Some("PyTorch bin (PEFT)".to_string()),
        file_size_bytes: Some(bin_size),
        peft_type: parse_peft_type(&adapter_dir.join("adapter_config.json")),
        reason: format!(
            "PyTorch .bin adapter ({:.2} MB) — llama.cpp requires GGUF format",
            bin_size as f64 / (1024.0 * 1024.0)
        ),
    })
}

/// Parses the `peft_type` field from an adapter_config.json file
fn parse_peft_type(config_path: &Path) -> Option<String> {
    if let Ok(content) = fs::read_to_string(config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            return json
                .get("peft_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
    }
    None
}

/// Validates all adapters for a model package and returns their statuses
pub fn validate_all_adapters(
    package_dir: &Path,
) -> std::collections::HashMap<String, AdapterValidationResult> {
    let mut results = std::collections::HashMap::new();
    let adapters_dir = package_dir.join("adapters");

    if !adapters_dir.exists() {
        return results;
    }

    if let Ok(entries) = fs::read_dir(&adapters_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let capability = entry
                    .file_name()
                    .to_string_lossy()
                    .to_string();
                let result = validate_adapter(&entry.path());
                log::info!(
                    "[LORA_VALIDATOR] {} adapter '{}': {} — {}",
                    package_dir.file_name().unwrap_or_default().to_string_lossy(),
                    capability,
                    result.status,
                    result.reason
                );
                results.insert(capability, result);
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_validate_nonexistent_dir() {
        let result = validate_adapter(Path::new("/nonexistent/adapter/dir"));
        assert_eq!(result.status, AdapterRuntimeStatus::NotPresent);
    }

    #[test]
    fn test_validate_empty_dir() {
        let temp_dir = std::env::temp_dir().join(format!(
            "sarathi_lora_test_empty_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        let result = validate_adapter(&temp_dir);
        assert_eq!(result.status, AdapterRuntimeStatus::NotPresent);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_small_safetensors() {
        let temp_dir = std::env::temp_dir().join(format!(
            "sarathi_lora_test_small_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        // Write a 1KB placeholder (like the bug we fixed)
        fs::write(temp_dir.join("adapter_model.safetensors"), vec![0u8; 1024]).unwrap();

        let result = validate_adapter(&temp_dir);
        assert_eq!(result.status, AdapterRuntimeStatus::Incompatible);
        assert!(result.reason.contains("too small"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_valid_safetensors() {
        let temp_dir = std::env::temp_dir().join(format!(
            "sarathi_lora_test_valid_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        // Write a >100KB safetensors file
        fs::write(
            temp_dir.join("adapter_model.safetensors"),
            vec![0u8; 200_000],
        )
        .unwrap();

        // Write adapter_config.json with LORA peft_type
        fs::write(
            temp_dir.join("adapter_config.json"),
            r#"{"peft_type": "LORA", "base_model_name_or_path": "meta-llama/Llama-3.2-1B"}"#,
        )
        .unwrap();

        let result = validate_adapter(&temp_dir);
        assert_eq!(result.status, AdapterRuntimeStatus::RequiresConversion);
        assert_eq!(result.adapter_format, Some("SafeTensors (PEFT)".to_string()));
        assert!(result.reason.contains("convert_lora_to_gguf"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_gguf_adapter() {
        let temp_dir = std::env::temp_dir().join(format!(
            "sarathi_lora_test_gguf_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        // Write a >100KB GGUF file
        fs::write(temp_dir.join("adapter.gguf"), vec![0u8; 200_000]).unwrap();

        let result = validate_adapter(&temp_dir);
        assert_eq!(result.status, AdapterRuntimeStatus::Compatible);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}

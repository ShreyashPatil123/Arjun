//! Capability Resolution — binding a decided capability to a real backend.
//!
//! Resolution is deliberately *degrading*: a capability always resolves to
//! something usable. If a shape-compatible GGUF LoRA adapter is installed for
//! the active base model it is used; otherwise the same capability is delivered
//! through its prompt profile. The inference path does not branch on which.
//!
//! This matters because public GGUF LoRA adapters for current base models are
//! scarce, and PEFT safetensors adapters — which is what HuggingFace discovery
//! actually finds — cannot be loaded by llama.cpp without conversion. Treating
//! LoRA as an *upgrade* rather than a precondition keeps the feature honest:
//! the capability is always real, its fidelity varies.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::adapter_manager::ModelPackageManifest;
use crate::capability::policy::GENERAL;
use crate::capability::profile::{CapabilityBackend, CapabilitySpec, DEFAULT_LORA_SCALE};

/// GGUF magic bytes (`GGUF`, 0x46554747 little-endian).
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// A capability bound to a concrete, usable backend.
#[derive(Debug, Clone)]
pub struct CapabilityResolution {
    /// Manifest capability key in force.
    pub capability: String,
    /// The specification (directive + sampling) for this capability.
    pub spec: CapabilitySpec,
    /// How it will actually be applied at inference time.
    pub backend: CapabilityBackend,
    /// Why this backend was chosen — surfaced in logs and the UI tooltip.
    pub backend_reason: String,
}

impl CapabilityResolution {
    /// The unmodified base model.
    pub fn base() -> Self {
        Self {
            capability: GENERAL.to_string(),
            spec: CapabilitySpec::builtin(GENERAL),
            backend: CapabilityBackend::Base,
            backend_reason: "General conversation handled natively by the base model".to_string(),
        }
    }

    /// Label for the chat UI badge, e.g. `Code · lora` or `Math · prompt-profile`.
    pub fn badge(&self) -> String {
        match self.backend {
            CapabilityBackend::Base => self.spec.display_name.clone(),
            _ => format!("{} · {}", self.spec.display_name, self.backend.label()),
        }
    }

    /// Path of the bound adapter, when the LoRA backend is active.
    pub fn adapter_path(&self) -> Option<&Path> {
        match &self.backend {
            CapabilityBackend::LoraAdapter { path, .. } => Some(path.as_path()),
            _ => None,
        }
    }
}

pub struct CapabilityResolver;

impl CapabilityResolver {
    /// Resolves a capability key against the active model's package manifest.
    ///
    /// Never fails — an unusable adapter degrades to the prompt profile, and an
    /// unknown capability degrades to the base model.
    pub fn resolve(
        capability: &str,
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> CapabilityResolution {
        if capability == GENERAL || capability.is_empty() {
            return CapabilityResolution::base();
        }

        let spec = CapabilitySpec::builtin(capability);
        if spec.is_noop() {
            log::debug!(
                "[CAPABILITY] Unknown capability '{}' — falling back to base model",
                capability
            );
            return CapabilityResolution::base();
        }

        let (backend, reason) = match Self::try_bind_adapter(capability, package_dir, manifest) {
            Ok((path, scale)) => {
                let backend = CapabilityBackend::LoraAdapter {
                    path: path.clone(),
                    scale,
                };
                let reason =
                    format!("Bound verified GGUF LoRA adapter at {:?} (scale {:.2})", path, scale);
                log::info!(
                    "[CAPABILITY] '{}' -> LoRA adapter {:?} at scale {:.2}",
                    capability, path, scale
                );
                (backend, reason)
            }
            Err(why) => {
                log::info!(
                    "[CAPABILITY] '{}' -> prompt profile ({})",
                    capability, why
                );
                (CapabilityBackend::PromptProfile, why)
            }
        };

        CapabilityResolution {
            capability: capability.to_string(),
            spec,
            backend,
            backend_reason: reason,
        }
    }

    /// Attempts to bind an installed, runtime-loadable GGUF adapter.
    ///
    /// Returns the adapter's path and the strength to bind it at. The reason for
    /// rejection comes back on the error path so the caller can report *why* a
    /// capability degraded rather than silently downgrading.
    fn try_bind_adapter(
        capability: &str,
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Result<(PathBuf, f32), String> {
        let info = manifest
            .adapters
            .get(capability)
            .ok_or_else(|| format!("no adapter registered for capability '{}'", capability))?;

        if !info.status.eq_ignore_ascii_case("installed") {
            return Err(format!("adapter status is '{}', not Installed", info.status));
        }

        // The discovery pipeline records PEFT safetensors adapters as installed
        // even though llama.cpp cannot load them. Honour that signal explicitly.
        if let Some(runtime_status) = &info.adapter_runtime_status {
            if runtime_status.eq_ignore_ascii_case("requires_conversion") {
                return Err(
                    "adapter is PEFT safetensors and needs GGUF conversion before it can be loaded"
                        .to_string(),
                );
            }
            if runtime_status.eq_ignore_ascii_case("incompatible") {
                return Err("adapter is marked incompatible with this base model".to_string());
            }
        }

        let rel = info
            .adapter_file
            .as_ref()
            .ok_or_else(|| "adapter record has no file path".to_string())?;

        let path = package_dir.join(rel);

        let is_gguf = path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            return Err(format!(
                "adapter file {:?} is not GGUF — llama.cpp cannot load it directly",
                path.file_name().unwrap_or_default()
            ));
        }

        if !path.is_file() {
            return Err(format!("adapter file missing on disk at {:?}", path));
        }

        // A truncated or partially-downloaded adapter would abort inside
        // llama.cpp; verifying the magic header keeps the failure in Rust.
        verify_gguf_magic(&path)?;

        // A hand-edited manifest is the only way a scale gets here, so a
        // nonsensical one is a typo rather than an attack. Non-finite values
        // would silently poison every weight the adapter touches, so they fall
        // back to the default loudly instead.
        let scale = match info.scale {
            Some(s) if s.is_finite() => s,
            Some(s) => {
                log::warn!(
                    "[CAPABILITY] '{}' has a non-finite adapter scale ({}); using {:.2}",
                    capability, s, DEFAULT_LORA_SCALE
                );
                DEFAULT_LORA_SCALE
            }
            None => DEFAULT_LORA_SCALE,
        };

        Ok((path, scale))
    }
}

/// Confirms a file begins with the GGUF magic bytes.
fn verify_gguf_magic(path: &Path) -> Result<(), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("cannot open adapter {:?}: {}", path, e))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| format!("adapter {:?} is truncated: {}", path, e))?;

    if &magic != GGUF_MAGIC {
        return Err(format!(
            "adapter {:?} has invalid GGUF magic {:?} — file is corrupt or not a GGUF adapter",
            path.file_name().unwrap_or_default(),
            String::from_utf8_lossy(&magic)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_manager::{AdapterManifestInfo, BaseManifestInfo};
    use std::collections::HashMap;
    use std::fs;

    fn manifest_with(capability: &str, info: AdapterManifestInfo) -> ModelPackageManifest {
        let mut adapters = HashMap::new();
        adapters.insert(capability.to_string(), info);
        ModelPackageManifest {
            package_id: "pkg".to_string(),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: "Qwen/Qwen3-4B".to_string(),
                model_name: "Qwen3 4B".to_string(),
                quantization: "Q4_K_M".to_string(),
                file_path: "base/model.gguf".to_string(),
                size_bytes: 1,
                checksum: None,
            },
            adapters,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    fn adapter_info(status: &str, file: Option<&str>, runtime: Option<&str>) -> AdapterManifestInfo {
        AdapterManifestInfo {
            capability: "coding".to_string(),
            status: status.to_string(),
            adapter_runtime_status: runtime.map(String::from),
            repo_id: Some("someone/coding-lora".to_string()),
            local_path: None,
            adapter_file: file.map(String::from),
            config_file: None,
            size_bytes: Some(1024),
            base_model_match: None,
            target_modules: vec![],
            peft_type: Some("LORA".to_string()),
            checksum: None,
            reason: None,
            ..Default::default()
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_cap_test_{}", name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Builds a package holding one bindable GGUF adapter and returns its dir.
    fn package_with_gguf_adapter(name: &str) -> PathBuf {
        let dir = temp_dir(name);
        let adapters = dir.join("adapters").join("coding");
        fs::create_dir_all(&adapters).unwrap();
        // Minimal GGUF: magic + version.
        fs::write(adapters.join("adapter.gguf"), b"GGUF\x03\x00\x00\x00").unwrap();
        dir
    }

    fn resolved_scale(resolution: &CapabilityResolution) -> f32 {
        match &resolution.backend {
            CapabilityBackend::LoraAdapter { scale, .. } => *scale,
            other => panic!("expected a LoRA backend, got {other:?}"),
        }
    }

    #[test]
    fn a_recorded_scale_reaches_the_binding() {
        let dir = package_with_gguf_adapter("scale_recorded");
        let mut info =
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible"));
        info.scale = Some(0.7);

        let r = CapabilityResolver::resolve("coding", &dir, &manifest_with("coding", info));

        assert_eq!(resolved_scale(&r), 0.7);
    }

    #[test]
    fn a_record_without_a_scale_uses_the_default() {
        let dir = package_with_gguf_adapter("scale_absent");
        let info =
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible"));

        let r = CapabilityResolver::resolve("coding", &dir, &manifest_with("coding", info));

        assert_eq!(resolved_scale(&r), DEFAULT_LORA_SCALE);
    }

    #[test]
    fn a_nonsense_scale_falls_back_rather_than_poisoning_the_weights() {
        // NaN would propagate silently through every weight the adapter touches
        // and produce garbage tokens with no error anywhere.
        let dir = package_with_gguf_adapter("scale_nan");
        let mut info =
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible"));
        info.scale = Some(f32::NAN);

        let r = CapabilityResolver::resolve("coding", &dir, &manifest_with("coding", info));

        assert_eq!(resolved_scale(&r), DEFAULT_LORA_SCALE);
    }

    #[test]
    fn general_resolves_to_the_base_model() {
        let dir = temp_dir("general");
        let m = manifest_with("coding", adapter_info("Installed", None, None));

        let r = CapabilityResolver::resolve(GENERAL, &dir, &m);

        assert_eq!(r.backend, CapabilityBackend::Base);
        assert_eq!(r.capability, GENERAL);
    }

    #[test]
    fn a_valid_gguf_adapter_binds_the_lora_backend() {
        let dir = temp_dir("valid_gguf");
        let adapters = dir.join("adapters").join("coding");
        fs::create_dir_all(&adapters).unwrap();
        // Minimal GGUF: magic + version.
        fs::write(adapters.join("adapter.gguf"), b"GGUF\x03\x00\x00\x00").unwrap();

        let m = manifest_with(
            "coding",
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible")),
        );

        let r = CapabilityResolver::resolve("coding", &dir, &m);

        assert!(r.backend.is_lora(), "expected LoRA backend, got {:?}", r.backend);
        assert!(r.adapter_path().is_some());
        assert_eq!(r.badge(), "Code · lora");
    }

    #[test]
    fn safetensors_adapter_degrades_to_prompt_profile() {
        // The realistic case: HuggingFace discovery finds a PEFT adapter.
        let dir = temp_dir("safetensors");
        let adapters = dir.join("adapters").join("coding");
        fs::create_dir_all(&adapters).unwrap();
        fs::write(adapters.join("adapter_model.safetensors"), vec![0u8; 2048]).unwrap();

        let m = manifest_with(
            "coding",
            adapter_info(
                "Installed",
                Some("adapters/coding/adapter_model.safetensors"),
                Some("requires_conversion"),
            ),
        );

        let r = CapabilityResolver::resolve("coding", &dir, &m);

        assert_eq!(r.backend, CapabilityBackend::PromptProfile);
        assert!(
            r.backend_reason.contains("GGUF conversion"),
            "reason should explain the degradation, got: {}",
            r.backend_reason
        );
        // The capability is still delivered, just at lower fidelity.
        assert!(!r.spec.is_noop());
    }

    #[test]
    fn a_corrupt_gguf_adapter_is_rejected_before_llama_cpp_sees_it() {
        let dir = temp_dir("corrupt");
        let adapters = dir.join("adapters").join("coding");
        fs::create_dir_all(&adapters).unwrap();
        fs::write(adapters.join("adapter.gguf"), b"NOPE----").unwrap();

        let m = manifest_with(
            "coding",
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible")),
        );

        let r = CapabilityResolver::resolve("coding", &dir, &m);

        assert_eq!(r.backend, CapabilityBackend::PromptProfile);
        assert!(r.backend_reason.contains("invalid GGUF magic"));
    }

    #[test]
    fn a_missing_adapter_file_degrades_cleanly() {
        let dir = temp_dir("missing");
        let m = manifest_with(
            "coding",
            adapter_info("Installed", Some("adapters/coding/adapter.gguf"), Some("compatible")),
        );

        let r = CapabilityResolver::resolve("coding", &dir, &m);

        assert_eq!(r.backend, CapabilityBackend::PromptProfile);
        assert!(r.backend_reason.contains("missing on disk"));
    }

    #[test]
    fn an_unregistered_capability_still_delivers_its_prompt_profile() {
        let dir = temp_dir("unregistered");
        let m = manifest_with("coding", adapter_info("Installed", None, None));

        let r = CapabilityResolver::resolve("mathematics", &dir, &m);

        assert_eq!(r.backend, CapabilityBackend::PromptProfile);
        assert_eq!(r.capability, "mathematics");
        assert!(r.spec.sampling.temperature.is_some());
    }

    #[test]
    fn resolution_never_panics_across_capability_keys() {
        let dir = temp_dir("all_keys");
        let m = manifest_with("coding", adapter_info("Unavailable", None, None));

        for key in CapabilitySpec::all_keys() {
            let r = CapabilityResolver::resolve(key, &dir, &m);
            assert_eq!(&r.capability, key);
        }
        // And for junk input.
        let r = CapabilityResolver::resolve("not-a-capability", &dir, &m);
        assert_eq!(r.backend, CapabilityBackend::Base);
    }
}

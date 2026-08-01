//! Model Intelligence Layer Module
//!
//! Provides source-driven metadata extraction, profile versioning,
//! dynamic capability management, intent detection, and dynamic adapter routing.

pub mod adapter_router;
pub mod extractor;
pub mod intent;
pub mod profile;

use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::adapter_manager::ModelPackageManifest;
pub use adapter_router::{AdapterRouteResult, AdapterRouter};
pub use extractor::MetadataExtractor;
pub use intent::{IntentDetector, PromptIntent};
pub use profile::{CapabilityRegistry, ModelFamily, ModelProfile, InferenceParameters, TokenConfig, CURRENT_PROFILE_VERSION};

pub struct ModelIntelligenceManager;

impl ModelIntelligenceManager {
    /// Loads existing `profile.json` or generates a new source-driven profile if missing or outdated
    pub fn get_or_create_profile(
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Result<ModelProfile> {
        let profile_path = package_dir.join("profile.json");
        if profile_path.exists() {
            if let Ok(content) = fs::read_to_string(&profile_path) {
                if let Ok(mut profile) = serde_json::from_str::<ModelProfile>(&content) {
                    if profile.migrate_if_needed() {
                        let _ = Self::write_profile(package_dir, &profile);
                    }
                    return Ok(profile);
                }
            }
        }

        // Generate new profile from package metadata sources
        let profile = MetadataExtractor::build_profile_from_package(package_dir, manifest)?;
        let _ = Self::write_profile(package_dir, &profile);
        Ok(profile)
    }

    /// Forces a metadata refresh from package sources without re-downloading GGUF or adapters
    pub fn refresh_profile(
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Result<ModelProfile> {
        log::info!(
            "[MODEL_INTELLIGENCE] Refreshing profile for model '{}' in {:?}",
            manifest.base_model.model_id, package_dir
        );
        let profile = MetadataExtractor::build_profile_from_package(package_dir, manifest)?;
        Self::write_profile(package_dir, &profile)?;
        Ok(profile)
    }

    /// Writes `profile.json` atomically to package directory
    pub fn write_profile(package_dir: &Path, profile: &ModelProfile) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        let path = package_dir.join("profile.json");
        let json = serde_json::to_string_pretty(profile)?;
        fs::write(&path, json)?;
        log::info!("[MODEL_INTELLIGENCE] Wrote ModelProfile to {:?}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_detection() {
        assert_eq!(IntentDetector::classify("Write a python function to compute fibonacci"), PromptIntent::Coding);
        assert_eq!(IntentDetector::classify("Solve for x: 3x + 5 = 20"), PromptIntent::Mathematics);
        assert_eq!(IntentDetector::classify("Think step by step and compare pros and cons"), PromptIntent::Reasoning);
        assert_eq!(IntentDetector::classify("Execute function call with json arguments"), PromptIntent::ToolCalling);
        assert_eq!(IntentDetector::classify("Summarize key findings of this literature paper"), PromptIntent::Research);
        assert_eq!(IntentDetector::classify("Hello how are you doing today?"), PromptIntent::GeneralChat);
    }

    #[test]
    fn test_profile_versioning_and_migration() {
        let mut prof = ModelProfile::new("pkg1", "meta-llama/Llama-3.2-1B", "Llama 3.2 1B");
        prof.profile_version = 0;
        assert!(prof.migrate_if_needed());
        assert_eq!(prof.profile_version, CURRENT_PROFILE_VERSION);
        assert!(!prof.migrate_if_needed());
    }

    #[test]
    fn test_capability_registry() {
        let mut reg = CapabilityRegistry::new();
        assert!(reg.is_supported("coding"));
        assert!(reg.is_supported("reasoning"));
        assert!(!reg.is_supported("vision"));

        reg.set_capability("audio", true, 0.9, "Audio processing");
        assert!(reg.is_supported("audio"));
    }
}

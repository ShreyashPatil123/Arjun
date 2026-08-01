//! Scalable Adapter Router
//!
//! Selects and routes compatible installed LoRA adapters dynamically
//! based on intent capability selection without reloading the base model.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::adapter_manager::ModelPackageManifest;
use crate::model_intelligence::intent::{IntentDetector, PromptIntent};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterRouteResult {
    pub intent: PromptIntent,
    pub target_capability: String,
    pub selected_adapter_name: Option<String>,
    pub adapter_file_path: Option<PathBuf>,
    pub is_auto_routed: bool,
    pub reasoning: String,
}

pub struct AdapterRouter;

impl AdapterRouter {
    /// Selects best adapter for user prompt given package manifest and directory
    pub fn select_adapter_for_prompt(
        package_dir: &Path,
        manifest: &ModelPackageManifest,
        prompt: &str,
        user_override_adapter: Option<&str>,
    ) -> AdapterRouteResult {
        // If user specified an explicit manual override in Settings
        if let Some(manual_cap) = user_override_adapter {
            if !manual_cap.is_empty() && manual_cap != "none" && manual_cap != "auto" {
                if let Some(adapter_info) = manifest.adapters.get(manual_cap) {
                    if adapter_info.status == "Installed" {
                        let path = adapter_info.adapter_file.as_ref().map(|f| package_dir.join(f));
                        return AdapterRouteResult {
                            intent: PromptIntent::GeneralChat,
                            target_capability: manual_cap.to_string(),
                            selected_adapter_name: Some(adapter_info.repo_id.clone().unwrap_or_else(|| manual_cap.to_string())),
                            adapter_file_path: path,
                            is_auto_routed: false,
                            reasoning: format!("Manual user override selected capability '{}'", manual_cap),
                        };
                    }
                }
            }
        }

        // Automatic intent classification
        let intent = IntentDetector::classify(prompt);
        let cap_name = intent.to_capability_name();

        if cap_name == "general" {
            return AdapterRouteResult {
                intent,
                target_capability: "general".to_string(),
                selected_adapter_name: None,
                adapter_file_path: None,
                is_auto_routed: true,
                reasoning: "Base model handles general prompt natively".to_string(),
            };
        }

        // Find matching installed adapter in manifest
        if let Some(adapter_info) = manifest.adapters.get(cap_name) {
            if adapter_info.status == "Installed" {
                let path = adapter_info.adapter_file.as_ref().map(|f| package_dir.join(f));
                return AdapterRouteResult {
                    intent,
                    target_capability: cap_name.to_string(),
                    selected_adapter_name: Some(adapter_info.repo_id.clone().unwrap_or_else(|| cap_name.to_string())),
                    adapter_file_path: path,
                    is_auto_routed: true,
                    reasoning: format!("Automatically routed prompt intent to installed '{}' LoRA adapter", cap_name),
                };
            }
        }

        AdapterRouteResult {
            intent,
            target_capability: cap_name.to_string(),
            selected_adapter_name: None,
            adapter_file_path: None,
            is_auto_routed: true,
            reasoning: format!("Capability '{}' requested by prompt intent, but no installed LoRA adapter found. Using base model natively.", cap_name),
        }
    }
}

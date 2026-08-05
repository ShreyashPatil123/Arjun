//! Capability Layer — intent-driven specialization of the active model.
//!
//! This module is the working replacement for what the build plan called the
//! "Dynamic LoRA Switching Engine". It resolves, per turn, *how* the model
//! should be specialized, and applies that specialization to the actual
//! inference call.
//!
//! The pipeline is:
//!
//! ```text
//! prompt -> classify (confidence) -> switch policy (hysteresis) -> resolve backend
//!        -> apply (system directive + sampling, and/or LoRA adapter binding)
//! ```
//!
//! Two properties are deliberate:
//!
//! - **It always applies.** The legacy path classified intent, displayed a
//!   badge, and then generated with the unmodified base model — the routing was
//!   never wired to inference. Everything here terminates in a real change to
//!   the prompt, the sampler, or the context's bound adapter.
//! - **It always degrades.** A capability with no loadable adapter is delivered
//!   through its prompt profile rather than being dropped, so behaviour is
//!   consistent whether or not adapters are installed.

pub mod classifier;
pub mod eval;
pub mod policy;
pub mod profile;
pub mod resolver;

use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::adapter_manager::ModelPackageManifest;
use crate::ai_engine::traits::{ChatMessage, GenerationParams};

pub use classifier::{ClassificationResult, IntentClassifier};
pub use policy::{CapabilityTracker, SwitchDecision, SwitchPolicy, GENERAL};
pub use profile::{CapabilityBackend, CapabilitySpec, SamplingOverrides, DEFAULT_LORA_SCALE};
pub use resolver::{CapabilityResolution, CapabilityResolver};

/// Everything decided for a single turn.
pub struct CapabilityTurn {
    pub classification: ClassificationResult,
    pub decision: SwitchDecision,
    pub resolution: CapabilityResolution,
}

impl CapabilityTurn {
    /// Serializable view for IPC and the `capability:changed` event.
    ///
    /// `effective` are the sampling parameters generation will actually run
    /// with, after capability overrides — so the diagnostics panel reports real
    /// values rather than assumed defaults.
    pub fn payload(&self, effective: &GenerationParams) -> CapabilityPayload {
        CapabilityPayload {
            capability: self.resolution.capability.clone(),
            display_name: self.resolution.spec.display_name.clone(),
            badge: self.resolution.badge(),
            backend: self.resolution.backend.label().to_string(),
            confidence: self.classification.confidence,
            switched: self.decision.is_switch(),
            reason: self.decision.reason().to_string(),
            backend_reason: self.resolution.backend_reason.clone(),
            adapter_path: self
                .resolution
                .adapter_path()
                .map(|p| p.to_string_lossy().to_string()),
            effective_temperature: effective.temperature,
            effective_top_p: effective.top_p,
            effective_repeat_penalty: effective.repeat_penalty,
        }
    }
}

/// What the UI is told about the active capability.
///
/// Emitted from the backend *after* the capability is actually applied, so the
/// badge can no longer claim a specialization that did not happen.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityPayload {
    pub capability: String,
    pub display_name: String,
    /// Pre-formatted UI label, e.g. `Code · lora`.
    pub badge: String,
    /// `base` | `prompt-profile` | `lora`.
    pub backend: String,
    pub confidence: f32,
    pub switched: bool,
    pub reason: String,
    pub backend_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<String>,
    /// Sampling values generation will actually use, post-override.
    pub effective_temperature: f32,
    pub effective_top_p: f32,
    pub effective_repeat_penalty: f32,
}

/// Thread-safe capability layer held in Tauri state alongside the runtime.
pub struct CapabilityLayer {
    tracker: Mutex<CapabilityTracker>,
}

impl CapabilityLayer {
    pub fn new(policy: SwitchPolicy) -> Self {
        Self {
            tracker: Mutex::new(CapabilityTracker::new(policy)),
        }
    }

    /// Clears conversation stickiness. Call on model unload or new chat.
    pub fn reset(&self) {
        if let Ok(mut t) = self.tracker.lock() {
            t.reset();
        }
    }

    /// The capability currently in force.
    pub fn active_capability(&self) -> String {
        self.tracker
            .lock()
            .map(|t| t.active_capability().to_string())
            .unwrap_or_else(|_| GENERAL.to_string())
    }

    /// Runs the full pipeline for one turn: classify, decide, resolve.
    ///
    /// `manual_override` pins a capability when the user selected one in
    /// Settings; `"auto"`, `"none"`, and empty fall through to classification.
    pub fn resolve_turn(
        &self,
        prompt: &str,
        package_dir: &Path,
        manifest: &ModelPackageManifest,
        manual_override: Option<&str>,
    ) -> CapabilityTurn {
        let classification = IntentClassifier::classify(prompt);

        let decision = {
            // A poisoned lock must not take inference down; recover the tracker.
            let mut tracker = self
                .tracker
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tracker.decide(&classification, manual_override)
        };

        let resolution =
            CapabilityResolver::resolve(decision.resulting_capability(), package_dir, manifest);

        log::info!(
            "[CAPABILITY] intent={:?} confidence={:.2} -> capability='{}' backend={} ({})",
            classification.intent,
            classification.confidence,
            resolution.capability,
            resolution.backend.label(),
            decision.reason()
        );

        CapabilityTurn {
            classification,
            decision,
            resolution,
        }
    }
}

impl Default for CapabilityLayer {
    fn default() -> Self {
        Self::new(SwitchPolicy::default())
    }
}

/// Layers a capability's system directive onto the message list.
///
/// Returns a new list; `messages` is never mutated. The directive is appended
/// to an existing system message when present — the memory engine has usually
/// already built one — so memory context and capability framing coexist rather
/// than overwrite each other.
pub fn apply_directive(messages: &[ChatMessage], spec: &CapabilitySpec) -> Vec<ChatMessage> {
    if spec.directive.is_empty() {
        return messages.to_vec();
    }

    let has_system = messages.iter().any(|m| m.role == "system");

    if !has_system {
        let mut out = Vec::with_capacity(messages.len() + 1);
        out.push(ChatMessage {
            role: "system".to_string(),
            content: spec.directive.clone(),
            timestamp: None,
        });
        out.extend(messages.iter().cloned());
        return out;
    }

    messages
        .iter()
        .map(|m| {
            if m.role == "system" {
                ChatMessage {
                    role: m.role.clone(),
                    content: format!("{}\n\n{}", m.content, spec.directive),
                    timestamp: m.timestamp.clone(),
                }
            } else {
                m.clone()
            }
        })
        .collect()
}

/// Layers a capability's sampling overrides onto the caller's parameters.
pub fn apply_sampling(params: &GenerationParams, spec: &CapabilitySpec) -> GenerationParams {
    spec.sampling.apply_to(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: None,
        }
    }

    #[test]
    fn directive_is_appended_to_an_existing_system_message() {
        let messages = vec![
            msg("system", "Recalled Context & Facts:\n1. User prefers Rust"),
            msg("user", "write a parser"),
        ];
        let spec = CapabilitySpec::builtin("coding");

        let out = apply_directive(&messages, &spec);

        assert_eq!(out.len(), 2, "no message should be added when one exists");
        assert!(
            out[0].content.contains("User prefers Rust"),
            "memory context must survive"
        );
        assert!(
            out[0].content.contains("code mode"),
            "capability directive must be present"
        );
        // Input untouched.
        assert!(!messages[0].content.contains("code mode"));
    }

    #[test]
    fn directive_creates_a_system_message_when_none_exists() {
        let messages = vec![msg("user", "write a parser")];
        let spec = CapabilitySpec::builtin("coding");

        let out = apply_directive(&messages, &spec);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn general_capability_leaves_messages_untouched() {
        let messages = vec![msg("user", "hi")];
        let out = apply_directive(&messages, &CapabilitySpec::builtin(GENERAL));

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "hi");
    }

    #[test]
    fn sampling_overrides_reach_the_generation_params() {
        let base = GenerationParams::default();
        let out = apply_sampling(&base, &CapabilitySpec::builtin("mathematics"));

        assert!(
            out.temperature < base.temperature,
            "math mode must be more deterministic than the default"
        );
        assert_eq!(out.max_tokens, base.max_tokens, "unrelated params inherit");
    }

    #[test]
    fn layer_reports_and_resets_active_capability() {
        let layer = CapabilityLayer::default();
        assert_eq!(layer.active_capability(), GENERAL);

        layer.reset();
        assert_eq!(layer.active_capability(), GENERAL);
    }
}

//! Capability Specifications — what it means to "be good at" a domain.
//!
//! A capability is realised through one of two interchangeable backends:
//!
//! - [`CapabilityBackend::PromptProfile`] — a system directive plus sampling
//!   overrides. Works on every model, requires zero downloads, applies instantly.
//! - [`CapabilityBackend::LoraAdapter`] — a GGUF LoRA adapter bound to the live
//!   llama.cpp context. Higher fidelity, but only available when a
//!   shape-compatible adapter has been installed for the active base model.
//!
//! The inference path treats both identically, so a capability transparently
//! upgrades from prompt-profile to LoRA the moment a valid adapter appears.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::ai_engine::traits::GenerationParams;

/// Sampling parameter overrides applied on top of the caller's base params.
///
/// Every field is optional: `None` means "inherit whatever the caller asked for".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
}

impl SamplingOverrides {
    /// Returns a new `GenerationParams` with these overrides layered on top.
    ///
    /// Never mutates `base`.
    pub fn apply_to(&self, base: &GenerationParams) -> GenerationParams {
        GenerationParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            min_p: self.min_p.unwrap_or(base.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(base.repeat_penalty),
            max_tokens: base.max_tokens,
            mirostat: base.mirostat,
        }
    }

    /// True when no override is set (the profile leaves sampling untouched).
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
            && self.repeat_penalty.is_none()
    }
}

/// How a capability is physically realised at inference time.
#[derive(Debug, Clone, PartialEq)]
pub enum CapabilityBackend {
    /// Base model, unmodified. Used for `general` and as the universal fallback.
    Base,
    /// System directive + sampling overrides. Always available.
    PromptProfile,
    /// A GGUF LoRA adapter bound to the live context at the given scale.
    LoraAdapter { path: PathBuf, scale: f32 },
}

impl CapabilityBackend {
    /// Short label for logs, telemetry, and the chat UI badge.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::PromptProfile => "prompt-profile",
            Self::LoraAdapter { .. } => "lora",
        }
    }

    pub fn is_lora(&self) -> bool {
        matches!(self, Self::LoraAdapter { .. })
    }
}

/// Default LoRA scale when a manifest does not specify one.
///
/// 1.0 applies the adapter at its trained strength; llama.cpp multiplies the
/// low-rank delta by this factor before adding it to the base weights.
pub const DEFAULT_LORA_SCALE: f32 = 1.0;

/// A capability definition: the directive, the sampling shape, and an optional
/// adapter binding discovered from the active model's package manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilitySpec {
    /// Canonical capability key — matches manifest adapter keys
    /// (`coding`, `reasoning`, `mathematics`, `tool-calling`, `research`, `general`).
    pub capability: String,
    /// Human-readable name for the UI badge.
    pub display_name: String,
    /// System-prompt directive injected when this capability is active.
    /// Empty for `general`.
    pub directive: String,
    /// Sampling shape appropriate to the domain.
    #[serde(default)]
    pub sampling: SamplingOverrides,
}

impl CapabilitySpec {
    /// Built-in specification for a capability key.
    ///
    /// These are deliberately conservative: low temperature and tight nucleus
    /// for deterministic domains (code, math, tool-calling), looser for
    /// open-ended ones (research, reasoning).
    pub fn builtin(capability: &str) -> Self {
        match capability {
            "coding" => Self {
                capability: "coding".to_string(),
                display_name: "Code".to_string(),
                directive: concat!(
                    "You are operating in code mode. Produce correct, runnable code. ",
                    "Match the language, style, and conventions already present in the user's context. ",
                    "Prefer standard library and well-established idioms over invented abstractions. ",
                    "State assumptions briefly rather than asking clarifying questions when the intent is clear. ",
                    "Do not add explanatory prose around code unless asked."
                )
                .to_string(),
                sampling: SamplingOverrides {
                    temperature: Some(0.20),
                    top_p: Some(0.90),
                    top_k: Some(40),
                    min_p: Some(0.05),
                    repeat_penalty: Some(1.05),
                },
            },
            "mathematics" => Self {
                capability: "mathematics".to_string(),
                display_name: "Math".to_string(),
                directive: concat!(
                    "You are operating in mathematics mode. Work step by step and show the ",
                    "reasoning that leads to each result. Define notation before using it. ",
                    "State the final answer explicitly and separately from the derivation. ",
                    "If a problem is underdetermined, say exactly what is missing."
                )
                .to_string(),
                sampling: SamplingOverrides {
                    temperature: Some(0.10),
                    top_p: Some(0.85),
                    top_k: Some(20),
                    min_p: Some(0.05),
                    repeat_penalty: Some(1.02),
                },
            },
            "reasoning" => Self {
                capability: "reasoning".to_string(),
                display_name: "Reasoning".to_string(),
                directive: concat!(
                    "You are operating in reasoning mode. Decompose the question before answering. ",
                    "Weigh alternatives explicitly and name the tradeoffs that decide between them. ",
                    "Distinguish what follows necessarily from what is a judgement call, ",
                    "and finish with a clear position rather than a survey."
                )
                .to_string(),
                sampling: SamplingOverrides {
                    temperature: Some(0.45),
                    top_p: Some(0.92),
                    top_k: Some(50),
                    min_p: Some(0.05),
                    repeat_penalty: Some(1.10),
                },
            },
            "tool-calling" => Self {
                capability: "tool-calling".to_string(),
                display_name: "Tools".to_string(),
                directive: concat!(
                    "You are operating in structured-output mode. Emit strictly valid, ",
                    "parseable output conforming to the requested schema. ",
                    "Do not wrap output in commentary, apologies, or markdown fences unless asked. ",
                    "Use null for genuinely unknown values rather than inventing plausible ones."
                )
                .to_string(),
                sampling: SamplingOverrides {
                    temperature: Some(0.05),
                    top_p: Some(0.80),
                    top_k: Some(10),
                    min_p: Some(0.10),
                    repeat_penalty: Some(1.00),
                },
            },
            "research" => Self {
                capability: "research".to_string(),
                display_name: "Research".to_string(),
                directive: concat!(
                    "You are operating in research mode. Summarise faithfully and preserve the ",
                    "source's own emphasis and hedging. Separate what the source claims from your ",
                    "own inference. Flag gaps and unsupported leaps rather than smoothing over them. ",
                    "Never fabricate citations, figures, or attributions."
                )
                .to_string(),
                sampling: SamplingOverrides {
                    temperature: Some(0.30),
                    top_p: Some(0.90),
                    top_k: Some(40),
                    min_p: Some(0.05),
                    repeat_penalty: Some(1.12),
                },
            },
            // "general" and anything unrecognised fall through to the base model.
            _ => Self {
                capability: "general".to_string(),
                display_name: "General".to_string(),
                directive: String::new(),
                sampling: SamplingOverrides::default(),
            },
        }
    }

    /// True when this spec would not change model behaviour at all.
    pub fn is_noop(&self) -> bool {
        self.directive.is_empty() && self.sampling.is_empty()
    }

    /// Every capability key Sarathi understands, in manifest-key form.
    pub fn all_keys() -> &'static [&'static str] {
        &["coding", "mathematics", "reasoning", "tool-calling", "research"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_layer_onto_base_without_mutating_it() {
        let base = GenerationParams::default();
        let overrides = SamplingOverrides {
            temperature: Some(0.2),
            repeat_penalty: Some(1.05),
            ..Default::default()
        };

        let merged = overrides.apply_to(&base);

        assert_eq!(merged.temperature, 0.2);
        assert_eq!(merged.repeat_penalty, 1.05);
        // Untouched fields inherit from base
        assert_eq!(merged.top_p, base.top_p);
        assert_eq!(merged.max_tokens, base.max_tokens);
        // Base is unchanged
        assert_eq!(base.temperature, 0.7);
    }

    #[test]
    fn empty_overrides_leave_params_identical() {
        let base = GenerationParams::default();
        let merged = SamplingOverrides::default().apply_to(&base);

        assert_eq!(merged.temperature, base.temperature);
        assert_eq!(merged.top_k, base.top_k);
        assert!(SamplingOverrides::default().is_empty());
    }

    #[test]
    fn coding_profile_is_more_deterministic_than_reasoning() {
        let coding = CapabilitySpec::builtin("coding");
        let reasoning = CapabilitySpec::builtin("reasoning");

        assert!(
            coding.sampling.temperature.unwrap() < reasoning.sampling.temperature.unwrap(),
            "code generation must be more deterministic than open reasoning"
        );
    }

    #[test]
    fn general_capability_is_a_noop() {
        assert!(CapabilitySpec::builtin("general").is_noop());
        assert!(CapabilitySpec::builtin("nonexistent-capability").is_noop());
    }

    #[test]
    fn every_advertised_key_has_a_real_spec() {
        for key in CapabilitySpec::all_keys() {
            let spec = CapabilitySpec::builtin(key);
            assert_eq!(&spec.capability, key);
            assert!(!spec.is_noop(), "capability '{}' must do something", key);
        }
    }
}

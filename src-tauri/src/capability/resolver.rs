//! Capability resolution through built-in prompt profiles.

use crate::capability::policy::GENERAL;
use crate::capability::profile::{CapabilityBackend, CapabilitySpec};

#[derive(Debug, Clone)]
pub struct CapabilityResolution {
    pub capability: String,
    pub spec: CapabilitySpec,
    pub backend: CapabilityBackend,
    pub backend_reason: String,
}

impl CapabilityResolution {
    pub fn base() -> Self {
        Self {
            capability: GENERAL.to_string(),
            spec: CapabilitySpec::builtin(GENERAL),
            backend: CapabilityBackend::Base,
            backend_reason: "General conversation handled natively by the base model".to_string(),
        }
    }

    pub fn badge(&self) -> String {
        match self.backend {
            CapabilityBackend::Base => self.spec.display_name.clone(),
            CapabilityBackend::PromptProfile => {
                format!("{} · {}", self.spec.display_name, self.backend.label())
            }
        }
    }
}

pub struct CapabilityResolver;

impl CapabilityResolver {
    pub fn resolve(capability: &str) -> CapabilityResolution {
        if capability == GENERAL || capability.is_empty() {
            return CapabilityResolution::base();
        }

        let spec = CapabilitySpec::builtin(capability);
        if spec.is_noop() {
            return CapabilityResolution::base();
        }

        CapabilityResolution {
            capability: capability.to_string(),
            spec,
            backend: CapabilityBackend::PromptProfile,
            backend_reason: "Applied Arjun's built-in capability profile".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_capability_uses_prompt_profile() {
        let resolved = CapabilityResolver::resolve("coding");
        assert_eq!(resolved.backend, CapabilityBackend::PromptProfile);
        assert_eq!(resolved.badge(), "Code · prompt-profile");
    }

    #[test]
    fn unknown_capability_uses_base_model() {
        let resolved = CapabilityResolver::resolve("unknown");
        assert_eq!(resolved.backend, CapabilityBackend::Base);
    }
}

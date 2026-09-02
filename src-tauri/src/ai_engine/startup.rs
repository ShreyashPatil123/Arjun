//! Startup selection for the model that runs the agent orchestrator.
//!
//! Nothing here names a model. The orchestrator is whatever an administrator
//! chose in Models → *Set as orchestrator*, and when nobody has chosen, the
//! selection falls back to what is demonstrably on the machine: the session
//! that was last running, or the sole installed model. Several unrelated
//! models with no choice recorded is deliberately not guessed between — the
//! router picks per prompt in that case, and picking one here would only mean
//! loading weights that the first prompt then evicts.

use crate::config::AiSettings;
use crate::download_manager::traits::InstalledModel;

/// The package coordinates needed by [`super::InferenceManager`] to load a
/// model. Keeping selection independent of Tauri makes the startup policy easy
/// to test without opening a window or allocating VRAM.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartupModelTarget {
    pub provider_id: String,
    pub model_id: String,
    pub quantization: String,
}

impl StartupModelTarget {
    pub fn from_installed(model: &InstalledModel) -> Self {
        Self {
            provider_id: model.provider_id.clone(),
            model_id: model.model_id.clone(),
            quantization: model.quantization.clone(),
        }
    }

    /// The coordinates a registry entry declares.
    ///
    /// The counterpart to [`Self::from_installed`], and the one to persist when
    /// a choice has to be matched against the registry later. The two disagree
    /// whenever a package manifest could not read a quantisation out of a file
    /// name and wrote the placeholder "GGUF" instead. Routing compares against
    /// the registry, so the registry's label is the one that has to be stored.
    ///
    /// An entry with no load coordinates cannot be loaded by the runtime at
    /// all, so there is nothing truthful to return for one.
    pub fn from_load(entry: &crate::registry::ModelEntry) -> Option<Self> {
        entry.load.as_ref().map(|load| Self {
            provider_id: load.provider_id.clone(),
            model_id: load.model_id.clone(),
            quantization: load.quantization.clone(),
        })
    }

    /// The administrator's choice, or `None` when nobody has made one.
    ///
    /// The point of returning an `Option` rather than a compiled-in default is
    /// that "no choice" and "this specific model" are different states, and
    /// only the first one may be resolved dynamically.
    pub fn configured(settings: &AiSettings) -> Option<Self> {
        settings.has_configured_orchestrator().then(|| Self {
            provider_id: settings.orchestrator_provider_id.trim().to_string(),
            model_id: settings.orchestrator_model_id.trim().to_string(),
            quantization: settings.orchestrator_quantization.trim().to_string(),
        })
    }

    pub fn matches_installed(&self, model: &InstalledModel) -> bool {
        model.provider_id == self.provider_id
            && model.model_id == self.model_id
            && model.quantization == self.quantization
    }

    /// The same package published under a cosmetically different id.
    ///
    /// Publishers routinely vary punctuation and casing on the same weights
    /// (`org/Model-GGUF` against `org/model_gguf`), so an exact string
    /// comparison alone would fail to recognise the administrator's own choice
    /// after a re-download. Punctuation and case are ignored; nothing else is,
    /// so two genuinely different models never collide.
    pub fn resembles_installed(&self, model: &InstalledModel) -> bool {
        equivalent(&model.provider_id, &self.provider_id)
            && equivalent(&model.model_id, &self.model_id)
            && equivalent(&model.quantization, &self.quantization)
    }
}

/// Chooses the startup model deterministically.
///
/// The administrator's configured orchestrator wins over session restore. If
/// it is not installed — or if there is no configured orchestrator at all — a
/// valid saved session is restored; one sole installed model is the final
/// fallback. Multiple unrelated models are never guessed between.
pub fn select_startup_model(
    installed: &[InstalledModel],
    preferred: Option<&StartupModelTarget>,
    last_session: Option<&StartupModelTarget>,
) -> Option<StartupModelTarget> {
    let ready: Vec<&InstalledModel> = installed
        .iter()
        .filter(|model| model.is_ready && model.size_bytes > 0)
        .collect();

    if let Some(preferred) = preferred {
        if let Some(model) = ready
            .iter()
            .find(|model| preferred.matches_installed(model))
            .or_else(|| ready.iter().find(|model| preferred.resembles_installed(model)))
        {
            return Some(StartupModelTarget::from_installed(model));
        }
    }

    if let Some(session) = last_session {
        if let Some(model) = ready.iter().find(|model| session.resembles_installed(model)) {
            return Some(StartupModelTarget::from_installed(model));
        }
    }

    match ready.as_slice() {
        [only] => Some(StartupModelTarget::from_installed(only)),
        _ => None,
    }
}

/// True only when this binary contains an accelerator backend. Runtime GPU
/// detection cannot compensate for a CPU-only llama.cpp build.
pub const fn gpu_backend_compiled() -> bool {
    cfg!(any(feature = "cuda", feature = "vulkan"))
}

/// Rejects a load that silently fell back to CPU when startup promised GPU
/// residency.
pub fn validate_gpu_residency(gpu_layers: u32) -> Result<(), String> {
    if gpu_layers == 0 {
        return Err(
            "the orchestrator loaded without any GPU-resident layers; use a CUDA or Vulkan build"
                .to_string(),
        );
    }
    Ok(())
}

fn equivalent(left: &str, right: &str) -> bool {
    normalize(left) == normalize(right)
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download_manager::traits::InstalledModel;

    fn installed(model_id: &str, model_name: &str, quantization: &str) -> InstalledModel {
        InstalledModel {
            id: format!("{model_id}_{quantization}"),
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            provider_id: "huggingface".to_string(),
            quantization: quantization.to_string(),
            format: "GGUF".to_string(),
            backend: "llama.cpp (GGUF)".to_string(),
            file_name: "model.gguf".to_string(),
            file_path: "/models/model.gguf".to_string(),
            size_bytes: 7_000_000_000,
            installed_at: String::new(),
            is_ready: true,
            checksum: None,
        }
    }

    #[test]
    fn nothing_is_selected_from_a_field_of_models_with_no_choice_recorded() {
        let first = installed("org/model-a-7b", "Model A 7B", "Q4_K_M");
        let second = installed("org/model-b-8b", "Model B 8B", "Q4_K_M");

        assert!(
            select_startup_model(&[first, second], None, None).is_none(),
            "with no administrator choice the startup loader must not pick a model itself"
        );
    }

    #[test]
    fn the_configured_orchestrator_wins_over_the_last_session() {
        let chosen = installed("org/chosen-4b", "Chosen 4B", "Q4_K_M");
        let previous = installed("org/previous-8b", "Previous 8B", "Q6_K");
        let session = StartupModelTarget::from_installed(&previous);
        let administrator_choice = StartupModelTarget::from_installed(&chosen);

        let target = select_startup_model(
            &[previous, chosen],
            Some(&administrator_choice),
            Some(&session),
        )
        .expect("the configured orchestrator should be selected");

        assert_eq!(target, administrator_choice);
    }

    #[test]
    fn the_same_package_republished_under_a_cosmetic_alias_still_resolves() {
        let republished = installed("org/Chosen-4B-GGUF", "Chosen 4B", "Q4_K_M");
        let administrator_choice = StartupModelTarget {
            provider_id: "huggingface".to_string(),
            model_id: "org/chosen_4b_gguf".to_string(),
            quantization: "q4_k_m".to_string(),
        };

        let target = select_startup_model(&[republished], Some(&administrator_choice), None)
            .expect("a cosmetic id difference should not lose the administrator choice");

        assert_eq!(target.model_id, "org/Chosen-4B-GGUF");
    }

    #[test]
    fn a_ready_last_session_is_the_fallback_when_the_choice_is_not_installed() {
        let previous = installed("org/previous-8b", "Previous 8B", "Q6_K");
        let session = StartupModelTarget::from_installed(&previous);
        let uninstalled = StartupModelTarget {
            provider_id: "huggingface".to_string(),
            model_id: "org/not-on-this-machine".to_string(),
            quantization: "Q4_K_M".to_string(),
        };

        let target = select_startup_model(
            std::slice::from_ref(&previous),
            Some(&uninstalled),
            Some(&session),
        )
        .expect("the installed previous model should be restored");

        assert_eq!(target.model_id, previous.model_id);
    }

    #[test]
    fn an_administrator_can_choose_any_exact_model_variant() {
        let q4 = installed("org/custom-orchestrator", "Custom Orchestrator", "Q4_K_M");
        let q6 = installed("org/custom-orchestrator", "Custom Orchestrator", "Q6_K");
        let administrator_choice = StartupModelTarget::from_installed(&q6);

        let target = select_startup_model(&[q4, q6], Some(&administrator_choice), None)
            .expect("the configured model variant should be selected");

        assert_eq!(target, administrator_choice);
    }

    #[test]
    fn a_sole_installed_model_is_used_when_nothing_is_configured() {
        let only = installed("org/only-model-4b", "Only Model 4B", "Q4_K_M");

        let target = select_startup_model(std::slice::from_ref(&only), None, None)
            .expect("one installed model is not a guess");

        assert_eq!(target.model_id, only.model_id);
    }

    #[test]
    fn an_unset_configuration_yields_no_target() {
        assert!(StartupModelTarget::configured(&AiSettings::default()).is_none());
    }

    #[test]
    fn a_configured_orchestrator_is_read_back_from_settings() {
        let settings = AiSettings {
            orchestrator_provider_id: "huggingface".to_string(),
            orchestrator_model_id: "org/chosen-4b".to_string(),
            orchestrator_quantization: "Q4_K_M".to_string(),
            ..AiSettings::default()
        };

        let target = StartupModelTarget::configured(&settings).expect("a complete choice");
        assert_eq!(target.model_id, "org/chosen-4b");
    }

    #[test]
    fn gpu_validation_rejects_a_cpu_loaded_orchestrator() {
        assert!(validate_gpu_residency(0).is_err());
        assert!(validate_gpu_residency(1).is_ok());
    }
}

//! Startup selection for the model that runs the agent orchestrator.

use crate::download_manager::traits::InstalledModel;

pub use crate::config::{
    DEFAULT_ORCHESTRATOR_MODEL_ID, DEFAULT_ORCHESTRATOR_PROVIDER_ID,
    DEFAULT_ORCHESTRATOR_QUANTIZATION,
};

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

    pub fn matches_installed(&self, model: &InstalledModel) -> bool {
        model.provider_id == self.provider_id
            && model.model_id == self.model_id
            && model.quantization == self.quantization
    }
}

/// Chooses the startup model deterministically.
///
/// The configured orchestrator wins over session restore. A differently
/// published Gemma 4 12B package is accepted when the product default is being
/// requested, because publishers routinely add repository and quantization
/// suffixes to the same model. If the orchestrator is not installed, a valid
/// saved session is restored; one sole installed model is the final backwards-
/// compatible fallback. Multiple unrelated models are never guessed between.
pub fn select_startup_model(
    installed: &[InstalledModel],
    preferred: &StartupModelTarget,
    last_session: Option<&StartupModelTarget>,
) -> Option<StartupModelTarget> {
    let ready: Vec<&InstalledModel> = installed
        .iter()
        .filter(|model| model.is_ready && model.size_bytes > 0)
        .collect();

    if let Some(model) = ready
        .iter()
        .find(|model| preferred.matches_installed(model))
    {
        return Some(StartupModelTarget::from_installed(model));
    }

    if equivalent(&preferred.model_id, DEFAULT_ORCHESTRATOR_MODEL_ID) {
        if let Some(model) = ready.iter().find(|model| is_gemma_4_12b(model)) {
            return Some(StartupModelTarget::from_installed(model));
        }
    }

    if let Some(session) = last_session {
        if let Some(model) = ready.iter().find(|model| {
            equivalent(&model.provider_id, &session.provider_id)
                && equivalent(&model.model_id, &session.model_id)
                && equivalent(&model.quantization, &session.quantization)
        }) {
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

fn is_gemma_4_12b(model: &InstalledModel) -> bool {
    let identity = normalize(&format!("{} {}", model.model_id, model.model_name));
    identity.contains("gemma4") && identity.contains("12b")
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

    fn default_target() -> StartupModelTarget {
        StartupModelTarget {
            provider_id: DEFAULT_ORCHESTRATOR_PROVIDER_ID.to_string(),
            model_id: DEFAULT_ORCHESTRATOR_MODEL_ID.to_string(),
            quantization: DEFAULT_ORCHESTRATOR_QUANTIZATION.to_string(),
        }
    }

    #[test]
    fn gemma_4_12b_is_the_default_orchestrator() {
        assert_eq!(
            DEFAULT_ORCHESTRATOR_MODEL_ID,
            "lmstudio-community/gemma-4-12B-it-QAT-GGUF"
        );
        assert_eq!(DEFAULT_ORCHESTRATOR_QUANTIZATION, "Q4_0");
        assert_eq!(DEFAULT_ORCHESTRATOR_PROVIDER_ID, "huggingface");
    }

    #[test]
    fn the_default_orchestrator_wins_over_the_last_session() {
        let gemma = installed(
            DEFAULT_ORCHESTRATOR_MODEL_ID,
            "Gemma 4 12B IT QAT",
            DEFAULT_ORCHESTRATOR_QUANTIZATION,
        );
        let previous = installed("Qwen/Qwen3-4B", "Qwen 3 4B", "Q6_K");
        let session = StartupModelTarget::from_installed(&previous);

        let target = select_startup_model(
            &[previous, gemma],
            &default_target(),
            Some(&session),
        )
        .expect("the default orchestrator should be selected");

        assert_eq!(target.model_id, DEFAULT_ORCHESTRATOR_MODEL_ID);
        assert_eq!(target.quantization, DEFAULT_ORCHESTRATOR_QUANTIZATION);
    }

    #[test]
    fn a_differently_published_gemma_4_12b_is_a_valid_default_alias() {
        let gemma = installed(
            "community/gemma-4-12b-it-GGUF",
            "Gemma 4 12B Instruct",
            "Q4_K_M",
        );

        let target = select_startup_model(&[gemma], &default_target(), None)
            .expect("a Gemma 4 12B package should satisfy the default");

        assert_eq!(target.model_id, "community/gemma-4-12b-it-GGUF");
    }

    #[test]
    fn a_ready_last_session_is_the_fallback_when_gemma_is_not_installed() {
        let previous = installed("Qwen/Qwen3-4B", "Qwen 3 4B", "Q6_K");
        let session = StartupModelTarget::from_installed(&previous);

        let target = select_startup_model(
            std::slice::from_ref(&previous),
            &default_target(),
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

        let target = select_startup_model(
            &[q4, q6],
            &administrator_choice,
            None,
        )
        .expect("the administrator's configured model should be selected");

        assert_eq!(target, administrator_choice);
    }

    #[test]
    fn several_non_default_models_are_not_guessed_between() {
        let first = installed("org/model-a-7b", "Model A 7B", "Q4_K_M");
        let second = installed("org/model-b-8b", "Model B 8B", "Q4_K_M");

        assert!(
            select_startup_model(&[first, second], &default_target(), None,).is_none()
        );
    }

    #[test]
    fn gpu_validation_rejects_a_cpu_loaded_orchestrator() {
        assert!(validate_gpu_residency(0).is_err());
        assert!(validate_gpu_residency(1).is_ok());
    }
}

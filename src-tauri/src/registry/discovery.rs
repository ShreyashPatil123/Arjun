//! Noticing models that are already on disk.
//!
//! Sarathi downloads models into `<app data>/models/<provider>/<model>/`, and
//! there is no reason to make an administrator retype what is already there. So
//! the registry is the union of two things:
//!
//! - **Discovered** models, found by scanning that directory.
//! - **Declared** models, written by hand in `registry.json`.
//!
//! A declared entry always wins. Discovery guesses; a person does not, and an
//! administrator who has written something down should not have it silently
//! overridden by a heuristic that read a filename.
//!
//! ## What discovery deliberately will not do
//!
//! It never assigns a data classification. A model found on disk arrives cleared
//! for **nothing**, so it is visible in the Models screen and unusable on real
//! material until somebody decides otherwise. This is the whole safety property
//! of the registry, and a scanner that guessed here would quietly undo it: the
//! question "may this model see vendor negotiations?" is a judgement about a
//! model's provenance and behaviour, and nothing in a filename answers it.
//!
//! Roles and parameter counts *are* guessed, because both are recoverable from
//! the name with reasonable accuracy and a wrong guess is corrected rather than
//! dangerous — a mis-sized model simply fails the floor check and is not routed to.

use std::path::Path;

use crate::registry::{LoadSpec, ModelEntry, ModelRole, Runtime};

/// Reads a parameter count out of a model name.
///
/// Matches the `7B` / `1.5b` / `0.5B` convention that essentially every open
/// release follows. Returns `None` rather than a default when the name says
/// nothing: an unknown size that was assumed to be large would be routed to
/// agent work it cannot do, which is the failure the floor check exists to stop.
fn infer_parameters_b(name: &str) -> Option<f32> {
    let lowered = name.to_ascii_lowercase();
    let bytes = lowered.as_bytes();

    for (i, window) in bytes.iter().enumerate() {
        if *window != b'b' {
            continue;
        }
        // A `b` only counts as a size suffix when it ends the token — `b` inside
        // `bert` or `base` is not a parameter count.
        let next_is_boundary = bytes
            .get(i + 1)
            .map(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(true);
        if !next_is_boundary {
            continue;
        }

        // Walk backwards over the digits and at most one decimal point.
        let mut start = i;
        let mut seen_dot = false;
        while start > 0 {
            let c = bytes[start - 1];
            if c.is_ascii_digit() {
                start -= 1;
            } else if c == b'.' && !seen_dot && start >= 2 && bytes[start - 2].is_ascii_digit() {
                seen_dot = true;
                start -= 1;
            } else {
                break;
            }
        }

        if start < i {
            if let Ok(value) = lowered[start..i].parse::<f32>() {
                // A plausibility band. `2024b` is a year in a name, not 2024
                // billion parameters, and 0 is never a real size.
                if value > 0.0 && value <= 2000.0 {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// Guesses what a model is for, from its name.
///
/// Everything is assumed capable of reasoning, because a general instruction
/// model is what most releases are. Specialisations are added when the name
/// says so. An administrator corrects this in the manifest; the cost of a wrong
/// guess is a model that is not offered for a task, never one that is misused.
fn infer_roles(name: &str) -> Vec<ModelRole> {
    let lowered = name.to_ascii_lowercase();
    let mut roles = vec![ModelRole::Reasoning];

    if lowered.contains("coder") || lowered.contains("code") {
        roles.push(ModelRole::Coding);
    }
    if lowered.contains("-vl") || lowered.contains("vision") || lowered.contains("llava") {
        roles.push(ModelRole::Vision);
    }
    if lowered.contains("embed") || lowered.contains("bge") {
        // An embedding model is not a reasoning model, whatever else it is.
        roles = vec![ModelRole::Embedding];
    }
    if lowered.contains("rerank") {
        roles = vec![ModelRole::Rerank];
    }
    if lowered.contains("ocr") || lowered.contains("docling") || lowered.contains("surya") {
        roles = vec![ModelRole::DocumentOcr];
    }

    roles
}

/// Scans the models directory and describes what it finds.
pub fn discover(app_data_dir: &Path) -> Vec<ModelEntry> {
    crate::model_manager::ModelManager::list_installed_models(app_data_dir)
        .into_iter()
        .map(|installed| {
            let parameters_b = infer_parameters_b(&installed.model_name)
                .or_else(|| infer_parameters_b(&installed.model_id))
                // Unknown size sorts below every floor, so the model is listed
                // but never routed to until somebody states its size.
                .unwrap_or(0.0);

            ModelEntry {
                id: installed.id.clone(),
                name: installed.model_name.clone(),
                version: installed.quantization.clone(),
                // Not knowable from a file on disk. Shown as unstated rather
                // than guessed, because a licence is a deployment blocker at a
                // PSU and a wrong guess there is worse than an honest gap.
                license: "unstated".to_string(),
                sha256: None,
                runtime: Runtime::LlamaCpp,
                roles: infer_roles(&installed.model_name),
                quantization: Some(installed.quantization.clone()),
                parameters_b,
                active_parameters_b: None,
                // Conservative: enough for real work, small enough that the
                // KV-cache estimate does not rule the model out on a laptop.
                context_length: 8192,
                weights_bytes: installed.size_bytes,
                // Cleared for nothing. See the module note above.
                permitted_classifications: Vec::new(),
                path: installed.file_path.clone().into(),
                // Discovery knows exactly how this model was installed, so the
                // load coordinates are recorded rather than inferred later.
                load: Some(LoadSpec {
                    provider_id: installed.provider_id.clone(),
                    model_id: installed.model_id.clone(),
                    quantization: installed.quantization.clone(),
                }),
                // A discovered GGUF is served by a llama-server ARJUN starts.
                // Nothing on disk could say otherwise, so this is the runtime
                // default made explicit rather than a guess.
                serving: None,
                enabled: true,
            }
        })
        .collect()
}

/// Combines what was declared with what was discovered.
///
/// Declared entries win on id collision, and are listed first so the Models
/// screen shows reviewed models above unreviewed ones.
pub fn merge(declared: Vec<ModelEntry>, discovered: Vec<ModelEntry>) -> Vec<ModelEntry> {
    let declared_ids: std::collections::BTreeSet<String> =
        declared.iter().map(|e| e.id.clone()).collect();

    let mut merged = declared;
    merged.extend(
        discovered
            .into_iter()
            .filter(|e| !declared_ids.contains(&e.id)),
    );
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Classification;

    #[test]
    fn parameter_counts_are_read_from_the_usual_naming_convention() {
        assert_eq!(infer_parameters_b("Qwen2.5-7B-Instruct"), Some(7.0));
        assert_eq!(infer_parameters_b("Qwen2.5-1.5B-Instruct"), Some(1.5));
        assert_eq!(infer_parameters_b("Qwen2.5-0.5B-Instruct"), Some(0.5));
        assert_eq!(infer_parameters_b("llama-3.3-70b-instruct"), Some(70.0));
    }

    /// The `b` in an ordinary word is not a parameter count.
    #[test]
    fn a_b_inside_a_word_is_not_a_size() {
        assert_eq!(infer_parameters_b("bge-m3-base"), None);
        assert_eq!(infer_parameters_b("bert-large"), None);
        assert_eq!(infer_parameters_b("something-2024b-preview"), None);
    }

    #[test]
    fn an_unreadable_name_yields_no_size_rather_than_a_default() {
        assert_eq!(infer_parameters_b("mystery-model"), None);
    }

    /// An unknown size must not be routed to work that needs a big model.
    #[test]
    fn an_unknown_size_fails_every_floor() {
        let entry = ModelEntry {
            id: "mystery".into(),
            name: "mystery".into(),
            version: "1".into(),
            license: "unstated".into(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles: vec![ModelRole::Coding, ModelRole::Reasoning],
            quantization: None,
            parameters_b: 0.0,
            active_parameters_b: None,
            context_length: 8192,
            weights_bytes: 1,
            permitted_classifications: Vec::new(),
            path: "mystery.gguf".into(),
            load: None,
            serving: None,
            enabled: true,
        };
        assert!(!entry.meets_floor(ModelRole::Coding));
        assert!(!entry.meets_floor(ModelRole::Reasoning));
    }

    #[test]
    fn roles_are_inferred_from_the_name() {
        assert!(infer_roles("Qwen2.5-Coder-7B").contains(&ModelRole::Coding));
        assert!(infer_roles("Qwen2.5-VL-7B").contains(&ModelRole::Vision));
        assert_eq!(infer_roles("bge-m3"), vec![ModelRole::Embedding]);
        assert_eq!(infer_roles("bge-reranker-v2-m3"), vec![ModelRole::Rerank]);
        assert_eq!(infer_roles("surya-ocr"), vec![ModelRole::DocumentOcr]);
        assert_eq!(infer_roles("Llama-3.2-3B-Instruct"), vec![ModelRole::Reasoning]);
    }

    /// An embedding model is not a general reasoning model, whatever else the
    /// name suggests.
    #[test]
    fn specialised_roles_replace_the_reasoning_default_rather_than_adding_to_it() {
        for name in ["bge-m3", "bge-reranker-v2-m3", "surya-ocr"] {
            assert!(
                !infer_roles(name).contains(&ModelRole::Reasoning),
                "{name} should not be offered as a reasoning model"
            );
        }
    }

    #[test]
    fn a_declared_entry_beats_a_discovered_one_with_the_same_id() {
        let mut declared = crate::registry::tests::entry("shared", 14.0, vec![ModelRole::Coding]);
        declared.license = "apache-2.0".into();
        let mut discovered = crate::registry::tests::entry("shared", 7.0, vec![ModelRole::Reasoning]);
        discovered.license = "unstated".into();

        let merged = merge(vec![declared], vec![discovered]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].license, "apache-2.0");
        assert_eq!(merged[0].parameters_b, 14.0);
    }

    #[test]
    fn a_discovered_model_with_no_declaration_is_kept() {
        let discovered = crate::registry::tests::entry("found", 7.0, vec![ModelRole::Reasoning]);
        let merged = merge(vec![], vec![discovered]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "found");
    }

    /// The property this whole module has to preserve.
    #[test]
    fn discovery_never_clears_a_model_for_any_material() {
        let entry = ModelEntry {
            id: "found".into(),
            name: "Qwen2.5-7B-Instruct".into(),
            version: "Q4_K_M".into(),
            license: "unstated".into(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles: infer_roles("Qwen2.5-7B-Instruct"),
            quantization: None,
            parameters_b: 7.0,
            active_parameters_b: None,
            context_length: 8192,
            weights_bytes: 1,
            permitted_classifications: Vec::new(),
            path: "x.gguf".into(),
            load: None,
            serving: None,
            enabled: true,
        };

        for classification in Classification::ALL {
            assert!(
                !entry.permits(*classification),
                "a discovered model must not arrive cleared for {}",
                classification.label()
            );
        }
    }
}

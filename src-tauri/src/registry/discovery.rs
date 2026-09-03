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

use std::path::{Path, PathBuf};

use crate::registry::capability::{infer_active_parameters_b, infer_modalities, infer_roles};
use crate::registry::{LoadSpec, ModelEntry, RoutingPreference, Runtime};

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

/// Context length recorded when the GGUF header does not state one.
///
/// The value discovery used unconditionally before the header was consulted, so
/// a file whose converter wrote no `context_length` key is registered exactly as
/// it was before.
const DEFAULT_CONTEXT_LENGTH: u32 = 8192;

/// Largest window this build will register from a header.
///
/// Not a judgement about the model. A KV cache scales linearly with the window,
/// and a frontier-length context costs more VRAM than the weights do — at which
/// point `plan_gpu_offload` correctly puts nothing on the GPU and the model
/// runs on the CPU at a fraction of the speed. An administrator who wants the
/// full window raises it on the entry, having decided to spend the VRAM.
const MAX_REGISTERED_CONTEXT_LENGTH: u32 = 32_768;

/// The window this model was trained for, read from its own header.
///
/// Falls back rather than failing: a header this build cannot parse is a reason
/// to use the old constant, not a reason to leave the model unregistered.
fn header_context_length(path: &Path) -> u32 {
    crate::ai_engine::gguf_meta::read_gguf_metadata(path)
        .ok()
        .and_then(|meta| meta.context_length)
        .filter(|value| *value > 0)
        .map(|value| value.min(MAX_REGISTERED_CONTEXT_LENGTH))
        .unwrap_or(DEFAULT_CONTEXT_LENGTH)
}

/// Whether an `mmproj-*.gguf` projector sits beside these weights.
///
/// A multimodal GGUF ships as two files, and the second one is what turns an
/// image into something the model can attend to. Discovery used to ignore the
/// question entirely — `infer_roles` here took only a name — so an installed
/// vision model was registered as text-only unless its *name* happened to say
/// otherwise, and images had no model to go to.
///
/// Checked at the weights' own directory, which is the layout the llama.cpp
/// loader expects and the one the downloader writes.
/// The `mmproj-*.gguf` sitting beside a model, if one is.
///
/// Returns the path rather than a yes-or-no, because both callers need it: the
/// role and modality inference needs to know a projector *exists*, and the
/// entry needs to record *which file* so the launcher can pass `--mmproj`.
/// Returning only the boolean is what left every discovered vision model
/// recorded with `projector: None` — inferred as vision-capable, and then
/// started blind, which llama.cpp does silently.
///
/// Where several are present the shortest name wins, deterministically. A
/// directory holding `mmproj-M-F16.gguf` beside `mmproj-M-F16-patched.gguf`
/// would otherwise pair differently depending on directory order, and a model
/// that reads pages correctly on one launch and not the next is far worse to
/// diagnose than one that is consistently wrong.
fn sibling_projector(weights: &str) -> Option<PathBuf> {
    let dir = Path::new(weights).parent()?;
    // An unreadable directory is not evidence of a projector. Text-only is the
    // answer that fails towards "not offered" rather than towards a model being
    // handed an image it cannot decode.
    let entries = std::fs::read_dir(dir).ok()?;
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| {
                    let lowered = name.to_ascii_lowercase();
                    lowered.starts_with("mmproj-") && lowered.ends_with(".gguf")
                })
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();
    found.sort_by_key(|path| {
        (
            path.file_name().map(|n| n.len()).unwrap_or(usize::MAX),
            path.clone(),
        )
    });
    found.into_iter().next()
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

            // Read once and used for both the role and the modality, so the two
            // cannot disagree about whether this model can see.
            let projector = sibling_projector(&installed.file_path);
            let has_projector = projector.is_some();

            // The window the file declares, not a constant.
            //
            // Every discovered model used to be recorded at 8192 tokens
            // whatever it was. That number is what the context meter shows and
            // what the agent loop compacts against, so a model trained for 128k
            // was compacted at 8k — discarding history it had room for on every
            // long task — while a model trained for 4k was told it had twice
            // the room it has.
            let context_length = header_context_length(Path::new(&installed.file_path));

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
                roles: infer_roles(&installed.model_name, has_projector),
                modalities: infer_modalities(&installed.model_name, has_projector),
                quantization: Some(installed.quantization.clone()),
                parameters_b,
                // What the model runs per token, where its name states it.
                //
                // `Qwen3.6-35B-A3B` is 35B of weights and 3B consulted per
                // token. Left as `None`, the router sized it at 35B, sorted it
                // ahead of every dense model and gave it the agent work a 3B
                // model cannot hold a tool-call format through — the exact
                // failure `ModelEntry::meets_floor` documents and could not
                // prevent while nothing populated this field.
                active_parameters_b: infer_active_parameters_b(&installed.model_name)
                    .or_else(|| infer_active_parameters_b(&installed.model_id)),
                // Conservative: enough for real work, small enough that the
                // KV-cache estimate does not rule the model out on a laptop.
                context_length,
                weights_bytes: installed.size_bytes,
                supports_structured_output: false,
                // Cleared for nothing. See the module note above.
                permitted_classifications: Vec::new(),
                path: installed.file_path.clone().into(),
                // The file the launcher passes to `--mmproj`.
                //
                // This was `None` while `has_projector` two lines up was being
                // used to advertise the model as vision-capable. A vision model
                // launched without its projector does not fail: llama.cpp loads
                // it and answers text-only, so every image sent to a discovered
                // vision model was silently unseen.
                projector,
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
                required_runtime_profile: None,
                enabled: true,
                routing: RoutingPreference::default(),
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
    use crate::registry::{Modality, ModelRole};
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
            modalities: vec![Modality::Text],
            quantization: None,
            parameters_b: 0.0,
            active_parameters_b: None,
            context_length: 8192,
            weights_bytes: 1,
            supports_structured_output: false,
            permitted_classifications: Vec::new(),
            path: "mystery.gguf".into(),
            projector: None,
            load: None,
            serving: None,
            required_runtime_profile: None,
            enabled: true,
            routing: RoutingPreference::default(),
        };
        assert!(!entry.meets_floor(ModelRole::Coding));
        assert!(!entry.meets_floor(ModelRole::Reasoning));
    }

    #[test]
    fn roles_are_inferred_from_the_name() {
        assert!(infer_roles("Qwen2.5-Coder-7B", false).contains(&ModelRole::Coding));
        assert!(infer_roles("Qwen2.5-VL-7B", false).contains(&ModelRole::Vision));
        assert_eq!(infer_roles("bge-m3", false), vec![ModelRole::Embedding]);
        assert_eq!(infer_roles("bge-reranker-v2-m3", false), vec![ModelRole::Rerank]);
        assert_eq!(infer_roles("surya-ocr", false), vec![ModelRole::DocumentOcr]);
    }

    /// A general instruction model is offered for code as well as for prose.
    ///
    /// This used to assert the opposite — that `Llama-3.2-3B-Instruct` held
    /// `[Reasoning]` and nothing else — which is the assertion that made the
    /// shipped behaviour look correct while every coding request on a machine
    /// full of general models was refused outright.
    #[test]
    fn a_general_instruction_model_is_offered_for_both_prose_and_code() {
        let roles = infer_roles("Llama-3.2-3B-Instruct", false);
        assert!(roles.contains(&ModelRole::Reasoning));
        assert!(roles.contains(&ModelRole::Coding));
        // Still not a vision model: nothing about this file says it can see.
        assert!(!roles.contains(&ModelRole::Vision));
    }

    /// An embedding model is not a general reasoning model, whatever else the
    /// name suggests.
    #[test]
    fn specialised_roles_replace_the_reasoning_default_rather_than_adding_to_it() {
        for name in ["bge-m3", "bge-reranker-v2-m3", "surya-ocr"] {
            assert!(
                !infer_roles(name, false).contains(&ModelRole::Reasoning),
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
            roles: infer_roles("Qwen2.5-7B-Instruct", false),
            modalities: vec![Modality::Text],
            quantization: None,
            parameters_b: 7.0,
            active_parameters_b: None,
            context_length: 8192,
            weights_bytes: 1,
            supports_structured_output: false,
            permitted_classifications: Vec::new(),
            path: "x.gguf".into(),
            projector: None,
            load: None,
            serving: None,
            required_runtime_profile: None,
            enabled: true,
            routing: RoutingPreference::default(),
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

#[cfg(test)]
mod projector_tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arjun-proj-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The pairing that was computed and then discarded.
    ///
    /// `has_sibling_projector` decided the model was vision-capable and the
    /// entry recorded `projector: None`, so the launcher had nothing to pass to
    /// `--mmproj`. llama.cpp answers text-only in that case without erroring,
    /// which made every image sent to a discovered vision model silently
    /// unseen.
    #[test]
    fn a_projector_beside_the_weights_is_recorded_not_merely_noticed() {
        let dir = temp_dir("paired");
        let weights = dir.join("some-vision-model-Q4_K_M.gguf");
        std::fs::write(&weights, b"weights").unwrap();
        std::fs::write(dir.join("mmproj-F16.gguf"), b"projector").unwrap();

        let found = sibling_projector(weights.to_str().unwrap());
        assert_eq!(
            found.as_deref(),
            Some(dir.join("mmproj-F16.gguf").as_path()),
            "the projector has to be named, not just counted"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_text_only_model_gets_no_projector() {
        let dir = temp_dir("textonly");
        let weights = dir.join("plain-model-Q4_K_M.gguf");
        std::fs::write(&weights, b"weights").unwrap();

        assert_eq!(sibling_projector(weights.to_str().unwrap()), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two projectors in one directory is the real case — a patched build kept
    /// beside the original — and the pairing must not depend on which one the
    /// filesystem happens to hand back first.
    #[test]
    fn the_choice_between_two_projectors_is_deterministic() {
        let dir = temp_dir("two");
        let weights = dir.join("m-Q6_K.gguf");
        std::fs::write(&weights, b"weights").unwrap();
        std::fs::write(dir.join("mmproj-M-F16-patched.gguf"), b"a").unwrap();
        std::fs::write(dir.join("mmproj-M-F16.gguf"), b"b").unwrap();

        let first = sibling_projector(weights.to_str().unwrap());
        let again = sibling_projector(weights.to_str().unwrap());
        assert_eq!(first, again);
        assert_eq!(
            first.and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string())),
            Some("mmproj-M-F16.gguf".to_string()),
            "the shortest name wins, so the pairing is stable across runs"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

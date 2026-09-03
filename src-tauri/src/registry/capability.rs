//! What a model file is good for, inferred from its name and its siblings.
//!
//! ## Why this module exists at all
//!
//! It used to exist twice. `scan.rs` (the library-folder walk) and
//! `discovery.rs` (the installed-model list) each carried their own
//! `infer_roles`, and the two had drifted: only one of them had ever heard of a
//! projector file, so the same model was a vision model when found by one path
//! and text-only when found by the other. Whichever path a deployment happened
//! to use decided what the machine could do. One implementation, used by both,
//! is the fix.
//!
//! ## The rule that changed
//!
//! Inference used to grant [`ModelRole::Coding`] only to a model whose *file
//! name* contained "coder" or "code". Every other model — including every
//! general instruction model anyone actually installs — was registered as
//! unable to write code, and the router answered a coding request with
//!
//! > No registered model is set up for coding work.
//!
//! on a machine holding six models that could all do it. That is not a routing
//! preference, it is a refusal, and it made auto-selection unmeetable out of
//! the box.
//!
//! A modern instruction-tuned model writes code. It is not as good at it as a
//! code specialist of the same size, and that is a question of *ranking* —
//! which the router already answers, by size and by the operator's preference
//! order — not a question of eligibility. So coding is granted alongside
//! reasoning, and [`ModelRole::minimum_parameters_b`] remains the floor that
//! keeps a model too small for the work out of it.
//!
//! ## Where the honesty line is
//!
//! Roles are granted on evidence, never on optimism:
//!
//! - **Text roles** are inferred from the name, because a model named
//!   "instruct" or named nothing in particular is an instruction model, and
//!   being wrong there costs a slower answer.
//! - **Vision is not.** A Gemma 3 GGUF without its `mmproj-*.gguf` projector
//!   cannot see; the image encoder weights are in the other file. So a model
//!   earns [`ModelRole::Vision`] by having a projector beside it, or by naming
//!   vision in its own file name — never by belonging to a family whose
//!   *upstream* release is multimodal. Guessing there would route a scanned
//!   page to a blind model and get back a confident description of nothing.

use crate::registry::{Modality, ModelRole};

/// Tokens that mean "this file reads images", when they appear in a model name.
///
/// A publisher that puts one of these in the file name is describing the file,
/// not the family: `Qwen2.5-VL-7B` is the vision build. Bare family names are
/// deliberately absent — see the module note on where the honesty line is.
const VISION_NAME_TOKENS: &[&str] = &[
    "-vl", "vision", "llava", "moondream", "minicpm-v", "internvl", "pixtral", "molmo", "idefics",
    "cogvlm", "glm-4v", "smolvlm", "granite-vision", "kimi-vl", "deepseek-vl", "ovis",
];

/// Tokens that mark a document-OCR specialist.
const OCR_NAME_TOKENS: &[&str] = &["ocr", "docling", "surya", "nanonets"];

/// Tokens that mark an embedding model.
const EMBEDDING_NAME_TOKENS: &[&str] = &["embed", "bge-", "gte-", "e5-", "minilm"];

/// Tokens that mark a reranker.
const RERANK_NAME_TOKENS: &[&str] = &["rerank", "cross-encoder"];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Whether the file name itself declares that this build reads images.
pub fn name_declares_vision(name: &str) -> bool {
    contains_any(&name.to_ascii_lowercase(), VISION_NAME_TOKENS)
}

/// What a model can be routed to.
///
/// `has_projector` is whether an `mmproj-*.gguf` sits beside the weights. It is
/// the difference between a multimodal model and the text half of one, so it is
/// a required argument rather than one with a default: a caller that does not
/// know has to say so by passing `false`, and gets a text-only answer.
pub fn infer_roles(name: &str, has_projector: bool) -> Vec<ModelRole> {
    let lowered = name.to_ascii_lowercase();

    // Specialists first, and they *replace* the set rather than adding to it.
    // An embedding model asked to reason produces an embedding, and a router
    // that offered it for chat would be offering a failure.
    //
    // Reranking is tested before embedding because the specific name sits
    // inside the general one: `bge-reranker-v2-m3` carries the `bge-` family
    // prefix and is not an embedding model. Whichever of the two is checked
    // first wins outright, so the narrower test goes first.
    if contains_any(&lowered, RERANK_NAME_TOKENS) {
        return vec![ModelRole::Rerank];
    }
    if contains_any(&lowered, EMBEDDING_NAME_TOKENS) {
        return vec![ModelRole::Embedding];
    }
    if contains_any(&lowered, OCR_NAME_TOKENS) {
        return vec![ModelRole::DocumentOcr];
    }

    // Everything else is an instruction model until something says otherwise.
    //
    // Coding is granted here, not gated behind "coder" in the file name. See
    // the module note: the parameter floor decides whether a candidate is good
    // enough, and the router's size ordering decides which of several is best.
    // Withholding the role turns both of those questions into a refusal.
    let mut roles = vec![ModelRole::Reasoning, ModelRole::Coding];

    if has_projector || contains_any(&lowered, VISION_NAME_TOKENS) {
        roles.push(ModelRole::Vision);
        // A model that can read a page is a candidate for reading a page. The
        // router ranks it below a real OCR specialist, and on a deployment that
        // has no specialist it is the difference between a degraded read and no
        // read at all.
        roles.push(ModelRole::DocumentOcr);
    }

    roles
}

/// Which kinds of input a model accepts.
///
/// Kept beside `infer_roles` because the two must agree: a model holding
/// [`ModelRole::Vision`] but not [`Modality::Image`] is filtered out by
/// `ModelRegistry::candidates` on the modality check and is unroutable for
/// vision work — the role is then a claim the registry itself contradicts.
pub fn infer_modalities(name: &str, has_projector: bool) -> Vec<Modality> {
    let mut modalities = vec![Modality::Text];
    if has_projector || name_declares_vision(name) {
        modalities.push(Modality::Image);
    }
    modalities
}

/// Active parameters per token, read from the `-A<n>B` naming convention.
///
/// A sparse mixture-of-experts release states both figures in its name:
/// `Qwen3.6-35B-A3B` is 35B of weights with 3B consulted per token. The router
/// sorts candidates by size and `meets_floor` judges them on it, and both want
/// the second number — a 35B-A3B holds a tool-call format about as well as a 3B
/// model does, which is the failure `ModelEntry::meets_floor` was written to
/// prevent and could not, because nothing ever populated `active_parameters_b`.
///
/// Returns `None` for a dense model, whose active count is its total and is
/// already recorded.
pub fn infer_active_parameters_b(name: &str) -> Option<f32> {
    let lowered = name.to_ascii_lowercase();
    let bytes = lowered.as_bytes();

    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'a' {
            continue;
        }
        // `a` only opens an active-parameter marker at a token boundary, or
        // `llama` and `gemma` would each be read as one.
        let opens = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
        if !opens {
            continue;
        }

        let mut cursor = index + 1;
        let digits_from = cursor;
        while cursor < bytes.len() && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b'.') {
            cursor += 1;
        }
        if cursor == digits_from || cursor >= bytes.len() || bytes[cursor] != b'b' {
            continue;
        }
        // And `b` has to end the token too, so `-a3bit` is not a size.
        let closes = bytes
            .get(cursor + 1)
            .map(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(true);
        if !closes {
            continue;
        }

        if let Ok(value) = lowered[digits_from..cursor].parse::<f32>() {
            if value > 0.0 && value <= 2000.0 {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this module was written for.
    ///
    /// Every one of these is a general instruction model with no "coder" in its
    /// name, and every one was previously registered as unable to write code —
    /// which is what produced "No registered model is set up for coding work"
    /// on a machine holding six usable models.
    #[test]
    fn a_general_instruction_model_is_a_coding_candidate() {
        for name in [
            "gemma-3-12b-it",
            "gemma-4-12b-it",
            "gemma-4-E4B-it",
            "Nemotron3-Nano-4B",
            "Qwen3.5-9B",
            "Qwen3.6-35B-A3B",
            "Mistral-Small-Instruct",
        ] {
            let roles = infer_roles(name, false);
            assert!(
                roles.contains(&ModelRole::Coding),
                "{name} was not offered for coding work"
            );
            assert!(
                roles.contains(&ModelRole::Reasoning),
                "{name} was not offered for reasoning work"
            );
        }
    }

    /// Eligibility is not quality. The floor is what keeps a small model out of
    /// coding work, and it still does — the role being present is what lets the
    /// registry say *that* rather than "nothing is set up for this".
    #[test]
    fn the_parameter_floor_still_governs_which_coding_candidate_runs() {
        assert!(infer_roles("Nemotron3-Nano-4B", false).contains(&ModelRole::Coding));
        assert!(4.0 < ModelRole::Coding.minimum_parameters_b());
        assert!(9.0 > ModelRole::Coding.minimum_parameters_b());
    }

    #[test]
    fn a_specialist_replaces_the_general_roles_rather_than_adding_to_them() {
        assert_eq!(infer_roles("bge-m3", false), vec![ModelRole::Embedding]);
        assert_eq!(
            infer_roles("deepseek-ocr-2", false),
            vec![ModelRole::DocumentOcr]
        );
    }

    /// `bge-reranker-v2-m3` is a reranker that carries an embedding family's
    /// name. Testing the family prefix first classifies it as an embedding
    /// model, and the knowledge index then reranks with something that cannot
    /// rank. The narrower token has to win.
    #[test]
    fn a_reranker_named_after_an_embedding_family_is_still_a_reranker() {
        for name in ["bge-reranker-v2-m3", "bge-reranker-large", "gte-multilingual-reranker"] {
            assert_eq!(
                infer_roles(name, false),
                vec![ModelRole::Rerank],
                "{name} was not classified as a reranker"
            );
        }
    }

    /// The honesty line: a projector is evidence, a family name is not.
    #[test]
    fn vision_needs_a_projector_or_a_name_that_says_so() {
        // Gemma 3 is a multimodal release. This *file* is its text half until a
        // projector turns up beside it.
        let text_only = infer_roles("gemma-3-12b-it", false);
        assert!(!text_only.contains(&ModelRole::Vision));
        assert!(!infer_modalities("gemma-3-12b-it", false).contains(&Modality::Image));

        let with_projector = infer_roles("gemma-3-12b-it", true);
        assert!(with_projector.contains(&ModelRole::Vision));
        assert!(infer_modalities("gemma-3-12b-it", true).contains(&Modality::Image));

        // A name that declares vision needs no projector to be believed.
        assert!(infer_roles("Qwen2.5-VL-7B-Instruct", false).contains(&ModelRole::Vision));
        assert!(infer_modalities("llava-1.6-mistral-7b", false).contains(&Modality::Image));
    }

    /// A sparse model is judged on what it actually runs per token.
    ///
    /// Nothing populated `active_parameters_b`, so `Qwen3.6-35B-A3B` was a 35B
    /// model to the router: it sorted ahead of every dense candidate and won
    /// agent work that a 3B model cannot hold a tool-call format through.
    #[test]
    fn a_mixture_of_experts_reports_the_parameters_it_actually_uses() {
        assert_eq!(infer_active_parameters_b("Qwen3.6-35B-A3B"), Some(3.0));
        assert_eq!(infer_active_parameters_b("Qwen3-235B-A22B"), Some(22.0));
        assert_eq!(infer_active_parameters_b("Qwen3-30B-A3B-Instruct"), Some(3.0));
    }

    /// The marker is specific, and a dense model must not acquire one by
    /// accident — every `a` in a model name would otherwise be a candidate.
    #[test]
    fn a_dense_model_reports_no_separate_active_count() {
        for name in [
            "gemma-3-12b-it",
            "Llama-3.2-3B-Instruct",
            "Mistral-Small-Instruct",
            "Nemotron3-Nano-4B",
            "Qwen2.5-Coder-7B",
        ] {
            assert_eq!(
                infer_active_parameters_b(name),
                None,
                "{name} was read as a mixture-of-experts release"
            );
        }
    }

    /// Roles and modalities cannot disagree, or the entry is unroutable.
    #[test]
    fn a_vision_role_always_comes_with_an_image_modality() {
        for (name, projector) in [
            ("gemma-3-12b-it", true),
            ("Qwen2.5-VL-7B", false),
            ("llava-1.6", false),
            ("Qwen3.5-9B", false),
        ] {
            let roles = infer_roles(name, projector);
            let modalities = infer_modalities(name, projector);
            assert_eq!(
                roles.contains(&ModelRole::Vision),
                modalities.contains(&Modality::Image),
                "{name} declares a vision role and an image modality inconsistently"
            );
        }
    }
}

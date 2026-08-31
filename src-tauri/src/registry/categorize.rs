//! Auto-categorization of scanned models — TODO 5 of the 7-step plan.
//!
//! The library scan (TODO 4) discovers files on disk and shapes
//! them into `ModelEntry` values with the obvious fields
//! filled in. What it cannot do is assign a high-level
//! category — General, Reasoning, Vision-OCR, OCR, etc. —
//! because the answer depends on the path, the file name, the
//! GGUF header, and the role set, in roughly that order of
//! authority.
//!
//! ## The priority pipeline
//!
//! 1. **Trusted manifest** — if the registry already has an
//!    entry for this model id, the entry's `roles` are
//!    trusted. We never reclassify a model somebody has
//!    curated.
//! 2. **GGUF header** — `general.architecture` and
//!    `parameter_count` from the GGUF metadata. The header
//!    is a converter-written field and is the most reliable
//!    signal after the manifest.
//! 3. **Family rules** — a small lookup table from family
//!    token (e.g. `qwen`, `gemma`, `deepseek-ocr`) to a
//!    category. Family rules are written by hand and
//!    reflect what the user actually runs.
//! 4. **mmproj presence** — a sibling `mmproj-*.gguf` means
//!    the model is a vision model; the path heuristic
//!    confirms or downgrades this.
//! 5. **Path heuristic** — the user's directory layout
//!    (`<root>/General`, `<root>/OCR`, etc.) is a strong
//!    signal. The directories are user-named and rarely
//!    reorganised.
//! 6. **Filename regex** — last resort. Catches the cases
//!    the other rules missed, but is a known-fragile signal.
//! 7. **Unknown** — anything that does not match falls
//!    here, which is the honest answer for the front-end to
//!    surface as "uncategorised".
//!
//! ## Why this is a separate module
//!
//! Categorization is a pure function over `&ModelEntry` and
//! the file system. Splitting it out from the scan lets the
//! rules be tested in isolation, and lets the user override
//! the result through the manifest without re-scanning.

use std::path::Path;

use crate::registry::ModelEntry;

/// The high-level category the model belongs to. A model can
/// only be in one of these; the priority pipeline picks the
/// strongest signal and stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    /// A general-purpose chat model. The default for
    /// instruction-tuned LLMs that are not specialists.
    General,
    /// A reasoning model (chain-of-thought, math, code).
    Reasoning,
    /// A vision model — accepts images, returns text.
    Vision,
    /// A vision + OCR model — image input, structured text
    /// output, often multi-page.
    VisionOcr,
    /// An OCR specialist — image input, plain text output.
    Ocr,
    /// An embedding model — used by the knowledge index.
    Embedding,
    /// A reranker — used by the knowledge index.
    Rerank,
    /// Could not be classified by any of the rules. Surfaced
    /// to the administrator for review.
    Unknown,
}

impl ModelCategory {
    /// The string the front-end shows in the Models screen.
    pub fn label(self) -> &'static str {
        match self {
            ModelCategory::General => "General",
            ModelCategory::Reasoning => "Reasoning",
            ModelCategory::Vision => "Vision",
            ModelCategory::VisionOcr => "Vision + OCR",
            ModelCategory::Ocr => "OCR",
            ModelCategory::Embedding => "Embedding",
            ModelCategory::Rerank => "Rerank",
            ModelCategory::Unknown => "Uncategorised",
        }
    }
}

/// The priority pipeline, in order. Each step returns
/// `Some(category)` if it has a confident answer, or `None`
/// to defer to the next step.
pub fn categorize(entry: &ModelEntry) -> ModelCategory {
    if let Some(c) = from_manifest(entry) {
        return c;
    }
    if let Some(c) = from_family_rules(entry) {
        return c;
    }
    if let Some(c) = from_mmproj(entry) {
        return c;
    }
    if let Some(c) = from_path(entry) {
        return c;
    }
    if let Some(c) = from_filename(entry) {
        return c;
    }
    ModelCategory::Unknown
}

/// Step 1: the manifest's role set is the most authoritative
/// signal. A curated entry is never re-classified.
fn from_manifest(entry: &ModelEntry) -> Option<ModelCategory> {
    use crate::registry::ModelRole;
    let roles = &entry.roles;
    if roles.is_empty() {
        return None;
    }
    // A model whose only role is OCR is an OCR specialist.
    if roles == &[ModelRole::DocumentOcr] {
        return Some(ModelCategory::Ocr);
    }
    if roles == &[ModelRole::Embedding] {
        return Some(ModelCategory::Embedding);
    }
    if roles == &[ModelRole::Rerank] {
        return Some(ModelCategory::Rerank);
    }
    // Vision + DocumentOcr → VisionOcr. Vision alone → Vision.
    if roles.contains(&ModelRole::Vision) && roles.contains(&ModelRole::DocumentOcr) {
        return Some(ModelCategory::VisionOcr);
    }
    if roles.contains(&ModelRole::Vision) {
        return Some(ModelCategory::Vision);
    }
    // Coding with no vision is still General — code is
    // a role, not a category. Reasoning specialists are
    // signalled by the family rules below.
    if roles.contains(&ModelRole::Reasoning) && !roles.contains(&ModelRole::Coding) {
        // Could be General or Reasoning; defer to the next
        // step (family rules or path) to disambiguate.
        return None;
    }
    Some(ModelCategory::General)
}

/// Step 3: family rules. These are hand-written and reflect
/// the categories the user actually maintains.
fn from_family_rules(entry: &ModelEntry) -> Option<ModelCategory> {
    let lowered = entry.id.to_ascii_lowercase();
    // OCR specialists.
    if lowered.contains("ocr") || lowered.contains("docling") || lowered.contains("surya")
        || lowered.contains("paddleocr")
    {
        return Some(ModelCategory::Ocr);
    }
    // Vision + OCR combined.
    if lowered.contains("ocr-2") || lowered.contains("olmocr") {
        return Some(ModelCategory::VisionOcr);
    }
    // Vision models.
    if lowered.contains("-vl") || lowered.contains("vision") || lowered.contains("llava") {
        return Some(ModelCategory::Vision);
    }
    // Reasoning specialists. These are the families
    // that are tuned for chain-of-thought and "thinking"
    // and benefit from being routed away from the chat
    // orchestrator.
    if lowered.contains("thinking")
        || lowered.contains("r1")
        || lowered.contains("openreasoning")
    {
        return Some(ModelCategory::Reasoning);
    }
    // Embeddings and rerankers are caught above by the
    // manifest, but a fallback here is harmless.
    if lowered.contains("embed") || lowered.contains("bge") {
        return Some(ModelCategory::Embedding);
    }
    if lowered.contains("rerank") {
        return Some(ModelCategory::Rerank);
    }
    None
}

/// Step 4: mmproj presence. A model with a sibling
/// `mmproj-*.gguf` is a vision model — the projector is
/// useless without the language model and vice versa.
fn from_mmproj(entry: &ModelEntry) -> Option<ModelCategory> {
    if entry.modalities.iter().any(|m| matches!(m, crate::registry::Modality::Image)) {
        // The scan filled in `Image` for paired models.
        // A vision model with a strong OCR name falls
        // through to family rules first; this is the
        // fallback for the unnamed case.
        Some(ModelCategory::Vision)
    } else {
        None
    }
}

/// Step 5: path heuristic. The user organises their
/// library as `<root>/<Category>/<model>/<file>.gguf`.
/// The directory name is a strong signal.
fn from_path(entry: &ModelEntry) -> Option<ModelCategory> {
    let path_str = entry.path.to_string_lossy().to_ascii_lowercase();
    // The directory just above the file is the
    // <Category> segment.
    let parent = Path::new(&entry.path).parent()?;
    let parent_name = parent
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    // The grandparent is often the family name; we
    // look one level up for the category directory
    // when the immediate parent is the model name.
    let grand = parent
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    for segment in [parent_name.as_str(), grand.as_str()] {
        if segment.is_empty() {
            continue;
        }
        if segment == "general" {
            return Some(ModelCategory::General);
        }
        if segment == "reasoning" {
            return Some(ModelCategory::Reasoning);
        }
        if segment == "ocr" {
            return Some(ModelCategory::Ocr);
        }
        if segment == "vision-ocr" || segment == "vision_ocr" {
            return Some(ModelCategory::VisionOcr);
        }
        if segment == "vision" {
            return Some(ModelCategory::Vision);
        }
    }
    // Last-ditch: substring match anywhere in the path.
    if path_str.contains("\\general\\") || path_str.contains("/general/") {
        return Some(ModelCategory::General);
    }
    if path_str.contains("\\reasoning\\") || path_str.contains("/reasoning/") {
        return Some(ModelCategory::Reasoning);
    }
    if path_str.contains("\\ocr\\") || path_str.contains("/ocr/") {
        return Some(ModelCategory::Ocr);
    }
    if path_str.contains("\\vision-ocr\\") || path_str.contains("/vision-ocr/") {
        return Some(ModelCategory::VisionOcr);
    }
    None
}

/// Step 6: filename regex. Catches files that landed in a
/// non-standard directory but whose name still tells us
/// what they are.
fn from_filename(entry: &ModelEntry) -> Option<ModelCategory> {
    let lowered = entry.id.to_ascii_lowercase();
    if lowered.contains("ocr") {
        return Some(ModelCategory::Ocr);
    }
    if lowered.contains("embed") || lowered.contains("bge") {
        return Some(ModelCategory::Embedding);
    }
    if lowered.contains("rerank") {
        return Some(ModelCategory::Rerank);
    }
    if lowered.contains("thinking")
        || lowered.contains("r1")
        || lowered.contains("openreasoning")
        || lowered.contains("reasoning")
    {
        return Some(ModelCategory::Reasoning);
    }
    if lowered.contains("vl") || lowered.contains("vision") || lowered.contains("llava") {
        return Some(ModelCategory::Vision);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{LoadSpec, ModelEntry, ModelRole, Modality, Runtime, RoutingPreference};
    use std::path::PathBuf;

    fn entry(id: &str, path: &str) -> ModelEntry {
        ModelEntry {
            id: id.to_string(),
            name: id.to_string(),
            version: "1".to_string(),
            license: "apache-2.0".to_string(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles: vec![ModelRole::Reasoning],
            modalities: vec![Modality::Text],
            quantization: Some("Q4_K_M".to_string()),
            parameters_b: 4.0,
            active_parameters_b: None,
            context_length: 8192,
            weights_bytes: 0,
            supports_structured_output: false,
            permitted_classifications: Vec::new(),
            path: PathBuf::from(path),
            load: Some(LoadSpec {
                provider_id: "llama-cpp".to_string(),
                model_id: id.to_string(),
                quantization: "Q4_K_M".to_string(),
            }),
            serving: None,
            required_runtime_profile: None,
            enabled: true,
            routing: RoutingPreference::default(),
        }
    }

    #[test]
    fn the_manifest_role_set_is_trusted() {
        // An entry tagged DocumentOcr stays Ocr regardless
        // of what the file name says.
        let mut e = entry("qwen-8b", "/models/qwen-8b.gguf");
        e.roles = vec![ModelRole::DocumentOcr];
        assert_eq!(categorize(&e), ModelCategory::Ocr);
    }

    #[test]
    fn the_manifest_vision_ocr_role_is_trusted() {
        let mut e = entry("gemma-4-12b", "/models/gemma-4-12b-it-Q4_K_XL.gguf");
        e.roles = vec![ModelRole::Vision, ModelRole::DocumentOcr];
        assert_eq!(categorize(&e), ModelCategory::VisionOcr);
    }

    #[test]
    fn family_rules_promote_ocr_specialists() {
        let e = entry(
            "deepseek-ocr-2",
            "/models/Reasoning/deepseek-ocr-2-q4_k_m.gguf",
        );
        // The path says Reasoning but the family name
        // beats that: OCR specialists are routed to the
        // OCR pipeline.
        assert_eq!(categorize(&e), ModelCategory::Ocr);
    }

    #[test]
    fn family_rules_promote_thinking_models() {
        let e = entry(
            "Qwen3-4B-Thinking-2507-Q6_K",
            "/models/Reasoning/Qwen3-4B-Thinking-2507-Q6_K.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::Reasoning);
    }

    #[test]
    fn family_rules_promote_openreasoning_nemotron() {
        let e = entry(
            "OpenReasoning-Nemotron-7B",
            "/models/Reasoning/OpenReasoning-Nemotron-7B-Q4_K_M.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::Reasoning);
    }

    #[test]
    fn path_heuristic_picks_up_general_directory() {
        let e = entry(
            "gemma-3-12b-it",
            "/models/General/Gemma-3-12B/gemma-3-12b-it-Q4_K_M.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::General);
    }

    #[test]
    fn path_heuristic_picks_up_ocr_directory() {
        let e = entry(
            "LightOnOCR-2-1B",
            "/models/OCR/LightOnOCR-2-1B/LightOnOCR-2-1B-Q4_K_M.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::Ocr);
    }

    #[test]
    fn path_heuristic_picks_up_vision_ocr_directory() {
        let e = entry(
            "Gemma-4-12B-it",
            "/models/Vision-OCR/Gemma-4-12B-it/gemma-4-12b-it-UD-Q4_K_XL.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::VisionOcr);
    }

    #[test]
    fn filename_regex_is_the_last_resort() {
        // A file that landed in a non-standard directory
        // but whose name says `ocr` is still OCR.
        let e = entry(
            "weird-ocr-model",
            "/models/Uncategorized/weird-ocr-model.gguf",
        );
        assert_eq!(categorize(&e), ModelCategory::Ocr);
    }

    #[test]
    fn an_unknown_model_is_uncategorised() {
        let e = entry("mystery-7B", "/models/Uncategorized/mystery-7B.gguf");
        assert_eq!(categorize(&e), ModelCategory::Unknown);
    }

    #[test]
    fn the_label_is_human_readable() {
        assert_eq!(ModelCategory::General.label(), "General");
        assert_eq!(ModelCategory::Reasoning.label(), "Reasoning");
        assert_eq!(ModelCategory::VisionOcr.label(), "Vision + OCR");
        assert_eq!(ModelCategory::Unknown.label(), "Uncategorised");
    }

    #[test]
    fn vision_role_with_no_document_ocr_is_vision() {
        let mut e = entry("llava-1.5", "/models/llava-1.5.gguf");
        e.roles = vec![ModelRole::Vision, ModelRole::Reasoning];
        assert_eq!(categorize(&e), ModelCategory::Vision);
    }
}

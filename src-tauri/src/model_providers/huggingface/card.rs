//! Model card presentation data.
//!
//! Everything a browsing card needs: publisher, licence, what the model is for,
//! how popular it is, how recently it was updated, and the quantizations you can
//! choose between.
//!
//! Categorisation is deliberately *multi-label*. A model is rarely one thing —
//! Qwen2.5-Coder-7B is both `Coding` and `SmallAndFast`, and a Mixtral is both
//! `MixtureOfExperts` and `General`. Forcing a single bucket would hide models
//! from the filter that most naturally describes them.
//!
//! Every signal here comes from HuggingFace metadata. Nothing is invented: when
//! a fact is unknown the field is `None` and the card omits it, rather than
//! showing a confident-looking guess.

use serde::{Deserialize, Serialize};

use crate::model_providers::huggingface::discovery::{GgufRepo, Quantization};

/// What a model is for. A model can carry several.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCategory {
    /// A LoRA adapter — a patch applied on top of a base model, not something
    /// that runs on its own. Listed first because it changes what the entry
    /// *is*, not merely what it is good at.
    LoraAdapter,
    Coding,
    Reasoning,
    Math,
    Agentic,
    Vision,
    MixtureOfExperts,
    LongContext,
    Multilingual,
    /// Small enough to run comfortably on modest hardware.
    SmallAndFast,
    /// No strong specialisation — a general assistant.
    General,
}

impl ModelCategory {
    pub fn label(&self) -> &'static str {
        match self {
            Self::LoraAdapter => "LoRA adapter",
            Self::Coding => "Coding",
            Self::Reasoning => "Reasoning",
            Self::Math => "Math",
            Self::Agentic => "Agentic",
            Self::Vision => "Vision",
            Self::MixtureOfExperts => "MoE",
            Self::LongContext => "Long context",
            Self::Multilingual => "Multilingual",
            Self::SmallAndFast => "Small & fast",
            Self::General => "General",
        }
    }

    /// Every category, in the order a sidebar should list them.
    pub fn all() -> &'static [ModelCategory] {
        &[
            Self::LoraAdapter,
            Self::Coding,
            Self::Reasoning,
            Self::Math,
            Self::Agentic,
            Self::Vision,
            Self::MixtureOfExperts,
            Self::LongContext,
            Self::Multilingual,
            Self::SmallAndFast,
            Self::General,
        ]
    }
}

/// Parameter count below which a model counts as small and fast.
///
/// Roughly the point where a 4-bit quantization fits a 4 GB card alongside its
/// KV cache, so it runs fully on GPU on modest laptops.
const SMALL_MODEL_PARAMS: u64 = 4_500_000_000;

/// Context length at or above which a model is notable for long context.
const LONG_CONTEXT_TOKENS: u32 = 100_000;

/// Classifies a model from its name, tags, and architecture.
///
/// Always returns at least one category — [`ModelCategory::General`] when
/// nothing more specific applies — so no model is unreachable from the sidebar.
pub fn categorize(
    name: &str,
    tags: &[String],
    total_params: u64,
    context_length: u32,
    architecture: &str,
) -> Vec<ModelCategory> {
    let hay = format!(
        "{} {} {}",
        name.to_lowercase(),
        tags.join(" ").to_lowercase(),
        architecture.to_lowercase()
    );
    let has = |needles: &[&str]| needles.iter().any(|n| hay.contains(n));

    let mut out = Vec::new();

    // Checked against the tags directly rather than the flattened haystack: a
    // model merely *named* "lora-tuned" is not an adapter, and mislabelling a
    // runnable model as one would hide it from the model list.
    if crate::model_providers::huggingface::discovery::is_lora_adapter(tags) {
        out.push(ModelCategory::LoraAdapter);
    }

    if has(&["coder", "code", "codegen", "starcoder", "codestral", "codellama", "sql"]) {
        out.push(ModelCategory::Coding);
    }
    if has(&["reasoning", "thinking", "-r1", "deepseek-r1", "qwq", "distill", "cot", "o1"]) {
        out.push(ModelCategory::Reasoning);
    }
    if has(&["math", "mathstral", "metamath"]) {
        out.push(ModelCategory::Math);
    }
    if has(&["agent", "tool-use", "tool_use", "function-calling", "hermes", "firefunction"]) {
        out.push(ModelCategory::Agentic);
    }
    if has(&["vision", "-vl", "llava", "image-text", "multimodal", "moondream"]) {
        out.push(ModelCategory::Vision);
    }
    if has(&["moe", "mixtral", "mixture-of-experts", "a3b", "a22b"]) {
        out.push(ModelCategory::MixtureOfExperts);
    }
    if has(&["multilingual", "aya", "bloom", "sea-lion"]) {
        out.push(ModelCategory::Multilingual);
    }

    if context_length >= LONG_CONTEXT_TOKENS {
        out.push(ModelCategory::LongContext);
    }
    if total_params > 0 && total_params <= SMALL_MODEL_PARAMS {
        out.push(ModelCategory::SmallAndFast);
    }

    // "General" means no specialisation, so size and context alone do not
    // disqualify it — a small general chat model is still general. An adapter,
    // however, is never a general assistant: it cannot run by itself.
    let specialised = out.iter().any(|c| {
        !matches!(
            c,
            ModelCategory::SmallAndFast | ModelCategory::LongContext
        )
    });
    if !specialised {
        out.push(ModelCategory::General);
    }

    out
}

/// What kind of thing an entry is, in terms a non-specialist can act on.
///
/// The distinction that matters is whether you can download it and run it. A
/// base model and a fine-tune both can; an adapter cannot — it needs a base
/// model underneath. Burying that in tags leaves people downloading files that
/// will not load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    /// Trained from scratch — the original.
    BaseModel,
    /// Retrained from a base model to specialise it.
    FineTuned,
    /// The same model, compressed to run on smaller hardware.
    Quantized,
    /// A patch applied on top of a base model. Cannot run alone.
    LoraAdapter,
}

impl ModelKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::BaseModel => "Base model",
            Self::FineTuned => "Fine-tuned",
            Self::Quantized => "Quantized",
            Self::LoraAdapter => "LoRA adapter",
        }
    }

    /// One line explaining what this means for the user.
    pub fn explanation(&self) -> &'static str {
        match self {
            Self::BaseModel => "The original model. Download and run it.",
            Self::FineTuned => "A base model retrained for a specific job. Download and run it.",
            Self::Quantized => "A compressed version of another model. Download and run it.",
            Self::LoraAdapter => {
                "An add-on for a base model. It cannot run on its own — you need the base model too."
            }
        }
    }
}

/// Determines the kind from HuggingFace's `base_model:` tags.
///
/// Order matters: a repo can carry both `base_model:X` and
/// `base_model:adapter:X`, and the more specific one is the truth.
pub fn model_kind(tags: &[String]) -> ModelKind {
    let has = |prefix: &str| tags.iter().any(|t| t.starts_with(prefix));

    if crate::model_providers::huggingface::discovery::is_lora_adapter(tags) {
        ModelKind::LoraAdapter
    } else if has("base_model:finetune:") || has("base_model:merge:") {
        ModelKind::FineTuned
    } else if has("base_model:quantized:") {
        ModelKind::Quantized
    } else {
        ModelKind::BaseModel
    }
}

/// Training datasets named in the tags, when they say anything useful.
///
/// Placeholder names that carry no information are dropped rather than shown —
/// "trained on: generator" tells a reader nothing.
pub fn datasets_from_tags(tags: &[String]) -> Vec<String> {
    const UNINFORMATIVE: &[&str] = &["generator", "custom", "private", "unknown", "none"];

    tags.iter()
        .filter_map(|t| t.strip_prefix("dataset:"))
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty() && !UNINFORMATIVE.contains(&d.to_lowercase().as_str()))
        .take(3)
        .collect()
}

/// Extracts an SPDX-ish licence id from HuggingFace tags (`license:apache-2.0`).
pub fn license_from_tags(tags: &[String]) -> Option<String> {
    tags.iter()
        .find_map(|t| t.strip_prefix("license:"))
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && l != "other" && l != "unknown")
}

/// Compact download count: `8M`, `52.3K`, `940`.
pub fn format_downloads(n: u64) -> String {
    match n {
        n if n >= 10_000_000 => format!("{}M", n / 1_000_000),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1_000_000.0),
        n if n >= 10_000 => format!("{}K", n / 1_000),
        n if n >= 1_000 => format!("{:.1}K", n as f64 / 1_000.0),
        n => n.to_string(),
    }
}

/// Compact age from an ISO-8601 timestamp: `3d`, `2mo`, `1y`.
///
/// Returns `None` when the timestamp is missing or unparseable, so the card can
/// omit the field rather than display something wrong.
pub fn format_age(last_modified: &str, now: chrono::DateTime<chrono::Utc>) -> Option<String> {
    let then = chrono::DateTime::parse_from_rfc3339(last_modified).ok()?;
    let days = now.signed_duration_since(then.with_timezone(&chrono::Utc)).num_days();
    if days < 0 {
        return None; // clock skew; better to say nothing
    }
    Some(match days {
        0 => "today".to_string(),
        1 => "1d".to_string(),
        d if d < 30 => format!("{d}d"),
        d if d < 365 => format!("{}mo", d / 30),
        d => format!("{}y", d / 365),
    })
}

/// One quantization choice, sized and described for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantizationOption {
    pub label: String,
    pub size_bytes: u64,
    /// Plain-English quality note, e.g. "Best quality", "Smallest".
    pub quality_note: String,
    /// Whether this option fits the machine's memory budget.
    pub fits: bool,
}

/// Presentation-ready model card.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCard {
    pub repo_id: String,
    /// HuggingFace org or user, shown as the publisher.
    pub publisher: String,
    pub name: String,
    /// Factual one-liner built from real metadata — never marketing copy.
    pub summary: String,
    pub categories: Vec<ModelCategory>,
    /// What this is — base model, fine-tune, quantization, or adapter.
    pub kind: ModelKind,
    /// Plain-language note on what that means for the user.
    pub kind_explanation: String,
    /// Datasets it was trained on, when named and informative.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    pub downloads: u64,
    pub downloads_label: String,
    pub likes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_label: Option<String>,
    pub is_finetune: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_parameters: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    pub quantizations: Vec<QuantizationOption>,
}

/// Human-readable parameter count: `7.6B`, `1.5B`, `350M`.
pub fn format_parameters(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else {
        format!("{}M", n / 1_000_000)
    }
}

/// Human-readable context: `32K`, `128K`, `2,048`.
pub fn format_context(n: u32) -> String {
    if n >= 1024 && n % 1024 == 0 {
        format!("{}K", n / 1024)
    } else {
        n.to_string()
    }
}

/// Builds the factual one-line summary shown on the card.
///
/// Assembled from metadata rather than scraped from a README: model card prose
/// is long-form marketing that would need fetching and trimming per model, and
/// would often say more about the base model than this repository.
fn build_summary(repo: &GgufRepo, categories: &[ModelCategory]) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(g) = &repo.gguf {
        if g.total_parameters > 0 {
            parts.push(format!("{} parameters", format_parameters(g.total_parameters)));
        }
        if g.context_length > 0 {
            parts.push(format!("{} context", format_context(g.context_length)));
        }
    }

    let focus: Vec<&str> = categories
        .iter()
        .filter(|c| !matches!(c, ModelCategory::General))
        .map(|c| c.label())
        .collect();
    if !focus.is_empty() {
        parts.push(format!("suited to {}", focus.join(", ").to_lowercase()));
    }

    if parts.is_empty() {
        return "GGUF model.".to_string();
    }
    format!("{}.", parts.join(" · "))
}

/// Quality note for a quantization label.
fn quality_note(label: &str) -> &'static str {
    let l = label.to_uppercase();
    if l.starts_with("F16") || l.starts_with("F32") || l.starts_with("BF16") {
        "Full precision — largest"
    } else if l.starts_with("Q8") {
        "Near-lossless"
    } else if l.starts_with("Q6") {
        "Very high quality"
    } else if l.starts_with("Q5") {
        "High quality"
    } else if l.starts_with("Q4") || l.starts_with("IQ4") {
        "Balanced — recommended"
    } else if l.starts_with("Q3") || l.starts_with("IQ3") {
        "Noticeable quality loss"
    } else {
        "Smallest — significant quality loss"
    }
}

fn to_option(q: &Quantization, budget_bytes: u64) -> QuantizationOption {
    QuantizationOption {
        label: q.label.clone(),
        size_bytes: q.size_bytes,
        quality_note: quality_note(&q.label).to_string(),
        fits: budget_bytes > 0 && q.size_bytes <= budget_bytes,
    }
}

/// Builds a card from a discovered repository.
///
/// `budget_bytes` is the memory this machine can give to model weights; pass 0
/// when unknown, and no option will be marked as fitting.
pub fn build_card(
    repo: &GgufRepo,
    budget_bytes: u64,
    now: chrono::DateTime<chrono::Utc>,
) -> ModelCard {
    let (params, context, arch) = repo
        .gguf
        .as_ref()
        .map(|g| (g.total_parameters, g.context_length, g.architecture.clone()))
        .unwrap_or((0, 0, String::new()));

    let categories = categorize(&repo.repo_id, &repo.tags, params, context, &arch);
    let kind = model_kind(&repo.tags);

    ModelCard {
        repo_id: repo.repo_id.clone(),
        publisher: repo.author.clone(),
        name: repo.display_name(),
        summary: build_summary(repo, &categories),
        categories,
        kind,
        kind_explanation: kind.explanation().to_string(),
        datasets: datasets_from_tags(&repo.tags),
        license: license_from_tags(&repo.tags),
        downloads: repo.downloads,
        downloads_label: format_downloads(repo.downloads),
        likes: repo.likes,
        age_label: format_age(&repo.last_modified, now),
        is_finetune: repo.is_finetune,
        base_model: repo.base_model.clone(),
        total_parameters: (params > 0).then_some(params),
        context_length: (context > 0).then_some(context),
        quantizations: repo.quantizations.iter().map(|q| to_option(q, budget_bytes)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::GgufMeta;

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn coding_models_are_categorised_as_coding() {
        let c = categorize("bartowski/Qwen2.5-Coder-7B-GGUF", &tags(&["code"]), 7_600_000_000, 32768, "qwen2");
        assert!(c.contains(&ModelCategory::Coding));
        assert!(!c.contains(&ModelCategory::General), "a specialised model is not 'general'");
    }

    #[test]
    fn a_model_can_hold_several_categories() {
        // Small coding model: both, because either filter should surface it.
        let c = categorize("x/Qwen2.5-Coder-1.5B-GGUF", &tags(&["code"]), 1_500_000_000, 32768, "qwen2");
        assert!(c.contains(&ModelCategory::Coding));
        assert!(c.contains(&ModelCategory::SmallAndFast));
    }

    #[test]
    fn mixture_of_experts_is_detected() {
        assert!(categorize("x/Mixtral-8x7B-GGUF", &tags(&[]), 46_000_000_000, 32768, "llama")
            .contains(&ModelCategory::MixtureOfExperts));
        assert!(categorize("x/Qwen3-30B-A3B-GGUF", &tags(&[]), 30_000_000_000, 32768, "qwen3moe")
            .contains(&ModelCategory::MixtureOfExperts));
    }

    #[test]
    fn reasoning_and_long_context_are_detected() {
        let c = categorize("x/DeepSeek-R1-Distill-8B", &tags(&[]), 8_000_000_000, 131072, "llama");
        assert!(c.contains(&ModelCategory::Reasoning));
        assert!(c.contains(&ModelCategory::LongContext));
    }

    #[test]
    fn lora_adapters_are_labelled_as_adapters() {
        // An adapter is a patch for a base model, not something you can load
        // and talk to. Labelling it plainly stops someone downloading one
        // expecting a chat model.
        let c = categorize(
            "someone/Llama-3-Coding-LoRA",
            &tags(&["base_model:adapter:meta-llama/Llama-3.1-8B", "lora"]),
            0,
            0,
            "",
        );

        assert!(c.contains(&ModelCategory::LoraAdapter));
        assert!(
            !c.contains(&ModelCategory::General),
            "an adapter cannot run on its own, so it is never a general assistant"
        );
    }

    #[test]
    fn a_peft_tag_alone_marks_an_adapter() {
        let c = categorize("x/y", &tags(&["peft"]), 0, 0, "");
        assert!(c.contains(&ModelCategory::LoraAdapter));
    }

    #[test]
    fn a_model_merely_named_lora_is_not_treated_as_an_adapter() {
        // Detection reads tags, not the name — mislabelling a runnable model as
        // an adapter would hide it from the model list.
        let c = categorize("someone/lora-tuned-chat-7B", &tags(&["en"]), 7_000_000_000, 8192, "llama");

        assert!(!c.contains(&ModelCategory::LoraAdapter));
        assert!(c.contains(&ModelCategory::General));
    }

    #[test]
    fn an_adapter_can_still_carry_what_it_is_for() {
        // "A coding LoRA" is exactly the useful description, so both apply.
        let c = categorize(
            "someone/Coder-LoRA",
            &tags(&["base_model:adapter:Qwen/Qwen2.5-7B", "code"]),
            0,
            0,
            "",
        );

        assert!(c.contains(&ModelCategory::LoraAdapter));
        assert!(c.contains(&ModelCategory::Coding));
    }

    #[test]
    fn quantizations_and_finetunes_are_not_adapters() {
        // These are runnable models; only `base_model:adapter:` is a patch.
        let quantized = categorize("x/y", &tags(&["base_model:quantized:Qwen/Qwen2.5-7B"]), 7_000_000_000, 8192, "");
        assert!(!quantized.contains(&ModelCategory::LoraAdapter));

        let finetuned = categorize("x/y", &tags(&["base_model:finetune:meta-llama/Llama-3.1-8B"]), 8_000_000_000, 8192, "");
        assert!(!finetuned.contains(&ModelCategory::LoraAdapter));
    }

    #[test]
    fn a_plain_chat_model_falls_back_to_general() {
        let c = categorize("x/Llama-3.1-8B-Instruct-GGUF", &tags(&["en"]), 8_000_000_000, 8192, "llama");
        assert_eq!(c, vec![ModelCategory::General]);
    }

    #[test]
    fn small_general_models_stay_general() {
        // Size alone must not strip the General label, or small chat models
        // vanish from the General filter.
        let c = categorize("x/Llama-3.2-1B-Instruct", &tags(&[]), 1_200_000_000, 8192, "llama");
        assert!(c.contains(&ModelCategory::SmallAndFast));
        assert!(c.contains(&ModelCategory::General));
    }

    #[test]
    fn every_model_gets_at_least_one_category() {
        // Nothing may be unreachable from the sidebar.
        for (name, params, ctx) in [("x/mystery", 0u64, 0u32), ("y/z", 7_000_000_000, 4096)] {
            assert!(!categorize(name, &tags(&[]), params, ctx, "").is_empty());
        }
    }

    #[test]
    fn model_kind_is_read_from_base_model_tags() {
        assert_eq!(model_kind(&tags(&["gguf", "en"])), ModelKind::BaseModel);
        assert_eq!(
            model_kind(&tags(&["base_model:quantized:Qwen/Qwen2.5-7B"])),
            ModelKind::Quantized
        );
        assert_eq!(
            model_kind(&tags(&["base_model:finetune:meta-llama/Llama-3.1-8B"])),
            ModelKind::FineTuned
        );
        assert_eq!(
            model_kind(&tags(&["base_model:adapter:meta-llama/Llama-3.1-8B"])),
            ModelKind::LoraAdapter
        );
    }

    #[test]
    fn the_more_specific_tag_wins() {
        // Adapters commonly carry both `base_model:X` and `base_model:adapter:X`.
        // Reading the plain one first would call an adapter a base model.
        let both = tags(&[
            "base_model:meta-llama/Llama-3.1-8B",
            "base_model:adapter:meta-llama/Llama-3.1-8B",
        ]);
        assert_eq!(model_kind(&both), ModelKind::LoraAdapter);
    }

    #[test]
    fn every_kind_explains_whether_it_can_run_alone() {
        // The distinction a non-specialist needs is exactly this one.
        assert!(ModelKind::LoraAdapter.explanation().contains("cannot run on its own"));
        for k in [ModelKind::BaseModel, ModelKind::FineTuned, ModelKind::Quantized] {
            assert!(k.explanation().contains("Download and run"), "{:?}", k);
        }
    }

    #[test]
    fn datasets_are_read_from_tags() {
        let found = datasets_from_tags(&tags(&["dataset:squad", "dataset:gsm8k", "gguf"]));
        assert_eq!(found, vec!["squad", "gsm8k"]);
    }

    #[test]
    fn placeholder_dataset_names_are_dropped() {
        // "Trained on: generator" tells a reader nothing, so it is not shown.
        assert!(datasets_from_tags(&tags(&["dataset:generator"])).is_empty());
        assert!(datasets_from_tags(&tags(&["dataset:custom", "dataset:unknown"])).is_empty());
        assert!(datasets_from_tags(&tags(&["gguf"])).is_empty());
    }

    #[test]
    fn licence_is_read_from_tags() {
        assert_eq!(license_from_tags(&tags(&["license:apache-2.0", "gguf"])).as_deref(), Some("apache-2.0"));
        assert_eq!(license_from_tags(&tags(&["gguf"])), None);
        // "other" carries no information, so it is treated as absent.
        assert_eq!(license_from_tags(&tags(&["license:other"])), None);
    }

    #[test]
    fn download_counts_are_compact() {
        assert_eq!(format_downloads(52_000_000), "52M");
        assert_eq!(format_downloads(1_500_000), "1.5M");
        assert_eq!(format_downloads(67_535), "67K");
        assert_eq!(format_downloads(1_200), "1.2K");
        assert_eq!(format_downloads(940), "940");
    }

    #[test]
    fn ages_are_compact_and_relative() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-08-05T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        let at = |s: &str| format_age(s, now);
        assert_eq!(at("2026-08-05T00:00:00Z").as_deref(), Some("today"));
        assert_eq!(at("2026-08-01T00:00:00Z").as_deref(), Some("4d"));
        assert_eq!(at("2026-06-05T00:00:00Z").as_deref(), Some("2mo"));
        assert_eq!(at("2024-08-05T00:00:00Z").as_deref(), Some("2y"));
    }

    #[test]
    fn unparseable_or_future_dates_yield_no_age() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-08-05T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        // Better to omit the field than to print something wrong.
        assert_eq!(format_age("", now), None);
        assert_eq!(format_age("not-a-date", now), None);
        assert_eq!(format_age("2027-01-01T00:00:00Z", now), None);
    }

    #[test]
    fn parameters_and_context_read_naturally() {
        assert_eq!(format_parameters(7_600_000_000), "7.6B");
        assert_eq!(format_parameters(350_000_000), "350M");
        assert_eq!(format_context(32768), "32K");
        assert_eq!(format_context(2048), "2K");
    }

    fn repo_for_card() -> GgufRepo {
        GgufRepo {
            repo_id: "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF".into(),
            author: "bartowski".into(),
            downloads: 67_535,
            likes: 68,
            last_modified: "2026-07-05T00:00:00Z".into(),
            quantizations: vec![
                Quantization { label: "Q2_K".into(), filename: "a.gguf".into(), size_bytes: 3_000_000_000, is_sharded: false },
                Quantization { label: "Q4_K_M".into(), filename: "b.gguf".into(), size_bytes: 4_700_000_000, is_sharded: false },
                Quantization { label: "Q8_0".into(), filename: "c.gguf".into(), size_bytes: 8_100_000_000, is_sharded: false },
            ],
            gguf: Some(GgufMeta {
                total_parameters: 7_600_000_000,
                architecture: "qwen2".into(),
                context_length: 32768,
                chat_template: None,
                bos_token: None,
                eos_token: None,
            }),
            base_model: Some("Qwen/Qwen2.5-Coder-7B-Instruct".into()),
            is_finetune: false,
            is_lora_adapter: false,
            tags: tags(&["license:apache-2.0", "code", "gguf"]),
        }
    }

    #[test]
    fn a_card_carries_everything_the_listing_shows() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-08-05T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        // 4 GB budget, as on the RTX 3050.
        let card = build_card(&repo_for_card(), 4_000_000_000, now);

        assert_eq!(card.publisher, "bartowski");
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(card.downloads_label, "67K");
        assert_eq!(card.age_label.as_deref(), Some("1mo"));
        assert!(card.categories.contains(&ModelCategory::Coding));
        assert!(card.summary.contains("7.6B"), "summary: {}", card.summary);
        assert!(card.summary.contains("32K"), "summary: {}", card.summary);
    }

    #[test]
    fn quantization_comparison_marks_only_what_fits() {
        let now = chrono::Utc::now();
        let card = build_card(&repo_for_card(), 4_000_000_000, now);

        let fitting: Vec<&str> = card
            .quantizations
            .iter()
            .filter(|q| q.fits)
            .map(|q| q.label.as_str())
            .collect();

        assert_eq!(fitting, vec!["Q2_K"], "only the 3 GB option fits a 4 GB budget");
        assert!(card.quantizations.iter().any(|q| q.quality_note.contains("recommended")));
    }

    #[test]
    fn an_unknown_budget_marks_nothing_as_fitting() {
        // Claiming a fit without knowing the hardware would be a guess.
        let card = build_card(&repo_for_card(), 0, chrono::Utc::now());
        assert!(card.quantizations.iter().all(|q| !q.fits));
    }
}

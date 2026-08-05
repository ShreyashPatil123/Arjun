//! Real HuggingFace GGUF discovery.
//!
//! Replaces the previous approach, where every search result was passed through
//! a hand-written `if repo.contains("llama-3.2-1b")` chain and **discarded if it
//! did not match**. Only models someone had typed in by hand could ever appear,
//! so the catalog was permanently stuck at 16 entries no matter what the search
//! returned — and no fine-tune, new release, or community quantization could
//! ever surface.
//!
//! Nothing here is hardcoded. Everything comes from the HuggingFace API:
//!
//! | Field | Source | Used for |
//! |---|---|---|
//! | Exact size per quantization | `?blobs=true` siblings | Does it fit in VRAM |
//! | Parameters, architecture, context | `gguf` object | Budgeting, display |
//! | Chat template, BOS/EOS | `gguf` object | Correct prompt formatting |
//! | Parent of a fine-tune | `base_model:` tag | Resolving fine-tune architecture |
//!
//! Bulk search does not include the `gguf` object, so discovery is two-phase:
//! one search call for candidates, then per-repo detail calls (run concurrently
//! by the caller).

use serde::{Deserialize, Serialize};

/// A single quantization available in a repo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Quantization {
    /// Label parsed from the filename, e.g. `Q4_K_M`, `IQ4_XS`.
    pub label: String,
    pub filename: String,
    /// Exact bytes. Authoritative for fit decisions — no estimation involved.
    pub size_bytes: u64,
    /// True when this is the first shard of a multi-file model.
    pub is_sharded: bool,
}

/// Architecture facts read from the repo's GGUF metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufMeta {
    /// Total parameters.
    pub total_parameters: u64,
    /// llama.cpp architecture id, e.g. `qwen2`, `llama`, `gemma2`.
    pub architecture: String,
    pub context_length: u32,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

/// A discovered GGUF repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufRepo {
    pub repo_id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: String,
    pub quantizations: Vec<Quantization>,
    pub gguf: Option<GgufMeta>,
    /// Parent model when this repo is a quantization or fine-tune of another.
    pub base_model: Option<String>,
    /// True when the repo is a fine-tune rather than a plain re-quantization.
    pub is_finetune: bool,
    /// True when the repo is a LoRA adapter — a patch for a base model, not
    /// something that can be loaded and run on its own.
    #[serde(default)]
    pub is_lora_adapter: bool,
    /// Free-form tags, kept for search and display.
    pub tags: Vec<String>,
}

impl GgufRepo {
    /// Smallest quantization, i.e. the one most likely to fit a small GPU.
    pub fn smallest(&self) -> Option<&Quantization> {
        self.quantizations.iter().min_by_key(|q| q.size_bytes)
    }

    /// Largest quantization that fits within `budget_bytes`.
    ///
    /// Bigger is better here: quantization quality rises with size, so the
    /// correct pick is the largest that still fits, not the smallest available.
    pub fn best_fitting(&self, budget_bytes: u64) -> Option<&Quantization> {
        self.quantizations
            .iter()
            .filter(|q| q.size_bytes <= budget_bytes)
            .max_by_key(|q| q.size_bytes)
    }

    /// Display name derived from the repo id.
    pub fn display_name(&self) -> String {
        let name = self.repo_id.split('/').next_back().unwrap_or(&self.repo_id);
        name.trim_end_matches("-GGUF")
            .trim_end_matches("-gguf")
            .replace('-', " ")
    }
}

// ─── Raw API shapes ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RawSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct RawGguf {
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub bos_token: Option<String>,
    #[serde(default)]
    pub eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawModelInfo {
    pub id: String,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub siblings: Vec<RawSibling>,
    #[serde(default)]
    pub gguf: Option<RawGguf>,
}

/// Pipeline tags Sarathi can serve.
///
/// The GGUF filter alone also returns OCR, text-to-speech, embedding, and
/// vision models — `PaddleOCR-VL`, `surya-ocr`, and `svara-tts` all ship GGUF
/// files but cannot answer a chat request. They were previously dropped only
/// because their filenames failed quantization parsing, which is luck rather
/// than a rule. Checking the declared task rejects them deliberately.
///
/// `None` is accepted: many quantization repos leave the tag unset, and
/// excluding those would discard a large part of the usable catalog.
pub fn is_servable_pipeline(tag: Option<&str>) -> bool {
    match tag {
        None => true,
        Some(t) => matches!(t, "text-generation" | "text2text-generation" | "conversational"),
    }
}

impl RawModelInfo {
    /// Builds a [`GgufRepo`], or `None` when the repo cannot serve chat requests.
    pub fn into_repo(self) -> Option<GgufRepo> {
        if !is_servable_pipeline(self.pipeline_tag.as_deref()) {
            return None;
        }

        let quantizations = parse_quantizations(&self.siblings);
        if quantizations.is_empty() {
            return None;
        }

        let author = self
            .author
            .clone()
            .unwrap_or_else(|| self.id.split('/').next().unwrap_or("").to_string());

        let gguf = self.gguf.map(|g| GgufMeta {
            total_parameters: g.total.unwrap_or(0),
            architecture: g.architecture.unwrap_or_default(),
            context_length: g.context_length.unwrap_or(0),
            chat_template: g.chat_template,
            bos_token: g.bos_token,
            eos_token: g.eos_token,
        });

        let base_model = base_model_from_tags(&self.tags);
        let is_finetune = is_finetune(&self.tags);
        let is_lora_adapter = is_lora_adapter(&self.tags);

        Some(GgufRepo {
            repo_id: self.id,
            author,
            downloads: self.downloads,
            likes: self.likes,
            last_modified: self.last_modified.unwrap_or_default(),
            quantizations,
            gguf,
            base_model,
            is_finetune,
            is_lora_adapter,
            tags: self.tags,
        })
    }
}

// ─── Parsing helpers ────────────────────────────────────────────────────────

/// Matches the `-00001-of-00003` shard suffix.
fn strip_shard_suffix(stem: &str) -> (String, bool) {
    // Looking for "-NNNNN-of-NNNNN" at the end.
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    // rsplitn yields reversed: [NNNNN, "of", NNNNN, rest]
    if parts.len() == 4
        && parts[1] == "of"
        && parts[0].len() == 5
        && parts[2].len() == 5
        && parts[0].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return (parts[3].to_string(), true);
    }
    (stem.to_string(), false)
}

/// Extracts the quantization label from a GGUF filename.
///
/// `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` -> `Q4_K_M`
pub fn quantization_label(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf")?;
    let (stem, _) = strip_shard_suffix(stem);
    let label = stem.rsplit('-').next()?;

    // Real labels start with Q or IQ and contain a digit: Q4_K_M, IQ4_XS, Q8_0.
    let upper = label.to_uppercase();
    let plausible = (upper.starts_with('Q') || upper.starts_with("IQ") || upper.starts_with('F') || upper.starts_with("BF"))
        && upper.chars().any(|c| c.is_ascii_digit());

    if plausible {
        Some(upper)
    } else {
        None
    }
}

/// Turns the file listing into quantization entries.
///
/// Later shards of a split model are skipped so a 3-part model appears once,
/// but its recorded size is the **total across all shards** — using only the
/// first shard's size would badly understate what has to fit in memory.
/// True for GGUF files that sit beside a model rather than being one.
///
/// The clearest case is the multimodal projector: `lmstudio-community/`
/// `gemma-4-12B-it-QAT-GGUF` ships `mmproj-gemma-4-12B-it-QAT-BF16.gguf` at
/// 167 MB next to a 6.6 GB Q4_0 model. Reading `BF16` out of that filename
/// offered a 167 MB "full precision" download of a 12B model, which comfortably
/// "fits" any machine — and which cannot be loaded as a model at all, because
/// it is the vision encoder's projection layer.
///
/// These files are real and needed for vision models, but they are companions
/// to a quantization, never a choice of one.
fn is_companion_file(filename: &str) -> bool {
    // Companions are kept in a subfolder in some repositories — the gemma-4
    // repo puts its multi-token-prediction module in `MTP/`. The model itself,
    // including every shard of it, sits at the top level in each layout seen
    // so far, so a nested GGUF is not a size a person can choose.
    if filename.contains('/') {
        return true;
    }

    let base = filename.to_ascii_lowercase();
    let has_token = |t: &str| {
        base.starts_with(t) || base.contains(&format!("-{t}")) || base.contains(&format!("_{t}"))
    };

    // Each of these is a documented llama.cpp side-car rather than a model:
    // the vision projector, the multi-token-prediction head, and the small
    // draft model used for speculative decoding.
    has_token("mmproj") || has_token("mtp") || has_token("draft") || has_token("projector")
}

pub fn parse_quantizations(siblings: &[RawSibling]) -> Vec<Quantization> {
    use std::collections::HashMap;

    let mut totals: HashMap<String, (String, u64, bool)> = HashMap::new();

    for s in siblings {
        if !s.rfilename.ends_with(".gguf") {
            continue;
        }
        if is_companion_file(&s.rfilename) {
            continue;
        }
        let Some(label) = quantization_label(&s.rfilename) else {
            continue;
        };
        let size = s.size.unwrap_or(0);
        let stem = s.rfilename.strip_suffix(".gguf").unwrap_or(&s.rfilename);
        let (_, sharded) = strip_shard_suffix(stem);

        let entry = totals
            .entry(label.clone())
            .or_insert_with(|| (s.rfilename.clone(), 0, sharded));
        entry.1 += size;
        entry.2 |= sharded;
        // Keep the first shard as the representative filename.
        if sharded && s.rfilename.contains("-00001-of-") {
            entry.0 = s.rfilename.clone();
        }
    }

    let mut out: Vec<Quantization> = totals
        .into_iter()
        .map(|(label, (filename, size_bytes, is_sharded))| Quantization {
            label,
            filename,
            size_bytes,
            is_sharded,
        })
        .collect();

    out.sort_by_key(|q| q.size_bytes);
    out
}

/// Reads the parent model from `base_model:` tags.
///
/// HuggingFace emits several forms; `base_model:quantized:X` and
/// `base_model:finetune:X` both carry the parent in the final segment.
pub fn base_model_from_tags(tags: &[String]) -> Option<String> {
    for tag in tags {
        if let Some(rest) = tag.strip_prefix("base_model:") {
            let parent = rest.rsplit(':').next().unwrap_or(rest);
            if !parent.is_empty() && parent.contains('/') {
                return Some(parent.to_string());
            }
        }
    }
    None
}

/// True when the repo is a LoRA adapter rather than a runnable model.
///
/// This matters more here than in a general model browser: an adapter is a
/// small patch applied *on top of* a base model, not something you can load and
/// talk to. Downloading one expecting a chat model gets you a file that will not
/// run, so adapters are labelled distinctly rather than folded in with
/// fine-tunes.
pub fn is_lora_adapter(tags: &[String]) -> bool {
    tags.iter().any(|t| {
        t.starts_with("base_model:adapter:")
            || t.eq_ignore_ascii_case("lora")
            || t.eq_ignore_ascii_case("peft")
    })
}

/// True when the repo is a fine-tune — a fully retrained model — rather than a
/// plain re-quantization or an adapter.
pub fn is_finetune(tags: &[String]) -> bool {
    tags.iter()
        .any(|t| t.starts_with("base_model:finetune:") || t.starts_with("base_model:merge:"))
}

/// Builds a HuggingFace search URL.
///
/// `full=true` is required for `siblings` to be present in list results.
pub fn search_url(query: Option<&str>, limit: u32, page: u32) -> String {
    let mut url = format!(
        "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit={}&full=true",
        limit.clamp(1, 100)
    );
    if page > 0 {
        url.push_str(&format!("&skip={}", page * limit));
    }
    if let Some(q) = query.map(str::trim).filter(|q| !q.is_empty()) {
        url.push_str(&format!("&search={}", urlencode(q)));
    }
    url
}

/// Per-repo detail URL. `blobs=true` is what makes file sizes available.
pub fn detail_url(repo_id: &str) -> String {
    format!("https://huggingface.co/api/models/{repo_id}?blobs=true")
}

/// Transformer shape needed for KV-cache maths.
///
/// The GGUF metadata HuggingFace exposes gives parameter count, architecture,
/// and context length, but not layer or head counts. Fetching each model's
/// `config.json` would cost one extra request per model, so these are inferred
/// from parameter count instead.
///
/// This is an **estimate**, and only ever affects KV-cache sizing. Whether a
/// model fits is decided by its exact `.gguf` file size, which is measured, not
/// guessed — so an imperfect estimate here cannot cause a model to be wrongly
/// recommended or rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchitectureEstimate {
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dimension: u32,
    pub hidden_size: u32,
}

/// Infers transformer shape from parameter count.
///
/// Bands follow the configurations the major open-weight families converged on
/// (Llama 3, Qwen 2.5/3, Mistral, Gemma 2, Phi 3). Grouped-query attention with
/// 8 KV heads and a 128-wide head is near-universal above ~3B parameters.
pub fn estimate_architecture(total_parameters: u64) -> ArchitectureEstimate {
    const B: u64 = 1_000_000_000;
    match total_parameters {
        p if p < 2 * B => ArchitectureEstimate {
            num_layers: 16,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 64,
            hidden_size: 2048,
        },
        p if p < 5 * B => ArchitectureEstimate {
            num_layers: 28,
            num_attention_heads: 24,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 3072,
        },
        p if p < 10 * B => ArchitectureEstimate {
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
        },
        p if p < 20 * B => ArchitectureEstimate {
            num_layers: 48,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 5120,
        },
        p if p < 40 * B => ArchitectureEstimate {
            num_layers: 64,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 5120,
        },
        _ => ArchitectureEstimate {
            num_layers: 80,
            num_attention_heads: 64,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 8192,
        },
    }
}

/// Human-readable family name derived from the repo and its tags.
pub fn family_name(repo: &GgufRepo) -> String {
    // A fine-tune belongs to its parent's family, which the base_model tag names.
    if let Some(base) = &repo.base_model {
        if let Some(name) = base.split('/').next_back() {
            return name.replace('-', " ");
        }
    }
    repo.gguf
        .as_ref()
        .map(|g| g.architecture.clone())
        .filter(|a| !a.is_empty())
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Minimal percent-encoding for query values (no extra dependency).
fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sib(name: &str, size: u64) -> RawSibling {
        RawSibling { rfilename: name.to_string(), size: Some(size) }
    }

    #[test]
    fn a_vision_projector_is_not_offered_as_a_quantization() {
        // The real file list of lmstudio-community/gemma-4-12B-it-QAT-GGUF.
        // Treating the projector as a quantization advertised a 167 MB "BF16"
        // build of a 12B model that fits any machine and loads on none.
        let files = vec![
            sib("gemma-4-12B-it-QAT-Q4_0.gguf", 6_976_000_000),
            sib("mmproj-gemma-4-12B-it-QAT-BF16.gguf", 175_000_000),
        ];

        let quants = parse_quantizations(&files);

        assert_eq!(quants.len(), 1, "only the model is a real choice: {quants:?}");
        assert_eq!(quants[0].label, "Q4_0");
        assert!(
            !quants.iter().any(|q| q.label == "BF16"),
            "the projector must not appear as a size option"
        );
    }

    #[test]
    fn quantization_labels_are_parsed_from_filenames() {
        assert_eq!(quantization_label("Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quantization_label("Qwen2.5-Coder-7B-Instruct-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quantization_label("Llama-3.2-1B-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quantization_label("model-F16.gguf").as_deref(), Some("F16"));
    }

    #[test]
    fn non_quantization_files_are_ignored() {
        assert_eq!(quantization_label("README.md"), None);
        assert_eq!(quantization_label(".gitattributes"), None);
        // A .gguf with no recognisable quant label.
        assert_eq!(quantization_label("mmproj-model.gguf"), None);
    }

    #[test]
    fn sharded_files_collapse_to_one_entry_with_the_total_size() {
        // A split model must report what actually has to fit in memory, not
        // just the size of its first shard.
        let siblings = vec![
            sib("Big-Model-Q4_K_M-00001-of-00003.gguf", 4_000_000_000),
            sib("Big-Model-Q4_K_M-00002-of-00003.gguf", 4_000_000_000),
            sib("Big-Model-Q4_K_M-00003-of-00003.gguf", 2_000_000_000),
        ];

        let quants = parse_quantizations(&siblings);
        assert_eq!(quants.len(), 1, "three shards are one quantization");
        assert_eq!(quants[0].label, "Q4_K_M");
        assert_eq!(quants[0].size_bytes, 10_000_000_000, "must sum all shards");
        assert!(quants[0].is_sharded);
        assert!(quants[0].filename.contains("-00001-of-"), "should point at the first shard");
    }

    #[test]
    fn quantizations_are_sorted_smallest_first() {
        let siblings = vec![
            sib("m-Q8_0.gguf", 8_000_000_000),
            sib("m-Q4_K_M.gguf", 4_000_000_000),
            sib("m-Q2_K.gguf", 2_000_000_000),
        ];
        let quants = parse_quantizations(&siblings);
        assert_eq!(quants.len(), 3);
        assert_eq!(quants[0].label, "Q2_K");
        assert_eq!(quants[2].label, "Q8_0");
    }

    fn repo_with(sizes: &[(&str, u64)]) -> GgufRepo {
        GgufRepo {
            repo_id: "someone/Test-GGUF".into(),
            author: "someone".into(),
            downloads: 0,
            likes: 0,
            last_modified: String::new(),
            quantizations: parse_quantizations(
                &sizes.iter().map(|(n, s)| sib(n, *s)).collect::<Vec<_>>(),
            ),
            gguf: None,
            base_model: None,
            is_finetune: false,
            is_lora_adapter: false,
            tags: vec![],
        }
    }

    #[test]
    fn best_fitting_picks_the_largest_that_fits_not_the_smallest() {
        // Quality rises with size, so the right pick is the biggest that fits.
        let repo = repo_with(&[
            ("m-Q2_K.gguf", 2_000_000_000),
            ("m-Q4_K_M.gguf", 4_000_000_000),
            ("m-Q8_0.gguf", 8_000_000_000),
        ]);

        let pick = repo.best_fitting(5_000_000_000).expect("something should fit");
        assert_eq!(pick.label, "Q4_K_M");
    }

    #[test]
    fn nothing_fits_a_tiny_budget() {
        let repo = repo_with(&[("m-Q4_K_M.gguf", 4_000_000_000)]);
        assert!(repo.best_fitting(1_000_000_000).is_none());
        assert_eq!(repo.smallest().unwrap().label, "Q4_K_M");
    }

    #[test]
    fn base_model_is_read_from_tags() {
        let tags = vec![
            "gguf".to_string(),
            "base_model:quantized:Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        ];
        assert_eq!(base_model_from_tags(&tags).as_deref(), Some("Qwen/Qwen2.5-Coder-7B-Instruct"));
        assert!(!is_finetune(&tags), "a quantization is not a fine-tune");
    }

    #[test]
    fn finetunes_are_detected() {
        let tags = vec!["base_model:finetune:meta-llama/Llama-3.1-8B".to_string()];
        assert!(is_finetune(&tags));
        assert_eq!(base_model_from_tags(&tags).as_deref(), Some("meta-llama/Llama-3.1-8B"));
    }

    #[test]
    fn tags_without_a_base_model_yield_none() {
        assert_eq!(base_model_from_tags(&["gguf".to_string(), "en".to_string()]), None);
        // Malformed: no org/name form.
        assert_eq!(base_model_from_tags(&["base_model:something".to_string()]), None);
    }

    #[test]
    fn search_urls_are_well_formed() {
        let url = search_url(None, 50, 0);
        assert!(url.contains("filter=gguf"));
        assert!(url.contains("limit=50"));
        assert!(url.contains("full=true"), "siblings require full=true");
        assert!(!url.contains("skip="), "first page needs no skip");

        let paged = search_url(None, 50, 2);
        assert!(paged.contains("skip=100"), "page 2 of 50 starts at 100");

        let searched = search_url(Some("qwen coder"), 20, 0);
        assert!(searched.contains("search=qwen+coder"));
    }

    #[test]
    fn search_limit_is_clamped_to_the_api_maximum() {
        assert!(search_url(None, 5000, 0).contains("limit=100"));
        assert!(search_url(None, 0, 0).contains("limit=1"));
    }

    #[test]
    fn detail_url_requests_blobs_so_sizes_are_present() {
        let url = detail_url("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert!(url.contains("blobs=true"), "without blobs there are no file sizes");
        assert!(url.contains("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"));
    }

    fn raw(pipeline: Option<&str>, siblings: Vec<RawSibling>) -> RawModelInfo {
        RawModelInfo {
            id: "someone/thing".into(),
            pipeline_tag: pipeline.map(String::from),
            author: None,
            downloads: 5,
            likes: 1,
            last_modified: None,
            tags: vec![],
            siblings,
            gguf: None,
        }
    }

    #[test]
    fn a_repo_without_gguf_files_is_rejected() {
        let r = raw(None, vec![sib("README.md", 100), sib("model.safetensors", 999)]);
        assert!(r.into_repo().is_none());
    }

    #[test]
    fn non_chat_models_are_rejected_even_when_they_ship_gguf() {
        // These are real examples seen in a live sweep. They carry .gguf files
        // but cannot answer a chat request.
        for tag in ["image-text-to-text", "text-to-speech", "feature-extraction", "automatic-speech-recognition"] {
            let r = raw(Some(tag), vec![sib("model-Q4_K_M.gguf", 1_000_000)]);
            assert!(
                r.into_repo().is_none(),
                "pipeline '{tag}' should not be offered as a chat model"
            );
        }
    }

    #[test]
    fn chat_capable_pipelines_are_accepted() {
        for tag in ["text-generation", "conversational", "text2text-generation"] {
            let r = raw(Some(tag), vec![sib("model-Q4_K_M.gguf", 1_000_000)]);
            assert!(r.into_repo().is_some(), "pipeline '{tag}' should be accepted");
        }
    }

    #[test]
    fn an_unset_pipeline_tag_is_accepted() {
        // Many quantization repos leave the tag blank; excluding them would
        // throw away a large part of the usable catalog.
        let r = raw(None, vec![sib("model-Q4_K_M.gguf", 1_000_000)]);
        assert!(r.into_repo().is_some());
    }

    #[test]
    fn display_names_are_readable() {
        let repo = repo_with(&[("m-Q4_K_M.gguf", 1)]);
        assert_eq!(repo.display_name(), "Test");
    }
}

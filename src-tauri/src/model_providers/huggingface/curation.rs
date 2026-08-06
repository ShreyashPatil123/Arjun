//! Which models Sarathi puts its name to.
//!
//! The Hub's popular sweep is not a shortlist. It mixes finished releases with
//! week-old merges, 1-bit experiments, and adapters, all sorted by download
//! count — which favours whatever went viral, not what will work on the user's
//! machine tonight. "Sarathi Recommended" is the answer to a narrower question:
//! *which of these would we hand to someone who just wants a working model?*
//!
//! Everything here judges a repository on evidence already fetched — its GGUF
//! metadata, tags, publisher, and measured sizes. Nothing calls out, and no
//! model is recommended for being fashionable.

use crate::model_providers::huggingface::discovery::GgufRepo;

/// Publishers whose GGUF conversions are dependable.
///
/// This is about *conversion quality*, not the merit of the underlying model:
/// these accounts publish complete quantization ranges, correct chat templates,
/// and metadata that matches the weights. A brilliant model converted carelessly
/// still loads badly, and the failure looks like a Sarathi bug.
///
/// Matched case-insensitively against the owner half of the repo id.
const TRUSTED_PUBLISHERS: &[&str] = &[
    // First-party releases.
    "google",
    "meta-llama",
    "mistralai",
    "qwen",
    "microsoft",
    "deepseek-ai",
    "allenai",
    "ibm-granite",
    "nvidia",
    // Conversion specialists with a long track record.
    "bartowski",
    "unsloth",
    "ggml-org",
    "lmstudio-community",
    "thebloke",
];

/// Markers that a model narrates its reasoning before answering.
///
/// These are matched against the GGUF's own chat template, which is the only
/// place that says what the model will actually emit. A template that opens an
/// `<think>` block commits every reply to carrying one.
///
/// This is not a judgement on reasoning models — they are often excellent. But
/// the tokens leak into anything consuming the gateway as plain text: a coding
/// agent receives a paragraph of deliberation where it expected a diff, and
/// tool-call parsing breaks on the surrounding prose.
const REASONING_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<thinking>",
    "<reasoning>",
    "<irm>",
    "<|thinking|>",
    "<|think|>",
    "<|reasoning|>",
    "<seed:think>",
    "<|start_of_thought|>",
    "◁think▷",
];

/// Families that reason aloud even when the template is unavailable.
///
/// A fallback for repositories whose GGUF metadata could not be read. Matched
/// against the repo id, so it catches re-quantizations that inherit the
/// behaviour along with the name.
const REASONING_FAMILIES: &[&str] = &[
    "deepseek-r1",
    "-r1-",
    "qwq",
    "qwen3", // thinking is on by default
    "marco-o1",
    "skywork-o1",
    "openthinker",
    "exaone-deep",
    "phi-4-reasoning",
    "magistral",
    "thinking",
    "-cot",
    "reasoner",
];

/// Whether this model narrates internal reasoning in its output.
///
/// Prefers the GGUF chat template — direct evidence of what the model emits —
/// and only falls back to the name when no template was published.
pub fn emits_reasoning_tokens(repo: &GgufRepo) -> bool {
    if let Some(template) = repo.gguf.as_ref().and_then(|g| g.chat_template.as_ref()) {
        let lower = template.to_lowercase();
        if REASONING_TOKENS.iter().any(|t| lower.contains(t)) {
            return true;
        }
        // A template was published and carries no reasoning markers. That is
        // stronger evidence than the name, so take it and stop.
        return false;
    }

    let id = repo.repo_id.to_lowercase();
    REASONING_FAMILIES.iter().any(|f| id.contains(f))
        || repo
            .tags
            .iter()
            .any(|t| REASONING_FAMILIES.iter().any(|f| t.to_lowercase().contains(f)))
}

/// Whether the publisher's conversions are known-good.
pub fn is_trusted_publisher(repo_id: &str) -> bool {
    let owner = repo_id.split('/').next().unwrap_or("").to_lowercase();
    TRUSTED_PUBLISHERS.iter().any(|p| owner == *p)
}

/// Words that mark a repository as work in progress.
///
/// Authors label these honestly, so the label is worth believing. None of them
/// belong in a list headed "recommended".
const EXPERIMENTAL_MARKERS: &[&str] = &[
    "test", "wip", "draft", "experimental", "untested", "broken", "alpha", "beta", "preview",
    "unstable", "scratch", "tmp", "temp", "debug", "abliterated", "uncensored", "frankenmerge",
];

fn looks_experimental(repo_id: &str) -> bool {
    let name = repo_id.split('/').next_back().unwrap_or("").to_lowercase();
    EXPERIMENTAL_MARKERS
        .iter()
        .any(|m| name.split(['-', '_', '.']).any(|part| part == *m))
}

/// Downloads below which a conversion has not been exercised enough to trust.
///
/// Popularity is a weak signal of quality but a strong one of *exposure*: a
/// conversion thousands of people have run has had its broken-template and
/// truncated-shard problems reported already.
const MIN_DOWNLOADS: u64 = 10_000;

/// Whether this repository belongs in "Sarathi Recommended".
///
/// `fits_here` says whether at least one quantization fits the machine's memory
/// budget — a recommendation the user cannot act on is not a recommendation.
/// Passing `true` when the budget is unknown keeps the section populated on
/// hardware Sarathi could not read, rather than emptying it for a detection
/// failure.
pub fn is_recommended(repo: &GgufRepo, fits_here: bool) -> bool {
    fits_here
        && is_trusted_publisher(&repo.repo_id)
        && !repo.is_lora_adapter
        && !looks_experimental(&repo.repo_id)
        && !emits_reasoning_tokens(repo)
        && repo.downloads >= MIN_DOWNLOADS
        // A repository with no readable GGUF metadata cannot be sized, templated,
        // or checked for reasoning tokens — too little is known to vouch for it.
        && repo.gguf.is_some()
        && !repo.quantizations.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::{GgufMeta, Quantization};

    fn repo(id: &str, downloads: u64, template: Option<&str>) -> GgufRepo {
        GgufRepo {
            repo_id: id.to_string(),
            author: id.split('/').next().unwrap_or("").to_string(),
            downloads,
            likes: 100,
            last_modified: "2026-01-01T00:00:00Z".to_string(),
            quantizations: vec![Quantization {
                label: "Q4_K_M".into(),
                filename: "m.gguf".into(),
                size_bytes: 4_000_000_000,
                is_sharded: false,
            }],
            gguf: Some(GgufMeta {
                total_parameters: 7_000_000_000,
                architecture: "qwen2".into(),
                context_length: 32768,
                chat_template: template.map(String::from),
                bos_token: None,
                eos_token: None,
            }),
            base_model: None,
            is_finetune: false,
            is_lora_adapter: false,
            tags: vec![],
        }
    }

    #[test]
    fn a_template_that_opens_a_think_block_is_caught() {
        let r = repo(
            "qwen/Qwen3-8B-GGUF",
            900_000,
            Some("{%- if enable_thinking %}<think>\n{%- endif %}"),
        );
        assert!(emits_reasoning_tokens(&r));
        assert!(!is_recommended(&r, true), "reasoning models stay out");
    }

    #[test]
    fn a_published_template_outranks_a_suspicious_name() {
        // Named like a reasoning family but its template emits none, which is
        // what the model will actually do.
        let r = repo(
            "bartowski/Qwen3-Coder-30B-GGUF",
            900_000,
            Some("{%- for m in messages %}<|im_start|>{{ m.role }}<|im_end|>{%- endfor %}"),
        );
        assert!(!emits_reasoning_tokens(&r), "the template is the evidence");
        assert!(is_recommended(&r, true));
    }

    #[test]
    fn the_name_decides_only_when_no_template_was_published() {
        let r = repo("someone/DeepSeek-R1-Distill-Qwen-7B-GGUF", 900_000, None);
        assert!(emits_reasoning_tokens(&r));
    }

    #[test]
    fn every_listed_reasoning_token_is_detected() {
        for token in REASONING_TOKENS {
            let template = format!("prefix {token} suffix");
            let r = repo("bartowski/Some-Model-GGUF", 900_000, Some(&template));
            assert!(emits_reasoning_tokens(&r), "missed {token}");
        }
    }

    #[test]
    fn an_unknown_publisher_is_not_recommended_however_popular() {
        let r = repo("randomuser/Some-Merge-GGUF", 5_000_000, Some("{{ messages }}"));
        assert!(!is_recommended(&r, true));
    }

    #[test]
    fn a_model_that_does_not_fit_is_never_recommended() {
        let r = repo("bartowski/Big-Model-GGUF", 900_000, Some("{{ messages }}"));
        assert!(is_recommended(&r, true));
        assert!(!is_recommended(&r, false), "cannot recommend what will not run");
    }

    #[test]
    fn experimental_names_are_refused() {
        for name in ["bartowski/Model-test-GGUF", "unsloth/Thing-experimental-GGUF"] {
            let r = repo(name, 900_000, Some("{{ messages }}"));
            assert!(!is_recommended(&r, true), "{name} should be refused");
        }

        // The marker must be a whole word: "latest" and "contest" are not "test".
        let fine = repo("bartowski/Model-latest-GGUF", 900_000, Some("{{ messages }}"));
        assert!(is_recommended(&fine, true), "substring matches must not fire");
    }

    #[test]
    fn adapters_and_barely_used_conversions_are_refused() {
        let mut adapter = repo("bartowski/Thing-GGUF", 900_000, Some("{{ messages }}"));
        adapter.is_lora_adapter = true;
        assert!(!is_recommended(&adapter, true));

        let obscure = repo("bartowski/Thing-GGUF", 12, Some("{{ messages }}"));
        assert!(!is_recommended(&obscure, true));
    }

    #[test]
    fn publisher_matching_is_exact_rather_than_a_prefix() {
        // "google-fan" must not inherit "google"'s standing.
        assert!(is_trusted_publisher("google/gemma-3-12b-it-GGUF"));
        assert!(!is_trusted_publisher("google-fan/gemma-3-12b-it-GGUF"));
        assert!(is_trusted_publisher("BARTOWSKI/Thing"), "matching is case-insensitive");
    }
}

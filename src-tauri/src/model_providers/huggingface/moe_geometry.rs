//! Expert geometry for Mixture-of-Experts models discovered on the Hub.
//!
//! The Hub's GGUF metadata (`discovery.rs::RawGguf`) exposes `total`,
//! `architecture` and `context_length` — but **not** expert counts. Without
//! them `estimator::split_moe_weights` cannot tell how much of a model is
//! routed experts, so a discovered MoE model is treated as dense and sized as
//! though every weight had to sit in VRAM.
//!
//! That mattered little while expert counts fed nothing. Now they decide
//! placement: how much of the model can live in system RAM, and therefore
//! whether it runs at all on a small card.
//!
//! This is a table of *known models*, matched on architecture plus parameter
//! count. It is deliberately small. Every entry is transcribed from the model's
//! published `config.json`, and an architecture that is not listed returns
//! `None` so the caller keeps the existing dense behaviour rather than
//! recommending against invented numbers.
//!
//! Once a model is downloaded, [`crate::ai_engine::gguf_meta`] reads the real
//! geometry out of the file header and this table stops being consulted. It
//! exists only to answer "will this run?" *before* committing to the download.

use crate::model_recommendation::traits::{ModelArchitecture, ModelMetadata};

/// A MoE model whose geometry has been verified against its published config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownMoe {
    /// GGUF `general.architecture`, lowercased.
    pub architecture: &'static str,
    /// Published parameter count, used to distinguish size variants that share
    /// an architecture string (`qwen3moe` covers both 30B-A3B and 235B-A22B).
    pub total_params: u64,
    pub num_experts: u32,
    pub active_experts: u32,
    /// Parameters used per token, as published.
    pub active_params: u64,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dimension: u32,
    pub hidden_size: u32,
}

/// How far a repo's reported parameter count may differ from the published
/// figure and still be considered the same model.
///
/// Quantized GGUF repos report totals that drift slightly from the original
/// checkpoint, but variants within one architecture differ by multiples, so a
/// loose band cannot confuse 30B-A3B with 235B-A22B.
const PARAM_MATCH_TOLERANCE: f64 = 0.15;

/// Verified MoE geometries.
///
/// Sources, all read from the model's `config.json` on the Hub:
///
/// - `openai/gpt-oss-20b` — `num_hidden_layers: 24`, `num_local_experts: 32`,
///   `num_experts_per_tok: 4`, `hidden_size: 2880`, `num_attention_heads: 64`,
///   `num_key_value_heads: 8`, `head_dim: 64`.
/// - `Qwen/Qwen3-30B-A3B` — `num_hidden_layers: 48`, `num_experts: 128`,
///   `num_experts_per_tok: 8`, `hidden_size: 2048`, `num_attention_heads: 32`,
///   `num_key_value_heads: 4`, `head_dim: 128`.
///
/// Mixtral is deliberately absent: its GGUF `general.architecture` is `llama`,
/// which would key against ordinary dense Llama models. It is already covered
/// by the static catalog entry.
const KNOWN_MOE: &[KnownMoe] = &[
    KnownMoe {
        architecture: "gpt-oss",
        total_params: 20_900_000_000,
        num_experts: 32,
        active_experts: 4,
        active_params: 3_600_000_000,
        num_layers: 24,
        num_attention_heads: 64,
        num_kv_heads: 8,
        head_dimension: 64,
        hidden_size: 2880,
    },
    KnownMoe {
        architecture: "qwen3moe",
        total_params: 30_500_000_000,
        num_experts: 128,
        active_experts: 8,
        active_params: 3_300_000_000,
        num_layers: 48,
        num_attention_heads: 32,
        num_kv_heads: 4,
        head_dimension: 128,
        hidden_size: 2048,
    },
];

/// Finds a verified geometry for a discovered repo.
///
/// `architecture` is the GGUF architecture string; `total_parameters` is the
/// count the Hub reports. Returns `None` for anything not verified, which the
/// caller must treat as "not known to be MoE" rather than "dense".
pub fn lookup(architecture: &str, total_parameters: u64) -> Option<&'static KnownMoe> {
    if total_parameters == 0 {
        return None;
    }
    let arch = architecture.trim().to_ascii_lowercase();

    KNOWN_MOE.iter().find(|entry| {
        entry.architecture == arch && within_tolerance(total_parameters, entry.total_params)
    })
}

fn within_tolerance(reported: u64, published: u64) -> bool {
    if published == 0 {
        return false;
    }
    let drift = (reported as f64 - published as f64).abs() / published as f64;
    drift <= PARAM_MATCH_TOLERANCE
}

impl KnownMoe {
    /// Applies this geometry to a model record built from Hub metadata.
    ///
    /// Overwrites the size-banded guesses `discovery::estimate_architecture`
    /// produces, because those are inferred from a parameter count and a MoE
    /// model's count is dominated by experts — a 30B MoE has the layer and head
    /// geometry of a far smaller dense model.
    pub fn apply(&self, model: &mut ModelMetadata) {
        model.architecture = ModelArchitecture::MixtureOfExperts {
            num_experts: self.num_experts,
            active_experts: self.active_experts,
        };
        model.active_parameters = Some(self.active_params);
        model.num_layers = self.num_layers;
        model.num_attention_heads = self.num_attention_heads;
        model.num_kv_heads = self.num_kv_heads;
        model.head_dimension = self.head_dimension;
        model.hidden_size = self.hidden_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_20b_is_recognised_as_moe() {
        let found = lookup("gpt-oss", 20_900_000_000).expect("gpt-oss-20b is a verified entry");

        assert_eq!(found.num_experts, 32);
        assert_eq!(found.active_experts, 4);
        assert_eq!(found.num_layers, 24);
    }

    #[test]
    fn qwen3_30b_a3b_is_recognised_as_moe() {
        let found = lookup("qwen3moe", 30_500_000_000).expect("Qwen3-30B-A3B is a verified entry");

        assert_eq!(found.num_experts, 128);
        assert_eq!(found.active_experts, 8);
        assert_eq!(found.num_layers, 48);
    }

    #[test]
    fn architecture_matching_is_case_insensitive_and_trims() {
        assert!(lookup("  GPT-OSS  ", 20_900_000_000).is_some());
    }

    /// Quantized repos report totals that drift from the original checkpoint.
    #[test]
    fn a_quantized_repos_drifting_parameter_count_still_matches() {
        for reported in [19_500_000_000u64, 20_900_000_000, 22_000_000_000] {
            assert!(
                lookup("gpt-oss", reported).is_some(),
                "{reported} should match gpt-oss-20b"
            );
        }
    }

    /// The reason the table is keyed on size as well as architecture.
    #[test]
    fn a_different_size_variant_of_the_same_architecture_does_not_match() {
        // Qwen3-235B-A22B shares the `qwen3moe` architecture but has entirely
        // different geometry, and is not a verified entry.
        assert!(lookup("qwen3moe", 235_000_000_000).is_none());
    }

    #[test]
    fn an_unknown_architecture_returns_nothing_rather_than_guessing() {
        assert!(lookup("llama", 8_000_000_000).is_none());
        assert!(lookup("deepseek2", 236_000_000_000).is_none());
        assert!(lookup("", 20_900_000_000).is_none());
    }

    #[test]
    fn a_repo_without_a_parameter_count_cannot_match() {
        assert!(lookup("gpt-oss", 0).is_none());
    }

    /// Mixtral's GGUF architecture is `llama`; keying it here would capture
    /// every dense Llama model.
    #[test]
    fn mixtral_is_not_keyed_under_the_llama_architecture() {
        assert!(lookup("llama", 46_700_000_000).is_none());
    }

    /// The geometry has to survive into the record the scorer reads, or the
    /// MoE path is still unreachable.
    #[test]
    fn applying_a_geometry_makes_the_record_report_moe() {
        let mut model = ModelMetadata {
            id: "unsloth/gpt-oss-20b-GGUF".to_string(),
            name: "gpt-oss 20B".to_string(),
            family: "gpt-oss".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 20_900_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 131072,
            vocab_size: 0,
            default_dtype: "unknown".to_string(),
            use_cases: vec![],
            catalog_version: "live".to_string(),
        };

        lookup("gpt-oss", model.total_parameters).unwrap().apply(&mut model);

        assert!(matches!(
            model.architecture,
            ModelArchitecture::MixtureOfExperts { num_experts: 32, active_experts: 4 }
        ));
        assert_eq!(model.active_parameters, Some(3_600_000_000));
        // The size-banded guesses must be replaced: a 20.9B parameter count
        // implies a much larger dense model than gpt-oss actually is.
        assert_eq!(model.num_layers, 24);
        assert_eq!(model.head_dimension, 64);
    }

    /// The split has to work on the applied record, since that is the whole
    /// reason the table exists.
    #[test]
    fn an_applied_geometry_produces_a_usable_expert_split() {
        use crate::model_recommendation::estimator::split_moe_weights;

        let mut model = ModelMetadata {
            id: "openai/gpt-oss-20b".to_string(),
            name: "gpt-oss 20B".to_string(),
            family: "gpt-oss".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 20_900_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 131072,
            vocab_size: 0,
            default_dtype: "unknown".to_string(),
            use_cases: vec![],
            catalog_version: "live".to_string(),
        };

        assert!(
            split_moe_weights(&model, 12_800_000_000).is_none(),
            "before the table is applied the record is dense — the bug this fixes"
        );

        lookup("gpt-oss", model.total_parameters).unwrap().apply(&mut model);

        let split = split_moe_weights(&model, 12_800_000_000).expect("now splittable");
        assert!(split.expert_bytes > split.resident_bytes * 4);
    }

    #[test]
    fn every_shipped_entry_is_internally_consistent() {
        for entry in KNOWN_MOE {
            assert!(entry.active_experts > 0, "{}", entry.architecture);
            assert!(
                entry.active_experts < entry.num_experts,
                "{}: active experts must be a strict subset",
                entry.architecture
            );
            assert!(
                entry.active_params < entry.total_params,
                "{}: active parameters must be fewer than total",
                entry.architecture
            );
            assert!(entry.num_layers > 0 && entry.num_kv_heads > 0, "{}", entry.architecture);
            assert!(
                entry.architecture.chars().all(|c| !c.is_ascii_uppercase()),
                "{}: keys are compared lowercased",
                entry.architecture
            );
        }
    }
}

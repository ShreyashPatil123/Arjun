//! Translating PEFT tensor names into the names llama.cpp expects.
//!
//! ## Why this exists
//!
//! A PEFT adapter and a GGUF adapter describe the same weights under different
//! names. PEFT follows the HuggingFace module path of the model it was trained
//! against:
//!
//! ```text
//! base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
//! ```
//!
//! llama.cpp names the same tensor by its role in the graph, and marks the two
//! low-rank factors with a suffix:
//!
//! ```text
//! blk.0.attn_q.weight.lora_a
//! ```
//!
//! Nothing about the weights changes — only the label. Getting the label wrong
//! is not a soft failure: `llama_adapter_lora_init` rejects an unrecognised
//! tensor name outright, so a mistranslation surfaces as an adapter that simply
//! refuses to load.
//!
//! ## Why the table is small
//!
//! Full model conversion has to name every tensor in the graph. A LoRA touches
//! only the projection matrices it was trained on — in practice the four
//! attention projections and the three feed-forward ones — so the mapping is a
//! short list per architecture rather than a full graph description.
//!
//! The names here follow llama.cpp's `gguf-py/gguf/tensor_mapping.py`. When a
//! new architecture is added upstream it must be added here too; an unknown
//! architecture is refused by name rather than guessed at, because a wrong guess
//! produces an adapter that loads and then degrades output in ways no error
//! message would explain.

use std::fmt;

/// The suffix llama.cpp gives the two factors of a low-rank update.
///
/// `delta_W = lora_b @ lora_a`, scaled by `alpha / rank`.
pub const LORA_A_SUFFIX: &str = "lora_a";
pub const LORA_B_SUFFIX: &str = "lora_b";

/// Which of the two low-rank factors a PEFT tensor holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraFactor {
    /// The down-projection, `[rank, in_features]`.
    A,
    /// The up-projection, `[out_features, rank]`.
    B,
}

impl LoraFactor {
    pub fn suffix(self) -> &'static str {
        match self {
            LoraFactor::A => LORA_A_SUFFIX,
            LoraFactor::B => LORA_B_SUFFIX,
        }
    }
}

/// A PEFT tensor name resolved to its llama.cpp equivalent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedTensor {
    /// The full GGUF tensor name, suffix included.
    pub gguf_name: String,
    pub factor: LoraFactor,
}

/// Why a PEFT tensor name could not be translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// The architecture has no mapping table here.
    UnknownArchitecture { arch: String },
    /// The name carries no `lora_A`/`lora_B` marker, so it is not a LoRA factor.
    NotALoraTensor { name: String },
    /// The layer index was missing or unparseable.
    MalformedLayerIndex { name: String },
    /// The module is real PEFT output but has no llama.cpp counterpart.
    UnmappedModule { name: String, module: String },
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Phrased for someone installing a skill, not someone reading GGUF
            // internals: the actionable fact is which model family is missing.
            MapError::UnknownArchitecture { arch } => write!(
                f,
                "This adapter is for a '{arch}' model, which Sarathi cannot convert yet."
            ),
            MapError::NotALoraTensor { name } => {
                write!(f, "'{name}' is not a LoRA weight.")
            }
            MapError::MalformedLayerIndex { name } => {
                write!(f, "'{name}' does not name a layer Sarathi can place.")
            }
            MapError::UnmappedModule { name, module } => write!(
                f,
                "This adapter modifies '{module}', which llama.cpp cannot apply \
                 as a LoRA. (from '{name}')"
            ),
        }
    }
}

impl std::error::Error for MapError {}

/// One architecture's module-name translations.
///
/// `layered` entries sit inside a numbered block and become `blk.{i}.{to}`.
/// `standalone` entries exist once per model and map to a bare name.
struct ArchMap {
    layered: &'static [(&'static str, &'static str)],
    standalone: &'static [(&'static str, &'static str)],
}

/// The projection names shared by every Llama-derived architecture.
///
/// Llama, Mistral, Qwen2/3, Gemma and Phi-3 all use HuggingFace's
/// `LlamaAttention`/`LlamaMLP` module names, so they share one table rather than
/// repeating it five times. Where a family genuinely diverges it gets its own
/// entry below.
const LLAMA_LIKE_LAYERED: &[(&str, &str)] = &[
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
];

/// Tensors that exist once per model rather than once per block.
///
/// Adapters that tune the embedding or the output head are uncommon but legal,
/// and PEFT writes them without a layer index.
const LLAMA_LIKE_STANDALONE: &[(&str, &str)] = &[
    ("embed_tokens", "token_embd"),
    ("lm_head", "output"),
];

/// Architectures this converter can translate, keyed by the
/// `general.architecture` value read from the base model's GGUF header.
fn arch_map(arch: &str) -> Option<ArchMap> {
    match arch {
        "llama" | "mistral" | "qwen2" | "qwen3" | "gemma" | "gemma2" | "phi3" => Some(ArchMap {
            layered: LLAMA_LIKE_LAYERED,
            standalone: LLAMA_LIKE_STANDALONE,
        }),
        _ => None,
    }
}

/// True when this converter can handle the given base architecture.
pub fn supports_architecture(arch: &str) -> bool {
    arch_map(arch).is_some()
}

/// Every architecture name this converter recognises.
///
/// Used to tell the user what *would* work, rather than only what did not.
pub fn supported_architectures() -> &'static [&'static str] {
    &["llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "phi3"]
}

/// Identifies which low-rank factor a PEFT name holds, and returns the name with
/// the factor marker and any `.weight` suffix removed.
///
/// PEFT writes `...lora_A.weight`; some exporters drop the trailing `.weight`,
/// so both spellings are accepted.
fn split_factor(name: &str) -> Option<(&str, LoraFactor)> {
    // Ordered longest-first so `.lora_A.weight` is tried before `.lora_A`.
    const MARKERS: &[(&str, LoraFactor)] = &[
        (".lora_A.weight", LoraFactor::A),
        (".lora_B.weight", LoraFactor::B),
        (".lora_A", LoraFactor::A),
        (".lora_B", LoraFactor::B),
    ];

    for (marker, factor) in MARKERS {
        if let Some(stem) = name.strip_suffix(marker) {
            return Some((stem, *factor));
        }
    }
    None
}

/// Strips the wrapper prefixes PEFT puts in front of the real module path.
///
/// A PEFT checkpoint nests the base model inside its own wrapper, which shows up
/// as a variable number of `base_model.` / `model.` segments before the part
/// that actually identifies the tensor. Trimming by prefix rather than by a
/// fixed count keeps this working across PEFT versions, which have not been
/// consistent about the depth.
fn strip_wrappers(stem: &str) -> &str {
    let mut rest = stem;
    loop {
        if let Some(next) = rest.strip_prefix("base_model.") {
            rest = next;
        } else if let Some(next) = rest.strip_prefix("model.") {
            rest = next;
        } else if let Some(next) = rest.strip_prefix("transformer.") {
            rest = next;
        } else {
            return rest;
        }
    }
}

/// Translates one PEFT tensor name into its llama.cpp equivalent.
///
/// `arch` is the `general.architecture` of the **base model**, not of the
/// adapter — a LoRA GGUF inherits the architecture of what it attaches to.
pub fn map_tensor_name(arch: &str, peft_name: &str) -> Result<MappedTensor, MapError> {
    let map = arch_map(arch).ok_or_else(|| MapError::UnknownArchitecture {
        arch: arch.to_string(),
    })?;

    let (stem, factor) = split_factor(peft_name).ok_or_else(|| MapError::NotALoraTensor {
        name: peft_name.to_string(),
    })?;

    let path = strip_wrappers(stem);

    // Layered tensors: `layers.{i}.{module}`.
    if let Some(after) = path.strip_prefix("layers.") {
        let (index_text, module) = after.split_once('.').ok_or_else(|| {
            MapError::MalformedLayerIndex { name: peft_name.to_string() }
        })?;

        let index: u32 = index_text.parse().map_err(|_| MapError::MalformedLayerIndex {
            name: peft_name.to_string(),
        })?;

        let target = map
            .layered
            .iter()
            .find(|(from, _)| *from == module)
            .map(|(_, to)| *to)
            .ok_or_else(|| MapError::UnmappedModule {
                name: peft_name.to_string(),
                module: module.to_string(),
            })?;

        return Ok(MappedTensor {
            gguf_name: format!("blk.{index}.{target}.weight.{}", factor.suffix()),
            factor,
        });
    }

    // Standalone tensors carry no layer index.
    if let Some((_, target)) = map.standalone.iter().find(|(from, _)| *from == path) {
        return Ok(MappedTensor {
            gguf_name: format!("{target}.weight.{}", factor.suffix()),
            factor,
        });
    }

    Err(MapError::UnmappedModule {
        name: peft_name.to_string(),
        module: path.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of name PEFT actually writes for a Llama-family adapter.
    #[test]
    fn a_standard_attention_projection_maps_to_its_block() {
        let mapped = map_tensor_name(
            "llama",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        )
        .unwrap();

        assert_eq!(mapped.gguf_name, "blk.0.attn_q.weight.lora_a");
        assert_eq!(mapped.factor, LoraFactor::A);
    }

    #[test]
    fn the_b_factor_is_distinguished_from_the_a_factor() {
        let a = map_tensor_name("llama", "base_model.model.model.layers.7.mlp.down_proj.lora_A.weight")
            .unwrap();
        let b = map_tensor_name("llama", "base_model.model.model.layers.7.mlp.down_proj.lora_B.weight")
            .unwrap();

        assert_eq!(a.gguf_name, "blk.7.ffn_down.weight.lora_a");
        assert_eq!(b.gguf_name, "blk.7.ffn_down.weight.lora_b");
        assert_ne!(a.factor, b.factor);
    }

    /// Every module in the table must translate — a gap here is an adapter that
    /// silently loses part of its training.
    #[test]
    fn every_llama_like_module_is_mappable() {
        for (peft_module, expected) in LLAMA_LIKE_LAYERED {
            let name =
                format!("base_model.model.model.layers.3.{peft_module}.lora_A.weight");
            let mapped = map_tensor_name("llama", &name)
                .unwrap_or_else(|e| panic!("{peft_module} should map: {e}"));

            assert_eq!(mapped.gguf_name, format!("blk.3.{expected}.weight.lora_a"));
        }
    }

    /// PEFT versions differ in how deeply they nest the base model, so the
    /// prefix trimming must not assume a fixed segment count.
    #[test]
    fn wrapper_depth_does_not_change_the_result() {
        let variants = [
            "base_model.model.model.layers.2.self_attn.k_proj.lora_A.weight",
            "base_model.model.layers.2.self_attn.k_proj.lora_A.weight",
            "model.layers.2.self_attn.k_proj.lora_A.weight",
            "layers.2.self_attn.k_proj.lora_A.weight",
        ];

        for name in variants {
            let mapped = map_tensor_name("llama", name)
                .unwrap_or_else(|e| panic!("{name} should map: {e}"));
            assert_eq!(mapped.gguf_name, "blk.2.attn_k.weight.lora_a", "for {name}");
        }
    }

    /// Some exporters omit the trailing `.weight`.
    #[test]
    fn the_weight_suffix_is_optional() {
        let mapped =
            map_tensor_name("llama", "base_model.model.model.layers.1.mlp.up_proj.lora_B").unwrap();
        assert_eq!(mapped.gguf_name, "blk.1.ffn_up.weight.lora_b");
    }

    #[test]
    fn double_digit_layer_indices_are_parsed() {
        let mapped = map_tensor_name(
            "llama",
            "base_model.model.model.layers.31.self_attn.v_proj.lora_A.weight",
        )
        .unwrap();
        assert_eq!(mapped.gguf_name, "blk.31.attn_v.weight.lora_a");
    }

    #[test]
    fn embedding_and_output_tensors_carry_no_block_index() {
        let embed =
            map_tensor_name("llama", "base_model.model.model.embed_tokens.lora_A.weight").unwrap();
        assert_eq!(embed.gguf_name, "token_embd.weight.lora_a");

        let head = map_tensor_name("llama", "base_model.model.lm_head.lora_B.weight").unwrap();
        assert_eq!(head.gguf_name, "output.weight.lora_b");
    }

    #[test]
    fn qwen_and_mistral_share_the_llama_table() {
        for arch in ["mistral", "qwen2", "qwen3", "gemma", "phi3"] {
            let mapped = map_tensor_name(
                arch,
                "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
            )
            .unwrap_or_else(|e| panic!("{arch} should map: {e}"));
            assert_eq!(mapped.gguf_name, "blk.0.attn_output.weight.lora_a");
        }
    }

    /// An unknown architecture is refused rather than guessed, and the message
    /// names the family so the user knows what they have.
    #[test]
    fn an_unknown_architecture_is_refused_by_name() {
        let err = map_tensor_name(
            "mamba",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        )
        .unwrap_err();

        assert_eq!(err, MapError::UnknownArchitecture { arch: "mamba".into() });
        let text = err.to_string();
        assert!(text.contains("mamba"), "should name the architecture: {text}");
        assert!(!text.contains("blk."), "should not leak GGUF internals: {text}");
    }

    /// PEFT checkpoints contain non-LoRA entries; they must be skipped, not
    /// mistranslated.
    #[test]
    fn a_non_lora_tensor_is_rejected() {
        let err = map_tensor_name("llama", "base_model.model.model.layers.0.self_attn.q_proj.weight")
            .unwrap_err();
        assert!(matches!(err, MapError::NotALoraTensor { .. }), "got {err:?}");
    }

    #[test]
    fn an_unmapped_module_names_what_it_could_not_place() {
        let err = map_tensor_name(
            "llama",
            "base_model.model.model.layers.0.self_attn.rotary_emb.lora_A.weight",
        )
        .unwrap_err();

        match &err {
            MapError::UnmappedModule { module, .. } => {
                assert_eq!(module, "self_attn.rotary_emb");
            }
            other => panic!("expected UnmappedModule, got {other:?}"),
        }
        assert!(err.to_string().contains("rotary_emb"));
    }

    #[test]
    fn a_non_numeric_layer_index_is_refused() {
        let err = map_tensor_name(
            "llama",
            "base_model.model.model.layers.first.self_attn.q_proj.lora_A.weight",
        )
        .unwrap_err();
        assert!(matches!(err, MapError::MalformedLayerIndex { .. }), "got {err:?}");
    }

    #[test]
    fn supported_architectures_all_actually_map() {
        for arch in supported_architectures() {
            assert!(supports_architecture(arch), "{arch} listed but unsupported");
        }
        assert!(!supports_architecture("mamba"));
    }
}

//! Reading `adapter_config.json`, and deciding early whether an adapter is
//! convertible at all.
//!
//! ## Why this runs before the weights are fetched
//!
//! Conversion is automatic, so nobody is watching when it fails. The cheapest
//! possible failure is one that happens before a hundred megabytes are
//! downloaded — and every disqualifying property of an adapter is stated in a
//! few kilobytes of JSON. Everything here is decidable from that file alone.
//!
//! ## What is refused, and why
//!
//! **DoRA** (`use_dora`) decomposes each weight into magnitude and direction and
//! trains them separately. It is not a low-rank update, carries an extra
//! magnitude vector per module, and llama.cpp's adapter path has nowhere to put
//! it. A DoRA checkpoint converted as if it were LoRA loads happily and produces
//! quietly wrong output, which is worse than refusing it.
//!
//! **Non-LoRA PEFT types** (prefix tuning, IA3, prompt tuning) are different
//! mechanisms entirely, not a different encoding of the same thing.
//!
//! **rsLoRA** (`use_rslora`) *is* convertible, and is handled rather than
//! refused — see [`PeftConfig::effective_alpha`].

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// The subset of `adapter_config.json` that affects conversion.
///
/// PEFT writes many more fields; unknown ones are ignored rather than rejected,
/// because the file gains keys across PEFT releases and an adapter should not
/// stop converting because a version added a setting we do not read.
#[derive(Debug, Clone, Deserialize)]
pub struct PeftConfig {
    #[serde(default)]
    pub peft_type: Option<String>,

    /// LoRA rank.
    #[serde(default)]
    pub r: Option<u32>,

    #[serde(default)]
    pub lora_alpha: Option<f32>,

    #[serde(default)]
    pub use_dora: bool,

    #[serde(default)]
    pub use_rslora: bool,

    #[serde(default)]
    pub base_model_name_or_path: Option<String>,

    #[serde(default)]
    pub target_modules: Vec<String>,
}

/// Why an adapter cannot be converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unconvertible {
    Dora,
    NotLora { peft_type: String },
}

impl std::fmt::Display for Unconvertible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Both messages name the thing the user chose, then say plainly that
            // it will not work here — no PEFT jargon beyond the name itself.
            Unconvertible::Dora => write!(
                f,
                "This is a DoRA adapter. Sarathi can only use LoRA adapters, and \
                 converting a DoRA one would change how the model answers without \
                 any sign that something is wrong."
            ),
            Unconvertible::NotLora { peft_type } => write!(
                f,
                "This is a '{peft_type}' adapter, not a LoRA one, so it cannot be \
                 added as a skill."
            ),
        }
    }
}

impl std::error::Error for Unconvertible {}

impl PeftConfig {
    /// Reads and parses `adapter_config.json` from an adapter directory.
    pub fn read_from_dir(adapter_dir: &Path) -> Result<Self> {
        let path = adapter_dir.join("adapter_config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("could not read {}", path.display()))?;

        serde_json::from_str(&text)
            .with_context(|| format!("{} is not valid adapter configuration", path.display()))
    }

    /// Decides whether this adapter can become a GGUF LoRA.
    ///
    /// A missing `peft_type` is accepted: some exporters omit it, and the
    /// tensor names are the real evidence of what the file holds.
    pub fn check_convertible(&self) -> Result<(), Unconvertible> {
        if self.use_dora {
            return Err(Unconvertible::Dora);
        }

        match self.peft_type.as_deref() {
            None => Ok(()),
            Some(kind) if kind.eq_ignore_ascii_case("lora") => Ok(()),
            Some(kind) => Err(Unconvertible::NotLora { peft_type: kind.to_string() }),
        }
    }

    /// The alpha to write into the GGUF, compensated for rsLoRA.
    ///
    /// llama.cpp always applies `scale = alpha / rank`. Standard LoRA agrees.
    /// rsLoRA instead uses `alpha / sqrt(rank)`, so writing its raw alpha would
    /// under-apply the adapter by a factor of `sqrt(rank)` — a subtle,
    /// output-only error with nothing to indicate it.
    ///
    /// Scaling alpha by `sqrt(rank)` makes llama.cpp's fixed formula produce
    /// rsLoRA's intended scale:
    ///
    /// ```text
    /// (alpha * sqrt(r)) / r  ==  alpha / sqrt(r)
    /// ```
    pub fn effective_alpha(&self) -> f32 {
        // PEFT's own default when the field is absent.
        let alpha = self.lora_alpha.unwrap_or(8.0);

        match (self.use_rslora, self.r) {
            (true, Some(rank)) if rank > 0 => alpha * (rank as f32).sqrt(),
            _ => alpha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> PeftConfig {
        serde_json::from_str(json).expect("should parse")
    }

    #[test]
    fn a_standard_lora_config_is_convertible() {
        let config = parse(r#"{"peft_type":"LORA","r":16,"lora_alpha":32.0}"#);
        assert!(config.check_convertible().is_ok());
        assert_eq!(config.effective_alpha(), 32.0);
    }

    #[test]
    fn lowercase_peft_type_is_accepted() {
        assert!(parse(r#"{"peft_type":"lora"}"#).check_convertible().is_ok());
    }

    /// Some exporters omit peft_type entirely; tensor names decide instead.
    #[test]
    fn a_missing_peft_type_is_not_a_refusal() {
        assert!(parse(r#"{"r":8}"#).check_convertible().is_ok());
    }

    #[test]
    fn a_dora_adapter_is_refused_before_download() {
        let config = parse(r#"{"peft_type":"LORA","use_dora":true,"r":16}"#);
        let err = config.check_convertible().unwrap_err();

        assert_eq!(err, Unconvertible::Dora);
        let text = err.to_string();
        assert!(text.contains("DoRA"), "should name it: {text}");
        assert!(
            text.contains("without any sign") || text.contains("wrong"),
            "should say why it is refused rather than converted: {text}"
        );
    }

    #[test]
    fn a_non_lora_peft_type_is_named_in_the_refusal() {
        let err = parse(r#"{"peft_type":"PREFIX_TUNING"}"#).check_convertible().unwrap_err();
        assert_eq!(err, Unconvertible::NotLora { peft_type: "PREFIX_TUNING".into() });
        assert!(err.to_string().contains("PREFIX_TUNING"));
    }

    /// The compensation must make llama.cpp's fixed `alpha / rank` reproduce
    /// rsLoRA's `alpha / sqrt(rank)`.
    #[test]
    fn rslora_alpha_is_compensated_for_llamacpp_scaling() {
        let config = parse(r#"{"peft_type":"LORA","r":16,"lora_alpha":32.0,"use_rslora":true}"#);

        let written = config.effective_alpha();
        let rank = 16.0f32;

        let llamacpp_scale = written / rank;
        let rslora_intended = 32.0 / rank.sqrt();

        assert!(
            (llamacpp_scale - rslora_intended).abs() < 1e-5,
            "llama.cpp would apply {llamacpp_scale}, rsLoRA intends {rslora_intended}"
        );
    }

    #[test]
    fn standard_lora_alpha_is_passed_through_untouched() {
        let config = parse(r#"{"peft_type":"LORA","r":64,"lora_alpha":16.0}"#);
        assert_eq!(config.effective_alpha(), 16.0);
    }

    /// A zero or absent rank must not produce NaN or a divide-by-zero.
    #[test]
    fn rslora_with_an_unusable_rank_falls_back_to_the_raw_alpha() {
        assert_eq!(
            parse(r#"{"lora_alpha":8.0,"use_rslora":true,"r":0}"#).effective_alpha(),
            8.0
        );
        assert_eq!(
            parse(r#"{"lora_alpha":8.0,"use_rslora":true}"#).effective_alpha(),
            8.0
        );
    }

    #[test]
    fn a_config_with_unknown_future_fields_still_parses() {
        let config = parse(
            r#"{"peft_type":"LORA","r":8,"some_future_setting":true,"nested":{"a":1}}"#,
        );
        assert!(config.check_convertible().is_ok());
        assert_eq!(config.r, Some(8));
    }

    #[test]
    fn target_modules_and_base_model_are_captured_for_reporting() {
        let config = parse(
            r#"{"peft_type":"LORA","base_model_name_or_path":"Qwen/Qwen3-4B",
                "target_modules":["q_proj","v_proj"]}"#,
        );
        assert_eq!(config.base_model_name_or_path.as_deref(), Some("Qwen/Qwen3-4B"));
        assert_eq!(config.target_modules, vec!["q_proj", "v_proj"]);
    }

    #[test]
    fn a_missing_config_file_reports_the_path() {
        let err = PeftConfig::read_from_dir(Path::new("not/a/real/dir")).unwrap_err().to_string();
        assert!(err.contains("could not read"), "got: {err}");
    }
}

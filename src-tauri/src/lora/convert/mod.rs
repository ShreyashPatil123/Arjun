//! Converting a PEFT safetensors LoRA adapter into a GGUF one, locally.
//!
//! ## Why this exists
//!
//! Almost every LoRA adapter published on HuggingFace ships as PEFT
//! safetensors. llama.cpp loads only GGUF. Until now
//! [`crate::lora::validator`] could recognise that gap and name it —
//! `RequiresConversion` — but nothing could close it, so the browse page told
//! users outright that Sarathi "cannot convert yet".
//!
//! This module closes it without Python, without torch, and without network
//! access: the whole conversion is a name translation plus a re-encode of
//! tensors that are already on disk.
//!
//! ## Pipeline
//!
//! ```text
//! adapter_config.json ──► peft_config  (refuse DoRA and non-LoRA early)
//! base model GGUF     ──► arch         (which family are we targeting?)
//! adapter_model.safetensors ──► safetensors_reader (weights, widened if bf16)
//!                              │
//!                              ▼
//!                          tensor_map   (PEFT names ──► llama.cpp names)
//!                              │
//!                              ▼
//!                          gguf_writer  (temp file ──► atomic rename)
//! ```
//!
//! ## Atomicity
//!
//! Conversion runs unattended as part of installing an adapter, so a crash or a
//! killed process must not leave a `.gguf` that looks installed. The file is
//! built under a temporary name in the destination directory and renamed only
//! once it is complete and closed — a rename within one directory is atomic on
//! both Windows and Unix, so the adapter either exists whole or not at all.

pub mod arch;
pub mod gguf_writer;
pub mod peft_config;
pub mod safetensors_reader;
pub mod tensor_map;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use gguf_writer::{GgufWriter, TensorData};
use peft_config::PeftConfig;
use tensor_map::{map_tensor_name, MapError};

/// The filename PEFT uses for adapter weights.
const ADAPTER_WEIGHTS: &str = "adapter_model.safetensors";

/// What the converted adapter is written as, inside the adapter's own folder.
pub const CONVERTED_FILENAME: &str = "adapter.gguf";

/// What a conversion produced, for logging and for the manifest.
#[derive(Debug, Clone)]
pub struct ConversionSummary {
    pub output: PathBuf,
    /// Base model family the adapter was converted against.
    pub architecture: String,
    /// How many LoRA tensors were written.
    pub tensors_written: usize,
    /// Entries in the safetensors file that were not LoRA factors and were
    /// skipped rather than translated.
    pub non_lora_skipped: usize,
    pub alpha: f32,
    pub output_bytes: u64,
    /// LoRA rank the adapter declared, when it declared one.
    ///
    /// Carried through for the manifest: it is the single most useful number for
    /// judging an adapter's weight, and re-reading it later would mean parsing
    /// `adapter_config.json` again after this function has already deleted the
    /// weights it belonged to.
    pub rank: Option<u32>,
    /// Modules the adapter declared it targets, e.g. `q_proj`, `v_proj`.
    pub target_modules: Vec<String>,
}

/// Converts the PEFT adapter in `adapter_dir` into a GGUF LoRA.
///
/// `base_gguf` is the model the adapter will attach to; its header supplies the
/// architecture. The result is written to `adapter_dir/adapter.gguf`.
pub fn convert_adapter(adapter_dir: &Path, base_gguf: &Path) -> Result<ConversionSummary> {
    let config = PeftConfig::read_from_dir(adapter_dir)?;

    // Cheapest check first: this is decidable from a few kilobytes of JSON.
    config.check_convertible()?;

    let architecture = arch::read_architecture(base_gguf)?;
    if !tensor_map::supports_architecture(&architecture) {
        bail!(
            "This adapter is for a '{}' model, which Sarathi cannot convert yet. \
             Supported model families: {}.",
            architecture,
            tensor_map::supported_architectures().join(", ")
        );
    }

    let weights_path = adapter_dir.join(ADAPTER_WEIGHTS);
    if !weights_path.is_file() {
        bail!(
            "No adapter weights found at {} — the download looks incomplete.",
            weights_path.display()
        );
    }

    let raw = safetensors_reader::read_adapter_tensors(&weights_path)?;

    let alpha = config.effective_alpha();
    let mut writer = GgufWriter::new();
    writer
        // A LoRA GGUF inherits the architecture of what it attaches to; llama.cpp
        // checks this against the loaded model before binding.
        .metadata_str("general.architecture", architecture.clone())
        .metadata_str("general.type", "adapter")
        .metadata_str("adapter.type", "lora")
        .metadata_f32("adapter.lora.alpha", alpha);

    let mut non_lora_skipped = 0usize;

    for tensor in raw {
        match map_tensor_name(&architecture, &tensor.name) {
            Ok(mapped) => {
                writer.add_tensor(TensorData {
                    name: mapped.gguf_name,
                    dims: tensor.dims,
                    dtype: tensor.dtype,
                    data: tensor.data,
                });
            }
            // PEFT checkpoints legitimately carry non-LoRA entries. Skipping
            // them is correct; failing on them would reject valid adapters.
            Err(MapError::NotALoraTensor { .. }) => non_lora_skipped += 1,
            Err(other) => {
                return Err(anyhow::Error::new(other).context(
                    "this adapter changes a part of the model Sarathi cannot convert",
                ))
            }
        }
    }

    if writer.tensor_count() == 0 {
        bail!(
            "{} holds no LoRA weights, so there is nothing to convert.",
            weights_path.display()
        );
    }

    let output = adapter_dir.join(CONVERTED_FILENAME);
    write_atomically(&writer, &output)?;

    let output_bytes = std::fs::metadata(&output)
        .with_context(|| format!("could not stat {}", output.display()))?
        .len();

    let summary = ConversionSummary {
        output,
        architecture,
        tensors_written: writer.tensor_count(),
        non_lora_skipped,
        alpha,
        output_bytes,
        rank: config.r,
        target_modules: config.target_modules.clone(),
    };

    log::info!(
        "[LORA_CONVERT] {} → {} tensors, arch {}, alpha {:.2}, {} bytes ({} non-LoRA entries skipped)",
        adapter_dir.display(),
        summary.tensors_written,
        summary.architecture,
        summary.alpha,
        summary.output_bytes,
        summary.non_lora_skipped,
    );

    Ok(summary)
}

/// Writes through a temporary file in the same directory, then renames.
///
/// Same directory matters: a rename across filesystems is a copy, which is not
/// atomic and would reintroduce the torn-file case this exists to prevent.
fn write_atomically(writer: &GgufWriter, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("{} has no parent directory", destination.display()))?;

    let temp = parent.join(format!(
        ".{}.partial",
        destination
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("adapter.gguf")
    ));

    // A leftover from an interrupted run must not be appended to or mistaken
    // for progress.
    if temp.exists() {
        std::fs::remove_file(&temp)
            .with_context(|| format!("could not clear {}", temp.display()))?;
    }

    writer.write_to(&temp)?;

    // Windows rename fails if the destination exists, so a reconversion has to
    // clear it first. The window between the two is why the temp file, not the
    // destination, is the one holding valid content until the last moment.
    if destination.exists() {
        std::fs::remove_file(destination)
            .with_context(|| format!("could not replace {}", destination.display()))?;
    }

    std::fs::rename(&temp, destination).with_context(|| {
        format!("could not move {} into place at {}", temp.display(), destination.display())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_manager::gguf::verify_is_lora_adapter;

    /// Minimal GGUF header declaring only an architecture — enough for `arch`.
    fn write_base_gguf(path: &Path, architecture: &str) {
        let key = "general.architecture";
        let mut kvs = Vec::new();
        kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
        kvs.extend_from_slice(key.as_bytes());
        kvs.extend_from_slice(&8u32.to_le_bytes()); // STRING
        kvs.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        kvs.extend_from_slice(architecture.as_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&kvs);

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// Builds a real safetensors file holding the named f32 tensors.
    fn write_safetensors(path: &Path, names: &[&str], rows: usize, cols: usize) {
        let element_count = rows * cols;
        let byte_len = element_count * 4;

        let mut header = String::from("{");
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let start = i * byte_len;
            let end = start + byte_len;
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":[{rows},{cols}],"data_offsets":[{start},{end}]}}"#
            ));
        }
        header.push('}');

        // safetensors requires the header to be padded to 8 bytes.
        while header.len() % 8 != 0 {
            header.push(' ');
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for i in 0..names.len() {
            for j in 0..element_count {
                bytes.extend_from_slice(&((i * element_count + j) as f32).to_le_bytes());
            }
        }

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    struct Fixture {
        adapter_dir: PathBuf,
        base_gguf: PathBuf,
    }

    /// Shape of each fixture tensor.
    ///
    /// Deliberately large enough that the file clears the 100 KB floor
    /// `validator::validate_adapter` uses to reject placeholder downloads —
    /// 128 x 256 f32 is 128 KB per tensor. A smaller fixture would be classified
    /// `Incompatible` before conversion ever ran, so the tests would not be
    /// exercising the path a real adapter takes.
    const FIXTURE_ROWS: usize = 128;
    const FIXTURE_COLS: usize = 256;

    fn fixture(name: &str, architecture: &str, config_json: &str, tensors: &[&str]) -> Fixture {
        let root = std::env::temp_dir().join(format!("sarathi_convert_{name}"));
        let _ = std::fs::remove_dir_all(&root);

        let adapter_dir = root.join("adapters").join("coding");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        std::fs::write(adapter_dir.join("adapter_config.json"), config_json).unwrap();
        write_safetensors(
            &adapter_dir.join(ADAPTER_WEIGHTS),
            tensors,
            FIXTURE_ROWS,
            FIXTURE_COLS,
        );

        let base_gguf = root.join("base").join("model.gguf");
        write_base_gguf(&base_gguf, architecture);

        Fixture { adapter_dir, base_gguf }
    }

    #[test]
    fn a_peft_adapter_becomes_a_loadable_lora_gguf() {
        let f = fixture(
            "happy",
            "llama",
            r#"{"peft_type":"LORA","r":8,"lora_alpha":16.0}"#,
            &[
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            ],
        );

        let summary = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();

        assert_eq!(summary.tensors_written, 2);
        assert_eq!(summary.architecture, "llama");
        assert_eq!(summary.alpha, 16.0);
        assert!(summary.output.is_file());

        // The decisive check: the app's own verifier must accept the result.
        verify_is_lora_adapter(&summary.output).unwrap();
    }

    /// The validator is what decides whether an adapter is installable, so a
    /// converted adapter must flip it from RequiresConversion to Compatible.
    #[test]
    fn conversion_makes_the_validator_report_compatible() {
        use crate::lora::validator::{validate_adapter, AdapterRuntimeStatus};

        let f = fixture(
            "validator",
            "qwen3",
            r#"{"peft_type":"LORA","r":8,"lora_alpha":16.0}"#,
            &["base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"],
        );

        let before = validate_adapter(&f.adapter_dir);
        assert_eq!(before.status, AdapterRuntimeStatus::RequiresConversion);

        convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();

        let after = validate_adapter(&f.adapter_dir);
        assert_eq!(after.status, AdapterRuntimeStatus::Compatible);
    }

    #[test]
    fn non_lora_entries_are_skipped_rather_than_failing_the_conversion() {
        let f = fixture(
            "skip",
            "llama",
            r#"{"peft_type":"LORA","r":8,"lora_alpha":16.0}"#,
            &[
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                // Real PEFT checkpoints carry entries like this.
                "base_model.model.model.layers.0.self_attn.q_proj.weight",
            ],
        );

        let summary = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();
        assert_eq!(summary.tensors_written, 1);
        assert_eq!(summary.non_lora_skipped, 1);
    }

    #[test]
    fn a_dora_adapter_is_refused_without_writing_anything() {
        let f = fixture(
            "dora",
            "llama",
            r#"{"peft_type":"LORA","use_dora":true,"r":8}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );

        let err = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap_err().to_string();
        assert!(err.contains("DoRA"), "got: {err}");
        assert!(
            !f.adapter_dir.join(CONVERTED_FILENAME).exists(),
            "nothing should be written on refusal"
        );
    }

    #[test]
    fn an_unsupported_architecture_names_what_is_supported() {
        let f = fixture(
            "mamba",
            "mamba",
            r#"{"peft_type":"LORA","r":8}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );

        let err = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap_err().to_string();
        assert!(err.contains("mamba"), "should name the family: {err}");
        assert!(err.contains("llama"), "should list what works: {err}");
    }

    #[test]
    fn an_adapter_with_no_lora_weights_is_reported_rather_than_written_empty() {
        let f = fixture(
            "empty",
            "llama",
            r#"{"peft_type":"LORA","r":8}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.weight"],
        );

        let err = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap_err().to_string();
        assert!(err.contains("nothing to convert"), "got: {err}");
        assert!(!f.adapter_dir.join(CONVERTED_FILENAME).exists());
    }

    #[test]
    fn missing_weights_are_reported_as_an_incomplete_download() {
        let f = fixture(
            "noweights",
            "llama",
            r#"{"peft_type":"LORA","r":8}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );
        std::fs::remove_file(f.adapter_dir.join(ADAPTER_WEIGHTS)).unwrap();

        let err = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap_err().to_string();
        assert!(err.contains("incomplete"), "got: {err}");
    }

    /// Reconversion must overwrite cleanly — Windows rename refuses an existing
    /// destination, so this exercises the replace path.
    #[test]
    fn converting_twice_replaces_the_previous_output() {
        let f = fixture(
            "twice",
            "llama",
            r#"{"peft_type":"LORA","r":8,"lora_alpha":16.0}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );

        let first = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();
        let second = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();

        assert_eq!(first.output_bytes, second.output_bytes);
        verify_is_lora_adapter(&second.output).unwrap();
    }

    /// No `.partial` file may survive a successful conversion.
    #[test]
    fn the_temporary_file_does_not_outlive_the_conversion() {
        let f = fixture(
            "temp",
            "llama",
            r#"{"peft_type":"LORA","r":8}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );

        convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();

        let leftovers: Vec<_> = std::fs::read_dir(&f.adapter_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("partial"))
            .collect();

        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    }

    #[test]
    fn rslora_alpha_compensation_reaches_the_written_file() {
        let f = fixture(
            "rslora",
            "llama",
            r#"{"peft_type":"LORA","r":16,"lora_alpha":32.0,"use_rslora":true}"#,
            &["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        );

        let summary = convert_adapter(&f.adapter_dir, &f.base_gguf).unwrap();
        // 32 * sqrt(16) = 128, so llama.cpp's alpha/rank gives 128/16 = 8 = 32/sqrt(16).
        assert!((summary.alpha - 128.0).abs() < 1e-4, "got {}", summary.alpha);
    }
}

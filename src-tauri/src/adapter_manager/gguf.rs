//! Reading just enough GGUF metadata to tell an adapter from a whole model.
//!
//! ## Why this exists
//!
//! A file ending in `.gguf` is not necessarily a LoRA adapter. Some repositories
//! are *tagged* as adapters on HuggingFace while actually shipping fully merged
//! weights — `ericflo/Llama-3.1-8B-ContinuedTraining` carries the
//! `base_model:adapter:` tag but contains `config.json`,
//! `model-00001-of-00007.safetensors`, and multi-gigabyte `Q8_0.gguf` files.
//!
//! Checking the magic bytes alone accepts that file happily: it *is* a valid
//! GGUF. Installing it would download several gigabytes into the adapters
//! folder and only fail later, when llama.cpp is asked to bind a whole model as
//! an adapter.
//!
//! A real LoRA GGUF declares itself in its metadata with `adapter.type = "lora"`
//! (llama.cpp writes this key when converting a PEFT adapter). That declaration
//! is what this module reads.
//!
//! ## Format
//!
//! Only the header is parsed — metadata sits at the front of the file, so the
//! answer is available after a few kilobytes even for a 20 GB model:
//!
//! ```text
//! magic: "GGUF"      4 bytes
//! version            u32
//! tensor_count       u64
//! metadata_kv_count  u64
//! then metadata_kv_count pairs of:
//!   key              u64 length + UTF-8 bytes
//!   value_type       u32
//!   value            depends on type
//! ```
//!
//! Values other than the one being looked for are skipped without allocating.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{anyhow, bail, Result};

/// The metadata key llama.cpp writes when converting a LoRA adapter.
const ADAPTER_TYPE_KEY: &str = "adapter.type";

/// Refuses absurd declared lengths rather than trying to allocate them.
///
/// Metadata strings are keys and short labels; anything past this is a
/// corrupt or hostile file, not a real name.
const MAX_STRING_LEN: u64 = 64 * 1024;

/// Bounds work on a malformed header claiming a huge number of entries.
const MAX_KV_ENTRIES: u64 = 100_000;

/// GGUF value type tags, from the format specification.
mod value_type {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// Fixed width in bytes of a scalar value type, or `None` for the variable-width
/// types (string, array) which must be walked instead of skipped.
fn scalar_width(vtype: u32) -> Option<i64> {
    match vtype {
        value_type::UINT8 | value_type::INT8 | value_type::BOOL => Some(1),
        value_type::UINT16 | value_type::INT16 => Some(2),
        value_type::UINT32 | value_type::INT32 | value_type::FLOAT32 => Some(4),
        value_type::UINT64 | value_type::INT64 | value_type::FLOAT64 => Some(8),
        _ => None,
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_LEN {
        bail!("metadata string of {len} bytes is implausible; file looks corrupt");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    // Lossy: a mangled label should not fail the whole check.
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Advances past a value without reading it into memory.
fn skip_value<R: Read + Seek>(r: &mut R, vtype: u32) -> Result<()> {
    if let Some(width) = scalar_width(vtype) {
        r.seek(SeekFrom::Current(width))?;
        return Ok(());
    }

    match vtype {
        value_type::STRING => {
            let len = read_u64(r)?;
            if len > MAX_STRING_LEN {
                bail!("metadata string of {len} bytes is implausible; file looks corrupt");
            }
            r.seek(SeekFrom::Current(len as i64))?;
            Ok(())
        }
        value_type::ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;

            // Fixed-width elements can be jumped in one go.
            if let Some(width) = scalar_width(elem_type) {
                let total = (count as i64).saturating_mul(width);
                r.seek(SeekFrom::Current(total))?;
                return Ok(());
            }

            // Arrays of strings must be walked one at a time. Token vocabularies
            // are stored this way and can hold hundreds of thousands of entries,
            // so this is bounded but not cheap; it is still only a seek per item.
            if elem_type == value_type::STRING {
                for _ in 0..count {
                    let len = read_u64(r)?;
                    if len > MAX_STRING_LEN {
                        bail!("metadata string of {len} bytes is implausible; file looks corrupt");
                    }
                    r.seek(SeekFrom::Current(len as i64))?;
                }
                return Ok(());
            }

            bail!("unsupported array element type {elem_type} in GGUF metadata")
        }
        other => bail!("unsupported GGUF metadata value type {other}"),
    }
}

/// Reads `adapter.type` from a GGUF file's metadata.
///
/// `Ok(None)` means the file parsed correctly but declares no adapter type —
/// which is what a full model looks like.
pub fn read_adapter_type(path: &Path) -> Result<Option<String>> {
    let file = File::open(path).map_err(|e| anyhow!("could not open {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| anyhow!("could not read file header: {e}"))?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }

    let _version = read_u32(&mut reader)?;
    let _tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;

    if kv_count > MAX_KV_ENTRIES {
        bail!("header claims {kv_count} metadata entries; file looks corrupt");
    }

    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let vtype = read_u32(&mut reader)?;

        if key == ADAPTER_TYPE_KEY {
            if vtype != value_type::STRING {
                bail!("`{ADAPTER_TYPE_KEY}` is not text; file looks malformed");
            }
            return Ok(Some(read_string(&mut reader)?));
        }

        skip_value(&mut reader, vtype)?;
    }

    Ok(None)
}

/// Confirms a downloaded file really is a LoRA adapter.
///
/// The error text is what a non-specialist will read, so it says what the file
/// turned out to be rather than naming the metadata key.
pub fn verify_is_lora_adapter(path: &Path) -> Result<()> {
    match read_adapter_type(path) {
        Ok(Some(kind)) if kind.eq_ignore_ascii_case("lora") => Ok(()),
        Ok(Some(kind)) => Err(anyhow!(
            "This file is a '{kind}' adapter, which Sarathi cannot load. Only LoRA adapters work."
        )),
        Ok(None) => Err(anyhow!(
            "This file is a complete model, not a LoRA adapter, even though the page \
             is labelled as one. Downloading it here would not add a skill to your \
             existing model — install it as a model instead."
        )),
        Err(e) => Err(anyhow!("This file is not a readable LoRA adapter: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GGUF header in memory so the parser can be tested without
    /// downloading gigabytes.
    struct GgufBuilder {
        kvs: Vec<u8>,
        count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { kvs: Vec::new(), count: 0 }
        }

        fn push_key(&mut self, key: &str) {
            self.kvs.extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(key.as_bytes());
        }

        fn string(mut self, key: &str, value: &str) -> Self {
            self.push_key(key);
            self.kvs.extend_from_slice(&value_type::STRING.to_le_bytes());
            self.kvs.extend_from_slice(&(value.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(value.as_bytes());
            self.count += 1;
            self
        }

        fn u32_val(mut self, key: &str, value: u32) -> Self {
            self.push_key(key);
            self.kvs.extend_from_slice(&value_type::UINT32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
            self
        }

        fn string_array(mut self, key: &str, values: &[&str]) -> Self {
            self.push_key(key);
            self.kvs.extend_from_slice(&value_type::ARRAY.to_le_bytes());
            self.kvs.extend_from_slice(&value_type::STRING.to_le_bytes());
            self.kvs.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
                self.kvs.extend_from_slice(v.as_bytes());
            }
            self.count += 1;
            self
        }

        fn write(self, name: &str) -> std::path::PathBuf {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            bytes.extend_from_slice(&3u32.to_le_bytes()); // version
            bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
            bytes.extend_from_slice(&self.count.to_le_bytes());
            bytes.extend_from_slice(&self.kvs);

            let path = std::env::temp_dir().join(format!("sarathi_gguf_{name}.gguf"));
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    #[test]
    fn a_lora_adapter_declares_itself() {
        let path = GgufBuilder::new()
            .string("general.architecture", "llama")
            .string("adapter.type", "lora")
            .write("lora");

        assert_eq!(read_adapter_type(&path).unwrap().as_deref(), Some("lora"));
        assert!(verify_is_lora_adapter(&path).is_ok());
    }

    #[test]
    fn a_full_model_is_rejected_in_plain_language() {
        // This is the real failure the magic-byte check missed: a valid GGUF
        // that is a whole model, published under an adapter tag.
        let path = GgufBuilder::new()
            .string("general.architecture", "llama")
            .u32_val("llama.block_count", 32)
            .write("full_model");

        assert_eq!(read_adapter_type(&path).unwrap(), None);

        let err = verify_is_lora_adapter(&path).unwrap_err().to_string();
        assert!(err.contains("complete model"), "should say what it is: {err}");
        assert!(!err.contains("adapter.type"), "should not name the metadata key: {err}");
    }

    #[test]
    fn metadata_after_a_token_vocabulary_is_still_found() {
        // Vocabularies are huge string arrays stored before other keys; the
        // parser has to walk past one to reach the adapter declaration.
        let vocab: Vec<&str> = vec!["the", "quick", "brown", "fox", "<|endoftext|>"];
        let path = GgufBuilder::new()
            .string("general.architecture", "llama")
            .string_array("tokenizer.ggml.tokens", &vocab)
            .string("adapter.type", "lora")
            .write("after_vocab");

        assert_eq!(read_adapter_type(&path).unwrap().as_deref(), Some("lora"));
    }

    #[test]
    fn a_non_gguf_file_is_rejected() {
        let path = std::env::temp_dir().join("sarathi_gguf_not_gguf.gguf");
        std::fs::write(&path, b"this is not a model at all").unwrap();

        let err = read_adapter_type(&path).unwrap_err().to_string();
        assert!(err.contains("not a GGUF"), "got: {err}");
    }

    #[test]
    fn a_truncated_file_errors_rather_than_panicking() {
        // Interrupted downloads are common; they must not bring the app down.
        let path = std::env::temp_dir().join("sarathi_gguf_truncated.gguf");
        std::fs::write(&path, b"GGUF\x03\x00\x00\x00\x00").unwrap();

        assert!(read_adapter_type(&path).is_err());
    }

    #[test]
    fn an_implausible_string_length_is_refused_without_allocating() {
        // A hostile header could otherwise ask for a huge allocation.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // key length

        let path = std::env::temp_dir().join("sarathi_gguf_huge.gguf");
        std::fs::write(&path, bytes).unwrap();

        let err = read_adapter_type(&path).unwrap_err().to_string();
        assert!(err.contains("implausible"), "got: {err}");
    }

    #[test]
    fn a_non_lora_adapter_type_is_named_in_the_error() {
        let path = GgufBuilder::new().string("adapter.type", "control_vector").write("control");

        let err = verify_is_lora_adapter(&path).unwrap_err().to_string();
        assert!(err.contains("control_vector"), "should name the kind: {err}");
    }
}

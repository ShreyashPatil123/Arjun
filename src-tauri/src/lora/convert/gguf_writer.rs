//! Writing a LoRA adapter out as GGUF.
//!
//! ## Why this exists
//!
//! [`crate::adapter_manager::gguf`] reads GGUF headers; nothing in the codebase
//! wrote one until adapters had to be converted locally. This is the smallest
//! writer that produces a file `llama_adapter_lora_init` accepts.
//!
//! ## Layout
//!
//! The format is the mirror of the reader's description:
//!
//! ```text
//! magic "GGUF"        4 bytes
//! version             u32
//! tensor_count        u64
//! metadata_kv_count   u64
//! metadata KV pairs
//! tensor info records  (name, n_dims, dims, type, offset)
//! padding to alignment
//! tensor data
//! ```
//!
//! Two details are easy to get wrong and produce a file that parses but will not
//! load:
//!
//! 1. **Tensor offsets are relative to the start of the data section**, not to
//!    the start of the file. The reader adds the (aligned) end of the header.
//! 2. **Every tensor must start on an `general.alignment` boundary.** llama.cpp
//!    memory-maps the data section and reads tensors in place, so a misaligned
//!    offset is not merely untidy.
//!
//! Dimensions are written in GGML order, which is the reverse of the row-major
//! shape a safetensors file reports — see [`TensorData::dims`].

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// GGUF version this writer emits. Version 3 is what current llama.cpp reads.
const GGUF_VERSION: u32 = 3;

/// Default data alignment. Mirrors llama.cpp's `GGUF_DEFAULT_ALIGNMENT`.
const ALIGNMENT: u64 = 32;

/// GGUF metadata value type tags, matching the reader's `value_type` module.
mod value_type {
    pub const UINT32: u32 = 4;
    pub const FLOAT32: u32 = 6;
    pub const STRING: u32 = 8;
}

/// The ggml tensor types this writer emits.
///
/// LoRA factors are small and are not quantised: keeping them at the precision
/// they were trained in avoids adding error to weights that are already a small
/// correction to the base model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
}

impl GgmlType {
    /// The numeric tag ggml uses on disk.
    fn tag(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
        }
    }
}

/// One tensor destined for the output file.
pub struct TensorData {
    /// Full GGUF name including the `.lora_a` / `.lora_b` suffix.
    pub name: String,
    /// Shape in **row-major (safetensors) order**, outermost dimension first.
    ///
    /// Written to disk reversed, because ggml orders dimensions
    /// fastest-varying first.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Raw little-endian element bytes, already in `dtype`.
    pub data: Vec<u8>,
}

impl TensorData {
    /// Number of elements implied by the shape.
    fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Checks the buffer actually holds the shape it claims.
    ///
    /// A mismatch here would otherwise be written out and only surface as
    /// garbled inference much later.
    fn validate(&self) -> Result<()> {
        let expected = self.element_count() as usize * self.dtype.bytes_per_element();
        anyhow::ensure!(
            self.data.len() == expected,
            "tensor '{}' claims shape {:?} ({} bytes) but holds {} bytes",
            self.name,
            self.dims,
            expected,
            self.data.len()
        );
        anyhow::ensure!(
            !self.dims.is_empty(),
            "tensor '{}' has no dimensions",
            self.name
        );
        Ok(())
    }
}

/// A metadata entry. Only the three value types a LoRA adapter needs.
enum MetaValue {
    Str(String),
    U32(u32),
    F32(f32),
}

/// Accumulates metadata and tensors, then writes them in one pass.
#[derive(Default)]
pub struct GgufWriter {
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<TensorData>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata_str(&mut self, key: &str, value: impl Into<String>) -> &mut Self {
        self.metadata.push((key.to_string(), MetaValue::Str(value.into())));
        self
    }

    pub fn metadata_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.metadata.push((key.to_string(), MetaValue::U32(value)));
        self
    }

    pub fn metadata_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.metadata.push((key.to_string(), MetaValue::F32(value)));
        self
    }

    pub fn add_tensor(&mut self, tensor: TensorData) -> &mut Self {
        self.tensors.push(tensor);
        self
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Writes the complete file to `path`.
    ///
    /// The caller is expected to write to a temporary path and rename, so that
    /// an interrupted conversion never leaves a file that looks installable.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        for tensor in &self.tensors {
            tensor.validate()?;
        }

        let file = File::create(path)
            .with_context(|| format!("could not create {}", path.display()))?;
        let mut out = BufWriter::new(file);

        out.write_all(b"GGUF")?;
        write_u32(&mut out, GGUF_VERSION)?;
        write_u64(&mut out, self.tensors.len() as u64)?;
        // `general.alignment` is emitted alongside the caller's entries.
        write_u64(&mut out, self.metadata.len() as u64 + 1)?;

        write_meta_entry(&mut out, "general.alignment", &MetaValue::U32(ALIGNMENT as u32))?;
        for (key, value) in &self.metadata {
            write_meta_entry(&mut out, key, value)?;
        }

        // Tensor offsets are relative to the data section, so they can be
        // computed before the header length is known.
        let mut offset: u64 = 0;
        for tensor in &self.tensors {
            write_string(&mut out, &tensor.name)?;
            write_u32(&mut out, tensor.dims.len() as u32)?;
            // ggml stores dimensions fastest-varying first.
            for dim in tensor.dims.iter().rev() {
                write_u64(&mut out, *dim)?;
            }
            write_u32(&mut out, tensor.dtype.tag())?;
            write_u64(&mut out, offset)?;

            offset = align_up(offset + tensor.data.len() as u64);
        }

        // Pad the header so the first tensor lands on an alignment boundary.
        let header_end = out.stream_position()?;
        write_padding(&mut out, align_up(header_end) - header_end)?;

        for tensor in &self.tensors {
            out.write_all(&tensor.data)?;
            let padding = align_up(tensor.data.len() as u64) - tensor.data.len() as u64;
            write_padding(&mut out, padding)?;
        }

        out.flush()?;
        Ok(())
    }
}

fn align_up(value: u64) -> u64 {
    value.div_ceil(ALIGNMENT) * ALIGNMENT
}

fn write_u32<W: Write>(w: &mut W, value: u32) -> Result<()> {
    w.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(w: &mut W, value: u64) -> Result<()> {
    w.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_string<W: Write>(w: &mut W, value: &str) -> Result<()> {
    write_u64(w, value.len() as u64)?;
    w.write_all(value.as_bytes())?;
    Ok(())
}

fn write_meta_entry<W: Write>(w: &mut W, key: &str, value: &MetaValue) -> Result<()> {
    write_string(w, key)?;
    match value {
        MetaValue::Str(s) => {
            write_u32(w, value_type::STRING)?;
            write_string(w, s)?;
        }
        MetaValue::U32(v) => {
            write_u32(w, value_type::UINT32)?;
            write_u32(w, *v)?;
        }
        MetaValue::F32(v) => {
            write_u32(w, value_type::FLOAT32)?;
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_padding<W: Write + Seek>(w: &mut W, count: u64) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let zeros = vec![0u8; count as usize];
    w.write_all(&zeros)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_manager::gguf::read_adapter_type;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("sarathi_writer_{name}.gguf"))
    }

    fn sample_tensor(name: &str) -> TensorData {
        // 2x3 f32, values chosen so byte order is checkable.
        let values: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        TensorData { name: name.to_string(), dims: vec![2, 3], dtype: GgmlType::F32, data }
    }

    /// The strongest available check: the header this writes must satisfy the
    /// reader the application already trusts to identify adapters.
    #[test]
    fn output_is_recognised_as_a_lora_by_the_existing_reader() {
        let path = temp_path("roundtrip");

        let mut writer = GgufWriter::new();
        writer
            .metadata_str("general.architecture", "llama")
            .metadata_str("general.type", "adapter")
            .metadata_str("adapter.type", "lora")
            .metadata_f32("adapter.lora.alpha", 16.0)
            .add_tensor(sample_tensor("blk.0.attn_q.weight.lora_a"));

        writer.write_to(&path).unwrap();

        assert_eq!(read_adapter_type(&path).unwrap().as_deref(), Some("lora"));
        // And the stricter check the download path uses.
        crate::adapter_manager::gguf::verify_is_lora_adapter(&path).unwrap();
    }

    #[test]
    fn the_file_starts_with_the_gguf_magic_and_version() {
        let path = temp_path("magic");
        let mut writer = GgufWriter::new();
        writer.metadata_str("adapter.type", "lora").add_tensor(sample_tensor("t.lora_a"));
        writer.write_to(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), GGUF_VERSION);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1, "tensor count");
    }

    /// llama.cpp maps the data section directly, so a misaligned tensor is a
    /// correctness bug rather than a cosmetic one.
    #[test]
    fn tensor_data_begins_on_an_alignment_boundary() {
        let path = temp_path("aligned");
        let mut writer = GgufWriter::new();
        writer.metadata_str("adapter.type", "lora");
        // Odd-length names make the header length awkward on purpose.
        writer.add_tensor(sample_tensor("blk.0.attn_q.weight.lora_a"));
        writer.add_tensor(sample_tensor("blk.11.ffn_down.weight.lora_b"));
        writer.write_to(&path).unwrap();

        let len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(len % ALIGNMENT, 0, "file should end padded to alignment, got {len}");
    }

    #[test]
    fn a_tensor_whose_buffer_disagrees_with_its_shape_is_refused() {
        let path = temp_path("mismatch");
        let mut writer = GgufWriter::new();
        writer.add_tensor(TensorData {
            name: "bad.lora_a".into(),
            dims: vec![4, 4], // claims 64 bytes
            dtype: GgmlType::F32,
            data: vec![0u8; 8], // holds 8
        });

        let err = writer.write_to(&path).unwrap_err().to_string();
        assert!(err.contains("claims shape"), "got: {err}");
    }

    #[test]
    fn dimensions_are_written_in_reversed_ggml_order() {
        let path = temp_path("dims");
        let mut writer = GgufWriter::new();
        writer.metadata_str("adapter.type", "lora").add_tensor(sample_tensor("t.lora_a"));
        writer.write_to(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // Walk to the tensor info record: header + 2 metadata entries.
        let name = b"t.lora_a";
        let pos = bytes
            .windows(name.len())
            .position(|w| w == name)
            .expect("tensor name should appear in the file");
        let after_name = pos + name.len();

        let n_dims = u32::from_le_bytes(bytes[after_name..after_name + 4].try_into().unwrap());
        assert_eq!(n_dims, 2);

        let d0 = u64::from_le_bytes(bytes[after_name + 4..after_name + 12].try_into().unwrap());
        let d1 = u64::from_le_bytes(bytes[after_name + 12..after_name + 20].try_into().unwrap());
        // Shape was [2, 3]; ggml order is reversed.
        assert_eq!((d0, d1), (3, 2), "dims should be reversed on disk");
    }

    #[test]
    fn an_f16_tensor_reports_half_the_byte_width() {
        assert_eq!(GgmlType::F16.bytes_per_element(), 2);
        assert_eq!(GgmlType::F32.bytes_per_element(), 4);

        let path = temp_path("f16");
        let mut writer = GgufWriter::new();
        writer.metadata_str("adapter.type", "lora").add_tensor(TensorData {
            name: "half.lora_b".into(),
            dims: vec![2, 2],
            dtype: GgmlType::F16,
            data: vec![0u8; 8],
        });
        writer.write_to(&path).unwrap();

        assert_eq!(read_adapter_type(&path).unwrap().as_deref(), Some("lora"));
    }

    #[test]
    fn alignment_rounds_up_only_when_needed() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 32);
        assert_eq!(align_up(32), 32);
        assert_eq!(align_up(33), 64);
    }
}

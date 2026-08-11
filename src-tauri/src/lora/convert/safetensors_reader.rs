//! Reading LoRA weights out of a PEFT `adapter_model.safetensors`.
//!
//! ## Why this exists
//!
//! PEFT publishes adapters as safetensors: an 8-byte header length, a JSON
//! header describing every tensor, then the raw little-endian element bytes.
//! The `safetensors` crate handles that framing; this module handles the part
//! specific to LoRA — deciding which tensors matter and getting their element
//! type into something GGUF can carry.
//!
//! ## Precision
//!
//! Adapters are trained in `f16`, `bf16` or `f32`. The first and last map
//! straight onto ggml types. `bf16` does not: ggml has a `BF16` type, but the
//! safe move for a format conversion is to widen to `f32` rather than
//! re-round — a LoRA is already a small correction to the base weights, and
//! rounding error introduced here is indistinguishable from bad training.
//!
//! Widening costs disk: a bf16 adapter doubles in size. LoRA files are tens of
//! megabytes, so this is a fair trade for not silently degrading the weights.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors};

use super::gguf_writer::GgmlType;

/// One LoRA tensor lifted out of a safetensors file.
///
/// `Debug` omits the weight bytes deliberately — a derived one would print tens
/// of megabytes into a log line on the first failed assertion.
pub struct RawTensor {
    /// The PEFT name, exactly as stored.
    pub name: String,
    /// Row-major shape, outermost dimension first.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Element bytes in `dtype`, little-endian.
    pub data: Vec<u8>,
}

impl std::fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTensor")
            .field("name", &self.name)
            .field("dims", &self.dims)
            .field("dtype", &self.dtype)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// Reads every tensor from a PEFT adapter file.
///
/// Returned sorted by name so a conversion is byte-for-byte reproducible:
/// safetensors iteration order follows its JSON header, which is not guaranteed
/// stable across producers.
pub fn read_adapter_tensors(path: &Path) -> Result<Vec<RawTensor>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("could not read {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&bytes).with_context(|| {
        format!("{} is not a readable safetensors file", path.display())
    })?;

    // BTreeMap rather than sorting afterwards: it also surfaces the (illegal)
    // case of a duplicated tensor name instead of silently keeping one.
    let mut collected: BTreeMap<String, RawTensor> = BTreeMap::new();

    for (name, view) in tensors.tensors() {
        // `to_string` rather than `clone`: the crate has returned both `&str`
        // and `String` keys across versions, and this compiles against either.
        let name = name.to_string();

        let (dtype, data) = convert_elements(&name, view.dtype(), view.data())?;

        let dims: Vec<u64> = view.shape().iter().map(|d| *d as u64).collect();
        if dims.is_empty() {
            bail!("tensor '{name}' in {} has no shape", path.display());
        }

        let already_present = collected
            .insert(name.clone(), RawTensor { name: name.clone(), dims, dtype, data })
            .is_some();

        if already_present {
            bail!("tensor '{name}' appears twice in {}", path.display());
        }
    }

    if collected.is_empty() {
        bail!("{} contains no tensors", path.display());
    }

    Ok(collected.into_values().collect())
}

/// Maps a safetensors element type onto a ggml one, widening `bf16` to `f32`.
fn convert_elements(name: &str, dtype: Dtype, data: &[u8]) -> Result<(GgmlType, Vec<u8>)> {
    match dtype {
        Dtype::F32 => Ok((GgmlType::F32, data.to_vec())),
        Dtype::F16 => Ok((GgmlType::F16, data.to_vec())),
        Dtype::BF16 => Ok((GgmlType::F32, bf16_to_f32_bytes(data)?)),
        other => bail!(
            "tensor '{name}' is stored as {other:?}, which Sarathi cannot convert. \
             LoRA adapters are expected in f16, bf16 or f32."
        ),
    }
}

/// Widens bfloat16 to f32.
///
/// bfloat16 is the top 16 bits of an f32 with the same exponent layout, so the
/// conversion is a shift — no table, no rounding, exactly representable.
fn bf16_to_f32_bytes(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 2 != 0 {
        bail!("bf16 tensor has an odd byte length ({}), so it is truncated", data.len());
    }

    let mut out = Vec::with_capacity(data.len() * 2);
    for pair in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let widened = (bits as u32) << 16;
        out.extend_from_slice(&f32::from_bits(widened).to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bf16 → f32 must be exact for values bf16 can represent.
    #[test]
    fn bf16_widening_is_exact_for_representable_values() {
        // 1.0f32 = 0x3F800000; its bf16 form is the top half, 0x3F80.
        let bytes = 0x3F80u16.to_le_bytes();
        let widened = bf16_to_f32_bytes(&bytes).unwrap();
        let value = f32::from_le_bytes(widened.try_into().unwrap());
        assert_eq!(value, 1.0);
    }

    #[test]
    fn bf16_widening_preserves_sign_and_zero() {
        for (bits, expected) in [(0x0000u16, 0.0f32), (0x8000u16, -0.0f32), (0xBF80u16, -1.0f32)] {
            let widened = bf16_to_f32_bytes(&bits.to_le_bytes()).unwrap();
            let value = f32::from_le_bytes(widened.try_into().unwrap());
            assert_eq!(value.to_bits(), expected.to_bits(), "for bits {bits:#06x}");
        }
    }

    #[test]
    fn bf16_widening_doubles_the_buffer() {
        let input = vec![0u8; 16];
        assert_eq!(bf16_to_f32_bytes(&input).unwrap().len(), 32);
    }

    /// A truncated download must be reported, not read past.
    #[test]
    fn an_odd_length_bf16_buffer_is_refused() {
        let err = bf16_to_f32_bytes(&[0u8; 3]).unwrap_err().to_string();
        assert!(err.contains("truncated"), "got: {err}");
    }

    #[test]
    fn f32_and_f16_pass_through_without_copying_precision() {
        let (ty, out) = convert_elements("t", Dtype::F32, &[1, 2, 3, 4]).unwrap();
        assert_eq!(ty, GgmlType::F32);
        assert_eq!(out, vec![1, 2, 3, 4]);

        let (ty, out) = convert_elements("t", Dtype::F16, &[1, 2]).unwrap();
        assert_eq!(ty, GgmlType::F16);
        assert_eq!(out, vec![1, 2]);
    }

    /// Quantised adapters exist; refusing them by name beats producing a file
    /// that loads and inferences incorrectly.
    #[test]
    fn an_unsupported_element_type_is_named_in_the_error() {
        let err = convert_elements("blk.0.attn_q.lora_A.weight", Dtype::I8, &[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("blk.0.attn_q.lora_A.weight"), "got: {err}");
        assert!(err.contains("f16"), "should say what is accepted: {err}");
    }

    #[test]
    fn a_missing_file_reports_the_path() {
        let missing = Path::new("definitely/not/here/adapter_model.safetensors");
        let err = read_adapter_tensors(missing).unwrap_err().to_string();
        assert!(err.contains("could not read"), "got: {err}");
    }

    #[test]
    fn a_non_safetensors_file_is_rejected() {
        let path = std::env::temp_dir().join("sarathi_not_safetensors.safetensors");
        std::fs::write(&path, b"this is not a tensor file at all").unwrap();

        let err = read_adapter_tensors(&path).unwrap_err().to_string();
        assert!(err.contains("not a readable safetensors"), "got: {err}");
    }
}

//! Steganographic watermark — **INTENTIONALLY NOT IMPLEMENTED**.
//!
//! ## Why this module exists
//!
//! An early ARJUN design draft proposed steganographic watermarking of
//! generated artifacts. PS 26117 does not mention watermarking of any kind. The function bodies below are placeholders that always return
//! [`StegoNotImplemented`]. They are not stubs to be filled in later: they
//! are a *refusal* of the feature, with the refusal written out so that a
//! future contributor reading the call sites cannot accidentally re-introduce
//! steganography by copying a working implementation from a different codebase.
//!
//! ## Why it is refused (so the decision survives the contributor who
//! inherits it)
//!
//! 1. **No honest use case in the local-only workbench.** A steganographic
//!    watermark is a forensic mark an inspector can recover from a leaked
//!    copy to prove provenance. In an air-gapped MRPL refinery inspection
//!    workflow, the artifacts are not at risk of being leaked to a public
//!    channel — they are produced for an internal approval chain. The threat
//!    steganography defends against (an attacker re-publishing a confidential
//!    document and denying its source) does not exist here.
//!
//! 2. **The implementation is fundamentally leaky in the wrong direction.**
//!    The only way a steganographic watermark works is that it survives
//!    reformatting, re-encoding, and re-printing. The only way that survives
//!    those operations is by being *invisible* to the reader. That property
//!    is exactly what makes steganography a vector for exfiltration in the
//!    *opposite* direction: a compromised model that we host could embed
//!    data into a document a reader carries out of the facility, with no
//!    way for the human reader to see it. Adding a stego *writer* alongside
//!    a (hypothetical) stego *reader* doubles the surface for that vector.
//!    PS 26117's sovereignty posture — the user owns the data, the model
//!    does not write covert channels out of the workbench — is directly
//!    contradicted by a working steganographic primitive in the build.
//!
//! 3. **It does not protect against the threat it appears to.** Even on
//!    platforms where steganography is the right answer (public-facing
//!    content distribution), the watermark is *attributable*, not
//!    *authenticating*: it says "this copy was once in ARJUN's hands" but
//!    not "this copy was not modified after ARJUN wrote it." Anyone who can
//!    copy the file can copy the watermark. For the property an MRPL
//!    reviewer actually needs (did the document change between ARJUN writing
//!    it and the reviewer signing it?), the [visible watermark] and the
//!    [HMAC-signed provenance] already provide the answer; the stego layer
//!    adds nothing verifiable on top.
//!
//! 4. **It cannot anchor a trust key in this topology.** A useful stego
//!    watermark is signed with a key the writer cannot also revoke. That key
//!    has to live somewhere the runtime can reach but the operator (and any
//!    attacker who reads the binary) cannot. In an air-gapped refinery
//!    laptop there is no such place: any on-disk key is owned by the same
//!    process that would forge the watermark, and any in-memory key is gone
//!    on next boot. The honest claim is that the visible watermark — which
//!    the *reader* sees, not the *attacker* — is the strongest attribution
//!    this topology supports.
//!
//! [visible watermark]:    ../artifacts/visible_watermark/index.html
//! [HMAC-signed provenance]: ../audit/provenance_hmac/index.html
//!
//! ## What to do if a future requirement genuinely demands steganography
//!
//! Do not copy from another project. The decision above is the decision, and
//! re-introducing the feature requires a fresh threat model and a new sign-off
//! from the security owner. If a requirement cannot be met without stego,
//! raise it as a design change, not as a code review.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::visible_watermark::VisibleStamp;

/// The one and only error this module returns, on every code path.
#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq)]
#[error(
    "Steganographic watermarking is intentionally not implemented in this build. \
     See `artifacts::stego_watermark` module docs for the reasoning. Use the \
     visible watermark (artifacts::visible_watermark) and the HMAC-signed \
     provenance (audit::provenance_hmac) for the traceability this topology \
     can honestly provide."
)]
pub struct StegoNotImplemented;

/// Carrier formats a steganographic watermark *would* target. Listed here so
/// callers can be told "this format is not supported, and here is why" rather
/// than getting a generic "not implemented" with no path forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StegoCarrier {
    Docx,
    Xlsx,
    Pptx,
    Pdf,
    Png,
    Jpeg,
    Wav,
}

impl StegoCarrier {
    /// A short, human-readable explanation of why this carrier has no stego
    /// support, in case a reviewer is wondering why a particular format is
    /// named.
    pub fn reason(self) -> &'static str {
        match self {
            StegoCarrier::Docx | StegoCarrier::Xlsx | StegoCarrier::Pptx => {
                "OOXML carriers are rebuilt from XML on every save, so a stego \
                 bit written into one revision rarely survives the next; the \
                 visible watermark is the honest attribution here."
            }
            StegoCarrier::Pdf => {
                "PDFs are post-process re-rendered for print, and re-rendering \
                 strips any carrier-level stego that does not survive \
                 resampling; the HMAC-signed provenance covers the integrity \
                 property a reviewer actually needs."
            }
            StegoCarrier::Png | StegoCarrier::Jpeg => {
                "Image carriers are the canonical stego channel, and for that \
                 reason they are the most likely vector for *outbound* covert \
                 channels from a compromised model. Adding a stego *writer* \
                 would double the surface for that vector, which is the \
                 opposite of PS 26117's sovereignty posture."
            }
            StegoCarrier::Wav => {
                "Audio carriers are not produced by ARJUN; the only audio \
                 flow in the workbench is speech-to-text, which already \
                 discards the carrier on ingestion."
            }
        }
    }
}

/// What a steganographic embedder *would* take. Surfaced in the error so
/// callers can decide what to do instead (most often: log and continue with
/// the visible watermark).
#[derive(Debug, Clone, PartialEq)]
pub struct StegoRequest {
    pub carrier: StegoCarrier,
    /// Owned (not borrowed) so the request does not need a lifetime
    /// parameter, which would require `VisibleStamp: Deserialize<'_>` and
    /// pull serialization constraints into a module that is supposed to
    /// compile even when the visible-watermark module is refactored.
    pub stamp: VisibleStamp,
    /// Placeholder for the payload a working stego embedder would write into
    /// the file. Held here so a caller that compiled against this signature
    /// fails to compile, rather than silently dropping the payload, the day
    /// someone tries to wire one up.
    pub payload: Vec<u8>,
}

/// The single public entry point. Always returns [`StegoNotImplemented`].
///
/// Marked `#[inline(never)]` so the symbol shows up in any binary the
/// compiler emits — a developer searching for "steg" or "stego" in a binary
/// will land on this function, not on a removed call site.
#[inline(never)]
pub fn embed(_request: StegoRequest) -> Result<Vec<u8>, StegoNotImplemented> {
    // Intentionally not implemented. See the module docs for the reasoning.
    Err(StegoNotImplemented)
}

/// The matching "extract a watermark" function. Also always returns
/// [`StegoNotImplemented`], for the same reason as [`embed`]: a stego
/// *reader* without a stego *writer* is meaningless, and a stego *writer* is
/// refused.
#[inline(never)]
pub fn extract(
    _carrier: StegoCarrier,
    _bytes: &[u8],
) -> Result<VisibleStamp, StegoNotImplemented> {
    Err(StegoNotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stamp() -> VisibleStamp {
        VisibleStamp {
            task_id: "task-1".to_string(),
            model: "m".to_string(),
            created_at: "2026-08-29T12:00:00Z".to_string(),
            classification: "Internal".to_string(),
            is_draft: false,
            claim: "x".to_string(),
        }
    }

    #[test]
    fn embed_always_returns_the_refusal_error() {
        let req = StegoRequest {
            carrier: StegoCarrier::Png,
            stamp: sample_stamp(),
            payload: vec![0u8; 32],
        };
        let result = embed(req);
        let err = result.expect_err("embed must always fail");
        assert_eq!(err, StegoNotImplemented);
    }

    #[test]
    fn extract_always_returns_the_refusal_error() {
        let result = extract(StegoCarrier::Png, b"some bytes");
        let err = result.expect_err("extract must always fail");
        assert_eq!(err, StegoNotImplemented);
    }

    #[test]
    fn refusal_message_names_an_alternative() {
        // The point of the refusal being written out (rather than the
        // function being absent) is that a developer who hits the error
        // path is pointed at the *visible* watermark and the HMAC
        // provenance, so the refusal is not a dead end.
        let err = StegoNotImplemented;
        let msg = err.to_string();
        assert!(msg.contains("visible_watermark"));
        assert!(msg.contains("provenance_hmac"));
    }

    #[test]
    fn every_carrier_has_a_written_reason() {
        // Forcing a developer to add a new variant to the enum is a useful
        // forcing function: a new carrier without a written `reason` would
        // not compile, and so cannot be silently added.
        for carrier in [
            StegoCarrier::Docx,
            StegoCarrier::Xlsx,
            StegoCarrier::Pptx,
            StegoCarrier::Pdf,
            StegoCarrier::Png,
            StegoCarrier::Jpeg,
            StegoCarrier::Wav,
        ] {
            let reason = carrier.reason();
            assert!(
                !reason.is_empty(),
                "carrier {carrier:?} must have a non-empty reason"
            );
        }
    }
}

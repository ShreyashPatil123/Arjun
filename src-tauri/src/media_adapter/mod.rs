//! Local media adapter for multimodal processing.
//!
//! Provides a single entry point for processing documents and images
//! locally, with:
//!
//! - PDF page rasterization
//! - Printed text OCR
//! - Image/document input to a local VLM where available
//! - Page/region evidence references
//! - Confidence and unreadable-region flags
//! - Byte, dimension and page limits
//! - No remote URLs (all processing is local)
//! - Clearance checks before model input
//!
//! The result is typed and serializable, suitable for both model consumption
//! and audit logs.

pub mod types;
pub mod adapter;
pub mod clearance;

pub use adapter::LocalMediaAdapter;
pub use types::{
    Finding, MultimodalResult, Region, RegionReference, Table, Uncertainty,
};

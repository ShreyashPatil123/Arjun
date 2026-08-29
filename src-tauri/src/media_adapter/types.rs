//! Typed multimodal result structures.
//!
//! These types are shared between the Rust core and the model runtime.
//! They are serializable to JSON for the model's consumption and for
//! audit logging.

use serde::{Deserialize, Serialize};

/// A region on a page, with coordinates in 0.0-1.0 normalized space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Region {
    /// Page number (1-based).
    pub page: u32,
    /// Region type: "text", "table", "figure", "formula", "header", "footer".
    pub kind: String,
    /// Left coordinate, 0.0-1.0.
    pub left: f32,
    /// Top coordinate, 0.0-1.0.
    pub top: f32,
    /// Right coordinate, 0.0-1.0.
    pub right: f32,
    /// Bottom coordinate, 0.0-1.0.
    pub bottom: f32,
    /// Confidence that this region was correctly identified, 0.0-1.0.
    pub confidence: f32,
}

/// A reference to a specific region for evidence purposes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RegionReference {
    pub page: u32,
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    /// Text or description of what was found in this region.
    pub text: String,
    /// Confidence that the content was correctly read, 0.0-1.0.
    pub confidence: f32,
}

/// A single finding from the multimodal analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// Category of finding: "entity", "fact", "number", "date", "condition", "observation".
    pub category: String,
    /// The extracted text/value.
    pub value: String,
    /// Page where this was found.
    pub page: u32,
    /// Optional region reference.
    pub region: Option<RegionReference>,
    /// Confidence, 0.0-1.0.
    pub confidence: f32,
}

/// A table extracted from the document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Table {
    /// Page where the table was found.
    pub page: u32,
    /// Optional region for the table.
    pub region: Option<RegionReference>,
    /// Table headers.
    pub headers: Vec<String>,
    /// Table rows, each row is a list of cell values.
    pub rows: Vec<Vec<String>>,
    /// Confidence, 0.0-1.0.
    pub confidence: f32,
}

/// An uncertainty or unreadable region.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Uncertainty {
    /// What is uncertain: "text", "figure", "table", "handwriting", "stamp".
    pub kind: String,
    /// Where the uncertainty is.
    pub page: u32,
    /// Optional region.
    pub region: Option<RegionReference>,
    /// Why it is uncertain: "low_quality", "handwritten", "damaged", "language_unknown".
    pub reason: String,
    /// Suggested action: "human_review", "ocr_fallback", "ignore".
    pub suggested_action: String,
}

/// The complete multimodal result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MultimodalResult {
    /// Key findings from the document/image.
    pub findings: Vec<Finding>,
    /// Tables extracted.
    pub tables: Vec<Table>,
    /// Detected regions on each page.
    pub regions: Vec<Region>,
    /// Evidence references for citation.
    pub evidence_refs: Vec<RegionReference>,
    /// Uncertainties and unreadable regions.
    pub uncertainty: Vec<Uncertainty>,
    /// True if a human should review this before its content is used.
    pub needs_human_review: bool,
    /// ID of the model that produced this result.
    pub model_id: String,
    /// Hash of the model that produced this result.
    pub model_hash: String,
    /// Engine used (e.g., "docling", "tesseract", "vlm").
    pub engine: String,
    /// Overall confidence, 0.0-1.0.
    pub confidence: f32,
    /// Pages that were processed.
    pub pages_processed: u32,
    /// Pages that were requested but not readable.
    pub pages_unreadable: u32,
}

/// Input to the multimodal processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaInput {
    /// Path to the document/image. Must be a local file path, not a URL.
    pub path: String,
    /// First page to process (1-based). None means start from page 1.
    pub from_page: Option<u32>,
    /// Last page to process (1-based, inclusive). None means process all.
    pub to_page: Option<u32>,
    /// Maximum file size in bytes. Default: 512 MB.
    pub max_file_bytes: Option<u64>,
    /// Maximum image dimension (width or height) in pixels.
    pub max_image_dimension: Option<u32>,
    /// Classification of the material being processed.
    pub classification: Option<crate::policy::Classification>,
}

//! Clearance checks before model input.
//!
//! Before any document or image is passed to a model, the clearance
//! pipeline verifies that:
//!
//! 1. The file is local (not a URL).
//! 2. The file size is within limits.
//! 3. The page range is valid.
//! 4. The classification of the material is allowed by policy.
//! 5. The user is authorized to process this material.
//!
//! Clearance failures are explicit errors, not silent fallbacks.

use std::path::Path;

use crate::policy::Classification;

/// Maximum file size for a single document (512 MB).
pub const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum image dimension (width or height) in pixels.
pub const MAX_IMAGE_DIMENSION: u32 = 8192;

/// Maximum number of pages to process in a single call.
pub const MAX_PAGES: u32 = 100;

/// Errors that can occur during clearance.
#[derive(Debug, Clone, PartialEq)]
pub enum ClearanceError {
    /// The path is a URL, not a local file.
    RemoteUrl(String),
    /// The file does not exist.
    FileNotFound(String),
    /// The file is too large.
    FileTooLarge { size: u64, max: u64 },
    /// The page range is invalid.
    InvalidPageRange { from: u32, to: u32 },
    /// The classification is not allowed.
    ClassificationNotAllowed(Classification),
    /// The image dimensions are too large.
    ImageDimensionsTooLarge { dimension: u32, max: u32 },
}

impl std::fmt::Display for ClearanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClearanceError::RemoteUrl(url) => {
                write!(f, "Remote URLs are not allowed: {}. Only local files can be processed.", url)
            }
            ClearanceError::FileNotFound(path) => {
                write!(f, "File not found: {}", path)
            }
            ClearanceError::FileTooLarge { size, max } => {
                write!(
                    f,
                    "File is {} MB, above the {} MB limit for processing",
                    size / 1024 / 1024,
                    max / 1024 / 1024
                )
            }
            ClearanceError::InvalidPageRange { from, to } => {
                write!(
                    f,
                    "Invalid page range: from={} to={}. from must be >= 1 and to >= from.",
                    from, to
                )
            }
            ClearanceError::ClassificationNotAllowed(c) => {
                write!(
                    f,
                    "Classification {:?} is not allowed for this model. An administrator must clear the model for this material.",
                    c
                )
            }
            ClearanceError::ImageDimensionsTooLarge { dimension, max } => {
                write!(
                    f,
                    "Image dimension {} exceeds the maximum allowed {}",
                    dimension, max
                )
            }
        }
    }
}

impl std::error::Error for ClearanceError {}

/// Validates that a path is a local file path and not a URL.
pub fn validate_local_path(path: &str) -> Result<&Path, ClearanceError> {
    // Reject URLs
    if path.starts_with("http://") || path.starts_with("https://") || path.starts_with("ftp://") {
        return Err(ClearanceError::RemoteUrl(path.to_string()));
    }

    // Reject file:// URLs
    if path.starts_with("file://") {
        return Err(ClearanceError::RemoteUrl(path.to_string()));
    }

    // Reject UNC paths that look like network paths
    if path.starts_with("\\\\") {
        return Err(ClearanceError::RemoteUrl(path.to_string()));
    }

    let p = Path::new(path);

    // Check that the file exists
    if !p.exists() {
        return Err(ClearanceError::FileNotFound(path.to_string()));
    }

    Ok(p)
}

/// Validates file size against the maximum allowed.
pub fn validate_file_size(path: &Path, max_bytes: u64) -> Result<u64, ClearanceError> {
    let metadata = std::fs::metadata(path)
        .map_err(|_| ClearanceError::FileNotFound(path.display().to_string()))?;
    let size = metadata.len();
    if size > max_bytes {
        return Err(ClearanceError::FileTooLarge {
            size,
            max: max_bytes,
        });
    }
    Ok(size)
}

/// Validates a page range.
pub fn validate_page_range(from: Option<u32>, to: Option<u32>) -> Result<(u32, u32), ClearanceError> {
    let from = from.unwrap_or(1);
    let to = to.unwrap_or(from);

    if from < 1 {
        return Err(ClearanceError::InvalidPageRange { from, to });
    }

    if to < from {
        return Err(ClearanceError::InvalidPageRange { from, to });
    }

    if to - from + 1 > MAX_PAGES {
        return Err(ClearanceError::InvalidPageRange { from, to });
    }

    Ok((from, to))
}

/// Validates that a classification is allowed.
pub fn validate_classification(
    classification: Option<Classification>,
    allowed_classifications: &[Classification],
) -> Result<(), ClearanceError> {
    if let Some(c) = classification {
        if !allowed_classifications.contains(&c) {
            return Err(ClearanceError::ClassificationNotAllowed(c));
        }
    }
    Ok(())
}

/// Validates image dimensions.
pub fn validate_image_dimensions(width: u32, height: u32) -> Result<(), ClearanceError> {
    if width > MAX_IMAGE_DIMENSION {
        return Err(ClearanceError::ImageDimensionsTooLarge {
            dimension: width,
            max: MAX_IMAGE_DIMENSION,
        });
    }
    if height > MAX_IMAGE_DIMENSION {
        return Err(ClearanceError::ImageDimensionsTooLarge {
            dimension: height,
            max: MAX_IMAGE_DIMENSION,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_urls_are_rejected() {
        let result = validate_local_path("http://example.com/file.pdf");
        assert!(matches!(result, Err(ClearanceError::RemoteUrl(_))));
    }

    #[test]
    fn https_urls_are_rejected() {
        let result = validate_local_path("https://example.com/file.pdf");
        assert!(matches!(result, Err(ClearanceError::RemoteUrl(_))));
    }

    #[test]
    fn ftp_urls_are_rejected() {
        let result = validate_local_path("ftp://server/file.pdf");
        assert!(matches!(result, Err(ClearanceError::RemoteUrl(_))));
    }

    #[test]
    fn file_urls_are_rejected() {
        let result = validate_local_path("file:///etc/passwd");
        assert!(matches!(result, Err(ClearanceError::RemoteUrl(_))));
    }

    #[test]
    fn unc_paths_are_rejected() {
        let result = validate_local_path("\\\\server\\share\\file.pdf");
        assert!(matches!(result, Err(ClearanceError::RemoteUrl(_))));
    }

    #[test]
    fn nonexistent_files_are_rejected() {
        let result = validate_local_path("./definitely-not-here-12345.pdf");
        assert!(matches!(result, Err(ClearanceError::FileNotFound(_))));
    }

    #[test]
    fn valid_page_range_accepted() {
        let (from, to) = validate_page_range(Some(1), Some(10)).unwrap();
        assert_eq!(from, 1);
        assert_eq!(to, 10);
    }

    #[test]
    fn page_range_below_one_rejected() {
        let result = validate_page_range(Some(0), Some(5));
        assert!(matches!(result, Err(ClearanceError::InvalidPageRange { .. })));
    }

    #[test]
    fn inverted_page_range_rejected() {
        let result = validate_page_range(Some(10), Some(5));
        assert!(matches!(result, Err(ClearanceError::InvalidPageRange { .. })));
    }

    #[test]
    fn too_many_pages_rejected() {
        let result = validate_page_range(Some(1), Some(200));
        assert!(matches!(result, Err(ClearanceError::InvalidPageRange { .. })));
    }

    #[test]
    fn no_classification_means_no_restriction() {
        let result = validate_classification(None, &[Classification::Internal]);
        assert!(result.is_ok());
    }

    #[test]
    fn allowed_classification_accepted() {
        let result = validate_classification(Some(Classification::Internal), &[Classification::Internal]);
        assert!(result.is_ok());
    }

    #[test]
    fn disallowed_classification_rejected() {
        let result = validate_classification(
            Some(Classification::VendorNegotiation),
            &[Classification::Internal],
        );
        assert!(matches!(result, Err(ClearanceError::ClassificationNotAllowed(_))));
    }
}

//! Evidence package export.
//!
//! After a document is produced, verified, and approved, an evidence package
//! is exported containing all the artifacts, hashes, and provenance needed
//! to audit the task.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::policy::Classification;

/// A single artifact in the evidence package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PackageArtifact {
    /// Type of artifact: "draft", "document", "approval", "calculation", "evidence".
    pub artifact_type: String,
    /// Path to the artifact.
    pub path: PathBuf,
    /// SHA-256 hash of the artifact content.
    pub sha256: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Timestamp when the artifact was created.
    pub created_at: String,
}

/// Provenance information for the task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    /// Task ID.
    pub task_id: String,
    /// Model that produced the draft.
    pub model_id: String,
    /// Skill that guided the draft.
    pub skill_id: String,
    /// Classification of the material.
    pub classification: Classification,
    /// Evidence IDs used.
    pub evidence_ids: Vec<String>,
    /// Calculation IDs used.
    pub calculation_ids: Vec<String>,
    /// Hash of the approved draft.
    pub draft_hash: String,
    /// Hash of the final artifact.
    pub artifact_hash: String,
    /// Timestamp when the package was exported.
    pub exported_at: String,
}

/// The complete evidence package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EvidencePackage {
    /// All artifacts included in this package.
    pub artifacts: Vec<PackageArtifact>,
    /// Provenance information.
    pub provenance: Provenance,
    /// SHA-256 hash of the entire package.
    pub package_hash: String,
}

impl EvidencePackage {
    /// Returns true if the package has the required artifacts.
    pub fn is_complete(&self) -> bool {
        let types: Vec<&str> = self
            .artifacts
            .iter()
            .map(|a| a.artifact_type.as_str())
            .collect();
        types.contains(&"document")
            && types.contains(&"approval")
            && types.contains(&"draft")
    }
}

/// Exports an evidence package to the given directory.
///
/// Computes SHA-256 hashes for all artifacts and writes a manifest.
pub fn export_evidence_package(
    output_dir: &Path,
    artifacts: Vec<PackageArtifact>,
    provenance: Provenance,
) -> std::io::Result<EvidencePackage> {
    // Compute package hash from all artifacts
    let mut all_hashes: Vec<&str> = artifacts.iter().map(|a| a.sha256.as_str()).collect();
    all_hashes.push(&provenance.artifact_hash);
    all_hashes.sort();
    let package_hash = compute_combined_hash(&all_hashes);

    let package = EvidencePackage {
        artifacts,
        provenance,
        package_hash,
    };

    // Write the manifest
    let manifest_path = output_dir.join("evidence-manifest.json");
    let manifest_json = serde_json::to_string_pretty(&package)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(manifest_path, manifest_json)?;

    Ok(package)
}

/// Computes a SHA-256 hash of a string.
pub fn compute_sha256(content: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // In production, use a real SHA-256. This is a deterministic placeholder.
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Computes a combined hash from multiple hash strings.
fn compute_combined_hash(hashes: &[&str]) -> String {
    let combined: String = hashes.join("|");
    compute_sha256(combined.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_artifact(artifact_type: &str) -> PackageArtifact {
        PackageArtifact {
            artifact_type: artifact_type.to_string(),
            path: PathBuf::from(format!("{}.bin", artifact_type)),
            sha256: format!("hash-{}", artifact_type),
            size_bytes: 100,
            created_at: "2026-01-15T10:00:00Z".to_string(),
        }
    }

    #[test]
    fn complete_package_has_required_artifacts() {
        let artifacts = vec![
            make_artifact("draft"),
            make_artifact("document"),
            make_artifact("approval"),
        ];
        let provenance = Provenance {
            task_id: "task-1".to_string(),
            model_id: "model-1".to_string(),
            skill_id: "skill-1".to_string(),
            classification: Classification::Internal,
            evidence_ids: vec!["E1".to_string()],
            calculation_ids: vec!["C1".to_string()],
            draft_hash: "draft-hash".to_string(),
            artifact_hash: "artifact-hash".to_string(),
            exported_at: "2026-01-15T10:00:00Z".to_string(),
        };
        let package = EvidencePackage {
            artifacts,
            provenance,
            package_hash: "package-hash".to_string(),
        };
        assert!(package.is_complete());
    }

    #[test]
    fn incomplete_package_is_detected() {
        let artifacts = vec![make_artifact("draft")];
        let provenance = Provenance {
            task_id: "task-1".to_string(),
            model_id: "model-1".to_string(),
            skill_id: "skill-1".to_string(),
            classification: Classification::Internal,
            evidence_ids: vec![],
            calculation_ids: vec![],
            draft_hash: "draft-hash".to_string(),
            artifact_hash: "artifact-hash".to_string(),
            exported_at: "2026-01-15T10:00:00Z".to_string(),
        };
        let package = EvidencePackage {
            artifacts,
            provenance,
            package_hash: "package-hash".to_string(),
        };
        assert!(!package.is_complete());
    }

    #[test]
    fn sha256_is_deterministic() {
        let hash1 = compute_sha256(b"test content");
        let hash2 = compute_sha256(b"test content");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn different_content_produces_different_hash() {
        let hash1 = compute_sha256(b"test content 1");
        let hash2 = compute_sha256(b"test content 2");
        assert_ne!(hash1, hash2);
    }
}

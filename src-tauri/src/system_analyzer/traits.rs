//! System analyzer trait and data structures for Phase 2: System Analyzer

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wrapper for values that can be manually overridden
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OverrideValue<T> {
    /// Originally detected hardware/software value
    pub detected: T,
    /// Manually overridden value if any
    pub overridden: Option<T>,
    /// Flag indicating whether the value is currently overridden
    pub is_overridden: bool,
}

impl<T> OverrideValue<T> {
    /// Creates a new `OverrideValue` with detected data and no override
    pub fn new(detected: T) -> Self {
        Self {
            detected,
            overridden: None,
            is_overridden: false,
        }
    }

    /// Creates an `OverrideValue` with an initial override
    pub fn with_override(detected: T, overridden: T) -> Self {
        Self {
            detected,
            overridden: Some(overridden),
            is_overridden: true,
        }
    }

    /// Gets a reference to the active value (overridden if present, otherwise detected)
    pub fn current(&self) -> &T {
        if self.is_overridden {
            if let Some(ref val) = self.overridden {
                return val;
            }
        }
        &self.detected
    }
}

/// Detailed information about the CPU
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuInfo {
    pub manufacturer: String,
    pub model: String,
    pub architecture: String,
    pub physical_cores: u32,
    pub logical_processors: u32,
    pub base_frequency_mhz: u64,
    pub boost_frequency_mhz: u64,
    pub cache_l1_kb: Option<u32>,
    pub cache_l2_kb: Option<u32>,
    pub cache_l3_kb: Option<u32>,
    pub virtualization_supported: bool,
    pub simd_capabilities: Vec<String>,
}

/// Detailed information about a GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub is_dedicated: bool,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
    pub cuda_supported: bool,
    pub rocm_supported: bool,
    pub directx_supported: bool,
    pub vulkan_supported: bool,
    pub opencl_supported: bool,
}

/// Information about system RAM
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
    pub memory_type: String,
    pub speed_mts: Option<u32>,
    pub total_slots: Option<u32>,
    pub populated_slots: Option<u32>,
}

/// Information about a storage drive
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageInfo {
    pub drive_name: String,
    pub mount_point: String,
    pub drive_type: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub file_system: String,
    pub is_ai_storage_ready: bool,
}

/// Operating system details
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OsInfo {
    pub name: String,
    pub edition: String,
    pub version: String,
    pub build_number: String,
    pub architecture: String,
    pub locale: String,
}

/// Status and info for individual software tools
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoftwareDetectorInfo {
    pub name: String,
    pub installed: bool,
    pub version: Option<String>,
    pub path: Option<String>,
}

/// Status of local development software dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoftwareEnvironment {
    pub python: SoftwareDetectorInfo,
    pub rust: SoftwareDetectorInfo,
    pub cargo: SoftwareDetectorInfo,
    pub git: SoftwareDetectorInfo,
    pub nodejs: SoftwareDetectorInfo,
    pub npm: SoftwareDetectorInfo,
    pub pnpm: SoftwareDetectorInfo,
    pub ollama: SoftwareDetectorInfo,
    pub cuda_toolkit: SoftwareDetectorInfo,
    pub vc_redistributable: SoftwareDetectorInfo,
    pub additional: Vec<SoftwareDetectorInfo>,
}

/// Local AI runtime status (e.g., Ollama, vLLM, Llama.cpp)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AIRuntimeInfo {
    pub name: String,
    pub status: String,
    pub version: Option<String>,
    pub endpoint: Option<String>,
    pub models_available: Vec<String>,
}

/// Key directories on the system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemPaths {
    pub user_home: String,
    pub downloads: String,
    pub documents: String,
    pub desktop: String,
    pub app_data: String,
    pub cache_dir: String,
    pub model_storage_dir: String,
}

/// AI model running capabilities computed based on hardware specs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AICapabilityProfile {
    pub max_recommended_model_size_bytes: Option<u64>,
    pub recommended_quantizations: Vec<String>,
    pub recommended_context_length: Option<u32>,
    pub preferred_inference_backend: Option<String>,
    pub multi_model_capable: bool,
    pub lora_ready: bool,
    pub vision_ready: bool,
    pub embedding_ready: bool,
    pub extra_capabilities: HashMap<String, serde_json::Value>,
}

/// Result of evaluating the system against Sarathi AI requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemValidationResult {
    pub is_ready_for_ai: bool,
    pub score: u32,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Comprehensive hardware and environment profile
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub id: String,
    pub profile_created_at: String,
    pub profile_updated_at: String,
    pub cpu: OverrideValue<CpuInfo>,
    pub gpus: OverrideValue<Vec<GpuInfo>>,
    pub memory: OverrideValue<MemoryInfo>,
    pub storage: OverrideValue<Vec<StorageInfo>>,
    pub os: OverrideValue<OsInfo>,
    pub software: OverrideValue<SoftwareEnvironment>,
    pub ai_runtimes: OverrideValue<Vec<AIRuntimeInfo>>,
    pub paths: SystemPaths,
    pub ai_capabilities: AICapabilityProfile,
    pub validation: SystemValidationResult,
}

/// Interface for system analysis components
pub trait SystemAnalyzer: Send + Sync {
    /// Performs a full system analysis gathering all specs
    fn collect_all(&self) -> Result<HardwareProfile>;

    /// Detects CPU specifications
    fn detect_cpu(&self) -> Result<CpuInfo>;

    /// Detects GPU device specifications
    fn detect_gpus(&self) -> Result<Vec<GpuInfo>>;

    /// Detects system RAM
    fn detect_memory(&self) -> Result<MemoryInfo>;

    /// Detects disk storage drives
    fn detect_storage(&self) -> Result<Vec<StorageInfo>>;

    /// Detects operating system version and architecture
    fn detect_os(&self) -> Result<OsInfo>;

    /// Detects developer software environment (Python, Rust, Git, Node, CUDA, etc.)
    fn detect_software(&self) -> Result<SoftwareEnvironment>;

    /// Detects active AI runtimes (e.g., Ollama HTTP API)
    fn detect_ai_runtimes(&self) -> Result<Vec<AIRuntimeInfo>>;

    /// Resolves standard system directories
    fn detect_paths(&self) -> Result<SystemPaths>;

    /// Assembles raw collector outputs into a normalized HardwareProfile
    fn normalize(
        &self,
        raw_cpu: CpuInfo,
        raw_gpus: Vec<GpuInfo>,
        raw_memory: MemoryInfo,
        raw_storage: Vec<StorageInfo>,
        raw_os: OsInfo,
        raw_software: SoftwareEnvironment,
        raw_ai_runtimes: Vec<AIRuntimeInfo>,
        raw_paths: SystemPaths,
    ) -> Result<HardwareProfile>;

    /// Validates hardware against local AI criteria
    fn validate(&self, profile: &HardwareProfile) -> Result<SystemValidationResult>;
}

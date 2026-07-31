//! Phase 3: Model Recommendation Engine — Data Structures
//!
//! All types used across the recommendation pipeline:
//! Budget → Catalog → Estimator → Scorer → Recommendation

use serde::{Deserialize, Serialize};

// ─── Memory Domain Model ────────────────────────────────────────────────────

/// Distinct memory domains — never pooled together.
/// Each GPU gets its own budget; system RAM is separate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryBudget {
    pub gpu_budgets: Vec<GpuMemoryBudget>,
    pub system_ram: SystemRamBudget,
}

/// Per-GPU memory budget with separate domains for dedicated VRAM,
/// GPU-accessible shared system memory, and acceleration capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuMemoryBudget {
    pub gpu_index: usize,
    pub gpu_model: String,
    pub gpu_type: GpuType,
    /// Physical VRAM on discrete GPU, or BIOS-reserved buffer on iGPU
    pub total_dedicated_vram: u64,
    /// Usable dedicated VRAM after driver/DWM reservation
    pub usable_dedicated_vram: u64,
    /// Shared system memory accessible by this GPU (from DXGI SharedSystemMemory)
    pub total_shared_memory: u64,
    /// Usable shared memory after OS reservation
    pub usable_shared_memory: u64,
    pub cuda_available: bool,
    pub rocm_available: bool,
    pub vulkan_available: bool,
    pub directml_available: bool,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuType {
    Dedicated,
    Integrated,
}

/// System RAM budget (separate from any GPU memory domain)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemRamBudget {
    pub total_bytes: u64,
    /// Current available from OS telemetry at scan time
    pub available_bytes: u64,
    /// Usable for inference after OS + Sarathi reservation
    pub usable_for_inference: u64,
    /// DDR speed if known (for future bandwidth estimation)
    pub ram_speed_mts: Option<u32>,
}

// ─── Model Catalog ──────────────────────────────────────────────────────────

/// Provider-independent model metadata for recommendation calculations.
/// All fields must be verified from authoritative model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    /// Canonical model identifier (e.g. "meta-llama/Llama-3.1-8B")
    pub id: String,
    /// Human-readable display name
    pub name: String,
    /// Model family (e.g. "Llama 3.1")
    pub family: String,
    /// Dense or MoE
    pub architecture: ModelArchitecture,
    /// Total parameter count (all experts for MoE)
    pub total_parameters: u64,
    /// Active parameters per forward pass (MoE only; None for Dense)
    pub active_parameters: Option<u64>,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of query attention heads
    pub num_attention_heads: u32,
    /// Number of KV heads (fewer for GQA/MQA; equal to query heads for MHA)
    pub num_kv_heads: u32,
    /// Per-head dimension in elements
    pub head_dimension: u32,
    /// Hidden size (embedding dimension)
    pub hidden_size: u32,
    /// Maximum context length supported by the model
    pub max_context_length: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Default weight dtype ("bf16", "fp16")
    pub default_dtype: String,
    /// Primary use cases (["chat", "code", "reasoning", "general"])
    pub use_cases: Vec<String>,
    /// Catalog version for freshness tracking
    pub catalog_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ModelArchitecture {
    Dense,
    MixtureOfExperts {
        num_experts: u32,
        active_experts: u32,
    },
}

// ─── Quantization ───────────────────────────────────────────────────────────

/// Specification for a single quantization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSpec {
    /// Display label (e.g. "Q4_K_M")
    pub label: String,
    /// Effective bits per weight including quantization metadata overhead
    pub bits_per_weight: f64,
    /// Quality ranking (higher = better; FP16=10, Q2_K=2)
    pub quality_rank: u32,
}

// ─── Run Mode & Backend ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode")]
pub enum RunMode {
    PureGpu { gpu_index: usize },
    GpuWithCpuOffload { gpu_index: usize, offload_fraction: f64 },
    MultiGpu { gpu_indices: Vec<usize> },
    PureCpu,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InferenceBackend {
    LlamaCppGguf,
    Ollama,
    VllmCuda,
    DirectML,
    VulkanCompute,
}

impl InferenceBackend {
    pub fn display_name(&self) -> &str {
        match self {
            Self::LlamaCppGguf => "llama.cpp (GGUF)",
            Self::Ollama => "Ollama",
            Self::VllmCuda => "vLLM (CUDA)",
            Self::DirectML => "DirectML",
            Self::VulkanCompute => "Vulkan Compute",
        }
    }

    /// Whether this backend supports CPU offloading of layers
    pub fn supports_cpu_offload(&self) -> bool {
        matches!(self, Self::LlamaCppGguf | Self::Ollama)
    }

    /// Whether this backend can use GPU shared system memory
    pub fn supports_shared_memory(&self) -> bool {
        matches!(self, Self::LlamaCppGguf | Self::Ollama | Self::DirectML | Self::VulkanCompute)
    }
}

// ─── Evaluated Configuration ────────────────────────────────────────────────

/// A single evaluated (quantization × context × backend × run_mode) configuration
#[derive(Debug, Clone)]
pub struct EvaluatedConfiguration {
    pub quantization: QuantizationSpec,
    pub context_length: u32,
    pub run_mode: RunMode,
    pub backend: InferenceBackend,
    pub weight_memory_bytes: u64,
    pub kv_cache_memory_bytes: u64,
    pub overhead_memory_bytes: u64,
    pub total_memory_bytes: u64,
    /// Portion assigned to GPU dedicated VRAM
    pub vram_required_bytes: u64,
    /// Portion assigned to system RAM (CPU offload or pure CPU)
    pub ram_required_bytes: u64,
    /// Portion assigned to GPU shared system memory
    pub shared_mem_required_bytes: u64,
    /// (usable - required) / usable for the binding memory domain
    pub headroom_ratio: f64,
    /// Whether this configuration fits within available resources
    pub fits: bool,
}

// ─── Fit Category ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FitCategory {
    /// Comfortable fit, safe headroom, expected good usability
    Recommended,
    /// Valid fit requiring more resources, offloading, or compromise
    Compatible,
    /// Theoretically feasible but insufficient confidence or tight resources
    MayRun,
}

impl FitCategory {
    pub fn display_name(&self) -> &str {
        match self {
            Self::Recommended => "Recommended",
            Self::Compatible => "Compatible",
            Self::MayRun => "May Run",
        }
    }
}

// ─── Model Recommendation (Output) ─────────────────────────────────────────

/// The output object consumed by the frontend and transferable to Phase 4.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRecommendation {
    // ── Identity (Phase 4 uses these to locate the model) ──
    pub model_id: String,
    pub model_name: String,
    pub model_family: String,
    pub provider_id: Option<String>,

    // ── Selected configuration (highest quality safe fit) ──
    pub quantization: String,
    pub quantization_bits_per_weight: f64,
    pub recommended_context: u32,
    pub max_possible_context: u32,
    pub backend: String,
    pub run_mode: String,

    // ── Resource estimates ──
    pub estimated_vram_bytes: u64,
    pub estimated_ram_bytes: u64,
    pub estimated_shared_mem_bytes: u64,
    pub estimated_total_memory_bytes: u64,
    pub headroom_percent: f64,

    // ── Scoring ──
    pub fit_score: f64,
    pub category: FitCategory,
    pub confidence: String,

    // ── Explainability ──
    pub explanation: String,
    pub warnings: Vec<String>,

    // ── Architecture info ──
    pub architecture: String,
    pub total_parameters: u64,
    pub active_parameters: Option<u64>,

    // ── Performance (Phase 3: always None) ──
    pub estimated_tokens_per_sec: Option<f64>,
    pub performance_note: String,
}

// ─── Estimator Configuration ────────────────────────────────────────────────

/// Configuration for the memory estimator.
/// The overhead factor is a configurable conservative heuristic (default 0.12 / 12%).
/// Future runtime measurements can calibrate/replace this value.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    /// Runtime/compute overhead factor applied to (weights + KV cache).
    /// Accounts for CUDA/Vulkan context buffers, GGUF metadata, activation
    /// tensors, and allocator fragmentation.
    /// Default: 0.12 (12%). Configurable for future calibration.
    pub overhead_factor: f64,
    /// Bytes per KV cache element (2 for FP16 KV cache, standard in
    /// llama.cpp / Ollama / vLLM)
    pub kv_cache_bytes_per_element: u32,
    /// Batch size for KV cache calculation (1 for single-user local inference)
    pub batch_size: u32,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            overhead_factor: 0.12,
            kv_cache_bytes_per_element: 2,
            batch_size: 1,
        }
    }
}

// ─── Budget Configuration ───────────────────────────────────────────────────

/// Configuration for the adaptive resource budget calculator.
/// All margins are configurable for future tuning.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Fraction of total RAM to reserve for OS + background tasks (default 0.10)
    pub ram_os_reserve_fraction: f64,
    /// Minimum OS RAM reservation in bytes (default 1 GB)
    pub ram_os_reserve_min: u64,
    /// Maximum OS RAM reservation in bytes (default 4 GB)
    pub ram_os_reserve_max: u64,
    /// Additional RAM reserved for Sarathi itself (default 256 MB)
    pub ram_sarathi_reserve: u64,
    /// Fraction of dedicated VRAM to reserve for DWM/driver (default 0.10)
    pub vram_reserve_fraction: f64,
    /// Minimum VRAM reservation in bytes (default 256 MB)
    pub vram_reserve_min: u64,
    /// Fraction of shared system memory usable for inference (default 0.50)
    pub shared_memory_usable_fraction: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            ram_os_reserve_fraction: 0.10,
            ram_os_reserve_min: 1_073_741_824,        // 1 GB
            ram_os_reserve_max: 4_294_967_296,        // 4 GB
            ram_sarathi_reserve: 268_435_456,          // 256 MB
            vram_reserve_fraction: 0.10,
            vram_reserve_min: 268_435_456,             // 256 MB
            shared_memory_usable_fraction: 0.50,
        }
    }
}

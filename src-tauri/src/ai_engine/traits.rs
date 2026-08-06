//! AI engine traits and data types
//!
//! Defines the abstract interface for AI inference backends,
//! runtime status tracking, generation parameters, and message types.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ─── Runtime Status ──────────────────────────────────────────────────────────

/// Represents the current state of the inference runtime
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RuntimeStatus {
    /// No model loaded, runtime idle
    NotLoaded,
    /// Model is being loaded into memory (step description included)
    Loading(String),
    /// Model loaded and ready for inference
    Ready,
    /// Actively generating tokens
    Generating,
    /// Model is being unloaded from memory
    Unloading,
    /// An error occurred
    Error(String),
}

impl std::fmt::Display for RuntimeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeStatus::NotLoaded => write!(f, "NotLoaded"),
            RuntimeStatus::Loading(step) => write!(f, "Loading: {}", step),
            RuntimeStatus::Ready => write!(f, "Ready"),
            RuntimeStatus::Generating => write!(f, "Generating"),
            RuntimeStatus::Unloading => write!(f, "Unloading"),
            RuntimeStatus::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

// ─── Backend Type Enum ───────────────────────────────────────────────────────

pub enum AIBackendType { Ollama, LlamaCpp, VLLM, Custom(String) }

// ─── Chat Message Types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponse {
    pub message: String,
    pub model: String,
    pub tokens_used: u32,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamChunk {
    /// The token text fragment
    pub text: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Running count of tokens generated so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_generated: Option<u32>,
    /// Reason generation finished (e.g., "stop", "length", "cancelled")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ─── Generation Parameters ───────────────────────────────────────────────────

/// Parameters controlling token generation/sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationParams {
    /// Sampling temperature (0.0 = greedy, higher = more random). Default: 0.7
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold. Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k sampling. Default: 40
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    /// Maximum tokens to generate. Default: 2048
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Min-p sampling threshold. Default: 0.05
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    /// Repeat penalty. Default: 1.1
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Mirostat sampling (0 = disabled, 1 = v1, 2 = v2). Default: 0
    #[serde(default = "default_mirostat")]
    pub mirostat: u32,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> u32 { 40 }
fn default_min_p() -> f32 { 0.05 }
fn default_max_tokens() -> u32 { 2048 }
fn default_repeat_penalty() -> f32 { 1.1 }
fn default_mirostat() -> u32 { 0 }

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            min_p: default_min_p(),
            max_tokens: default_max_tokens(),
            repeat_penalty: default_repeat_penalty(),
            mirostat: default_mirostat(),
        }
    }
}

// ─── Model Load Configuration ────────────────────────────────────────────────

/// Configuration passed to the runtime when loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadConfig {
    /// Absolute path to the GGUF file
    pub model_path: String,
    /// Model identifier (e.g., "meta-llama/Llama-3.2-1B")
    pub model_id: String,
    /// Human-readable model name
    pub model_name: String,
    /// Quantization label (e.g., "Q8_0")
    pub quantization: String,
    /// Context length in tokens (from Phase 3 recommendation or dynamic calculation)
    pub context_length: u32,
    /// Number of model layers to offload to GPU (0 = CPU-only)
    pub gpu_layers: u32,
    /// Number of CPU threads for inference
    pub threads: u32,
    /// Model chat template (e.g., "chatml", "llama3", "gemma", "mistral")
    #[serde(default = "default_chat_template")]
    pub chat_template: String,
    /// Stop tokens for generation cutoff
    #[serde(default)]
    pub stop_tokens: Vec<String>,
}

fn default_chat_template() -> String {
    "chatml".to_string()
}

// ─── Loaded Model Info ───────────────────────────────────────────────────────

/// Information about the currently loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedModelInfo {
    pub model_id: String,
    pub model_name: String,
    pub quantization: String,
    pub file_path: String,
    pub context_length: u32,
    pub gpu_layers: u32,
    pub threads: u32,
    pub backend_used: String,
    pub loaded_at: String,
    pub chat_template: String,
    /// Where the prompt formatting actually came from: `"gguf"` when the model's
    /// own template is in use, or `"fallback:<name>"` when a hand-written one is.
    #[serde(default = "default_template_source")]
    pub template_source: String,
    pub stop_tokens: Vec<String>,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    pub active_adapter: Option<String>,
}

fn default_model_family() -> String {
    "Generic".to_string()
}

fn default_template_source() -> String {
    "unknown".to_string()
}

// ─── Inference Status Payload (for Tauri events) ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceStatusPayload {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<LoadedModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ─── Abstract Backend Trait ──────────────────────────────────────────────────

pub trait AIBackend: Send + Sync {
    fn name(&self) -> &'static str { "Unknown" }
    fn backend_type(&self) -> AIBackendType { AIBackendType::Ollama }
    fn load_model(&self, _config: ModelLoadConfig) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unload_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn is_model_loaded(&self, _model_id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat_stream(&self, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_loaded_models(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_model_info(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait ContextManager: Send + Sync {
    fn add_message(&self, _message: ChatMessage) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_context(&self) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn clear_context(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn trim_context(&self, _max_tokens: u32) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_token_count(&self, _messages: Vec<ChatMessage>) -> Result<u32> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait MemoryManager: Send + Sync {
    fn save_conversation(&self, _id: &str, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn load_conversation(&self, _id: &str) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_conversations(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn delete_conversation(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

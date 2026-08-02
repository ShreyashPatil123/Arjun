//! Memory Provider Abstraction Trait
//! Defines the decoupled interface for all memory processing engines.

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractedFact {
    pub content: String,
    #[serde(alias = "memory_type")]
    pub memory_type: String,
    pub key: Option<String>,
    pub value: Option<String>,
    #[serde(alias = "importance_score")]
    pub importance_score: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoredCandidate {
    pub id: String,
    pub content: String,
    #[serde(alias = "memory_type")]
    pub memory_type: String,
    #[serde(alias = "project_id")]
    pub project_id: Option<String>,
    #[serde(alias = "importance_score")]
    pub importance_score: f64,
    #[serde(alias = "recency_timestamp")]
    pub recency_timestamp: i64,
    pub similarity: f64,
    #[serde(alias = "recency_score")]
    pub recency_score: Option<f64>,
    #[serde(alias = "final_score")]
    pub final_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompressedBlock {
    #[serde(alias = "compressed_text")]
    pub compressed_text: String,
    #[serde(alias = "tokens_used")]
    pub tokens_used: usize,
    #[serde(alias = "evicted_turns")]
    pub evicted_turns: usize,
    #[serde(alias = "retained_turns")]
    pub retained_turns: usize,
    #[serde(alias = "provider_used")]
    pub provider_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummaryResult {
    pub summary: String,
    #[serde(alias = "provider_used")]
    pub provider_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub content: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderHealthStatus {
    pub status: String,
    #[serde(alias = "registered_providers")]
    pub registered_providers: Vec<String>,
    pub capabilities: serde_json::Value,
}

#[async_trait::async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Provider ID
    fn provider_id(&self) -> &str;

    /// Health check
    async fn check_health(&self) -> Result<ProviderHealthStatus>;

    /// Extract facts from user message
    async fn extract_facts(&self, text: &str, context: Option<&str>) -> Result<Vec<ExtractedFact>>;

    /// Compress context window turns
    async fn compress_context(&self, messages: &[serde_json::Value], max_tokens: usize) -> Result<CompressedBlock>;

    /// Summarize session turns
    async fn summarize_session(&self, messages: &[serde_json::Value]) -> Result<SessionSummaryResult>;

    /// Rank memory candidates
    async fn calculate_rankings(&self, candidates: &[ScoredCandidate], query: &str) -> Result<Vec<ScoredCandidate>>;

    /// Chunk document for RAG
    async fn chunk_document(&self, text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<DocumentChunk>>;
}

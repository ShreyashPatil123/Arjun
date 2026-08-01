//! PythonSidecarProvider — Default MemoryProvider implementation using Stdio NDJSON-RPC
//! Routes requests to the Python sidecar process.

use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

use crate::memory_engine::adapters::sidecar_adapter::SidecarAdapter;
use crate::memory_engine::traits::*;

pub struct PythonSidecarProvider {
    adapter: Arc<SidecarAdapter>,
}

impl PythonSidecarProvider {
    pub fn new(app_data_dir: &PathBuf) -> Result<Self> {
        let adapter = SidecarAdapter::spawn(app_data_dir)?;
        Ok(Self {
            adapter: Arc::new(adapter),
        })
    }
}

#[async_trait::async_trait]
impl MemoryProvider for PythonSidecarProvider {
    fn provider_id(&self) -> &str {
        "python_sidecar"
    }

    async fn check_health(&self) -> Result<ProviderHealthStatus> {
        let res = self.adapter.call_rpc("health_check", json!({}))?;
        let health: ProviderHealthStatus = serde_json::from_value(res)?;
        Ok(health)
    }

    async fn extract_facts(&self, text: &str, context: Option<&str>) -> Result<Vec<ExtractedFact>> {
        let res = self.adapter.call_rpc(
            "extract_facts",
            json!({
                "text": text,
                "context": context
            }),
        )?;

        let facts_val = res.get("facts").cloned().unwrap_or(json!([]));
        let facts: Vec<ExtractedFact> = serde_json::from_value(facts_val)?;
        Ok(facts)
    }

    async fn compress_context(&self, messages: &[serde_json::Value], max_tokens: usize) -> Result<CompressedBlock> {
        let res = self.adapter.call_rpc(
            "compress_context",
            json!({
                "messages": messages,
                "max_tokens": max_tokens
            }),
        )?;

        let block: CompressedBlock = serde_json::from_value(res)?;
        Ok(block)
    }

    async fn summarize_session(&self, messages: &[serde_json::Value]) -> Result<SessionSummaryResult> {
        let res = self.adapter.call_rpc(
            "summarize_session",
            json!({
                "messages": messages
            }),
        )?;

        let summary: SessionSummaryResult = serde_json::from_value(res)?;
        Ok(summary)
    }

    async fn calculate_rankings(&self, candidates: &[ScoredCandidate], query: &str) -> Result<Vec<ScoredCandidate>> {
        let res = self.adapter.call_rpc(
            "calculate_rankings",
            json!({
                "candidates": candidates,
                "query": query
            }),
        )?;

        let ranked_val = res.get("ranked_candidates").cloned().unwrap_or(json!([]));
        let ranked: Vec<ScoredCandidate> = serde_json::from_value(ranked_val)?;
        Ok(ranked)
    }

    async fn chunk_document(&self, text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<DocumentChunk>> {
        let res = self.adapter.call_rpc(
            "chunk_document",
            json!({
                "text": text,
                "chunk_size": chunk_size,
                "overlap": overlap
            }),
        )?;

        let chunks_val = res.get("chunks").cloned().unwrap_or(json!([]));
        let chunks: Vec<DocumentChunk> = serde_json::from_value(chunks_val)?;
        Ok(chunks)
    }
}

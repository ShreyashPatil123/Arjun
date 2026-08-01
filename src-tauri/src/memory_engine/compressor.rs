//! Context Window Token Compressor
//! Manages token budgeting and compresses overflowing context windows via MemoryProvider.

use anyhow::Result;
use std::sync::Arc;
use serde_json::json;
use crate::memory_engine::traits::{CompressedBlock, MemoryProvider};

pub struct Compressor {
    provider: Arc<dyn MemoryProvider>,
}

impl Compressor {
    pub fn new(provider: Arc<dyn MemoryProvider>) -> Self {
        Self { provider }
    }

    pub async fn compress(&self, messages: &[crate::ai_engine::traits::ChatMessage], max_tokens: usize) -> Result<CompressedBlock> {
        let json_msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();

        self.provider.compress_context(&json_msgs, max_tokens).await
    }
}

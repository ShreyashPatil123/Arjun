//! Rolling Session Summarizer
//! Distills conversation history turns into rolling summaries via MemoryProvider.

use anyhow::Result;
use std::sync::Arc;
use serde_json::json;
use crate::memory_engine::traits::MemoryProvider;

pub struct Summarizer {
    provider: Arc<dyn MemoryProvider>,
}

impl Summarizer {
    pub fn new(provider: Arc<dyn MemoryProvider>) -> Self {
        Self { provider }
    }

    pub async fn summarize_turns(&self, messages: &[crate::ai_engine::traits::ChatMessage]) -> Result<String> {
        let json_msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();

        let res = self.provider.summarize_session(&json_msgs).await?;
        Ok(res.summary)
    }
}

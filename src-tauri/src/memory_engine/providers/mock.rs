//! MockProvider — Pure Rust mock memory provider for unit testing.

use anyhow::Result;
use serde_json::json;
use crate::memory_engine::traits::*;

pub struct MockProvider;

impl MockProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl MemoryProvider for MockProvider {
    fn provider_id(&self) -> &str {
        "mock"
    }

    async fn check_health(&self) -> Result<ProviderHealthStatus> {
        Ok(ProviderHealthStatus {
            status: "healthy".to_string(),
            registered_providers: vec!["mock".to_string()],
            capabilities: json!({"extraction": true, "summarization": true}),
        })
    }

    async fn extract_facts(&self, text: &str, _context: Option<&str>) -> Result<Vec<ExtractedFact>> {
        let mut facts = Vec::new();
        let lower = text.to_lowercase();

        if lower.contains("my name is") || lower.contains("i am ") || lower.contains("call me ") {
            // Extract the name after the trigger phrase
            let name = if let Some(pos) = lower.find("my name is") {
                text[pos + "my name is".len()..].trim().trim_matches(&['.', ',', '!', '?'][..]).to_string()
            } else if let Some(pos) = lower.find("call me ") {
                text[pos + "call me ".len()..].trim().trim_matches(&['.', ',', '!', '?'][..]).to_string()
            } else if let Some(pos) = lower.find("i am ") {
                text[pos + "i am ".len()..].trim().trim_matches(&['.', ',', '!', '?'][..]).to_string()
            } else {
                text.split_whitespace().last().unwrap_or("User").to_string()
            };
            facts.push(ExtractedFact {
                content: format!("User's name is {}", name),
                memory_type: "user_fact".to_string(),
                key: Some("name".to_string()),
                value: Some(name),
                importance_score: 0.98,
                confidence: 0.99,
            });
        }

        if lower.contains("birthday") || lower.contains("born on") {
            facts.push(ExtractedFact {
                content: text.to_string(),
                memory_type: "user_fact".to_string(),
                key: Some("birthday".to_string()),
                value: Some(text.to_string()),
                importance_score: 0.95,
                confidence: 0.98,
            });
        }

        if lower.contains("study at") || lower.contains("college") || lower.contains("university") {
            facts.push(ExtractedFact {
                content: text.to_string(),
                memory_type: "user_fact".to_string(),
                key: Some("education".to_string()),
                value: Some(text.to_string()),
                importance_score: 0.92,
                confidence: 0.96,
            });
        }

        if lower.contains("laptop") || lower.contains("pc") || lower.contains("device") {
            facts.push(ExtractedFact {
                content: text.to_string(),
                memory_type: "user_fact".to_string(),
                key: Some("device".to_string()),
                value: Some(text.to_string()),
                importance_score: 0.90,
                confidence: 0.95,
            });
        }

        if lower.contains("like ") || lower.contains("love ") || lower.contains("prefer ") {
            facts.push(ExtractedFact {
                content: text.to_string(),
                memory_type: "preference".to_string(),
                key: Some("preference".to_string()),
                value: Some(text.to_string()),
                importance_score: 0.88,
                confidence: 0.94,
            });
        }

        Ok(facts)
    }

    async fn compress_context(&self, messages: &[serde_json::Value], _max_tokens: usize) -> Result<CompressedBlock> {
        Ok(CompressedBlock {
            compressed_text: format!("Mock compressed context for {} messages", messages.len()),
            tokens_used: 10,
            evicted_turns: messages.len().saturating_sub(2),
            retained_turns: 2,
            provider_used: "mock".to_string(),
        })
    }

    async fn summarize_session(&self, messages: &[serde_json::Value]) -> Result<SessionSummaryResult> {
        Ok(SessionSummaryResult {
            summary: format!("Mock summary of {} turns", messages.len()),
            provider_used: "mock".to_string(),
        })
    }

    async fn calculate_rankings(&self, candidates: &[ScoredCandidate], _query: &str) -> Result<Vec<ScoredCandidate>> {
        let mut ranked = candidates.to_vec();
        for item in &mut ranked {
            item.final_score = Some(item.similarity * 0.5 + item.importance_score * 0.5);
        }
        ranked.sort_by(|a, b| b.final_score.unwrap_or(0.0).partial_cmp(&a.final_score.unwrap_or(0.0)).unwrap());
        Ok(ranked)
    }

    async fn chunk_document(&self, text: &str, _chunk_size: usize, _overlap: usize) -> Result<Vec<DocumentChunk>> {
        Ok(vec![DocumentChunk {
            chunk_id: "chunk_0".to_string(),
            content: text.to_string(),
            token_count: text.len() / 4,
        }])
    }
}

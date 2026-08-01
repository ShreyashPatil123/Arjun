//! Multi-Tier Hybrid Memory Retriever
//! Queries SQLite for candidate memory nodes, filters by project ID, and ranks using MemoryProvider.

use anyhow::Result;
use std::sync::Arc;

use crate::memory_engine::persistence::PersistenceManager;
use crate::memory_engine::ranking::RankingEngine;
use crate::memory_engine::traits::{MemoryProvider, ScoredCandidate};

pub struct Retriever {
    provider: Arc<dyn MemoryProvider>,
    persistence: Arc<PersistenceManager>,
}

impl Retriever {
    pub fn new(provider: Arc<dyn MemoryProvider>, persistence: Arc<PersistenceManager>) -> Self {
        Self {
            provider,
            persistence,
        }
    }

    /// Retrieves relevant memory nodes scoped strictly to project_id
    pub async fn retrieve(&self, query: &str, project_id: Option<&str>, limit: usize) -> Result<Vec<ScoredCandidate>> {
        let records = self.persistence.get_memories_by_project(project_id, 50)?;

        let candidates: Vec<ScoredCandidate> = records
            .into_iter()
            .map(|r| {
                let sim = Self::calculate_text_similarity(query, &r.content);
                ScoredCandidate {
                    id: r.id,
                    content: r.content,
                    memory_type: r.memory_type,
                    project_id: r.project_id,
                    importance_score: r.importance_score,
                    recency_timestamp: r.recency_timestamp,
                    similarity: sim,
                    recency_score: None,
                    final_score: None,
                }
            })
            .collect();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Attempt provider ranking with fallback to pure Rust RankingEngine
        let ranked = match self.provider.calculate_rankings(&candidates, query).await {
            Ok(res) if !res.is_empty() => res,
            _ => RankingEngine::rank_candidates(&candidates, query),
        };

        Ok(ranked.into_iter().take(limit).collect())
    }

    fn calculate_text_similarity(query: &str, text: &str) -> f64 {
        let clean_q = query.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect::<String>();
        let clean_t = text.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect::<String>();

        let q_words: Vec<&str> = clean_q.split_whitespace().collect();
        let t_words: Vec<&str> = clean_t.split_whitespace().collect();
        if q_words.is_empty() || t_words.is_empty() {
            return 0.2;
        }

        let matches = q_words.iter().filter(|w| t_words.contains(w)).count();
        let score = (matches as f64 / q_words.len() as f64).clamp(0.1, 0.95);
        score
    }
}

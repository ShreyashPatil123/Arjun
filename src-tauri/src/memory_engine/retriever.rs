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
        let start_time = std::time::Instant::now();
        log::info!(
            "[MEMORY_RETRIEVAL] Query entered: \"{}\" | Project Scope: {:?} | Limit: {}",
            query, project_id, limit
        );

        let records = self.persistence.get_memories_by_project(project_id, 50)?;
        log::info!(
            "[MEMORY_RETRIEVAL] Fetched {} raw candidate(s) from SQLite persistence for project scope {:?}",
            records.len(), project_id
        );

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
            log::info!("[MEMORY_RETRIEVAL] No candidates found in database. Returning empty set. Latency: {}ms", start_time.elapsed().as_millis());
            return Ok(Vec::new());
        }

        for (i, c) in candidates.iter().enumerate().take(10) {
            log::info!(
                "[MEMORY_RETRIEVAL_CANDIDATE] [{}] ID='{}' | Type='{}' | Sim={:.3} | Imp={:.2} | Content=\"{}\"",
                i + 1, c.id, c.memory_type, c.similarity, c.importance_score, &c.content[..c.content.len().min(60)]
            );
        }

        // Attempt provider ranking with fallback to pure Rust RankingEngine
        let ranked = match self.provider.calculate_rankings(&candidates, query).await {
            Ok(res) if !res.is_empty() => {
                log::info!("[MEMORY_RETRIEVAL] Provider '{}' calculated candidate rankings successfully", self.provider.provider_id());
                res
            }
            _ => {
                log::info!("[MEMORY_RETRIEVAL_FALLBACK] Provider ranking unavailable/empty, using Rust RankingEngine fallback");
                RankingEngine::rank_candidates(&candidates, query)
            }
        };

        let selected: Vec<ScoredCandidate> = ranked.into_iter().take(limit).collect();
        let latency_ms = start_time.elapsed().as_millis();

        log::info!(
            "[MEMORY_RETRIEVAL_SUCCESS] Selected Top-{} ranked memory candidate(s) in {}ms:",
            selected.len(), latency_ms
        );
        for (i, s) in selected.iter().enumerate() {
            log::info!(
                "[MEMORY_RETRIEVAL_SELECTED] #{} ID='{}' | Score={:.3} | Type='{}' | Content=\"{}\"",
                i + 1, s.id, s.final_score.unwrap_or(s.similarity), s.memory_type, &s.content[..s.content.len().min(80)]
            );
        }

        Ok(selected)
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

//! Scoring & Ranking Engine — Multi-Factor Score Calculation
//! Combines semantic similarity, exponential recency decay, and importance score.

use crate::memory_engine::traits::ScoredCandidate;

pub struct RankingEngine;

impl RankingEngine {
    /// Pure Rust fallback scoring formula: Score = 0.5 * similarity + 0.3 * recency_decay + 0.2 * importance
    pub fn rank_candidates(candidates: &[ScoredCandidate], _query: &str) -> Vec<ScoredCandidate> {
        let now = chrono::Utc::now().timestamp();
        let decay_lambda = 0.005; // half-life ~138h

        let mut ranked: Vec<ScoredCandidate> = candidates
            .iter()
            .map(|c| {
                let hours_elapsed = ((now - c.recency_timestamp) as f64 / 3600.0).max(0.0);
                let recency_score = (-decay_lambda * hours_elapsed).exp();
                let final_score = (0.50 * c.similarity) + (0.30 * recency_score) + (0.20 * c.importance_score);

                let mut item = c.clone();
                item.recency_score = Some(recency_score);
                item.final_score = Some(final_score);
                item
            })
            .collect();

        ranked.sort_by(|a, b| b.final_score.unwrap_or(0.0).partial_cmp(&a.final_score.unwrap_or(0.0)).unwrap());
        ranked
    }
}

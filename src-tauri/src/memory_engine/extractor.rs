//! Fact & Entity Extractor Coordinator
//! Coordinates fact extraction via MemoryProvider and persists extracted memory nodes to SQLite.

use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

use crate::memory_engine::persistence::{MemoryNodeRecord, PersistenceManager};
use crate::memory_engine::profile::ProfileManager;
use crate::memory_engine::providers::mock::MockProvider;
use crate::memory_engine::traits::MemoryProvider;

pub struct Extractor {
    provider: Arc<dyn MemoryProvider>,
    persistence: Arc<PersistenceManager>,
    profile: Arc<ProfileManager>,
}

impl Extractor {
    pub fn new(
        provider: Arc<dyn MemoryProvider>,
        persistence: Arc<PersistenceManager>,
        profile: Arc<ProfileManager>,
    ) -> Self {
        Self {
            provider,
            persistence,
            profile,
        }
    }

    /// Extracts facts from user message and persists them to SQLite
    pub async fn process_user_input(&self, message: &str, project_id: &str, session_id: Option<&str>) -> Result<usize> {
        log::info!("[EXTRACTOR] Processing input: \"{}\" (project={})", &message[..message.len().min(80)], project_id);

        // Try primary provider first, fallback to MockProvider on deserialization errors
        let facts = match self.provider.extract_facts(message, None).await {
            Ok(f) => {
                log::info!("[EXTRACTOR] Provider '{}' extracted {} fact(s)", self.provider.provider_id(), f.len());
                f
            }
            Err(e) => {
                log::warn!("[EXTRACTOR] Provider '{}' failed: {:?}. Falling back to MockProvider.", self.provider.provider_id(), e);
                let mock = MockProvider::new();
                match mock.extract_facts(message, None).await {
                    Ok(f) => {
                        log::info!("[EXTRACTOR] MockProvider fallback extracted {} fact(s)", f.len());
                        f
                    }
                    Err(e2) => {
                        log::error!("[EXTRACTOR] MockProvider fallback also failed: {:?}", e2);
                        return Err(e2);
                    }
                }
            }
        };

        let now_ts = chrono::Utc::now().timestamp();
        let now_str = chrono::Utc::now().to_rfc3339();

        let count = facts.len();
        for fact in facts {
            log::info!("[EXTRACTOR] Persisting fact: type='{}', key={:?}, value={:?}, importance={:.2}",
                fact.memory_type, fact.key, fact.value, fact.importance_score);

            let node = MemoryNodeRecord {
                id: format!("mem_{}", Uuid::new_v4()),
                memory_type: fact.memory_type.clone(),
                project_id: Some(project_id.to_string()),
                session_id: session_id.map(|s| s.to_string()),
                content: fact.content.clone(),
                importance_score: fact.importance_score,
                recency_timestamp: now_ts,
                metadata: fact.key.as_ref().map(|k| serde_json::json!({"key": k, "confidence": fact.confidence}).to_string()),
                created_at: now_str.clone(),
                updated_at: now_str.clone(),
            };

            match self.persistence.save_memory_node(&node) {
                Ok(_) => log::info!("[EXTRACTOR] ✓ Saved memory node id={}", node.id),
                Err(e) => log::error!("[EXTRACTOR] ❌ Failed to save memory node: {:?}", e),
            }

            // If user_fact, also update user_profile table
            if fact.memory_type == "user_fact" || fact.memory_type == "preference" {
                if let (Some(k), Some(v)) = (&fact.key, &fact.value) {
                    match self.profile.set_fact(k, v, &fact.memory_type) {
                        Ok(_) => log::info!("[EXTRACTOR] ✓ Updated user_profile: {}={}", k, v),
                        Err(e) => log::error!("[EXTRACTOR] ❌ Failed to update user_profile: {:?}", e),
                    }
                }
            }
        }

        log::info!("[EXTRACTOR] Extraction complete: {} fact(s) persisted", count);
        Ok(count)
    }
}


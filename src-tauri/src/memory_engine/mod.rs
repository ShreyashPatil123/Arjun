//! Phase 6 Hybrid Local Memory Engine Module Root

pub mod traits;
pub mod adapters;
pub mod providers;
pub mod persistence;
pub mod profile;
pub mod project;
pub mod session;
pub mod ranking;
pub mod extractor;
pub mod retriever;
pub mod summarizer;
pub mod compressor;
pub mod injector;
pub mod manager;
pub mod api;

pub use manager::MemoryManager;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use traits::*;
    use providers::mock::MockProvider;
    use persistence::PersistenceManager;
    use profile::ProfileManager;
    use project::ProjectManager;
    use injector::PromptInjector;
    use crate::ai_engine::traits::ChatMessage;

    #[tokio::test]
    async fn test_mock_provider_capabilities() {
        let mock = MockProvider::new();
        let health = mock.check_health().await.unwrap();
        assert_eq!(health.status, "healthy");

        let facts = mock.extract_facts("I build apps in Rust", None).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value.as_deref(), Some("Rust"));
    }

    #[test]
    fn test_sqlite_persistence_roundtrip() {
        let temp_dir = std::env::temp_dir().join("sarathi_test_memory");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let persistence = Arc::new(PersistenceManager::new(&temp_dir).unwrap());
        let profile = ProfileManager::new(persistence.clone());
        let project = ProjectManager::new(persistence.clone());

        profile.set_fact("favorite_editor", "VSCode", "preferences").unwrap();
        let records = profile.get_profile().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, "favorite_editor");
        assert_eq!(records[0].value, "VSCode");

        let proj = project.create_project("TrackOcean", Some("SIH Project")).unwrap();
        assert_eq!(proj.name, "TrackOcean");

        let projects = project.list_projects().unwrap();
        assert!(projects.iter().any(|p| p.name == "TrackOcean"));
    }

    #[test]
    fn test_prompt_injector_preserves_context() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "What is my favorite language?".to_string(),
            timestamp: None,
        }];

        let user_profile = "- language: Rust\n- editor: VSCode";
        let project_name = "TrackOcean";
        let memories = vec![ScoredCandidate {
            id: "1".to_string(),
            content: "User prefers Rust for performance".to_string(),
            memory_type: "preference".to_string(),
            project_id: Some("proj_trackocean".to_string()),
            importance_score: 0.9,
            recency_timestamp: 1000,
            similarity: 0.85,
            recency_score: Some(0.9),
            final_score: Some(0.88),
        }];

        let injected = PromptInjector::inject_memory_into_messages(
            &messages,
            user_profile,
            project_name,
            &memories,
            Some("Previous context summary"),
        );

        assert_eq!(injected.len(), 2);
        assert_eq!(injected[0].role, "system");
        assert!(injected[0].content.contains("[SARATHI MEMORY ENGINE - ACTIVE PROJECT: TRACKOCEAN]"));
        assert!(injected[0].content.contains("language: Rust"));
        assert!(injected[0].content.contains("User prefers Rust for performance"));
    }
}

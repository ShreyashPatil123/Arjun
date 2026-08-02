//! MemoryManager — Supreme Orchestrator & Unified Entrypoint for Sarathi Memory Engine
//! Wraps MemoryProvider, PersistenceManager, ProfileManager, ProjectManager, Extractor, Retriever, Summarizer, Compressor, and PromptInjector.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ai_engine::traits::ChatMessage;
use crate::memory_engine::compressor::Compressor;
use crate::memory_engine::extractor::Extractor;
use crate::memory_engine::injector::PromptInjector;
use crate::memory_engine::persistence::{MemoryNodeRecord, PersistenceManager, ProjectRecord, UserProfileRecord};
use crate::memory_engine::profile::ProfileManager;
use crate::memory_engine::project::ProjectManager;
use crate::memory_engine::providers::mock::MockProvider;
use crate::memory_engine::providers::python_sidecar::PythonSidecarProvider;
use crate::memory_engine::retriever::Retriever;
use crate::memory_engine::summarizer::Summarizer;
use crate::memory_engine::traits::{MemoryProvider, ProviderHealthStatus, ScoredCandidate};

pub struct MemoryManager {
    provider: Arc<dyn MemoryProvider>,
    persistence: Arc<PersistenceManager>,
    profile: Arc<ProfileManager>,
    project: Arc<ProjectManager>,
    extractor: Arc<Extractor>,
    retriever: Arc<Retriever>,
    summarizer: Arc<Summarizer>,
    compressor: Arc<Compressor>,
}

impl MemoryManager {
    /// Initializes MemoryManager with default PythonSidecarProvider (or MockProvider fallback)
    pub fn new(app_data_dir: &PathBuf) -> Self {
        let persistence = Arc::new(PersistenceManager::new(app_data_dir).expect("Failed to init SQLite persistence"));
        let profile = Arc::new(ProfileManager::new(persistence.clone()));
        let project = Arc::new(ProjectManager::new(persistence.clone()));

        // Try initializing PythonSidecarProvider, fallback gracefully to MockProvider
        let provider: Arc<dyn MemoryProvider> = match PythonSidecarProvider::new(app_data_dir) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                log::warn!("[MEMORY_MANAGER WARN] Failed to spawn Python sidecar provider (falling back to MockProvider): {}", e);
                Arc::new(MockProvider::new())
            }
        };

        let extractor = Arc::new(Extractor::new(provider.clone(), persistence.clone(), profile.clone()));
        let retriever = Arc::new(Retriever::new(provider.clone(), persistence.clone()));
        let summarizer = Arc::new(Summarizer::new(provider.clone()));
        let compressor = Arc::new(Compressor::new(provider.clone()));

        Self {
            provider,
            persistence,
            profile,
            project,
            extractor,
            retriever,
            summarizer,
            compressor,
        }
    }

    /// Health check
    pub async fn check_health(&self) -> Result<ProviderHealthStatus> {
        self.provider.check_health().await
    }

    /// Process user message turn: extract facts & save to SQLite
    pub async fn process_user_turn(&self, message: &str, session_id: Option<&str>) -> Result<usize> {
        let project_id = self.project.get_active_project_id();
        self.extractor.process_user_input(message, &project_id, session_id).await
    }

    /// Prepares injected messages before inference turn
    pub async fn prepare_injected_messages(
        &self,
        messages: &[ChatMessage],
        query: &str,
    ) -> Result<Vec<ChatMessage>> {
        let project_id = self.project.get_active_project_id();
        let user_profile_summary = self.profile.build_profile_summary();
        let retrieved = self.retriever.retrieve(query, Some(&project_id), 5).await.unwrap_or_default();
        let summary = if messages.len() > 6 {
            self.summarizer.summarize_turns(messages).await.ok()
        } else {
            None
        };

        let injected = PromptInjector::inject_memory_into_messages(
            messages,
            &user_profile_summary,
            &project_id,
            &retrieved,
            summary.as_deref(),
        );

        Ok(injected)
    }

    /// Profile CRUD
    pub fn get_user_profile(&self) -> Result<Vec<UserProfileRecord>> {
        self.profile.get_profile()
    }

    pub fn set_user_profile_fact(&self, key: &str, value: &str, category: &str) -> Result<()> {
        self.profile.set_fact(key, value, category)
    }

    /// Projects CRUD
    pub fn get_active_project_id(&self) -> String {
        self.project.get_active_project_id()
    }

    pub fn set_active_project_id(&self, project_id: &str) {
        self.project.set_active_project_id(project_id)
    }

    pub fn create_project(&self, name: &str, description: Option<&str>) -> Result<ProjectRecord> {
        self.project.create_project(name, description)
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectRecord>> {
        self.project.list_projects()
    }

    /// Memory Search & Delete
    pub async fn search_memories(&self, query: &str, project_id: Option<&str>) -> Result<Vec<ScoredCandidate>> {
        self.retriever.retrieve(query, project_id, 20).await
    }

    pub fn delete_memory(&self, id: &str) -> Result<()> {
        self.persistence.delete_memory_node(id)
    }

    pub fn provider_id(&self) -> &str {
        self.provider.provider_id()
    }

    pub fn get_counts(&self) -> Result<(usize, usize, usize)> {
        self.persistence.get_memory_counts()
    }
}

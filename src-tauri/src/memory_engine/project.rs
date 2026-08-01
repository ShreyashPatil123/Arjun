//! Project Memory Manager
//! Manages isolated project scopes (Sarathi, TrackOcean, SIH, College, Personal, etc.).

use anyhow::Result;
use std::sync::{Arc, Mutex};
use crate::memory_engine::persistence::{PersistenceManager, ProjectRecord};

pub struct ProjectManager {
    persistence: Arc<PersistenceManager>,
    active_project_id: Arc<Mutex<String>>,
}

impl ProjectManager {
    pub fn new(persistence: Arc<PersistenceManager>) -> Self {
        Self {
            persistence,
            active_project_id: Arc::new(Mutex::new("proj_general".to_string())),
        }
    }

    pub fn get_active_project_id(&self) -> String {
        self.active_project_id.lock().unwrap().clone()
    }

    pub fn set_active_project_id(&self, project_id: &str) {
        let mut lock = self.active_project_id.lock().unwrap();
        *lock = project_id.to_string();
    }

    pub fn create_project(&self, name: &str, description: Option<&str>) -> Result<ProjectRecord> {
        let id = format!("proj_{}", name.to_lowercase().replace(' ', "_"));
        self.persistence.create_project(&id, name, description)
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectRecord>> {
        self.persistence.list_projects()
    }
}

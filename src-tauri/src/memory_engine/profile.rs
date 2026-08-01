//! User Profile Memory Manager
//! Manages key user facts, preferences, background, and skills.

use anyhow::Result;
use std::sync::Arc;
use crate::memory_engine::persistence::{PersistenceManager, UserProfileRecord};

pub struct ProfileManager {
    persistence: Arc<PersistenceManager>,
}

impl ProfileManager {
    pub fn new(persistence: Arc<PersistenceManager>) -> Self {
        Self { persistence }
    }

    pub fn set_fact(&self, key: &str, value: &str, category: &str) -> Result<()> {
        self.persistence.save_user_profile_fact(key, value, category)
    }

    pub fn get_profile(&self) -> Result<Vec<UserProfileRecord>> {
        self.persistence.get_user_profile()
    }

    pub fn build_profile_summary(&self) -> String {
        let profile = self.get_profile().unwrap_or_default();
        if profile.is_empty() {
            return "No prior user preferences recorded.".to_string();
        }

        let mut lines = Vec::new();
        for item in profile {
            lines.push(format!("- {}: {}", item.key, item.value));
        }

        lines.join("\n")
    }
}

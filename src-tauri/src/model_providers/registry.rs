//! Provider registry

use super::provider::ModelProvider;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct ProviderRegistry {
    providers: Arc<Mutex<HashMap<String, Box<dyn ModelProvider>>>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self { providers: Arc::new(Mutex::new(HashMap::new())) }
    }
    pub fn register_provider(&self, id: String, provider: Box<dyn ModelProvider>) -> Result<()> {
        self.providers.lock().unwrap().insert(id, provider);
        Ok(())
    }
    pub fn get_provider(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    pub fn list_providers(&self) -> Result<Vec<String>> {
        Ok(self.providers.lock().unwrap().keys().cloned().collect())
    }
    pub fn get_default_provider(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

//! Provider traits and types

use anyhow::Result;

pub enum ProviderType { HuggingFace, OllamaLibrary, Local, Custom(String) }

pub struct ModelMetadata {
    pub id: String, pub name: String, pub description: String, pub size_bytes: u64,
    pub format: String, pub quantization: String, pub tags: Vec<String>,
    pub provider: String, pub download_url: Option<String>, pub sha256: Option<String>,
}

pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &'static str { "Unknown" }
    fn provider_type(&self) -> ProviderType { ProviderType::Local }
    fn authenticate(&self, _token: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn search(&self, _query: &str) -> Result<Vec<ModelMetadata>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_model_metadata(&self, _id: &str) -> Result<ModelMetadata> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_download_url(&self, _id: &str) -> Result<String> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn verify_model(&self, _id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn is_available(&self) -> bool { false }
}

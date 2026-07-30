//! Ollama Library provider stub

use crate::model_providers::provider::{ModelProvider, ProviderType, ModelMetadata};
use anyhow::Result;

pub struct OllamaLibraryProvider;

impl ModelProvider for OllamaLibraryProvider {
    fn name(&self) -> &'static str { "Ollama Library" }
    fn provider_type(&self) -> ProviderType { ProviderType::OllamaLibrary }
}

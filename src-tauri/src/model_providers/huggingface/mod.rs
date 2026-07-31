//! HuggingFace provider module

pub mod resolver;
pub mod catalog_provider;

use crate::model_providers::provider::{ModelProvider, ProviderType};

pub struct HuggingFaceProvider;

impl ModelProvider for HuggingFaceProvider {
    fn name(&self) -> &'static str { "HuggingFace" }
    fn provider_type(&self) -> ProviderType { ProviderType::HuggingFace }
}

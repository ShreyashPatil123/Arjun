//! HuggingFace provider module

pub mod resolver;
pub mod catalog_provider;
pub mod adapter_provider;
pub mod discovery;
pub mod live_catalog;
pub mod card;

use crate::model_providers::provider::{ModelProvider, ProviderType};

pub struct HuggingFaceProvider;

impl ModelProvider for HuggingFaceProvider {
    fn name(&self) -> &'static str { "HuggingFace" }
    fn provider_type(&self) -> ProviderType { ProviderType::HuggingFace }
}

//! HuggingFace provider stub

use crate::model_providers::provider::{ModelProvider, ProviderType, ModelMetadata};
use anyhow::Result;

pub struct HuggingFaceProvider;

impl ModelProvider for HuggingFaceProvider {
    fn name(&self) -> &'static str { "HuggingFace" }
    fn provider_type(&self) -> ProviderType { ProviderType::HuggingFace }
}

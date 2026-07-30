//! Local provider stub

use crate::model_providers::provider::{ModelProvider, ProviderType, ModelMetadata};
use anyhow::Result;

pub struct LocalProvider;

impl ModelProvider for LocalProvider {
    fn name(&self) -> &'static str { "Local" }
    fn provider_type(&self) -> ProviderType { ProviderType::Local }
}

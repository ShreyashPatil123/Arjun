//! LoRA traits

use anyhow::Result;

pub struct LoRAAdapter {
    pub id: String, pub name: String, pub base_model_id: String,
    pub file_path: String, pub rank: u32, pub alpha: f32,
    pub target_modules: Vec<String>, pub metadata: Option<String>,
}

pub struct LoRAComposition { pub adapters: Vec<(String, f32)> }

pub trait LoRAManager: Send + Sync {
    fn load_adapter(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unload_adapter(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn switch_adapter(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_loaded_adapters(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_adapter_info(&self, _id: &str) -> Result<LoRAAdapter> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait LoRARegistry: Send + Sync {
    fn register_adapter(&self, _adapter: LoRAAdapter) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unregister_adapter(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_adapter(&self, _id: &str) -> Result<LoRAAdapter> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_adapters(&self) -> Result<Vec<LoRAAdapter>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn search_adapters(&self, _query: &str) -> Result<Vec<LoRAAdapter>> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait LoRARouter: Send + Sync {
    fn select_adapter(&self, _prompt: &str) -> Result<String> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_routing_rules(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn set_routing_rules(&self, _rules: ()) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait LoRAComposer: Send + Sync {
    fn compose_adapters(&self, _composition: LoRAComposition) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_composition(&self) -> Result<LoRAComposition> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn set_weights(&self, _adapter_id: &str, _weight: f32) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

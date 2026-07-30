//! Model manager traits

use anyhow::Result;

pub struct ModelCompatibility { pub model_id: String, pub score: f32, pub reasons: Vec<String>, pub warnings: Vec<String> }
pub struct ModelRecommendation { pub model_id: String, pub score: f32, pub hardware_match: f32, pub size_fit: f32, pub performance_estimate: f32 }

pub trait ModelRegistry: Send + Sync {
    fn register_model(&self, _model: ()) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unregister_model(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_model(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_models(&self) -> Result<Vec<()>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn search_models(&self, _query: &str) -> Result<Vec<()>> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait CompatibilityScorer: Send + Sync {
    fn score_model(&self, _model_id: &str, _profile: &()) -> Result<ModelCompatibility> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn check_compatibility(&self, _model_id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_requirements(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait RecommendationEngine: Send + Sync {
    fn recommend_models(&self, _profile: &()) -> Result<Vec<ModelRecommendation>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn rank_models(&self, _models: Vec<String>, _profile: &()) -> Result<Vec<ModelRecommendation>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn estimate_performance(&self, _model_id: &str, _profile: &()) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait ModelInstaller: Send + Sync {
    fn install_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn uninstall_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn verify_model(&self, _model_id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn update_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_installed_models(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
}

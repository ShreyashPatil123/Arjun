//! Phase 3: Model Recommendation Engine
//!
//! Takes Phase 2's HardwareProfile and deterministically calculates which
//! LLMs the user's machine can realistically run.
//!
//! Architecture:
//!   HardwareProfile → Budget Calculator → Model Catalog → Memory Estimator
//!   → Multi-Config Evaluator → Deterministic Scorer → Ranked Recommendations
//!
//! This module does NOT download models or launch inference backends.

pub mod traits;
pub mod budget;
pub mod catalog;
pub mod estimator;
pub mod runtime;
pub mod scorer;

use std::path::Path;
use crate::system_analyzer::traits::HardwareProfile;
use crate::model_providers::huggingface::catalog_provider::HuggingFaceCatalogProvider;
use traits::*;

/// Generate model recommendations from a HardwareProfile and dynamic Hugging Face catalog.
/// This is the main entry point for the recommendation engine.
pub async fn generate_recommendations(
    profile: &HardwareProfile,
    app_data_dir: Option<&Path>,
    force_refresh: bool,
) -> Vec<ModelRecommendation> {
    log::info!("[RECOMMENDATION] 🚀 Starting Model Recommendation Engine");

    // 1. Calculate adaptive resource budget
    let budget_config = BudgetConfig::default();
    let memory_budget = budget::calculate_budget(profile, &budget_config);

    // 2. Discover model catalog from Hugging Face Hub (or cached/bootstrap fallback)
    let models = if let Some(data_dir) = app_data_dir {
        HuggingFaceCatalogProvider::fetch_catalog(data_dir, force_refresh).await
    } else {
        catalog::bootstrap_models()
    };
    log::info!("[RECOMMENDATION] Discovered/loaded {} models from catalog", models.len());

    // 3. Evaluate all models against budget and score
    let estimator_config = EstimatorConfig::default();
    let recommendations = scorer::generate_all_recommendations(
        &models,
        &memory_budget,
        &estimator_config,
    );

    log::info!("[RECOMMENDATION] ✓ Engine complete: {} recommendations generated", recommendations.len());
    recommendations
}

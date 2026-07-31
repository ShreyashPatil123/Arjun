//! Phase 3: Tauri IPC Command for Model Recommendations

use crate::model_recommendation;
use crate::model_recommendation::traits::ModelRecommendation;
use crate::system_analyzer;

/// Get model recommendations based on a fresh hardware scan.
/// Returns ranked recommendations sorted by fit_score descending.
#[tauri::command]
pub async fn get_model_recommendations() -> Result<Vec<ModelRecommendation>, String> {
    log::info!("[RECOMMENDATION CMD] get_model_recommendations invoked");

    // Get fresh HardwareProfile from Phase 2 system analyzer
    let analyzer = system_analyzer::get_system_analyzer_manager();
    let profile = analyzer.analyze_system().map_err(|e| format!("System analysis failed: {}", e))?;

    // Generate recommendations
    let recommendations = model_recommendation::generate_recommendations(&profile);

    Ok(recommendations)
}

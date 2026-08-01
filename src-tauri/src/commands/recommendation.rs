use tauri::Manager;
use crate::model_recommendation;
use crate::model_recommendation::traits::ModelRecommendation;
use crate::system_analyzer;

/// Get model recommendations based on a fresh hardware scan and Hugging Face model discovery.
/// Returns ranked recommendations sorted by fit_score descending.
#[tauri::command]
pub async fn get_model_recommendations(
    app_handle: tauri::AppHandle,
    force_refresh: Option<bool>,
) -> Result<Vec<ModelRecommendation>, String> {
    log::info!("[RECOMMENDATION CMD] get_model_recommendations invoked (force_refresh={:?})", force_refresh);

    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    let force = force_refresh.unwrap_or(false);

    // Get HardwareProfile from Phase 2 system analyzer (use cached profile if available unless force_refresh is requested)
    let analyzer = system_analyzer::get_system_analyzer_manager();
    let profile = if !force {
        if let Some(cached) = analyzer.get_profile() {
            cached
        } else {
            analyzer.analyze_system().map_err(|e| format!("System analysis failed: {}", e))?
        }
    } else {
        analyzer.analyze_system().map_err(|e| format!("System analysis failed: {}", e))?
    };
    let recommendations = model_recommendation::generate_recommendations(&profile, Some(&app_data_dir), force).await;

    Ok(recommendations)
}

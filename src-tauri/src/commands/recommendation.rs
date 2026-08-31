use tauri::{Manager, State};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::model_recommendation;
use crate::model_recommendation::traits::ModelRecommendation;
use crate::system_analyzer;

/// Get model recommendations based on a fresh hardware scan and Hugging Face model discovery.
/// Returns ranked recommendations sorted by fit_score descending.
#[tauri::command]
pub async fn get_model_recommendations(
    app_handle: tauri::AppHandle,
    session: State<'_, CurrentSession>,
    force_refresh: Option<bool>,
) -> Result<Vec<ModelRecommendation>, String> {
    // Recommendations are model-catalog reads. The matrix does not gate
    // them beyond sign-in.
    require_session(&session)?;
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

#[tauri::command]
pub async fn get_package_certification(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
    model_id: String,
) -> Result<Option<model_recommendation::certified_catalog::PackageCertification>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_package_certification(&model_id))
}

#[tauri::command]
pub async fn get_all_package_certifications(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<model_recommendation::certified_catalog::PackageCertification>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_all_certifications())
}

#[tauri::command]
pub async fn get_runtime_profile(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
    profile_id: String,
) -> Result<Option<model_recommendation::certified_catalog::RuntimeProfile>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_runtime_profile(&profile_id))
}

#[tauri::command]
pub async fn get_recommended_packages(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<model_recommendation::certified_catalog::PackageCertification>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_recommended_models())
}

#[tauri::command]
pub async fn get_compatible_packages(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<model_recommendation::certified_catalog::PackageCertification>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_compatible_models())
}

#[tauri::command]
pub async fn get_experimental_packages(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<model_recommendation::certified_catalog::PackageCertification>, String> {
    require_session(&session)?;
    Ok(pack_mgr.get_experimental_models())
}

#[tauri::command]
pub async fn reload_certification_packs(
    pack_mgr: tauri::State<'_, std::sync::Arc<model_recommendation::pack_manager::PackManager>>,
    session: State<'_, CurrentSession>,
) -> Result<(), String> {
    // Reloading the pack catalog changes the model catalog. That is
    // model-management work, so the matrix puts it under `ImportModel`.
    require_permission(&session, Permission::ImportModel)?;
    pack_mgr.reload_all_packs().map_err(|e| e.to_string())
}

//! Sarathi Main Library
//! Wires together all modules and sets up the Tauri application.

pub mod core;
pub mod database;
pub mod config;
pub mod logging;
pub mod commands;

// Phase modules
pub mod system_analyzer;
pub mod model_recommendation;
pub mod model_manager;
pub mod model_providers;
pub mod download_manager;
pub mod adapter_manager;
pub mod ai_engine;
pub mod model_intelligence;
pub mod lora;
pub mod installer;
pub mod plugins;
pub mod memory_engine;

use std::sync::Arc;
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};
use tauri_plugin_sql::Builder as SqlBuilder;
use log::info;

use download_manager::DownloadManager;
use ai_engine::InferenceManager;
use memory_engine::MemoryManager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Set up crash handler
    logging::setup_panic_handler();

    // Initialize core and managers
    let sarathi_core = core::init();
    let download_manager = Arc::new(DownloadManager::new());
    let inference_manager = Arc::new(InferenceManager::new());

    // Configure SQL plugin with migrations
    let migrations = database::get_migrations();
    let sql_plugin = SqlBuilder::default()
        .add_migrations("sqlite:sarathi.db", migrations)
        .build();

    // Configure Log plugin
    let log_plugin = tauri_plugin_log::Builder::new()
        .targets([
            Target::new(TargetKind::Stdout),
            Target::new(TargetKind::LogDir { file_name: Some("sarathi".into()) }),
            Target::new(TargetKind::Webview),
        ])
        .build();

    // Build and run the app
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(sql_plugin)
        .plugin(log_plugin)
        .manage(sarathi_core)
        .manage(download_manager)
        .manage(inference_manager)
        .setup(|app| {
            info!("Sarathi application starting...");

            // Resolve app_data_dir dynamically from Tauri app handle
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("./app_data"));
            let memory_manager = Arc::new(MemoryManager::new(&app_data_dir));
            app.manage(memory_manager);

            let pack_manager = Arc::new(crate::model_recommendation::pack_manager::PackManager::new(&app_data_dir).expect("Failed to initialize PackManager"));
            app.manage(pack_manager);

            // Initial event publication
            let event_bus = core::event_bus::get_event_bus();
            event_bus.publish(core::event_bus::SarathiEvent::ApplicationStarted, None);

            // Startup scan for local model packages and LoRA adapters
            if let Ok(app_data_dir) = app.path().app_data_dir() {
                std::thread::spawn(move || {
                    adapter_manager::AdapterRegistry::perform_startup_scan(&app_data_dir);
                });
            }

            // Run initial system analysis task on a blocking thread (not a tokio async worker)
            // so it doesn't occupy the async runtime while running PowerShell/DXGI detection
            std::thread::spawn(move || {
                let analyzer = system_analyzer::get_system_analyzer_manager();
                if let Err(e) = analyzer.analyze_system() {
                    log::error!("Initial system analysis failed: {}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Config commands
            commands::config::get_config,
            commands::config::set_config,
            commands::config::get_config_value,
            commands::config::set_config_value,
            commands::config::get_default_config,
            commands::config::reset_config,
            commands::config::get_app_paths,

            // System commands
            commands::system::get_app_info,
            commands::system::get_app_state_info,
            commands::system::log_activity,
            commands::system::get_hardware_profile,
            commands::system::analyze_system,
            commands::system::override_hardware_value,
            commands::system::revert_hardware_override,
            commands::system::validate_system,

            // Recommendation & Certification commands (Phase 3 & Ecosystem)
            commands::recommendation::get_model_recommendations,
            commands::recommendation::get_package_certification,
            commands::recommendation::get_all_package_certifications,
            commands::recommendation::get_runtime_profile,
            commands::recommendation::reload_certification_packs,

            // Download & Storage Management commands (Phase 4)
            commands::download::start_model_download,
            commands::download::pause_model_download,
            commands::download::cancel_model_download,
            commands::download::get_active_downloads,
            commands::download::get_installed_models,
            commands::download::delete_installed_model,
            commands::download::get_storage_summary,

            // LoRA Capability Adapter commands
            commands::adapter::discover_model_adapters,
            commands::adapter::get_model_package_manifest,
            commands::adapter::list_installed_model_packages,

            // Phase 5 Inference Commands
            commands::inference::load_installed_model,
            commands::inference::unload_active_model,
            commands::inference::get_inference_status,
            commands::inference::send_chat_message,
            commands::inference::stop_chat_generation,
            commands::inference::restore_last_session,

            // Model Intelligence Layer Commands
            commands::intelligence::get_model_profile,
            commands::intelligence::update_model_profile,
            commands::intelligence::refresh_model_profile,
            commands::intelligence::route_prompt_capability,

            // Phase 6 Memory Engine Commands
            memory_engine::api::get_memory_health_status,
            memory_engine::api::get_user_profile_memory,
            memory_engine::api::update_user_profile_fact,
            memory_engine::api::list_memory_projects,
            memory_engine::api::create_memory_project,
            memory_engine::api::switch_active_project,
            memory_engine::api::get_active_project,
            memory_engine::api::search_memory_nodes,
            memory_engine::api::delete_memory_node_by_id,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

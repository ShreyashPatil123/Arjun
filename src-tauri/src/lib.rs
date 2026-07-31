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
pub mod lora;
pub mod installer;
pub mod plugins;

use std::sync::Arc;
use tauri_plugin_log::{Target, TargetKind};
use tauri_plugin_sql::Builder as SqlBuilder;
use log::info;

use download_manager::DownloadManager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Set up crash handler
    logging::setup_panic_handler();

    // Initialize core
    let sarathi_core = core::init();
    let download_manager = Arc::new(DownloadManager::new());

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
        .setup(|_app| {
            info!("Sarathi application starting...");

            // Initial event publication
            let event_bus = core::event_bus::get_event_bus();
            event_bus.publish(core::event_bus::SarathiEvent::ApplicationStarted, None);

            // Run initial system analysis task asynchronously on startup
            tauri::async_runtime::spawn(async move {
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

            // Recommendation commands (Phase 3)
            commands::recommendation::get_model_recommendations,

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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

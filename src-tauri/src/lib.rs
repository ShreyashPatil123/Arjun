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
pub mod capability;
pub mod gateway;
pub mod launcher;
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

            // Serialize all model access behind one worker, then start the local
            // gateway so external tools (Claude Code, opencode, openclaw) can use
            // whichever model this app has loaded.
            let inference_for_gateway = app.state::<Arc<InferenceManager>>().inner().clone();
            let scheduler = Arc::new(ai_engine::scheduler::GenerationScheduler::start(
                inference_for_gateway.clone(),
            ));
            app.manage(scheduler.clone());

            let gateway_state = Arc::new(gateway::GatewayState::new(
                scheduler,
                inference_for_gateway,
                gateway::GatewayConfig::default(),
            ));
            app.manage(gateway_state.clone());

            // Tracks tools the Launch screen started, so cards can show Running.
            app.manage(Arc::new(launcher::LaunchedProcesses::default()));

            let app_for_gateway = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match gateway::start_gateway(gateway_state).await {
                    Ok(handle) => {
                        info!(
                            "Sarathi gateway ready on http://127.0.0.1:{} — point Claude Code at /v1/messages, opencode at /v1/chat/completions",
                            handle.port
                        );
                        // Hand the handle to Tauri so it lives as long as the
                        // app. Letting it drop here would drop the shutdown
                        // sender, resolving the graceful-shutdown future and
                        // closing the server immediately after it announced
                        // itself — the logs would claim it was listening while
                        // every connection was refused.
                        app_for_gateway.manage(handle);
                    }
                    // A busy port must not stop the desktop app from opening;
                    // the user needs the UI to change the port.
                    Err(e) => log::error!("Gateway failed to start: {e:#}"),
                }
            });

            // Bring a model up on launch so the gateway can answer immediately.
            //
            // Sarathi serves other tools rather than hosting its own chat, so
            // nothing in the UI would otherwise trigger a load — a user could
            // install a model, point Claude Code at the gateway, and get
            // "no model loaded" with no obvious way to fix it.
            //
            // Prefers the last model used. Falls back to the only installed one.
            // With several installed and no previous session it loads nothing,
            // because guessing which model someone wants resident in VRAM is
            // worse than letting them choose.
            {
                let inference = app.state::<Arc<InferenceManager>>().inner().clone();
                let app_data = app.path().app_data_dir().ok();

                tauri::async_runtime::spawn(async move {
                    let Some(dir) = app_data else { return };

                    let restore = ai_engine::session::SessionManager::load_session(&dir)
                        .ok()
                        .flatten()
                        .filter(|s| s.auto_restore_enabled)
                        .map(|s| (s.provider_id, s.model_id, s.quantization));

                    let target = restore.or_else(|| {
                        let packages = adapter_manager::AdapterRegistry::list_installed_packages(&dir);
                        match packages.len() {
                            1 => {
                                let p = &packages[0];
                                Some((
                                    p.provider_id.clone(),
                                    p.base_model.model_id.clone(),
                                    p.base_model.quantization.clone(),
                                ))
                            }
                            0 => None,
                            n => {
                                info!("{n} models installed — select one to load; not guessing.");
                                None
                            }
                        }
                    });

                    let Some((provider, model, quant)) = target else { return };
                    info!("Auto-loading '{model}' ({quant}) so the gateway can serve requests");

                    let res = tokio::task::spawn_blocking(move || {
                        inference.load_installed_model_direct(&dir, &provider, &model, &quant)
                    })
                    .await;

                    match res {
                        Ok(Ok(info)) => info!(
                            "Model ready: {} via {} — gateway can now serve requests",
                            info.model_name, info.backend_used
                        ),
                        // A load failure must not take the app down; the UI still
                        // needs to open so the user can pick a different model.
                        Ok(Err(e)) => log::error!("Auto-load failed: {e:#}"),
                        Err(e) => log::error!("Auto-load task panicked: {e}"),
                    }
                });
            }

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
            commands::recommendation::get_recommended_packages,
            commands::recommendation::get_compatible_packages,
            commands::recommendation::get_experimental_packages,
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

            // Launch section — start coding tools already connected
            commands::launcher::get_launch_overview,
            commands::launcher::preview_tool_install,
            commands::launcher::install_tool,
            commands::launcher::launch_tool,
            commands::launcher::forget_tool_process,
            commands::launcher::user_tools_file,

            // Model browsing by category
            commands::catalog::browse_model_cards,
            commands::catalog::list_model_categories,
            commands::catalog::find_model_adapters,

            // Adapter downloads and management
            commands::adapters::list_installed_adapters,
            commands::adapters::download_adapter,
            commands::adapters::remove_adapter,
            commands::adapter_details::get_adapter_details,

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
            memory_engine::api::get_memory_diagnostics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

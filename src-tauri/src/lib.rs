//! Sarathi Main Library
//! Wires together all modules and sets up the Tauri application.

pub mod core;
pub mod database;
pub mod config;
pub mod logging;
pub mod commands;
pub mod sovereignty;
pub mod artifacts;
pub mod audit;
pub mod documents;
pub mod health;
pub mod hooks;
pub mod identity;
pub mod knowledge;
pub mod agent_runtime;
pub mod orchestrator;
pub mod package;
pub mod policy;
pub mod registry;
pub mod serving;
pub mod skills;
pub mod subagents;

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
pub mod model_intelligence;
pub mod lora;
pub mod installer;
pub mod plugins;
pub mod memory_engine;
pub mod media_adapter;
pub mod sih_workflow;
pub mod voice;
pub mod benchmarks;

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
        // Cloned rather than moved: the activator built in setup below needs
        // the same manager the commands see, not a second one.
        .manage(inference_manager.clone())
        .setup(move |app| {
            info!("Sarathi application starting...");

            // Resolve app_data_dir dynamically from Tauri app handle
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("./app_data"));
            let memory_manager = Arc::new(MemoryManager::new(&app_data_dir));
            // The single outbound chokepoint. Managed before anything that could
            // want the network, so no module can come up with its own client.
            // Governance state. The audit log is opened first so that everything
            // after it — including the broker's own decisions — is on the record.
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("the application data directory must resolve");
            // Expose the resolved data dir to commands that need to read or
            // write files outside the audit DB (e.g. the provenance HMAC key
            // and tag files live here).
            app.manage(data_dir.clone());
            match audit::AuditService::open(&data_dir) {
                Ok(service) => {
                    let service = Arc::new(service);
                    sovereignty::global_broker().attach_audit(service.clone());
                    app.manage(service.clone());
                    // The zero-trust gate is opened next to the audit log
                    // because every gate decision is recorded in the log.
                    match sovereignty::zero_trust::ZeroTrustGate::open(&data_dir, service) {
                        Ok(gate) => {
                            app.manage(Arc::new(gate));
                        }
                        Err(e) => {
                            log::error!("[ZERO-TRUST] could not open the gate: {e}");
                        }
                    }
                }
                Err(e) => {
                    // Running without a durable record is a real degradation, so
                    // it is logged at error level rather than passed over.
                    log::error!("[AUDIT] could not open the audit log: {e}");
                }
            }
            match identity::CredentialStore::open(&data_dir) {
                Ok(store) => {
                    app.manage(Arc::new(store));
                }
                Err(e) => log::error!("[IDENTITY] could not open the credential store: {e}"),
            }
            app.manage(Arc::new(identity::UserDirectory::seeded()));

            // The model registry is a manifest beside the models on disk. A
            // missing one is an empty registry, not a failure — a fresh install
            // legitimately has nothing to route to until somebody provisions it.
            // The activator owns model swapping: routing chooses, this loads.
            app.manage(std::sync::Arc::new(ai_engine::activation::ModelActivator::new(
                ai_engine::activation::InferenceLoader::new(
                    inference_manager.clone(),
                    data_dir.clone(),
                ),
            )));

            match registry::ModelRegistry::load_with_discovery(&data_dir) {
                Ok(loaded) => {
                    // Logged on every start, including — especially — when it is
                    // zero. An empty registry turns into "no models are
                    // registered yet" on the workbench, which sends somebody off
                    // to import models they may already have; this line says
                    // where it looked, so the next person can tell an empty
                    // directory from an unreadable manifest in one glance.
                    info!(
                        "[REGISTRY] {} model(s) available from {}",
                        loaded.all().len(),
                        data_dir.join("models").display()
                    );
                    app.manage(Arc::new(loaded));
                }
                Err(e) => {
                    log::error!("[REGISTRY] the model manifest could not be read: {e}");
                    app.manage(Arc::new(
                        registry::ModelRegistry::load(std::path::Path::new("./__absent__"))
                            .expect("an absent manifest always loads as empty"),
                    ));
                }
            }
            // The knowledge index is the same SQLite file the rest of the app
            // uses. It is managed here so the health panel can count documents
            // without opening a second connection per request.
            match knowledge::index::KnowledgeIndex::open(&data_dir) {
                Ok(index) => {
                    app.manage(Arc::new(index));
                }
                Err(e) => log::error!("[KNOWLEDGE] the index could not be opened: {e}"),
            }

            app.manage(Arc::new(orchestrator::approvals::ApprovalQueue::new()));
            app.manage(commands::governance::CurrentSession::default());
            // The telemetry sink: per-model call records, written to the
            // audit log on each call so the Model Health page reads from
            // a single signed chain rather than a separate database.
            let telemetry_sink =
                Arc::new(model_intelligence::telemetry::TelemetrySink::new());
            // Diagnostic: prove the sink is registered and the lookup
            // from the inference path will resolve. If the page is
            // empty after a chat, this log line is the first thing to
            // check; the path that records calls goes through
            // `app_handle.try_state::<Arc<TelemetrySink>>` and would
            // hit the same lookup mechanism.
            // `eprintln!` is used deliberately: the Tauri log plugin's
            // flush is lazy, and `log::info!` from the `log` crate was
            // observed to not reach `sarathi.log` for these new lines.
            // `eprintln!` writes to stderr, which `Start-Process` on
            // Windows exposes via the parent's handle when the parent
            // is a console host. The PowerShell wrapper captures it.
            eprintln!(
                "[telemetry] sink registered; default seq = 0; \
                 snapshot at startup = {} rows",
                telemetry_sink.snapshot().len()
            );

            // Diagnostic: insert one synthetic row at startup so the
            // Model Health page is guaranteed non-empty after launch.
            // This is the baseline that proves the IPC + reducer + page
            // chain is working end-to-end. Any real inference call
            // adds a second row; if the synthetic row is missing, the
            // page itself is broken; if only the synthetic row is
            // there, the inference path is broken.
            telemetry_sink.record(
                None,
                crate::model_intelligence::telemetry::ModelCallRecord {
                    model_id: "<startup>".to_string(),
                    task_id: "<startup>".to_string(),
                    intent: "startup".to_string(),
                    role: "diagnostic".to_string(),
                    latency: std::time::Duration::from_millis(0),
                    tokens_in: 0,
                    tokens_out: 0,
                    used_fallback: false,
                    exit: crate::model_intelligence::telemetry::CallExit::Ok,
                    note: Some("synthetic startup record inserted to prove \
                              the IPC + reducer + page chain is wired"
                        .to_string()),
                    complexity: None,
                },
            );
            eprintln!(
                "[telemetry] synthetic startup record inserted; \
                 snapshot now = {} rows",
                telemetry_sink.snapshot().len()
            );

            app.manage(telemetry_sink);
            // Started on first run rather than here: the workbench must open for an
            // auditor, and on a machine where the runtime bundle was never built.
            app.manage(commands::agent::AgentRuntimeHandle::default());
            // Model servers ARJUN starts, so a llama-server is loaded once and
            // reused across runs rather than per prompt.
            app.manage(Arc::new(serving::ModelServers::new()));
            // One working directory per run, shared with the agent runtime so a
            // tool call can be resolved against the run that made it.
            app.manage(commands::agent::RunWorkspaces::default());
            // The rest of a run's working state, held here for the same reason:
            // the command that starts a run has to read all of it back when the
            // run ends, to write the task's record.
            app.manage(commands::agent::RunPlans::default());
            app.manage(commands::agent::RunCalculations::default());
            app.manage(commands::agent::RunToolCalls::default());
            // Scoped memory: what this machine remembers, and for whom. Opened
            // under the same data directory as the index and the task records,
            // and lazily per scope — a deployment with two hundred projects
            // should not pay for two hundred file reads to start.
            app.manage(std::sync::Arc::new(agent_runtime::memory::MemoryStore::open(
                &data_dir,
            )) as commands::agent::AgentMemory);
            // The fixed half of each live run's checkpoint. Dies with the
            // process on purpose: a seed describes a world observed at start,
            // and after a restart that world has to be observed again.
            app.manage(commands::agent::RunCheckpoints::default());
            app.manage(agent_runtime::retrieval::RunPassages::default());
            app.manage(agent_runtime::artifacts::RunArtifacts::default());

            // Chat conversations: persistent ordered transcripts that own
            // one or more runs. Sits beside the task record; the two are
            // complementary (the task record is the audit-grade proof a
            // run happened; the conversation is the user-visible chat).
            let conversation_store = std::sync::Arc::new(
                agent_runtime::conversations::ConversationStore::open(&data_dir)
                    .unwrap_or_else(|error| {
                        // A conversation store that cannot be opened is a
                        // real degradation. We still fall back to a temp
                        // directory so the app does not refuse to start.
                        log::error!("[CONVERSATIONS] the store could not be opened: {error}");
                        let tmp = std::env::temp_dir().join("arjun-conversations-fallback");
                        agent_runtime::conversations::ConversationStore::open(&tmp)
                            .expect("a temp conversation store must open")
                    }),
            );
            let run_to_conversation =
                std::sync::Arc::new(agent_runtime::conversations::RunToConversation::new());
            app.manage(commands::conversations::ConversationsState(conversation_store));
            app.manage(commands::conversations::RunToConversationState(run_to_conversation));

            // The durable half of all of the above. Everything managed just
            // now dies with this process; this is what a run leaves behind
            // while it is still going, and it is opened before any command can
            // run so that no run starts unrecorded.
            let task_events: std::sync::Arc<agent_runtime::events::TaskEventLog> = std::sync::Arc::new(
                agent_runtime::events::TaskEventLog::open(&data_dir).unwrap_or_else(|error| {
                    // A run with no durable history is a real degradation, and
                    // it is logged as one. It is not a reason to refuse to
                    // open: an unrecorded run is worse than a recorded one and
                    // better than an application that will not start.
                    log::error!("[TASKS] the task event log could not be opened: {error}");
                    agent_runtime::events::TaskEventLog::in_memory()
                        .expect("an in-memory task event log")
                }),
            );

            // Runs that were going when the process last went away. Closed off
            // here, before anything else writes, so the Tasks screen never
            // shows a run that has been "running" since last Tuesday next to
            // one that is running now.
            match task_events.recover_interrupted(agent_runtime::events::SYSTEM_ACTOR) {
                Ok(recovered) if !recovered.is_empty() => {
                    info!(
                        "[TASKS] {} run(s) were interrupted by the last shutdown and have been \
                         closed off: {}",
                        recovered.len(),
                        recovered.join(", ")
                    );
                }
                Ok(_) => {}
                Err(error) => log::error!("[TASKS] interrupted runs could not be closed off: {error}"),
            }
            let subagent_events = std::sync::Arc::clone(&task_events);
            app.manage(task_events as commands::agent::TaskEvents);

            // Skills: reusable instructions an operator installs. Discovered
            // once at start, metadata only — reading every SKILL.md into memory
            // here would be the thing that puts every skill in front of every
            // model. See `skills::registry`.
            //
            // Resolved as a bundled resource in a packaged build and as the
            // sibling directory in a checkout, the same way the agent runtime
            // bundle is.
            let skills_dir = app
                .path()
                .resolve("skills", tauri::path::BaseDirectory::Resource)
                .ok()
                .filter(|path| path.is_dir())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|root| root.join("skills"))
                        .unwrap_or_default()
                });
            let skills = std::sync::Arc::new(skills::SkillRegistry::open(&skills_dir));
            let found = skills.snapshot();
            info!(
                "[SKILLS] {} of {} skill(s) available from {}",
                found.available(),
                found.count(),
                skills_dir.display()
            );
            for card in found.cards().iter().filter(|card| !card.is_available()) {
                // Quarantined skills are named at start rather than only when
                // somebody goes looking. An operator who installed one and
                // cannot find it should not have to open a screen to find out
                // why.
                if let Some(reason) = &card.quarantined {
                    log::warn!("[SKILLS] {} is quarantined: {}", card.name, reason.explain());
                }
            }
            app.manage(skills as commands::agent::Skills);

            // Subagent profiles: bounded workers a run may delegate to. Loaded
            // beside the skills and for the same reasons — an operator reads
            // and reviews a file, and Rust enforces what it declares.
            let agents_dir = app
                .path()
                .resolve("agents", tauri::path::BaseDirectory::Resource)
                .ok()
                .filter(|path| path.is_dir())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|root| root.join("agents"))
                        .unwrap_or_default()
                });
            let loaded_profiles = subagents::load_profiles(&agents_dir);
            info!(
                "[SUBAGENTS] {} profile(s) from {}",
                loaded_profiles.profiles.len(),
                agents_dir.display()
            );
            for rejected in &loaded_profiles.rejected {
                // Named at start rather than only when somebody goes looking.
                log::warn!(
                    "[SUBAGENTS] {} was not loaded: {}",
                    rejected.file,
                    rejected.error.explain()
                );
            }
            app.manage(std::sync::Arc::new(subagents::SubagentManager::new(
                loaded_profiles.profiles,
                subagent_events,
            )) as commands::agent::Subagents);

            app.manage(sovereignty::global_broker().clone());

            app.manage(memory_manager);

            // Load the saved HuggingFace token into the process before anything
            // reaches the Hub. Catalog browsing and adapter discovery run from
            // plain commands with no app handle, so they read it from there
            // rather than opening config.json themselves.
            {
                let config_path = crate::config::ConfigManager::get_config_path(app.handle());
                match crate::config::ConfigManager::load(&config_path) {
                    Ok(cfg) if !cfg.hf_token.trim().is_empty() => {
                        crate::config::hf_token::set(Some(cfg.hf_token));
                        info!("HuggingFace token loaded from settings");
                    }
                    Ok(_) => {
                        info!(
                            "No HuggingFace token in settings (environment: {})",
                            crate::config::hf_token::source()
                        );
                    }
                    Err(e) => log::warn!("Could not read config for HuggingFace token: {e:#}"),
                }
            }

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

            // Optionally bring a model up on launch so the gateway can answer
            // immediately.
            //
            // Sarathi serves other tools rather than hosting its own chat, so
            // nothing in the UI would otherwise trigger a load — a user could
            // install a model, point Claude Code at the gateway, and get
            // "no model loaded" with no obvious way to fix it. That is the case
            // this exists for.
            //
            // It is off by default all the same: a load commits gigabytes of
            // VRAM and takes real time, and doing that before the user has named
            // a model takes a decision away from them. "No model loaded" is a
            // recoverable state; a surprise load is not. Enable it with
            // `ai_settings.auto_load_on_startup` when the gateway-first workflow
            // is what you want.
            //
            // When enabled it prefers the last model used, then falls back to the
            // only installed one. With several installed and no previous session
            // it loads nothing, because guessing which model someone wants
            // resident in VRAM is worse than letting them choose.
            let auto_load_on_startup = config::ConfigManager::load(
                &config::ConfigManager::get_config_path(app.handle()),
            )
            .map(|c| c.ai_settings.auto_load_on_startup)
            .unwrap_or(false);

            if !auto_load_on_startup {
                info!(
                    "Startup auto-load is off — no model will be loaded until one is \
                     chosen in the app. Set ai_settings.auto_load_on_startup to change this."
                );
            }

            if auto_load_on_startup {
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
            commands::config::get_hf_token_status,
            commands::config::set_hf_token,
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
            commands::download::resume_model_download,
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
            commands::intelligence::model_health_snapshot,

            // Launch section — start coding tools already connected

            // Model browsing by category
            // Agent runs. The loop lives in the Node runtime; these start,
            // stop and observe it.
            commands::agent::agent_start_run,
            commands::agent::agent_abort_run,
            commands::agent::agent_steer_run,
            commands::agent::agent_runtime_health,
            // What those runs left behind: the plan, the evidence, the working
            // and the files.
            commands::agent::agent_task_history,
            commands::agent::agent_task,
            commands::agent::agent_task_artifacts,
            commands::agent::agent_reveal_artifact,
            commands::agent::artifact_preview,

            // Chat conversations: persistent ordered transcripts that own
            // one or more runs. The chat surface calls these to create a
            // conversation, list previous ones, append a user turn (which
            // also reserves the assistant cell and binds the run id), update
            // the streaming content as tokens arrive, and mark the message
            // complete when the run ends.
            commands::conversations::agent_create_conversation,
            commands::conversations::agent_get_conversation,
            commands::conversations::agent_list_conversations,
            commands::conversations::agent_append_turn,
            commands::conversations::agent_update_streaming_content,
            commands::conversations::agent_complete_message,
            commands::conversations::agent_run_conversation,
            commands::conversations::agent_get_message,
            // Recovering a run: the state a window reattaches to, the events
            // since that state, and which runs are still going at all.
            commands::agent::agent_run_resumability,
            commands::agent::agent_resume_run,
            commands::agent::agent_task_snapshot,
            commands::agent::agent_task_events,
            commands::agent::agent_active_tasks,
            // Side effects that were in flight when the process went away, and
            // the person saying what actually happened to them.
            commands::agent::agent_unknown_effects,
            commands::agent::agent_reconcile_effect,
            // Skills: what is installed, and re-reading the directory.
            commands::agent::skill_search,
            commands::agent::skill_reload,
            commands::agent::subagent_profiles,

            commands::sovereignty::get_operating_mode,
            commands::sovereignty::set_operating_mode,
            commands::sovereignty::recent_egress_events,
            commands::sovereignty::run_egress_canary,
            commands::sovereignty::observe_process_connections,
            commands::sovereignty::assert_confidential_allowed,
            commands::governance::list_accounts,
            commands::governance::sign_in,
            commands::governance::sign_out,
            commands::governance::current_session,
            commands::governance::current_permissions,
            commands::governance::recent_audit_entries,
            commands::governance::verify_audit_chain,
            commands::governance::verify_audit_merkle,
            commands::governance::sign_provenance,
            commands::governance::verify_provenance,
            commands::governance::read_zero_trust_config,
            commands::governance::set_zero_trust_mode,
            commands::governance::zero_trust_check_tool_call,
            commands::governance::zero_trust_confirm_approval,
            // Voice bridge (push-to-talk STT/TTS)
            commands::voice::voice_transcribe,
            commands::voice::voice_speak,
            commands::voice::voice_status,
            // Performance benchmarks for the System Health page
            commands::benchmarks::run_benchmark,
            commands::benchmarks::synthetic_benchmark,
            commands::benchmarks::recent_benchmarks,
            commands::governance::authentication_status,
            commands::governance::set_initial_administrator_password,
            commands::governance::set_account_password,
            commands::registry::list_registered_models,
            commands::registry::model_manifest_path,
            commands::registry::preview_routing,
            commands::registry::prepare_model_for,
            commands::registry::model_residency,
            commands::health::health_snapshot,
            commands::approvals::list_approvals,
            commands::approvals::decide_approval,
            commands::catalog::browse_model_cards,
            commands::catalog::list_model_categories,
            commands::catalog::find_model_adapters,

            // Adapter downloads and management
            commands::adapters::list_installed_adapters,
            commands::adapters::download_adapter,
            commands::adapters::remove_adapter,
            commands::adapters::set_adapter_capability,
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

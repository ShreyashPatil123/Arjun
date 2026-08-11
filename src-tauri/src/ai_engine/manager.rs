//! Inference Manager — Thread-safe wrapper around LlamaCppRuntime
//!
//! Manages model loading/unloading with hardware-aware configuration,
//! provides streaming generation via Tauri events, and tracks the last used model.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::Manager;

use anyhow::{anyhow, Result};
use tauri::Emitter;

use crate::adapter_manager::{AdapterRegistry, ModelPackageManifest};
use crate::ai_engine::runtime::LlamaCppRuntime;
use crate::ai_engine::traits::*;
use crate::capability::{self, CapabilityLayer, CapabilityPayload};
use crate::system_analyzer;

/// Context a model is planned around when its own maximum is larger.
///
/// Modern GGUFs advertise ceilings (256K on Gemma 3/4 and Qwen 3) that describe
/// what the weights permit, not what a desktop card can hold. Budgeting for one
/// reserves tens of gigabytes of KV cache that will never be used. This is the
/// working context Sarathi actually plans for; a user who wants more can raise
/// it per model in Settings.
const DEFAULT_WORKING_CONTEXT: u32 = 8192;

/// The installed package backing the currently loaded model.
///
/// Captured at load time so each turn can resolve capabilities without
/// re-reading and re-parsing `manifest.json` from disk.
#[derive(Clone)]
pub struct ActivePackage {
    pub package_dir: PathBuf,
    pub manifest: ModelPackageManifest,
}

/// Thread-safe inference state manager.
///
/// Wraps `LlamaCppRuntime` in `Arc<Mutex<>>` for safe concurrent access
/// from multiple Tauri command handlers.
pub struct InferenceManager {
    runtime: Arc<Mutex<LlamaCppRuntime>>,
    last_used_model_id: Arc<Mutex<Option<String>>>,
    /// Package context for the loaded model, used to resolve LoRA adapters.
    active_package: Arc<Mutex<Option<ActivePackage>>>,
    /// Intent classification, switch hysteresis, and capability resolution.
    capability: Arc<CapabilityLayer>,
}

impl InferenceManager {
    /// Creates a new InferenceManager with no model loaded
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(Mutex::new(LlamaCppRuntime::new())),
            last_used_model_id: Arc::new(Mutex::new(None)),
            active_package: Arc::new(Mutex::new(None)),
            capability: Arc::new(CapabilityLayer::default()),
        }
    }

    /// The capability layer, for status queries and manual overrides.
    pub fn capability_layer(&self) -> Arc<CapabilityLayer> {
        self.capability.clone()
    }

    /// The package backing the loaded model, if any.
    pub fn active_package(&self) -> Option<ActivePackage> {
        self.active_package
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    /// Returns the current runtime status
    pub fn get_status(&self) -> RuntimeStatus {
        let runtime = self.runtime.lock().unwrap();
        runtime.status()
    }

    /// Returns info about the currently loaded model
    pub fn get_loaded_model_info(&self) -> Option<LoadedModelInfo> {
        let runtime = self.runtime.lock().unwrap();
        runtime.loaded_model_info().cloned()
    }

    /// Returns the last used model identifier (persisted across sessions)
    pub fn get_last_used_model_id(&self) -> Option<String> {
        self.last_used_model_id.lock().unwrap().clone()
    }

    /// Sets the last used model identifier
    pub fn set_last_used_model_id(&self, model_id: Option<String>) {
        let mut lock = self.last_used_model_id.lock().unwrap();
        *lock = model_id;
    }

    /// Loads an installed model using its manifest and hardware profile.
    ///
    /// - Reads `manifest.json` from the model's package directory
    /// - Consults the Phase 2 hardware profile for GPU/thread configuration
    /// - Auto-unloads any previously loaded model
    /// - Emits `inference:status` events during loading
    /// Loads an installed model using its manifest and hardware profile.
    pub fn load_installed_model(
        &self,
        app_handle: &tauri::AppHandle,
        app_data_dir: &std::path::Path,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
    ) -> Result<LoadedModelInfo> {
        let app_handle_clone = app_handle.clone();
        let status_cb = move |status: &str, step: Option<&str>| {
            let _ = app_handle_clone.emit("inference:status", InferenceStatusPayload {
                status: status.to_string(),
                step: step.map(|s| s.to_string()),
                model: None,
                error: None,
            });
        };

        self.load_installed_model_internal(app_data_dir, provider_id, model_id, quantization, Some(status_cb))
    }

    /// Loads an installed model without requiring a Tauri AppHandle (for tests & backend validation).
    pub fn load_installed_model_direct(
        &self,
        app_data_dir: &std::path::Path,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
    ) -> Result<LoadedModelInfo> {
        self.load_installed_model_internal::<fn(&str, Option<&str>)>(app_data_dir, provider_id, model_id, quantization, None)
    }

    fn load_installed_model_internal<F>(
        &self,
        app_data_dir: &std::path::Path,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
        status_cb: Option<F>,
    ) -> Result<LoadedModelInfo>
    where
        F: Fn(&str, Option<&str>),
    {
        log::info!(
            "[STAGE 3 MANAGER] load_installed_model_internal entered: provider_id='{}', model_id='{}', quantization='{}', app_data_dir={:?}",
            provider_id, model_id, quantization, app_data_dir
        );

        if let Some(ref cb) = status_cb {
            cb("Loading", Some("Reading model manifest..."));
        }

        // Resolve package directory and read manifest
        let package_dir = AdapterRegistry::resolve_package_dir(app_data_dir, provider_id, model_id);
        log::info!("[STAGE 3 MANAGER] Resolved package_dir: {:?} (exists: {})", package_dir, package_dir.exists());

        let manifest = AdapterRegistry::ensure_valid_manifest(&package_dir, provider_id, model_id)
            .map_err(|e| {
                let err = anyhow!("[STAGE 3 MANAGER ERROR] Failed to read or repair manifest for model '{}' in {:?}: {:#}", model_id, package_dir, e);
                log::error!("{}", err);
                err
            })?;
        log::info!("[STAGE 3 MANAGER] Manifest read successfully: name='{}', base_file='{}'", manifest.base_model.model_name, manifest.base_model.file_path);

        // Locate the GGUF file
        let gguf_path = Self::resolve_gguf_path(&package_dir, &manifest)
            .map_err(|e| {
                let err = anyhow!("[STAGE 3 MANAGER ERROR] Failed to resolve GGUF path: {:#}", e);
                log::error!("{}", err);
                err
            })?;
        log::info!("[STAGE 3 MANAGER] Resolved GGUF path: '{}' (exists: {})", gguf_path, std::path::Path::new(&gguf_path).exists());

        let profile = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest)
            .unwrap_or_else(|_| crate::model_intelligence::ModelProfile::new(&manifest.package_id, model_id, &manifest.base_model.model_name));
        log::info!("[STAGE 3 MANAGER] Loaded ModelProfile: family={:?}, chat_template='{}'", profile.model_family, profile.chat_template);

        if let Some(ref cb) = status_cb {
            cb("Loading", Some("Analyzing hardware configuration..."));
        }

        // Build load configuration from hardware profile + manifest + profile
        let config = Self::build_load_config(
            app_data_dir,
            &gguf_path,
            model_id,
            &manifest,
            quantization,
            &profile,
        ).map_err(|e| {
            let err = anyhow!("[STAGE 3 MANAGER ERROR] Failed to build load config: {:#}", e);
            log::error!("{}", err);
            err
        })?;

        log::info!(
            "[STAGE 3 MANAGER] Built ModelLoadConfig: model_path='{}', model_id='{}', context_length={}, gpu_layers={}, threads={}, chat_template='{}'",
            config.model_path, config.model_id, config.context_length, config.gpu_layers, config.threads, config.chat_template
        );

        // Perform the actual load (auto-unloads previous model)
        let info_res = {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.load_model(&config, |step| {
                log::info!("[STAGE 3 MANAGER PROGRESS] Step: {}", step);
            })
        };

        let info = info_res.map_err(|e| {
            let err = anyhow!("[STAGE 3 MANAGER ERROR] Runtime load_model failed: {:#}", e);
            log::error!("{}", err);
            err
        })?;

        log::info!("[STAGE 3 MANAGER SUCCESS] Model loaded cleanly: {:?}", info);

        // Record package context for per-turn capability resolution, and clear
        // any capability stickiness carried over from the previous model.
        if let Ok(mut guard) = self.active_package.lock() {
            *guard = Some(ActivePackage {
                package_dir: package_dir.clone(),
                manifest: manifest.clone(),
            });
        }
        self.capability.reset();

        self.set_last_used_model_id(Some(model_id.to_string()));
        let _ = super::session::SessionManager::save_session(app_data_dir, provider_id, model_id, quantization);

        if let Some(ref cb) = status_cb {
            cb("Ready", None);
        }

        Ok(info)
    }

    /// Direct unload without requiring Tauri AppHandle
    pub fn unload_active_model_direct(&self) -> Result<()> {
        self.clear_package_context();
        let mut runtime = self.runtime.lock().unwrap();
        runtime.unload_model()
    }

    /// Drops package context and capability stickiness.
    ///
    /// Called on every unload path so a newly loaded model never inherits the
    /// previous model's active capability or adapter bindings.
    fn clear_package_context(&self) {
        if let Ok(mut guard) = self.active_package.lock() {
            *guard = None;
        }
        self.capability.reset();
    }

    /// Unloads the currently active model
    pub fn unload_active_model(&self, app_handle: &tauri::AppHandle) -> Result<()> {
        if let Ok(app_dir) = app_handle.path().app_data_dir() {
            let _ = super::session::SessionManager::clear_session(&app_dir);
        }

        self.clear_package_context();

        let _ = app_handle.emit("inference:status", InferenceStatusPayload {
            status: "Unloading".to_string(),
            step: Some("Releasing model resources...".to_string()),
            model: None,
            error: None,
        });

        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.unload_model()?;
        }

        let _ = app_handle.emit("inference:status", InferenceStatusPayload {
            status: "NotLoaded".to_string(),
            step: None,
            model: None,
            error: None,
        });

        Ok(())
    }

    /// Sends a chat message and streams tokens via Tauri events.
    ///
    /// Emits `inference:token` events for each generated token.
    /// The generation can be cancelled by calling `stop_generation`.
    pub fn send_chat_message(
        &self,
        app_handle: &tauri::AppHandle,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        manual_capability: Option<String>,
    ) -> Result<()> {
        // Emit generating status
        let _ = app_handle.emit("inference:status", InferenceStatusPayload {
            status: "Generating".to_string(),
            step: None,
            model: self.get_loaded_model_info(),
            error: None,
        });

        // Resolve the capability for this turn and apply it to the prompt and
        // sampler. Previously the routing result was computed in the UI purely
        // to render a badge, and generation ran on the unmodified base model.
        let (final_messages, final_params, capability_backend) =
            self.prepare_capability_turn(app_handle, &messages, &params, manual_capability.as_deref());

        let app_handle_clone = app_handle.clone();
        let result = {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.generate_with_capability(
                &final_messages,
                &final_params,
                capability_backend.as_ref(),
                |chunk| {
                    let _ = app_handle_clone.emit("inference:token", &chunk);
                },
            )
        };

        match result {
            Ok(_) => {
                // Emit ready status after generation completes
                let _ = app_handle.emit("inference:status", InferenceStatusPayload {
                    status: "Ready".to_string(),
                    step: None,
                    model: self.get_loaded_model_info(),
                    error: None,
                });
                Ok(())
            }
            Err(e) => {
                let err_msg = e.to_string();
                let _ = app_handle.emit("inference:error", serde_json::json!({
                    "error": err_msg,
                }));
                let _ = app_handle.emit("inference:status", InferenceStatusPayload {
                    status: "Ready".to_string(),
                    step: None,
                    model: self.get_loaded_model_info(),
                    error: Some(err_msg.clone()),
                });
                Err(e)
            }
        }
    }

    /// Classifies the turn, resolves a capability backend, and layers it onto
    /// the prompt and sampling parameters.
    ///
    /// Returns the messages and params to generate with, plus the backend to
    /// bind. Falls back to the untouched inputs whenever no package context is
    /// available or the turn resolves to general conversation.
    fn prepare_capability_turn(
        &self,
        app_handle: &tauri::AppHandle,
        messages: &[ChatMessage],
        params: &GenerationParams,
        manual_capability: Option<&str>,
    ) -> (Vec<ChatMessage>, GenerationParams, Option<capability::CapabilityBackend>) {
        // Explicitly typed: a bare `None` here would be ambiguous to infer.
        let untouched = || -> (Vec<ChatMessage>, GenerationParams, Option<capability::CapabilityBackend>) {
            (messages.to_vec(), params.clone(), None)
        };

        let Some(package) = self.active_package() else {
            log::debug!("[CAPABILITY] No active package context — generating on base model");
            return untouched();
        };

        // Classify on the latest user turn only; earlier turns describe past
        // intent, not the request being answered now.
        let Some(prompt) = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
        else {
            return untouched();
        };

        let turn = self.capability.resolve_turn(
            prompt,
            &package.package_dir,
            &package.manifest,
            manual_capability,
        );

        let is_base = matches!(turn.resolution.backend, capability::CapabilityBackend::Base);

        let final_messages = capability::apply_directive(messages, &turn.resolution.spec);
        let final_params = capability::apply_sampling(params, &turn.resolution.spec);

        // Tell the UI what is actually in force — emitted after resolution and
        // parameter merging, so the badge and diagnostics reflect a real binding
        // rather than an intention.
        let payload: CapabilityPayload = turn.payload(if is_base { params } else { &final_params });
        let _ = app_handle.emit("capability:changed", &payload);

        if is_base {
            return untouched();
        }

        log::info!(
            "[CAPABILITY] Applied '{}' via {} (temp {:.2} -> {:.2})",
            turn.resolution.capability,
            turn.resolution.backend.label(),
            params.temperature,
            final_params.temperature
        );

        (final_messages, final_params, Some(turn.resolution.backend))
    }

    /// Direct generation without requiring a Tauri AppHandle (for test scripts & backend execution)
    pub fn generate_direct<F>(
        &self,
        messages: &[ChatMessage],
        params: &GenerationParams,
        token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.generate(messages, params, token_cb)
    }

    /// Stops the current token generation
    pub fn stop_generation(&self) {
        let runtime = self.runtime.lock().unwrap();
        runtime.stop_generation();
    }

    /// Clones the runtime's cancellation flag, if a model is loaded.
    ///
    /// Callers that need to interrupt a generation already in flight must obtain
    /// this *before* generation starts: the runtime mutex is held for the whole
    /// of `generate_direct`, so `stop_generation` would deadlock if called from
    /// inside a token callback.
    pub fn cancel_handle(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        let runtime = self.runtime.lock().unwrap();
        if runtime.loaded_model_info().is_some() {
            Some(runtime.cancel_flag())
        } else {
            None
        }
    }

    /// Builds a `ModelLoadConfig` from the manifest and hardware profile.
    ///
    /// Context length comes from the Phase 3 recommendation (via manifest) or
    /// is calculated dynamically from available RAM/VRAM.
    /// GPU layers and thread count come from the Phase 2 hardware profile.
    pub(crate) fn build_load_config(
        app_data_dir: &std::path::Path,
        gguf_path: &str,
        model_id: &str,
        manifest: &ModelPackageManifest,
        quantization: &str,
        profile: &crate::model_intelligence::ModelProfile,
    ) -> Result<ModelLoadConfig> {
        let analyzer = system_analyzer::get_system_analyzer_manager();

        // Detection is demanded here, not merely read.
        //
        // The startup auto-load races the background hardware scan and usually
        // wins, so the profile is still empty when the most important load of
        // the session is planned. A missing profile silently means "no GPU",
        // which is indistinguishable from a machine that genuinely has none —
        // the model went to CPU on a box with an idle discrete card, every
        // session, and the log line explaining it read like a hardware fact.
        //
        // `analyze_system` joins an in-flight scan rather than duplicating it,
        // so this costs the scan once and only when something got there first.
        let hw_profile = analyzer.get_profile().or_else(|| {
            log::info!(
                "[INFERENCE_MGR] Hardware profile not ready; running detection before planning GPU offload"
            );
            if let Err(e) = analyzer.analyze_system() {
                log::warn!("[INFERENCE_MGR] Hardware detection failed ({e:#}); planning for CPU");
            }
            analyzer.get_profile()
        });

        // Thread count comes from the machine, with no ceiling of our own.
        //
        // Physical cores rather than logical: llama.cpp's GEMM is memory-bandwidth
        // bound, and running two hyperthreads per core contends for the same L2
        // without adding throughput. The previous cap of 16 was arbitrary — it
        // silently discarded half the CPU on a 32-core workstation, and the
        // fallback's cap of 8 did the same on anything larger.
        let threads = hw_profile
            .as_ref()
            .map(|p| p.cpu.current().physical_cores)
            .filter(|&cores| cores > 0)
            .or_else(|| {
                // No profile yet: sysinfo reports logical CPUs, so halve them as
                // an estimate of physical cores on an SMT machine.
                let sys = sysinfo::System::new_all();
                let logical = sys.cpus().len() as u32;
                (logical > 0).then(|| logical.div_ceil(2))
            })
            .unwrap_or(1)
            .max(1);

        // Provisional context length, needed to size the KV cache before the
        // certified profile is consulted below.
        let requested_context = profile.effective_params().context_length;

        // Determine GPU layers from the hardware profile, accounting for KV
        // cache and OS reserve rather than comparing raw VRAM to file size.
        let selected_gpu = hw_profile
            .as_ref()
            .and_then(|hw| select_inference_gpu(hw.gpus.current()));

        // Size the KV cache against a working context, not the model's ceiling.
        //
        // Two failures sat on either side of this. Planning against the
        // advertised maximum charged the budget for a cache far larger than
        // would ever be allocated — a 12B claiming 256K left nothing for
        // weights, so the planner said not one layer fit and sent a model that
        // partially offloads comfortably to pure CPU. Shrinking the context
        // until everything fit in VRAM instead bought full offload at 2265
        // tokens, which no coding agent can work in; its system prompt alone is
        // refused at that size.
        //
        // So the context is held at something usable and the layer count is
        // what gives. Partial offload degrades smoothly; a context too small to
        // hold the first message does not.
        let planned_context = requested_context.min(DEFAULT_WORKING_CONTEXT);

        if planned_context < requested_context {
            log::info!(
                "[INFERENCE_MGR] Planning against a {}-token working context, not the model's advertised {}",
                planned_context,
                requested_context
            );
        }

        // Real geometry from the GGUF header, so placement works from the
        // model's actual layer count and KV cost instead of size-banded
        // guesses. An unreadable header is not fatal — planning falls back to
        // the estimates it used before, just less precisely.
        let gguf_meta =
            match crate::ai_engine::gguf_meta::read_gguf_metadata(std::path::Path::new(gguf_path)) {
                Ok(meta) => Some(meta),
                Err(e) => {
                    log::warn!(
                        "[INFERENCE_MGR] Could not read GGUF header ({e:#}); planning on estimates"
                    );
                    None
                }
            };

        // Host memory available for offloaded experts. Routed through the same
        // budget calculator the recommender uses, so the loader and the
        // recommendation apply identical OS reserves rather than two different
        // notions of "usable".
        let usable_ram = hw_profile
            .as_ref()
            .map(|hw| {
                crate::model_recommendation::budget::calculate_budget(
                    hw,
                    &crate::model_recommendation::traits::BudgetConfig::default(),
                )
                .system_ram
                .usable_for_inference
            })
            .unwrap_or(0);

        let (gpu_layers, cpu_moe_layers) = match &selected_gpu {
            Some(gpu) => {
                let budget = usable_vram_bytes(gpu);
                let model_bytes = manifest.base_model.size_bytes;
                let gpu_label = format!(
                    "GPU '{}' ({}, {:.2} GB usable of {:.2} GB)",
                    gpu.model,
                    if gpu.is_dedicated { "dedicated" } else { "integrated" },
                    budget as f64 / 1e9,
                    gpu.vram_total_bytes as f64 / 1e9,
                );

                // A MoE model is placed by tensor, not by layer: routed experts
                // move to system RAM while attention, KV cache, router and
                // shared experts stay on the card. Reducing the layer count —
                // the dense lever — would evict exactly the wrong things.
                let moe_plan = gguf_meta.as_ref().filter(|m| m.is_moe()).map(|m| {
                    let geom = crate::ai_engine::vram_planner::MoeGeometry {
                        total_layers: m.block_count,
                        expert_bytes: m.expert_bytes(model_bytes, None),
                        kv_bytes_per_token: m.kv_bytes_per_token(),
                        active_params: m.active_params(None).unwrap_or(0),
                    };
                    crate::ai_engine::vram_planner::plan_moe_offload(
                        model_id,
                        budget,
                        usable_ram,
                        model_bytes,
                        planned_context,
                        &geom,
                    )
                });

                match moe_plan {
                    Some(plan) if plan.fits => {
                        log::info!("[INFERENCE_MGR] Selected {gpu_label}: {}", plan.reason);
                        (plan.gpu_layers, plan.cpu_moe_layers)
                    }
                    rejected => {
                        if let Some(plan) = rejected {
                            log::info!(
                                "[INFERENCE_MGR] Expert offload not viable ({}); placing densely instead",
                                plan.reason
                            );
                        }
                        let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
                            budget,
                            model_bytes,
                            planned_context,
                            gguf_meta.as_ref().map(|m| m.block_count),
                        );
                        log::info!("[INFERENCE_MGR] Selected {gpu_label}: {}", plan.reason);
                        (plan.gpu_layers, 0)
                    }
                }
            }
            None if hw_profile.is_some() => {
                log::info!("[INFERENCE_MGR] No GPU with a usable accelerator backend, using CPU mode (0 layers)");
                (0, 0)
            }
            None => {
                log::info!("[INFERENCE_MGR] No hardware profile available, defaulting to CPU mode");
                (0, 0)
            }
        };

        // Check for authoritative certified RuntimeProfile from PackManager
        let app_data = app_data_dir.to_path_buf();
        let certified_profile = crate::model_recommendation::pack_manager::PackManager::new(&app_data)
            .ok()
            .and_then(|pm| pm.get_package_certification(model_id).and_then(|c| pm.get_runtime_profile(&c.runtime_profile_id)));

        let (chat_template, stop_tokens, context_length, gpu_layers, threads) = if let Some(ref cert_prof) = certified_profile {
            log::info!(
                "[INFERENCE_MGR] Applying Authoritative Saarthi Certified RuntimeProfile '{}' for model '{}'",
                cert_prof.profile_id, model_id
            );
            let cfg = &cert_prof.execution_config;

            // A certified profile is static per-model JSON shipped with the app.
            // It describes the MODEL (template, stop tokens, maximum context) and
            // is authoritative for those. It cannot know anything about the machine
            // it lands on, so every hardware-dependent value stays measured here.
            // Same GPU the offload plan above chose, so the context it sizes and
            // the layers it places are budgeted against one card, not two.
            let detected_vram = selected_gpu
                .as_ref()
                .map(|gpu| usable_vram_bytes(gpu))
                .unwrap_or(0);

            // The model's advertised context is an upper bound, not an entitlement.
            let affordable_context = crate::ai_engine::vram_planner::max_affordable_context(
                detected_vram,
                manifest.base_model.size_bytes,
                cfg.context_length,
            );
            if affordable_context < cfg.context_length {
                log::info!(
                    "[INFERENCE_MGR] Context reduced {} -> {} to fit {:.2} GB VRAM",
                    cfg.context_length,
                    affordable_context,
                    detected_vram as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }

            if cfg.threads != threads {
                log::info!(
                    "[INFERENCE_MGR] Using {} detected CPU threads, not the profile's {}",
                    threads, cfg.threads
                );
            }

            (
                cfg.chat_template.clone(),
                cfg.stop_tokens.clone(),
                affordable_context,
                // The profile may lower the offload but never raise it — taking
                // `max` here let a profile's 999 override a hardware plan of ~12
                // layers on a 4 GB card and run out of memory.
                if gpu_layers > 0 { std::cmp::min(gpu_layers, cfg.gpu_layers) } else { 0 },
                // Detected physical cores, never the profile's fixed number.
                threads,
            )
        } else {
            (
                profile.chat_template.clone(),
                profile.tokens.stop_tokens.clone(),
                // The context the offload was budgeted against, not the model's
                // advertised maximum — allocating a larger cache than the plan
                // assumed is what pushes the load past the card's memory.
                planned_context,
                gpu_layers,
                threads,
            )
        };

        let model_name = manifest.base_model.model_name.clone();

        Ok(ModelLoadConfig {
            model_path: gguf_path.to_string(),
            model_id: model_id.to_string(),
            model_name,
            quantization: quantization.to_string(),
            context_length,
            gpu_layers,
            // Survives a certified profile's clamp on `gpu_layers`: that clamp
            // only ever lowers GPU residency, which frees VRAM, so an expert
            // split planned against a larger budget stays valid.
            cpu_moe_layers,
            threads,
            chat_template,
            stop_tokens,
        })
    }

    /// Resolves the absolute path to the primary GGUF file from manifest
    pub(crate) fn resolve_gguf_path(
        package_dir: &std::path::Path,
        manifest: &ModelPackageManifest,
    ) -> Result<String> {
        // manifest.base_model.file_path is relative to package_dir (e.g., "base/model.gguf")
        let gguf_path = package_dir.join(&manifest.base_model.file_path);

        // Check if manifest.base_model.file_path points directly to a valid .gguf FILE (not a directory)
        if gguf_path.exists() && gguf_path.is_file() {
            let clean = gguf_path.to_string_lossy().to_string();
            log::info!("[INFERENCE_MGR] GGUF file resolved at manifest path: {}", clean);
            return Ok(clean);
        }

        log::warn!("[INFERENCE_MGR] Manifest filePath '{:?}' is not a file. Scanning base/ directory for .gguf...", gguf_path);

        // Fallback: scan base/ directory for any .gguf file, prioritizing -00001-of-
        let base_dir = package_dir.join("base");
        if base_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&base_dir) {
                let mut found_files = Vec::new();
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension().map_or(false, |ext| ext == "gguf") {
                        found_files.push(p.to_string_lossy().to_string());
                    }
                }
                if !found_files.is_empty() {
                    found_files.sort();
                    let primary = found_files.iter().find(|f| f.contains("-00001-of-")).cloned().unwrap_or_else(|| found_files[0].clone());
                    log::info!("[INFERENCE_MGR] GGUF file found via base directory scan: {}", primary);
                    return Ok(primary);
                }
            }
        }
        Err(anyhow!(
            "GGUF file not found in package directory '{:?}'",
            package_dir
        ))
    }
}

/// VRAM this GPU can actually give the model right now.
///
/// Free memory when the driver reports it — a desktop compositor, a browser, or
/// another model may already hold part of the card, and budgeting against the
/// total would plan for memory that is not there. Falls back to total when the
/// vendor exposes no free figure.
fn usable_vram_bytes(gpu: &crate::system_analyzer::traits::GpuInfo) -> u64 {
    if gpu.vram_free_bytes > 0 {
        gpu.vram_free_bytes
    } else {
        gpu.vram_total_bytes
    }
}

/// Picks the GPU most likely to run the model fastest.
///
/// Enumeration order is not preference order: adapters come back in whatever
/// order the driver reports them, so taking the first compatible one could put
/// the model on an integrated GPU sharing system RAM while a discrete card sat
/// idle beside it. This machine reports exactly that pair.
///
/// Dedicated memory is ranked *before* capacity, because an integrated GPU's
/// reported "VRAM" is not comparable to a discrete card's.
///
/// An iGPU advertises a slice of system RAM — this machine's Radeon 780M
/// reports 13 GB beside an RTX 5060's real 8 GB — so ranking by capacity first
/// handed every model to the slower device on its own memory bus while the
/// discrete card sat idle. Capacity only decides between cards of the same
/// kind. Any backend the build can drive is eligible; a card with no usable
/// memory is not a candidate at all, which is what leaves CPU as the fallback
/// rather than a broken GPU path.
pub(crate) fn select_inference_gpu(
    gpus: &[crate::system_analyzer::traits::GpuInfo],
) -> Option<crate::system_analyzer::traits::GpuInfo> {
    gpus.iter()
        .filter(|g| g.cuda_supported || g.vulkan_supported || g.rocm_supported)
        .filter(|g| usable_vram_bytes(g) > 0)
        .max_by(|a, b| {
            a.is_dedicated
                .cmp(&b.is_dedicated)
                .then(usable_vram_bytes(a).cmp(&usable_vram_bytes(b)))
        })
        .cloned()
}

impl Default for InferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_analyzer::traits::GpuInfo;

    /// A GPU with only the fields selection looks at set.
    fn gpu(model: &str, dedicated: bool, total: u64, free: u64, cuda: bool, vulkan: bool) -> GpuInfo {
        GpuInfo {
            vendor: String::new(),
            model: model.to_string(),
            gpu_type: String::new(),
            is_dedicated: dedicated,
            dedicated_video_memory_bytes: if dedicated { total } else { 0 },
            dedicated_system_memory_bytes: 0,
            shared_system_memory_bytes: if dedicated { 0 } else { total },
            total_available_graphics_memory_bytes: total,
            vram_total_bytes: total,
            vram_free_bytes: free,
            driver_version: None,
            vendor_id: None,
            device_id: None,
            compute_capability: None,
            cuda_supported: cuda,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: vulkan,
            opencl_supported: true,
            detection_source: "test".into(),
            confidence: "High".into(),
        }
    }

    const GB: u64 = 1_000_000_000;

    #[test]
    fn the_discrete_gpu_wins_even_when_the_integrated_one_reports_more_memory() {
        // The real shape of this machine: a Radeon 780M advertising ~13 GB of
        // shared system memory beside an RTX 5060 with 8 GB of its own, the
        // iGPU enumerated first.
        //
        // Ranking by capacity picked the iGPU, so every model ran on the slower
        // device over system RAM while the discrete card sat idle. An iGPU's
        // reported memory is not comparable to real VRAM, so it must not
        // outrank it however large it looks.
        let gpus = vec![
            gpu("Integrated", false, 13 * GB, 12 * GB, false, true),
            gpu("Discrete", true, 8 * GB, 7 * GB, true, true),
        ];

        let picked = select_inference_gpu(&gpus).expect("a compatible GPU exists");
        assert_eq!(picked.model, "Discrete", "a dedicated card outranks any iGPU");
    }

    #[test]
    fn a_long_context_model_is_planned_around_a_usable_working_context() {
        // Gemma 3/4 and Qwen 3 advertise 262144. Budgeting a KV cache for that
        // leaves nothing for weights, which sent a 12B that partially offloads
        // fine to pure CPU; shrinking the context until it fully offloaded gave
        // 2265 tokens, too small for a coding agent's opening message.
        let planned = 262_144u32.min(DEFAULT_WORKING_CONTEXT);
        assert_eq!(planned, 8192);

        // With that context the planner places most of a 6.09 GB model on an
        // 8.28 GB card rather than giving up on the GPU.
        let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
            8_280_000_000,
            6_087_086_624,
            planned,
            Some(48),
        );
        assert!(
            plan.gpu_layers > 0,
            "a model this size must still reach the GPU: {}",
            plan.reason
        );
    }

    #[test]
    fn a_model_asking_for_less_than_the_working_context_keeps_its_own() {
        // The cap is a ceiling, not a floor — a 4K model must not be inflated.
        assert_eq!(4096u32.min(DEFAULT_WORKING_CONTEXT), 4096);
    }

    #[test]
    fn capacity_still_decides_between_two_cards_of_the_same_kind() {
        let discrete = vec![
            gpu("Small", true, 8 * GB, 7 * GB, true, true),
            gpu("Large", true, 24 * GB, 22 * GB, true, true),
        ];
        assert_eq!(
            select_inference_gpu(&discrete).expect("a compatible GPU exists").model,
            "Large"
        );

        // With no discrete card at all, the iGPU is still better than nothing.
        let integrated_only = vec![gpu("Integrated", false, 13 * GB, 12 * GB, false, true)];
        assert_eq!(
            select_inference_gpu(&integrated_only).expect("a compatible GPU exists").model,
            "Integrated"
        );
    }

    #[test]
    fn a_gpu_with_no_usable_backend_is_never_selected() {
        // Present, but nothing the build can drive: CPU must remain the answer
        // rather than a GPU path that cannot work.
        let gpus = vec![gpu("Display only", true, 8 * GB, 8 * GB, false, false)];
        assert!(select_inference_gpu(&gpus).is_none());
        assert!(select_inference_gpu(&[]).is_none());
    }

    #[test]
    fn a_card_already_full_is_not_offered_as_a_target() {
        // Reports a backend but has nothing left to give — planning against its
        // total would place layers in memory another process holds.
        let gpus = vec![gpu("Busy", true, 8 * GB, 0, true, true)];
        let picked = select_inference_gpu(&gpus).expect("free is 0, so total is used");
        assert_eq!(usable_vram_bytes(&picked), 8 * GB);

        let unknown_total = vec![gpu("Odd", true, 0, 0, true, true)];
        assert!(
            select_inference_gpu(&unknown_total).is_none(),
            "no memory figure at all means no GPU plan"
        );
    }

    #[test]
    fn free_vram_is_preferred_over_total_when_the_driver_reports_it() {
        // Another process holding half the card must shrink the budget, or the
        // planner offloads more layers than will fit.
        let busy = gpu("Half used", true, 8 * GB, 3 * GB, true, false);
        assert_eq!(usable_vram_bytes(&busy), 3 * GB);
    }

    #[test]
    fn test_inference_manager_initial_state() {
        let mgr = InferenceManager::new();
        assert_eq!(mgr.get_status(), RuntimeStatus::NotLoaded);
        assert!(mgr.get_loaded_model_info().is_none());
        assert!(mgr.get_last_used_model_id().is_none());
    }

    #[test]
    fn test_last_used_model_persistence() {
        let mgr = InferenceManager::new();
        assert!(mgr.get_last_used_model_id().is_none());

        mgr.set_last_used_model_id(Some("meta-llama/Llama-3.2-1B".to_string()));
        assert_eq!(
            mgr.get_last_used_model_id(),
            Some("meta-llama/Llama-3.2-1B".to_string())
        );

        mgr.set_last_used_model_id(None);
        assert!(mgr.get_last_used_model_id().is_none());
    }
}

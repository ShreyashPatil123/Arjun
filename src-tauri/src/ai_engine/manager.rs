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
        let hw_profile = analyzer.get_profile();

        // Determine thread count from hardware profile
        let threads = if let Some(ref profile) = hw_profile {
            let cpu = profile.cpu.current();
            // Use physical cores (not logical/HT) for inference, capped at reasonable value
            let physical = cpu.physical_cores;
            std::cmp::min(physical, 16) // Cap at 16 threads
        } else {
            // Fallback: use half of available logical processors
            let sys = sysinfo::System::new_all();
            let cpus = sys.cpus().len() as u32;
            std::cmp::max(1, std::cmp::min(cpus / 2, 8))
        };

        // Provisional context length, needed to size the KV cache before the
        // certified profile is consulted below.
        let planned_context = profile.effective_params().context_length;

        // Determine GPU layers from the hardware profile, accounting for KV
        // cache and OS reserve rather than comparing raw VRAM to file size.
        let gpu_layers = if let Some(ref hw) = hw_profile {
            let gpus = hw.gpus.current();
            if let Some(gpu) = gpus.iter().find(|g| g.cuda_supported || g.vulkan_supported) {
                let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
                    gpu.vram_total_bytes,
                    manifest.base_model.size_bytes,
                    planned_context,
                    None, // GGUF layer count not parsed at load time
                );
                log::info!(
                    "[INFERENCE_MGR] GPU '{}': {}",
                    gpu.model, plan.reason
                );
                plan.gpu_layers
            } else {
                log::info!("[INFERENCE_MGR] No CUDA/Vulkan capable GPU detected, using CPU mode (0 layers)");
                0
            }
        } else {
            log::info!("[INFERENCE_MGR] No hardware profile available, defaulting to CPU mode");
            0
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
            let detected_vram = hw_profile
                .as_ref()
                .and_then(|hw| {
                    hw.gpus
                        .current()
                        .iter()
                        .find(|g| g.cuda_supported || g.vulkan_supported)
                        .map(|g| g.vram_total_bytes)
                })
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
                profile.effective_params().context_length,
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
            let clean = gguf_path.to_string_lossy().replace('/', "\\");
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
                        found_files.push(p.to_string_lossy().replace('/', "\\"));
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

impl Default for InferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

//! Inference Manager — Thread-safe wrapper around LlamaCppRuntime
//!
//! Manages model loading/unloading with hardware-aware configuration,
//! provides streaming generation via Tauri events, and tracks the last used model.

use std::sync::{Arc, Mutex};
use tauri::Manager;

use anyhow::{anyhow, Result};
use tauri::Emitter;

use crate::adapter_manager::{AdapterRegistry, ModelPackageManifest};
use crate::ai_engine::runtime::LlamaCppRuntime;
use crate::ai_engine::traits::*;
use crate::system_analyzer;

/// Thread-safe inference state manager.
///
/// Wraps `LlamaCppRuntime` in `Arc<Mutex<>>` for safe concurrent access
/// from multiple Tauri command handlers.
pub struct InferenceManager {
    runtime: Arc<Mutex<LlamaCppRuntime>>,
    last_used_model_id: Arc<Mutex<Option<String>>>,
}

impl InferenceManager {
    /// Creates a new InferenceManager with no model loaded
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(Mutex::new(LlamaCppRuntime::new())),
            last_used_model_id: Arc::new(Mutex::new(None)),
        }
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
        self.set_last_used_model_id(Some(model_id.to_string()));
        let _ = super::session::SessionManager::save_session(app_data_dir, provider_id, model_id, quantization);

        if let Some(ref cb) = status_cb {
            cb("Ready", None);
        }

        Ok(info)
    }

    /// Direct unload without requiring Tauri AppHandle
    pub fn unload_active_model_direct(&self) -> Result<()> {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.unload_model()
    }

    /// Unloads the currently active model
    pub fn unload_active_model(&self, app_handle: &tauri::AppHandle) -> Result<()> {
        if let Ok(app_dir) = app_handle.path().app_data_dir() {
            let _ = super::session::SessionManager::clear_session(&app_dir);
        }

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
    ) -> Result<()> {
        // Emit generating status
        let _ = app_handle.emit("inference:status", InferenceStatusPayload {
            status: "Generating".to_string(),
            step: None,
            model: self.get_loaded_model_info(),
            error: None,
        });

        let app_handle_clone = app_handle.clone();
        let result = {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.generate(&messages, &params, |chunk| {
                let _ = app_handle_clone.emit("inference:token", &chunk);
            })
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

        // Determine GPU layers from hardware profile dynamically
        let gpu_layers = if let Some(ref profile) = hw_profile {
            let gpus = profile.gpus.current();
            // Find any GPU (dedicated or integrated) with CUDA or Vulkan acceleration
            if let Some(gpu) = gpus.iter().find(|g| g.cuda_supported || g.vulkan_supported) {
                let vram_gb = gpu.vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let model_size_gb = manifest.base_model.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                log::info!(
                    "[INFERENCE_MGR] Detected GPU '{}' ({:.2} GB VRAM) for model '{:.2} GB'",
                    gpu.model, vram_gb, model_size_gb
                );

                if vram_gb < 1.5 {
                    log::info!("[INFERENCE_MGR] Low VRAM (<1.5 GB), defaulting to CPU mode");
                    0
                } else if vram_gb >= model_size_gb + 1.0 || vram_gb >= 6.0 {
                    // Full GPU offload when VRAM comfortably fits model + KV cache
                    log::info!("[INFERENCE_MGR] Full GPU offload selected (999 layers)");
                    999
                } else {
                    // Partial layer offload proportional to available VRAM
                    let ratio = (vram_gb / (model_size_gb + 0.5)).clamp(0.1, 0.9);
                    let layers = (ratio * 32.0).round() as u32;
                    log::info!("[INFERENCE_MGR] Partial GPU offload calculated: {} layers", layers);
                    layers
                }
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
            (
                cfg.chat_template.clone(),
                cfg.stop_tokens.clone(),
                cfg.context_length,
                if gpu_layers > 0 { std::cmp::max(gpu_layers, cfg.gpu_layers) } else { 0 },
                cfg.threads,
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

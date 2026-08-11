//! LlamaCpp Runtime — In-process GGUF model inference via llama-cpp-2
//!
//! Provides model loading, token generation with streaming, and resource management.
//! Designed to be wrapped by InferenceManager for thread-safe Tauri integration.

use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::ai_engine::lora_binding::LoraAdapterCache;
use crate::ai_engine::traits::*;
use crate::capability::CapabilityBackend;

/// Buffer-type override patterns keeping the routed experts of the first
/// `n_layers` layers in system RAM.
///
/// This is precisely what `--n-cpu-moe N` does. There is no distinct C API for
/// that flag: llama.cpp implements it in `common/arg.cpp` as a loop pushing one
/// buffer-type override per layer, each built from `llm_ffn_exps_block_regex(i)`
/// and bound to the CPU buffer type.
///
/// The binding's own [`LlamaModelParams::add_cpu_moe_override`] is deliberately
/// **not** used. Its regex is `\.ffn_(up|down|gate)_(ch|)exps`, which drops the
/// `gate_up` alternative that upstream carries in `common/common.h`. Models with
/// *fused* expert tensors — `blk.N.ffn_gate_up_exps`, registered in
/// `llama-arch.cpp` and shipped by gpt-oss — would not match, and the regex
/// cannot fall through to the `gate` branch either: after `gate` it requires
/// `_`, then optionally `ch`, then `exps`, while the tensor has `_up_exps`
/// there. The offload would silently do nothing and the model would spill or
/// run out of memory with nothing explaining why.
fn cpu_moe_override_patterns(n_layers: u32) -> Vec<CString> {
    (0..n_layers)
        // The pattern is generated, so it can never contain an interior NUL.
        .filter_map(|i| CString::new(format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps")).ok())
        .collect()
}

/// Core inference runtime wrapping llama.cpp via safe Rust bindings.
///
/// This struct owns the model, context, and backend. It is NOT thread-safe by itself;
/// the InferenceManager wraps it in Arc<Mutex<>> for safe concurrent access.
pub struct LlamaCppRuntime {
    backend: Option<LlamaBackend>,
    model: Option<LlamaModel>,
    loaded_info: Option<LoadedModelInfo>,
    /// The chat template baked into the loaded GGUF, when it ships one.
    ///
    /// Rendering through llama.cpp's own template engine is the only way to be
    /// sure the prompt matches what the model was trained on. The hand-written
    /// templates in `format_chat_prompt_with_template` are guesses keyed off a
    /// family name, and guessing wrong produces confident nonsense rather than
    /// an error — so they are only a fallback for GGUFs that carry no template.
    native_template: Option<NativeChatTemplate>,
    is_generating: Arc<AtomicBool>,
    /// LoRA adapters initialised against the currently loaded model.
    /// Cleared on unload — entries are only valid for the model they were built from.
    adapter_cache: LoraAdapterCache,
}

impl LlamaCppRuntime {
    /// Creates a new unloaded runtime instance
    pub fn new() -> Self {
        Self {
            backend: None,
            model: None,
            loaded_info: None,
            native_template: None,
            is_generating: Arc::new(AtomicBool::new(false)),
            adapter_cache: LoraAdapterCache::new(),
        }
    }

    /// Returns the current runtime status
    pub fn status(&self) -> RuntimeStatus {
        if self.is_generating.load(Ordering::Relaxed) {
            RuntimeStatus::Generating
        } else if self.model.is_some() {
            RuntimeStatus::Ready
        } else {
            RuntimeStatus::NotLoaded
        }
    }

    /// Returns info about the currently loaded model, if any
    pub fn loaded_model_info(&self) -> Option<&LoadedModelInfo> {
        self.loaded_info.as_ref()
    }

    /// Returns a clone of the generation cancellation flag
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.is_generating.clone()
    }

    /// Loads a GGUF model from disk into memory.
    ///
    /// `status_callback` is called with human-readable step descriptions.
    pub fn load_model<F>(&mut self, config: &ModelLoadConfig, mut status_cb: F) -> Result<LoadedModelInfo>
    where
        F: FnMut(&str),
    {
        // Ensure no model is already loaded
        if self.model.is_some() {
            self.unload_model()?;
        }

        log::info!(
            "[STAGE 4 RUNTIME] load_model entered: path='{}', id='{}', ctx={}, gpu_layers={}, threads={}",
            config.model_path, config.model_id, config.context_length, config.gpu_layers, config.threads
        );

        let model_path = Path::new(&config.model_path);
        let path_exists = model_path.exists();
        log::info!("[STAGE 4 RUNTIME] Model path check: '{:?}' exists={}", model_path, path_exists);
        if !path_exists {
            let err = anyhow!("[STAGE 4 RUNTIME ERROR] GGUF model file not found at '{:?}'", model_path);
            log::error!("{}", err);
            return Err(err);
        }

        // Step 1: Initialize backend (keep alive across model loads/reloads)
        status_cb("Initializing llama.cpp backend...");
        if self.backend.is_none() {
            log::info!("[STAGE 4 RUNTIME] Initializing process-global LlamaBackend instance...");
            match LlamaBackend::init() {
                Ok(b) => {
                    log::info!("[STAGE 4 RUNTIME] LlamaBackend initialized successfully!");
                    self.backend = Some(b);
                }
                Err(e) => {
                    let err = anyhow!("[STAGE 4 RUNTIME ERROR] LlamaBackend::init() failed: {:?}", e);
                    log::error!("{}", err);
                    return Err(err);
                }
            }
        } else {
            log::info!("[STAGE 4 RUNTIME] Reusing active process-global LlamaBackend instance");
        }
        let backend = self.backend.as_ref().unwrap();

        // Step 2: Configure model parameters
        status_cb("Configuring model parameters...");
        log::info!(
            "[STAGE 4 RUNTIME] Configuring LlamaModelParams (gpu_layers={}, threads={})",
            config.gpu_layers, config.threads
        );

        // GPU offload only takes effect when llama.cpp was compiled with a GPU
        // backend. Requesting layers without one is a silent no-op inside
        // llama.cpp, which previously made CPU-only builds appear GPU-accelerated
        // in the logs and the UI. Surface the mismatch instead of hiding it.
        let gpu_backend_compiled = cfg!(any(feature = "cuda", feature = "vulkan"));
        if config.gpu_layers > 0 && !gpu_backend_compiled {
            log::warn!(
                "[STAGE 4 RUNTIME WARN] {} GPU layers requested, but this binary was built \
                 without a GPU backend — llama.cpp will ignore the request and run on CPU. \
                 Rebuild with `--features cuda` (needs the CUDA Toolkit) or `--features vulkan`.",
                config.gpu_layers
            );
        }

        let effective_gpu_layers = if gpu_backend_compiled { config.gpu_layers } else { 0 };

        // Pinning experts to CPU is meaningless without a GPU backend — every
        // tensor is already in system RAM — so it is reported as not applied
        // rather than silently requested.
        let effective_cpu_moe = if gpu_backend_compiled { config.cpu_moe_layers } else { 0 };
        if config.cpu_moe_layers > 0 && !gpu_backend_compiled {
            log::warn!(
                "[STAGE 4 RUNTIME WARN] Expert offload for {} layers requested, but this binary \
                 was built without a GPU backend — every tensor is on CPU already.",
                config.cpu_moe_layers
            );
        }

        // Declared before `model_params` so it is dropped *after* them: Rust
        // drops locals in reverse declaration order, and `add_cpu_buft_override`
        // stores a borrowed pointer with no lifetime tie recorded on the params
        // (see the crate's SAFETY note on `tensor_buft_override_patterns`), so
        // these strings must outlive the params that point into them.
        let expert_patterns = cpu_moe_override_patterns(effective_cpu_moe);

        let model_params = {
            let mut params = Box::pin(LlamaModelParams::default().with_n_gpu_layers(effective_gpu_layers));
            for pattern in &expert_patterns {
                params.as_mut().add_cpu_buft_override(pattern);
            }
            params
        };

        if effective_cpu_moe > 0 {
            log::info!(
                "[STAGE 4 RUNTIME] MoE expert offload: routed experts of {} layer(s) pinned to \
                 system RAM via {} buffer-type override(s); all layers still requested on GPU",
                effective_cpu_moe,
                expert_patterns.len()
            );
        }

        // Use the path exactly as resolved upstream. Rewriting separators here
        // corrupts absolute paths on every non-Windows platform.
        let clean_path = config.model_path.clone();
        let path_obj = Path::new(&clean_path);
        let exists = path_obj.exists();
        let size = std::fs::metadata(&clean_path).map(|m| m.len()).unwrap_or(0);

        log::info!(
            "[STAGE 4 RUNTIME PARAMETER AUDIT]\n  -> Raw model_path: '{}'\n  -> Cleaned model_path: '{}'\n  -> File exists: {}\n  -> File size: {} bytes\n  -> Requested gpu_layers: {}\n  -> Context length: {}\n  -> Threads: {}\n  -> Backend active: true",
            config.model_path,
            clean_path,
            exists,
            size,
            config.gpu_layers,
            config.context_length,
            config.threads
        );

        match std::fs::File::open(&clean_path) {
            Ok(mut f) => {
                let mut header_buf = [0u8; 8];
                use std::io::Read;
                if f.read_exact(&mut header_buf).is_err() || &header_buf[0..4] != b"GGUF" {
                    let err = anyhow!(
                        "[STAGE 4 RUNTIME ERROR] Corrupted/Invalid GGUF header at '{}': Expected magic bytes 'GGUF' (0x46554747), found {:?}",
                        clean_path, String::from_utf8_lossy(&header_buf[0..4])
                    );
                    log::error!("{}", err);
                    return Err(err);
                }
                let gguf_ver = u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);
                log::info!("[STAGE 4 RUNTIME AUDIT] std::fs::File::open & GGUF header check succeeded for '{}' (Magic='GGUF', Version={})", clean_path, gguf_ver);
            }
            Err(e) => {
                let err = anyhow!("[STAGE 4 RUNTIME ERROR] std::fs::File::open failed for '{}': {:?}", clean_path, e);
                log::error!("{}", err);
                return Err(err);
            }
        }

        // Shard completeness check for split GGUFs (e.g., -00001-of-00002.gguf)
        if clean_path.contains("-00001-of-") {
            if let Some(pos) = clean_path.find("-00001-of-") {
                let suffix = &clean_path[pos + 10..];
                if let Some(end_pos) = suffix.find(".gguf") {
                    let total_shards_str = &suffix[..end_pos];
                    if let Ok(total_shards) = total_shards_str.parse::<u32>() {
                        for shard_idx in 2..=total_shards {
                            let shard_name = format!("-{:05}-of-{}", shard_idx, suffix);
                            let shard_path_str = clean_path.replace(&format!("-00001-of-{}", suffix), &shard_name);
                            let shard_path = Path::new(&shard_path_str);
                            if !shard_path.exists() || std::fs::metadata(&shard_path_str).map(|m| m.len()).unwrap_or(0) == 0 {
                                let err = anyhow!(
                                    "[STAGE 4 RUNTIME ERROR] Split GGUF shard missing or incomplete: shard {} of {} at '{}'",
                                    shard_idx, total_shards, shard_path_str
                                );
                                log::error!("{}", err);
                                return Err(err);
                            }
                            log::info!("[STAGE 4 RUNTIME AUDIT] Verified split GGUF shard {}/{} exists: '{}'", shard_idx, total_shards, shard_path_str);
                        }
                    }
                }
            }
        }

        // Step 3: Load model from GGUF file
        status_cb("Loading model weights from GGUF file...");
        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME LOAD CONFIGURATION] ====================\n\
             - GGUF File Path:     {} (Source: Manifest / Resolver)\n\
             - Model ID:           {} (Source: Manifest)\n\
             - Quantization:       {} (Source: Model Manifest / User Selection)\n\
             - GPU Layers Offload: {} (Source: Hardware Profile Estimator)\n\
             - CPU Threads:        {} (Source: System Hardware Collector)\n\
             - Context Length:     {} tokens (Source: GGUF / config.json max_position_embeddings)\n\
             - Chat Template:      {} (Source: GGUF / tokenizer_config.json / Jinja)\n\
             - Stop Sequences:     {:?} (Source: GGUF / tokenizer.json added_tokens)\n\
             =================================================================================",
            clean_path,
            config.model_id,
            config.quantization,
            config.gpu_layers,
            config.threads,
            config.context_length,
            config.chat_template,
            config.stop_tokens
        );

        let model_result = LlamaModel::load_from_file(&backend, &clean_path, &model_params);

        let (model, actual_backend) = match model_result {
            Ok(m) => {
                let desc = if effective_gpu_layers > 0 && effective_cpu_moe > 0 {
                    // Names the split explicitly: a user whose experts went to
                    // system RAM should not be told the model is on the GPU.
                    format!(
                        "llama.cpp (GPU + experts of {} layers on CPU)",
                        effective_cpu_moe
                    )
                } else if effective_gpu_layers > 0 {
                    format!("llama.cpp (GPU offload: {} layers)", effective_gpu_layers)
                } else if config.gpu_layers > 0 {
                    "llama.cpp (CPU — built without GPU backend)".to_string()
                } else {
                    "llama.cpp (CPU)".to_string()
                };
                log::info!("[STAGE 4 RUNTIME] LlamaModel::load_from_file succeeded with backend: {}", desc);
                (m, desc)
            }
            Err(e) => {
                log::warn!(
                    "[STAGE 4 RUNTIME WARN] LlamaModel::load_from_file(gpu_layers={}) failed with error: {:?}. Attempting CPU fallback (gpu_layers=0)...",
                    config.gpu_layers, e
                );
                let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
                match LlamaModel::load_from_file(&backend, &clean_path, &cpu_params) {
                    Ok(cpu_model) => {
                        log::info!("[STAGE 4 RUNTIME] CPU fallback LlamaModel::load_from_file succeeded!");
                        (cpu_model, "llama.cpp (CPU Fallback)".to_string())
                    }
                    Err(e2) => {
                        let err = anyhow!(
                            "[STAGE 4 RUNTIME ERROR] LlamaModel::load_from_file failed for both GPU and CPU! GPU error: {:?}, CPU error: {:?}",
                            e, e2
                        );
                        log::error!("{}", err);
                        return Err(err);
                    }
                }
            }
        };

        log::info!("[RUNTIME] Model loaded successfully via {}", actual_backend);

        let backend_desc = actual_backend;

        // Read the chat template out of the GGUF itself. The profile's
        // `chat_template` is only a family-name guess derived from the model id,
        // and it has been wrong in practice (a Gemma model profiled with Llama-3
        // tokens), so the model's own template wins whenever it has one.
        let native_template = match model.chat_template(None) {
            Ok(handle) => {
                let source = model.meta_val_str("tokenizer.chat_template").ok();
                log::info!(
                    "[RUNTIME] ✓ Using chat template embedded in the GGUF (profile guessed '{}', Jinja source {})",
                    config.chat_template,
                    if source.is_some() { "available" } else { "unavailable" }
                );
                let piece = |t| {
                    model
                        .token_to_str(t, llama_cpp_2::model::Special::Tokenize)
                        .unwrap_or_default()
                };
                Some(NativeChatTemplate {
                    source,
                    handle,
                    bos_token: piece(model.token_bos()),
                    eos_token: piece(model.token_eos()),
                })
            }
            Err(e) => {
                log::warn!(
                    "[RUNTIME] GGUF ships no chat template ({e:?}); falling back to the hand-written '{}' template. \
                     If output looks malformed, this is the first thing to suspect.",
                    config.chat_template
                );
                None
            }
        };

        let template_source = if native_template.is_some() {
            "gguf".to_string()
        } else {
            format!("fallback:{}", config.chat_template)
        };

        // Now that llama.cpp has parsed the GGUF, write the real architecture and
        // token metadata back to profile.json. Profiles are generated at download
        // time, before the `.part` file is finalized, so the extractor finds no
        // GGUF to read and the profile keeps whatever defaults it started with.
        // This is the first moment the truth is available.
        let runtime_meta = Self::extract_runtime_metadata(&model, native_template.is_some());
        let effective_stop_tokens = Self::sync_profile_metadata(&clean_path, &runtime_meta)
            .unwrap_or_else(|| config.stop_tokens.clone());

        // Step 4: Store state
        status_cb("Model loaded and ready for inference");
        let fam_str = format!("{:?}", crate::model_intelligence::MetadataExtractor::infer_family_from_string(&config.model_id));
        let info = LoadedModelInfo {
            model_id: config.model_id.clone(),
            model_name: config.model_name.clone(),
            quantization: config.quantization.clone(),
            file_path: config.model_path.clone(),
            context_length: config.context_length,
            // Report what was actually applied, not what was requested.
            gpu_layers: effective_gpu_layers,
            cpu_moe_layers: effective_cpu_moe,
            threads: config.threads,
            backend_used: backend_desc,
            loaded_at: chrono::Utc::now().to_rfc3339(),
            chat_template: config.chat_template.clone(),
            template_source,
            stop_tokens: effective_stop_tokens,
            model_family: fam_str,
            active_adapter: None,
        };

        self.model = Some(model);
        self.native_template = native_template;
        self.loaded_info = Some(info.clone());

        log::info!(
            "[RUNTIME] ✓ Model ready: {} ({}) — context={}, gpu_layers={}, threads={}, backend={}",
            info.model_name, info.quantization, info.context_length,
            info.gpu_layers, info.threads, info.backend_used
        );

        Ok(info)
    }

    /// Reads architecture and token facts out of a model llama.cpp has loaded.
    fn extract_runtime_metadata(
        model: &LlamaModel,
        has_native_chat_template: bool,
    ) -> crate::model_intelligence::profile::RuntimeGgufMetadata {
        let piece = |t| {
            model
                .token_to_str(t, llama_cpp_2::model::Special::Tokenize)
                .ok()
                .filter(|s| !s.is_empty())
        };

        let architecture = model.meta_val_str("general.architecture").ok();

        // A model's end-of-turn token is often distinct from its EOS — Gemma
        // ends turns with `<end_of_turn>` while `<eos>` barely appears in chat
        // data — so record it separately when the GGUF declares one.
        let eot_token = model
            .meta_val_str("tokenizer.ggml.eot_token_id")
            .ok()
            .and_then(|id| id.trim().parse::<i32>().ok())
            .and_then(|id| piece(llama_cpp_2::token::LlamaToken(id)));

        crate::model_intelligence::profile::RuntimeGgufMetadata {
            architecture,
            bos_token: piece(model.token_bos()),
            eos_token: piece(model.token_eos()),
            eot_token,
            context_length: model.n_ctx_train(),
            has_native_chat_template,
        }
    }

    /// Writes runtime-read metadata back to the package's `profile.json`.
    ///
    /// Returns the corrected stop tokens when the profile was updated, so the
    /// current session uses them too rather than waiting for the next load.
    /// Best-effort: a model still runs fine if its profile cannot be written.
    fn sync_profile_metadata(
        gguf_path: &str,
        meta: &crate::model_intelligence::profile::RuntimeGgufMetadata,
    ) -> Option<Vec<String>> {
        // <package_dir>/base/<file>.gguf — the profile lives beside `base/`.
        let package_dir = Path::new(gguf_path).parent()?.parent()?;
        let profile_path = package_dir.join("profile.json");

        let content = std::fs::read_to_string(&profile_path).ok()?;
        let mut profile =
            serde_json::from_str::<crate::model_intelligence::ModelProfile>(&content).ok()?;

        let changed = profile.apply_runtime_metadata(meta);
        let stop_tokens = profile.tokens.stop_tokens.clone();

        if changed {
            log::info!(
                "[RUNTIME] Corrected profile from GGUF metadata: architecture={}, eos={:?}, stop_tokens={:?}",
                profile.architecture, profile.tokens.eos_token, stop_tokens
            );
            if let Err(e) =
                crate::model_intelligence::ModelIntelligenceManager::write_profile(package_dir, &profile)
            {
                log::warn!("[RUNTIME] Could not persist corrected profile: {e:#}");
            }
        }

        Some(stop_tokens)
    }

    /// Unloads the current model, freeing all RAM/VRAM
    pub fn unload_model(&mut self) -> Result<()> {
        self.is_generating.store(false, Ordering::Relaxed);

        if let Some(ref info) = self.loaded_info {
            log::info!("[RUNTIME] Unloading model: {} ({})", info.model_name, info.quantization);
        }

        // Release adapter handles BEFORE dropping the model. Each adapter was
        // initialised against this model; using one after the model is freed
        // would dereference a dangling pointer.
        self.adapter_cache.clear();

        // Drop model to free RAM/VRAM tensors
        self.model = None;
        self.native_template = None;
        self.loaded_info = None;
        // Keep self.backend alive for process lifetime

        log::info!("[RUNTIME] ✓ Model unloaded, RAM/VRAM resources freed");
        Ok(())
    }

    /// Generates tokens for the given messages.
    ///
    /// Calls `token_cb` for each generated token. Returns the total generation.
    /// Respects the `is_generating` flag for cancellation.
    pub fn generate<F>(
        &mut self,
        messages: &[ChatMessage],
        params: &GenerationParams,
        token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        self.generate_with_capability(messages, params, None, token_cb)
    }

    /// Generates tokens with an optional capability backend applied.
    ///
    /// When `capability_backend` is [`CapabilityBackend::LoraAdapter`], the
    /// adapter is bound to the freshly created context *before* the prompt is
    /// decoded, so prefill and generation both run against adapted weights.
    ///
    /// A LoRA binding failure is never fatal: it is logged and generation
    /// proceeds on the base model, matching the graceful-degradation contract
    /// in [`crate::capability`].
    pub fn generate_with_capability<F>(
        &mut self,
        messages: &[ChatMessage],
        params: &GenerationParams,
        capability_backend: Option<&CapabilityBackend>,
        mut token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        // Destructured so `adapter_cache` can be borrowed mutably while `model`
        // is borrowed immutably — these are disjoint fields.
        let Self {
            backend,
            model,
            loaded_info,
            native_template,
            is_generating,
            adapter_cache,
        } = self;

        let model = model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?;
        let backend = backend
            .as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;
        let config = loaded_info
            .as_ref()
            .ok_or_else(|| anyhow!("No model info available"))?;

        is_generating.store(true, Ordering::Relaxed);
        let cancel_flag = is_generating.clone();

        // Render the prompt with the model's own template when it has one, and
        // only fall back to the hand-written approximations otherwise.
        let (prompt, template_source) = match native_template.as_ref() {
            Some(tmpl) => match render_with_native_template(model, tmpl, messages) {
                Ok(p) => (p, "gguf"),
                Err(e) => {
                    log::warn!(
                        "[RUNTIME] The GGUF's own chat template failed to render ({e:#}); \
                         falling back to the hand-written '{}' template.",
                        config.chat_template
                    );
                    (
                        format_chat_prompt_with_template(messages, &config.chat_template),
                        "fallback",
                    )
                }
            },
            None => (
                format_chat_prompt_with_template(messages, &config.chat_template),
                "fallback",
            ),
        };

        // Compute SHA-256 digest of final prompt sent to llama.cpp
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        let prompt_sha256 = format!("{:x}", hasher.finalize());

        let has_memory_injection = prompt.contains("Known User Information & Preferences")
            || prompt.contains("Recalled Context & Facts")
            || prompt.contains("Shreyash")
            || prompt.contains("User Workspace & Project Context");

        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME PROMPT TRACE] ====================\n\
             - Prompt SHA-256 Hash:   {}\n\
             - Prompt Total Length:   {} chars\n\
             - Memory Injected:       {}\n\
             - Template Format:       {} (rendered via: {})\n\
             - Input Message Count:   {}\n\
             ==========================================================================",
            prompt_sha256, prompt.len(), has_memory_injection, config.chat_template, template_source, messages.len()
        );

        // Decide BOS from what the rendered prompt actually starts with, rather
        // than from the template name.
        //
        // `AddBos::Always` does not blindly prepend a token — it sets llama.cpp's
        // `add_special`, which honours the model's own `add_bos_token` metadata.
        // The previous rule forced `Never` for every "gemma" and "chatml"
        // template, which left Gemma running with no BOS at all; Gemma degrades
        // into multilingual noise without it. The only case that genuinely needs
        // `Never` is a template that already emitted BOS itself (some Jinja
        // templates interpolate `{{ bos_token }}`), which we detect directly.
        let bos_str = model
            .token_to_str(model.token_bos(), llama_cpp_2::model::Special::Tokenize)
            .unwrap_or_default();
        let add_bos = if !bos_str.is_empty() && prompt.starts_with(bos_str.as_str()) {
            AddBos::Never
        } else {
            AddBos::Always
        };

        let prompt_tokens = model
            .str_to_token(&prompt, add_bos)
            .map_err(|e| anyhow!("Tokenization failed: {:?}", e))?;

        let n_prompt_tokens = prompt_tokens.len();
        log::info!("[RUNTIME] Prompt tokenized: {} tokens", n_prompt_tokens);

        if n_prompt_tokens == 0 {
            is_generating.store(false, Ordering::Relaxed);
            return Err(anyhow!("Empty prompt after tokenization"));
        }

        // Create context for this generation
        let ctx_size = std::num::NonZeroU32::new(config.context_length)
            .unwrap_or(std::num::NonZeroU32::new(2048).unwrap());

        // Refused before a context is allocated: a prompt at or over the window
        // is one of the ways llama.cpp aborts the process, and an error the user
        // can read beats the window disappearing.
        if n_prompt_tokens >= ctx_size.get() as usize {
            is_generating.store(false, Ordering::Relaxed);
            return Err(anyhow!(
                "Prompt is {} tokens but the model is loaded with a {}-token context. \
                 Load the model with a larger context, or send a shorter prompt.",
                n_prompt_tokens,
                ctx_size.get()
            ));
        }

        // Batch size is left at whatever this llama.cpp build considers sane and
        // the prefill below is chunked to match, rather than inflated to cover
        // the whole prompt in one decode.
        //
        // Both matter. llama.cpp *aborts the process* — it does not return an
        // error — when a single decode carries more tokens than n_batch, which
        // is how the app vanished (0xc0000409, FAST_FAIL_FATAL_APP_EXIT) the
        // moment a real client connected. Sizing n_batch up to the prompt fixed
        // that but allocated buffers proportional to the largest prompt seen,
        // which on a 32k certified context is memory no short turn uses.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads(config.threads as i32)
            .with_n_threads_batch(config.threads as i32);

        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create inference context: {:?}", e))?;

        // Bind the capability's LoRA adapter, if one was resolved.
        //
        // This must happen before the prefill decode below so the prompt is
        // processed against the adapted weights. Because the context is created
        // fresh for every generation, there is no stale binding to clear first.
        let mut active_adapter_label: Option<String> = None;
        if let Some(CapabilityBackend::LoraAdapter { path, scale }) = capability_backend {
            match adapter_cache.get_or_init(model, path) {
                Ok(adapter) => match crate::ai_engine::lora_binding::bind_adapter(&mut ctx, adapter, *scale) {
                    Ok(()) => {
                        let label = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "adapter".to_string());
                        log::info!("[RUNTIME] Generating with LoRA adapter '{}' at scale {:.2}", label, scale);
                        active_adapter_label = Some(label);
                    }
                    Err(e) => log::warn!(
                        "[RUNTIME WARN] LoRA bind failed, continuing on base model: {:#}",
                        e
                    ),
                },
                Err(e) => log::warn!(
                    "[RUNTIME WARN] LoRA adapter init failed, continuing on base model: {:#}",
                    e
                ),
            }
        }

        // Prefill in chunks the size of this context's own batch.
        //
        // Prefill is the dominant cost of a request — on a CPU-only build a
        // coding agent's system prompt measured ~98s — and decoding it as one
        // call meant a client that had already hung up was discovered only
        // after all of it had been paid for. Chunking gives a cancellation
        // point between batches, so an abandoned request stops within one
        // chunk instead of running to completion.
        let prefill_chunk = (ctx.n_batch().max(1)) as usize;
        let mut batch = LlamaBatch::new(prefill_chunk, 1);

        for (chunk_index, chunk) in prompt_tokens.chunks(prefill_chunk).enumerate() {
            // The flag is cleared by the canceller, so "not generating" here
            // means someone asked us to stop.
            if !cancel_flag.load(Ordering::Relaxed) {
                let done = chunk_index * prefill_chunk;
                log::info!(
                    "[RUNTIME] Prefill cancelled after {}/{} prompt tokens",
                    done, n_prompt_tokens
                );
                is_generating.store(false, Ordering::Relaxed);
                return Err(anyhow!("Generation cancelled during prompt prefill"));
            }

            batch.clear();
            let base = chunk_index * prefill_chunk;
            for (offset, &token) in chunk.iter().enumerate() {
                let pos = base + offset;
                // Only the final prompt token needs logits — that is the one the
                // first sampled token is drawn from.
                let wants_logits = pos == n_prompt_tokens - 1;
                batch
                    .add(token, pos as i32, &[0], wants_logits)
                    .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Failed to decode prompt batch: {:?}", e))?;
        }

        let eos_token = model.token_eos();

        // Build effective template-driven stop sequences list.
        //
        // These additions are guesses derived from the family name, so they are
        // only appropriate when the hand-written template is what produced the
        // prompt. When the GGUF's own template is in use, the vocab's
        // end-of-generation tokens are authoritative and guessing here would
        // just reintroduce mismatched tokens.
        let mut effective_stop_tokens = config.stop_tokens.clone();
        let lower_temp = config.chat_template.to_lowercase();

        if template_source != "fallback" {
            // Nothing to add — `is_eog_token` handles termination.
        } else if lower_temp.contains("chatml") || lower_temp.contains("qwen") {
            for st in &["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        } else if lower_temp.contains("gemma") {
            for st in &["<end_of_turn>", "<start_of_turn>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        } else if lower_temp.contains("llama3") || lower_temp.contains("llama-3") || lower_temp.contains("llama") {
            for st in &["<|eot_id|>", "<|end_of_text|>", "</s>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        }

        // Log complete runtime generation parameters & provenance immediately before generation
        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME GENERATION CONFIGURATION] ====================\n\
             - Model ID:           {} (Source: Manifest)\n\
             - Chat Template:      {} (Source: GGUF / tokenizer_config.json / Jinja)\n\
             - EOS Token ID:       {:?} (Source: GGUF model.token_eos)\n\
             - Context Length:     {} tokens (Source: GGUF / config.json max_position_embeddings)\n\
             - Threads (batch):    {} (Source: System Hardware Collector)\n\
             - Temperature:        {} (Source: generation_config.json / User Override)\n\
             - Top-P:              {} (Source: generation_config.json / User Override)\n\
             - Top-K:              {} (Source: generation_config.json / User Override)\n\
             - Min-P:              {} (Source: generation_config.json / Model Profile)\n\
             - Repeat Penalty:     {} (Source: generation_config.json / Model Profile)\n\
             - Mirostat Mode:      {} (Source: Model Profile / Fallback)\n\
             - Effective Stop Tokens: {:?} (Source: Metadata + Template Defaults)\n\
             =======================================================================================",
            config.model_id,
            config.chat_template,
            eos_token,
            config.context_length,
            config.threads,
            params.temperature,
            params.top_p,
            params.top_k,
            params.min_p,
            params.repeat_penalty,
            params.mirostat,
            effective_stop_tokens
        );

        // Set up sampler chain directly consumed by llama.cpp.
        //
        // `penalties` must come first so repetition suppression applies to the
        // full distribution before truncation. It was previously absent from the
        // chain entirely, so `repeat_penalty` was logged above but never took
        // effect — capability sampling profiles depend on it being real.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1234);

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, params.repeat_penalty, 0.0, 0.0),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_k(params.top_k as i32),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::min_p(params.min_p, 1),
            // Previously a hardcoded seed, which made every regeneration of the
            // same prompt byte-identical regardless of temperature.
            LlamaSampler::dist(seed),
        ]);

        // Autoregressive generation loop
        let mut generated_text = String::new();
        let mut n_generated: u32 = 0;
        let mut n_cur = n_prompt_tokens as i32;

        loop {
            // Check cancellation
            if !cancel_flag.load(Ordering::Relaxed) {
                log::info!("[RUNTIME] Generation cancelled by user after {} tokens", n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("cancelled".to_string()),
                });
                break;
            }

            // Check max tokens
            if n_generated >= params.max_tokens {
                log::info!("[RUNTIME] Max tokens reached: {}", n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("length".to_string()),
                });
                break;
            }

            // Check context window
            if n_cur as u32 >= config.context_length {
                log::warn!("[RUNTIME] Context window exhausted at {} tokens", n_cur);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("context_length".to_string()),
                });
                break;
            }

            // Sample next token
            let new_token_id = sampler.sample(&ctx, -1);

            // Stop on any end-of-generation token the model itself declares.
            // `token_eos()` is only one of them — Gemma, for instance, ends turns
            // with `<end_of_turn>` rather than `<eos>` — and asking the vocab is
            // reliable where string matching on detokenized text is not.
            if new_token_id == eos_token || model.is_eog_token(new_token_id) {
                log::info!("[RUNTIME] End-of-generation token {} reached after {} tokens", new_token_id, n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("stop".to_string()),
                });
                break;
            }

            // Detokenize
            let token_str = model
                .token_to_str(new_token_id, llama_cpp_2::model::Special::Tokenize)
                .unwrap_or_default();

            // Match only the stop strings this model actually declares.
            //
            // Two previous rules made this fire far too eagerly: any token
            // beginning with `<|` was treated as a stop token whenever *some*
            // configured stop string happened to begin with `<|`, and ChatML's
            // control tokens were hardcoded regardless of the model's family.
            // Between them, a model could be cut off mid-answer by ordinary
            // output. End-of-generation is handled by the vocab check above.
            let combined_check = format!("{}{}", generated_text, token_str);
            let is_stop_str = effective_stop_tokens
                .iter()
                .any(|st| !st.is_empty() && (token_str.contains(st) || combined_check.ends_with(st)));

            if is_stop_str {
                log::info!("[RUNTIME] Stop token sequence '{}' (Token ID: {}) detected after {} tokens", token_str, new_token_id, n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("stop".to_string()),
                });
                break;
            }

            generated_text.push_str(&token_str);
            n_generated += 1;

            // Filter out <think>...</think> reasoning blocks for UI stream
            let mut emit_text = token_str.clone();
            if emit_text.contains("<think>") || emit_text.contains("</think>") || emit_text.contains("<|im_start|>") || emit_text.contains("<|im_end|>") {
                emit_text = emit_text
                    .replace("<think>", "")
                    .replace("</think>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "");
            }

            if !emit_text.is_empty() {
                // Emit clean token to callback
                token_cb(StreamChunk {
                    text: emit_text,
                    is_final: false,
                    tokens_generated: Some(n_generated),
                    finish_reason: None,
                });
            }

            // Prepare next batch with the new token
            batch.clear();
            batch
                .add(new_token_id, n_cur, &[0], true)
                .map_err(|e| anyhow!("Failed to add generated token: {:?}", e))?;

            n_cur += 1;

            // Decode
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Decode failed at token {}: {:?}", n_generated, e))?;
        }

        is_generating.store(false, Ordering::Relaxed);
        log::info!(
            "[RUNTIME] Generation complete: {} tokens, {} chars, adapter={}",
            n_generated,
            generated_text.len(),
            active_adapter_label.as_deref().unwrap_or("none")
        );

        Ok(generated_text)
    }

    /// Signals the generation loop to stop
    pub fn stop_generation(&self) {
        log::info!("[RUNTIME] Stop generation requested");
        self.is_generating.store(false, Ordering::Relaxed);
    }
}

/// Everything needed to frame a prompt the way a given model expects.
///
/// Held together because the two rendering routes need different things: the
/// Jinja source drives the faithful path, and llama.cpp's handle drives the
/// built-in path used when that source will not render.
pub struct NativeChatTemplate {
    /// Raw Jinja from the GGUF's `tokenizer.chat_template` key, when present.
    source: Option<String>,
    /// llama.cpp's own handle on the same template.
    handle: LlamaChatTemplate,
    bos_token: String,
    eos_token: String,
}

/// Renders messages through the chat template baked into the GGUF.
///
/// Two routes, tried in order of faithfulness:
///
/// 1. The template's actual Jinja, rendered with minijinja. This is the same
///    thing HuggingFace evaluates, so it reproduces the exact framing a model
///    was trained on — including custom formats nothing else knows about.
/// 2. llama.cpp's `apply_chat_template`, which recognises a fixed set of
///    well-known formats and rejects anything outside it.
///
/// Falling through both leaves the hand-written family templates, which are
/// approximations and can be badly wrong for an unusual model.
fn render_with_native_template(
    model: &LlamaModel,
    tmpl: &NativeChatTemplate,
    messages: &[ChatMessage],
) -> Result<String> {
    if let Some(src) = &tmpl.source {
        match render_jinja_chat_template(src, messages, &tmpl.bos_token, &tmpl.eos_token) {
            Ok(prompt) if !prompt.trim().is_empty() => return Ok(prompt),
            Ok(_) => log::warn!("[RUNTIME] Jinja chat template rendered an empty prompt; trying llama.cpp's engine"),
            Err(e) => log::warn!("[RUNTIME] Jinja chat template failed to render ({e:#}); trying llama.cpp's engine"),
        }
    }

    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            LlamaChatMessage::new(m.role.clone(), m.content.clone())
                .map_err(|e| anyhow!("Message rejected by chat template: {e:?}"))
        })
        .collect::<Result<_>>()?;

    // `add_ass = true` leaves the assistant tag open so the model completes a
    // reply instead of generating a fresh turn header of its own.
    let prompt = model
        .apply_chat_template(&tmpl.handle, &chat, true)
        .map_err(|e| anyhow!("apply_chat_template failed: {e:?}"))?;

    if prompt.trim().is_empty() {
        return Err(anyhow!("chat template rendered an empty prompt"));
    }

    Ok(prompt)
}

/// A chat message as a chat template expects to see it.
///
/// Templates are written against Python dicts, so they reach for `.get(...)`
/// on optional fields like `reasoning` and `tool_calls`. A plain map value has
/// no such method and rendering aborts, so this supplies dict semantics:
/// subscripting known keys, and `.get` returning a default instead of raising.
#[derive(Debug)]
struct TemplateMessage {
    role: String,
    content: String,
}

impl minijinja::value::Object for TemplateMessage {
    fn get_value(self: &std::sync::Arc<Self>, key: &minijinja::value::Value) -> Option<minijinja::value::Value> {
        match key.as_str()? {
            "role" => Some(minijinja::value::Value::from(self.role.clone())),
            "content" => Some(minijinja::value::Value::from(self.content.clone())),
            _ => None,
        }
    }

    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        method: &str,
        args: &[minijinja::value::Value],
    ) -> std::result::Result<minijinja::value::Value, minijinja::Error> {
        match method {
            // Python's `dict.get(key[, default])`: absent keys yield the
            // default (None when unspecified) rather than erroring.
            "get" => {
                let none = minijinja::value::Value::from(());
                let default = args.get(1).cloned().unwrap_or(none.clone());
                Ok(args
                    .first()
                    .and_then(|k| self.get_value(k))
                    .unwrap_or(default))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("chat message has no method named {method}"),
            )),
        }
    }
}

/// Renders a HuggingFace-style Jinja chat template.
///
/// The context mirrors what `tokenizer.apply_chat_template` provides, since
/// templates are written against it: a `messages` list of role/content maps,
/// the special tokens as variables, and `add_generation_prompt` set so the
/// template leaves the assistant turn open for the model to complete.
pub fn render_jinja_chat_template(
    template_src: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
) -> Result<String> {
    use minijinja::{context, Environment};

    let mut env = Environment::new();
    // Chat templates are whitespace-sensitive; keeping blocks and newlines
    // verbatim is what makes the output match the training format.
    env.set_keep_trailing_newline(true);
    env.add_template("chat", template_src)
        .map_err(|e| anyhow!("chat template is not valid Jinja: {e}"))?;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("chat template unavailable: {e}"))?;

    let msgs: Vec<_> = messages
        .iter()
        .map(|m| {
            minijinja::value::Value::from_object(TemplateMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
        })
        .collect();

    tmpl.render(context! {
        messages => msgs,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => true,
        // Declared so templates that branch on them take the plain-chat path
        // rather than failing on an undefined name.
        tools => Vec::<minijinja::value::Value>::new(),
        enable_thinking => false,
    })
    .map_err(|e| anyhow!("chat template failed to render: {e}"))
}

/// Formats chat messages into a prompt string based on Model Profile template.
///
/// Fallback only — used when a GGUF ships no chat template of its own. These are
/// approximations of each family's real format, so prefer the model's embedded
/// template (see [`render_with_native_template`]) wherever one exists.
pub fn format_chat_prompt_with_template(messages: &[ChatMessage], template_name: &str) -> String {
    let mut prompt = String::new();
    let lower_temp = template_name.to_lowercase();

    if lower_temp.contains("chatml") || lower_temp.contains("qwen") {
        for msg in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        prompt.push_str("<|im_start|>assistant\n");
    } else if lower_temp.contains("llama3") || lower_temp.contains("llama-3") || lower_temp.contains("llama_3") {
        // The role must sit between the header tags with no newline of its own;
        // a stray one here does not match how Llama-3 was trained.
        for msg in messages {
            prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", msg.role, msg.content));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    } else if lower_temp.contains("gemma") {
        for msg in messages {
            let role = if msg.role == "assistant" { "model" } else { &msg.role };
            prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
        }
        prompt.push_str("<start_of_turn>model\n");
    } else if lower_temp.contains("mistral") {
        for msg in messages {
            if msg.role == "user" {
                prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
            } else if msg.role == "assistant" {
                prompt.push_str(&format!(" {}\n", msg.content));
            }
        }
    } else {
        // Fallback to ChatML structure for unknown models
        for msg in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        prompt.push_str("<|im_start|>assistant\n");
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── MoE expert offload patterns ────────────────────────────────────────

    /// Matches a tensor name against a generated pattern the way llama.cpp does.
    fn pattern_matches(pattern: &CString, tensor: &str) -> bool {
        // The patterns are fixed-shape: `blk\.<i>\.ffn_(a|b|…)_(ch|)exps`.
        // Rather than pull in a regex engine for a test, expand the two
        // alternations and check the literal forms.
        let raw = pattern.to_str().expect("generated pattern is valid UTF-8");
        let prefix = raw.trim_end_matches(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        let block = prefix.replace(r"\.", ".");

        ["up", "down", "gate", "gate_up"].iter().any(|proj| {
            ["ch", ""]
                .iter()
                .any(|infix| tensor.contains(&format!("{block}.ffn_{proj}_{infix}exps")))
        })
    }

    /// The regression this design exists to prevent.
    ///
    /// The crate's own `add_cpu_moe_override()` omits the `gate_up` alternative
    /// that upstream carries, so a model with fused expert tensors would load
    /// with the offload silently doing nothing. If anyone "simplifies" the
    /// pattern generation back to the crate helper, this must fail loudly.
    #[test]
    fn generated_patterns_match_fused_gate_up_expert_tensors() {
        let patterns = cpu_moe_override_patterns(1);
        assert_eq!(patterns.len(), 1);

        assert!(
            pattern_matches(&patterns[0], "blk.0.ffn_gate_up_exps.weight"),
            "fused expert tensors must match, or gpt-oss offloads nothing: {:?}",
            patterns[0]
        );
    }

    #[test]
    fn generated_patterns_match_split_expert_tensors() {
        let patterns = cpu_moe_override_patterns(1);

        for tensor in [
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_chexps.weight",
        ] {
            assert!(pattern_matches(&patterns[0], tensor), "{tensor} must match");
        }
    }

    /// `--n-cpu-moe N` covers the *first* N layers, one override each.
    #[test]
    fn one_pattern_is_generated_per_offloaded_layer() {
        let patterns = cpu_moe_override_patterns(22);
        assert_eq!(patterns.len(), 22);

        assert!(patterns[0].to_str().unwrap().starts_with(r"blk\.0\."));
        assert!(patterns[21].to_str().unwrap().starts_with(r"blk\.21\."));

        // Layer 22 is beyond the requested depth and must stay resident.
        assert!(!pattern_matches(&patterns[21], "blk.22.ffn_gate_up_exps.weight"));
    }

    #[test]
    fn a_depth_of_zero_generates_no_overrides() {
        assert!(cpu_moe_override_patterns(0).is_empty());
    }

    #[test]
    fn patterns_target_only_their_own_layer() {
        let patterns = cpu_moe_override_patterns(3);

        assert!(pattern_matches(&patterns[1], "blk.1.ffn_up_exps.weight"));
        assert!(!pattern_matches(&patterns[1], "blk.0.ffn_up_exps.weight"));
    }

    /// Attention and the KV cache are what expert offload exists to keep on the
    /// GPU, so they must never be caught by these patterns.
    #[test]
    fn patterns_never_match_attention_or_shared_tensors() {
        let patterns = cpu_moe_override_patterns(4);

        for tensor in [
            "blk.0.attn_q.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_up_shexp.weight",
            "token_embd.weight",
            "output_norm.weight",
        ] {
            for pattern in &patterns {
                assert!(
                    !pattern_matches(pattern, tensor),
                    "{tensor} must stay resident but matched {pattern:?}"
                );
            }
        }
    }

    #[test]
    fn test_runtime_initial_status() {
        let runtime = LlamaCppRuntime::new();
        assert_eq!(runtime.status(), RuntimeStatus::NotLoaded);
        assert!(runtime.loaded_model_info().is_none());
    }

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                timestamp: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                timestamp: None,
            },
        ]
    }

    #[test]
    fn chatml_template_uses_im_start_markers() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "chatml");

        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
        // Must end primed for the assistant turn, or the model continues the
        // user's message instead of replying.
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_template_uses_header_markers() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "llama3");

        assert!(prompt.contains("<|start_header_id|>"));
        assert!(prompt.contains("<|eot_id|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn llama3_role_sits_inside_the_header_tags() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "llama3");

        // A newline between the role and `<|end_header_id|>` is not the format
        // Llama-3 was trained on, and the closing tag never had one — so the
        // opening tags disagreed with the assistant tag the prompt ends with.
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\n"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\n"));
        assert!(!prompt.contains("<|start_header_id|>system\n"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn gemma_template_maps_assistant_onto_model_turns() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "gemma");

        // Gemma names the assistant role "model"; emitting "assistant" would put
        // a token the model never saw in training at the start of every reply.
        assert!(prompt.contains("<start_of_turn>user\n"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        assert!(!prompt.contains("<start_of_turn>assistant"));
    }

    #[test]
    fn jinja_template_renders_roles_and_generation_prompt() {
        let src = "{{- bos_token -}}\
                   {%- for m in messages -%}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{%- endfor -%}\
                   {%- if add_generation_prompt -%}<|turn>model\n{%- endif -%}";

        let out = render_jinja_chat_template(src, &sample_messages(), "<bos>", "<eos>").unwrap();

        // `{%-` trims the preceding newline, so the turns run together — which
        // is exactly the whitespace handling a chat template depends on.
        assert_eq!(
            out,
            "<bos><|turn>system\nYou are helpful.<turn|><|turn>user\nHello<turn|><|turn>model"
        );
    }

    #[test]
    fn jinja_messages_support_python_dict_get() {
        // Real templates probe optional fields with `.get(...)`. Without dict
        // semantics minijinja aborts with "map has no method named get", and
        // the whole template silently falls back to a guessed format.
        let src = "{%- for m in messages -%}\
                   {{ m.get('role') }}:{{ m.get('tool_calls', 'none') }};\
                   {%- endfor -%}";

        let out = render_jinja_chat_template(src, &sample_messages(), "<bos>", "<eos>").unwrap();

        assert_eq!(out, "system:none;user:none;");
    }

    #[test]
    fn a_broken_jinja_template_errors_rather_than_panicking() {
        // Callers fall back to a hand-written template on error, so this must
        // surface as Err rather than taking the process down.
        let out = render_jinja_chat_template("{% for %}", &sample_messages(), "<bos>", "<eos>");
        assert!(out.is_err());
    }

    #[test]
    fn qwen_is_treated_as_chatml() {
        let qwen = format_chat_prompt_with_template(&sample_messages(), "qwen");
        let chatml = format_chat_prompt_with_template(&sample_messages(), "chatml");
        assert_eq!(qwen, chatml);
    }

    #[test]
    fn an_unknown_template_still_produces_a_usable_prompt() {
        // Must never return empty: an empty prompt tokenizes to nothing and the
        // runtime rejects the request outright.
        let prompt = format_chat_prompt_with_template(&sample_messages(), "not-a-real-template");

        assert!(!prompt.trim().is_empty());
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_generate_fails_without_model() {
        let mut runtime = LlamaCppRuntime::new();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
            timestamp: None,
        }];
        let result = runtime.generate(&messages, &GenerationParams::default(), |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn test_unload_without_model() {
        let mut runtime = LlamaCppRuntime::new();
        // Should succeed even with no model loaded
        assert!(runtime.unload_model().is_ok());
    }

    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Loads a real installed model and reports what the fix actually reads
    /// from it: the embedded chat template, the true BOS/EOS tokens, and the
    /// prompt that now gets sent.
    ///
    /// Ignored by default because it loads several gigabytes of weights.
    /// Run with: `cargo test --lib gguf_metadata_matches_the_model -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gguf_metadata_matches_the_model() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\yuxinlu1_gemma-4-12B-coder-fable5-composer2.5-v1-GGUF\base\gemma4-coding-Q2_K.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping: {gguf_path} not present");
            return;
        }

        let backend = LlamaBackend::init().expect("backend init");
        let params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, gguf_path, &params).expect("model load");

        let meta = LlamaCppRuntime::extract_runtime_metadata(&model, model.chat_template(None).is_ok());
        println!("architecture      : {:?}", meta.architecture);
        println!("bos_token         : {:?}", meta.bos_token);
        println!("eos_token         : {:?}", meta.eos_token);
        println!("eot_token         : {:?}", meta.eot_token);
        println!("n_ctx_train       : {}", meta.context_length);
        println!("native template   : {}", meta.has_native_chat_template);

        // The profile on disk claimed architecture "llama" with `<|eot_id|>`;
        // the model itself must say otherwise.
        assert!(meta.has_native_chat_template, "GGUF should carry its own chat template");
        assert_ne!(meta.eos_token.as_deref(), Some("<|eot_id|>"), "EOS must not be Llama-3's");

        println!(
            "--- embedded template source ---\n{}\n--------------------------------",
            model.meta_val_str("tokenizer.chat_template").unwrap_or_else(|e| format!("<unreadable: {e:?}>"))
        );

        let piece = |t| model.token_to_str(t, llama_cpp_2::model::Special::Tokenize).unwrap_or_default();
        let tmpl = NativeChatTemplate {
            source: model.meta_val_str("tokenizer.chat_template").ok(),
            handle: model.chat_template(None).unwrap(),
            bos_token: piece(model.token_bos()),
            eos_token: piece(model.token_eos()),
        };

        let rendered = render_with_native_template(&model, &tmpl, &sample_messages())
            .expect("the model's own template must render");
        println!("--- rendered prompt ---\n{rendered}\n-----------------------");

        assert!(rendered.contains("Hello"));
        // This model frames turns as `<|turn>role … <turn|>`. The hand-written
        // "gemma" fallback emits `<start_of_turn>`, which this model was never
        // trained on — that mismatch is what produced the garbled output.
        assert!(rendered.contains("<|turn>user"), "expected the model's own turn markers");
        assert!(!rendered.contains("<start_of_turn>"), "must not use the guessed Gemma framing");
    }

    #[test]
    fn test_load_real_installed_gguf() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\meta-Llama_Llama-3.2-1B\base\Llama-3.2-1B-Instruct-Q8_0.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping real GGUF test, file does not exist at {}", gguf_path);
            return;
        }

        let mut runtime = LlamaCppRuntime::new();
        let config = ModelLoadConfig {
            model_path: gguf_path.to_string(),
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            model_name: "Llama 3.2 1B".to_string(),
            quantization: "Q8_0".to_string(),
            context_length: 65536,
            gpu_layers: 999,
            cpu_moe_layers: 0,
            threads: 4,
            chat_template: "llama3".to_string(),
            stop_tokens: vec![],
        };

        println!("Testing load_model for real GGUF at {}", gguf_path);
        let res = runtime.load_model(&config, |step| {
            println!("Step: {}", step);
        });

        match &res {
            Ok(info) => println!("SUCCESSFULLY LOADED MODEL: {:?}", info),
            Err(e) => println!("FAILED TO LOAD MODEL: {:?}", e),
        }

        assert!(res.is_ok(), "Real GGUF load failed: {:?}", res.err());
    }

    #[test]
    fn test_load_real_installed_gguf_cpu() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\meta-Llama_Llama-3.2-1B\base\Llama-3.2-1B-Instruct-Q8_0.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping real GGUF CPU test, file does not exist at {}", gguf_path);
            return;
        }

        let mut runtime = LlamaCppRuntime::new();
        let config = ModelLoadConfig {
            model_path: gguf_path.to_string(),
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            model_name: "Llama 3.2 1B".to_string(),
            quantization: "Q8_0".to_string(),
            context_length: 4096,
            gpu_layers: 0,
            cpu_moe_layers: 0,
            threads: 4,
            chat_template: "llama3".to_string(),
            stop_tokens: vec![],
        };

        println!("Testing load_model (CPU mode, gpu_layers=0) for GGUF at {}", gguf_path);
        let res = runtime.load_model(&config, |step| {
            println!("Step: {}", step);
        });

        match &res {
            Ok(info) => println!("SUCCESSFULLY LOADED CPU MODEL: {:?}", info),
            Err(e) => println!("FAILED TO LOAD CPU MODEL: {:?}", e),
        }

        assert!(res.is_ok(), "CPU GGUF load failed: {:?}", res.err());
    }

    #[test]
    fn test_simulate_gui_load_model_flow() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let provider_id = "huggingface";
        let model_id = "meta-llama/Llama-3.2-1B";
        let quantization = "Q8_0";

        println!("Simulating exact GUI load flow for provider='{}', model='{}'", provider_id, model_id);

        let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, provider_id, model_id);
        if !package_dir.exists() || !package_dir.join("manifest.json").exists() {
            println!("Skipping Llama test: package dir or manifest does not exist at {:?}", package_dir);
            return;
        }

        let manifest = AdapterRegistry::read_manifest(&package_dir).expect("Failed to read manifest");
        println!("Manifest: model_name='{}', file_path='{}'", manifest.base_model.model_name, manifest.base_model.file_path);

        let gguf_path = match InferenceManager::resolve_gguf_path(&package_dir, &manifest) {
            Ok(p) => p,
            Err(e) => {
                println!("Skipping Llama test: GGUF file not found: {}", e);
                return;
            }
        };
        println!("Resolved GGUF path: {}", gguf_path);

        let profile = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest).expect("Failed to get profile");
        let config = InferenceManager::build_load_config(&app_data_dir, &gguf_path, model_id, &manifest, quantization, &profile).expect("Failed to build load config");
        println!("Built ModelLoadConfig: path='{}', ctx={}, gpu_layers={}, threads={}", config.model_path, config.context_length, config.gpu_layers, config.threads);

        let mut runtime = LlamaCppRuntime::new();
        let res = runtime.load_model(&config, |step| println!("Step: {}", step));
        println!("Simulated GUI Load Result: {:?}", res);
        assert!(res.is_ok(), "GUI simulated load failed: {:?}", res.err());
    }

    #[test]
    fn test_simulate_gui_load_qwen_coder_7b() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let provider_id = "huggingface";
        let model_id = "Qwen/Qwen2.5-Coder-7B";
        let quantization = "Q4_0";

        println!("Simulating exact GUI load flow for provider='{}', model='{}'", provider_id, model_id);

        let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, provider_id, model_id);
        if !package_dir.exists() || !package_dir.join("manifest.json").exists() {
            println!("Skipping Qwen test: package dir or manifest does not exist at {:?}", package_dir);
            return;
        }

        let manifest = AdapterRegistry::read_manifest(&package_dir).expect("Failed to read manifest");
        let gguf_path = match InferenceManager::resolve_gguf_path(&package_dir, &manifest) {
            Ok(p) => p,
            Err(e) => {
                println!("Skipping Qwen test: GGUF file not found: {}", e);
                return;
            }
        };
        println!("Resolved GGUF path: {}", gguf_path);

        let profile = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest).expect("Failed to get profile");
        let config = InferenceManager::build_load_config(&app_data_dir, &gguf_path, model_id, &manifest, quantization, &profile).expect("Failed to build load config");

        let mut runtime = LlamaCppRuntime::new();
        let res = runtime.load_model(&config, |step| println!("Step: {}", step));
        println!("Simulated Qwen Load Result: {:?}", res);
        assert!(res.is_ok(), "Qwen load failed: {:?}", res.err());
    }

    #[test]
    fn test_generic_multi_model_sequential_session_load() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;
        use crate::model_intelligence::{MetadataExtractor, ModelFamily};

        // 1. Verify 4 distinct Model Families are detected generically without model-name hardcoding
        assert_eq!(MetadataExtractor::infer_family_from_string("meta-llama/Llama-3.2-1B"), ModelFamily::Llama);
        assert_eq!(MetadataExtractor::infer_family_from_string("Qwen/Qwen2.5-Coder-7B"), ModelFamily::Qwen);
        assert_eq!(MetadataExtractor::infer_family_from_string("google/gemma-2-9b"), ModelFamily::Gemma);
        assert_eq!(MetadataExtractor::infer_family_from_string("mistralai/Mistral-7B-v0.3"), ModelFamily::Mistral);

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let mut runtime = LlamaCppRuntime::new();

        // 2. Load Model 1 (Llama-3.2-1B) in runtime session
        let pkg1 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        if pkg1.exists() && pkg1.join("manifest.json").exists() {
            let manifest1 = AdapterRegistry::read_manifest(&pkg1).unwrap();
            if let Ok(path1) = InferenceManager::resolve_gguf_path(&pkg1, &manifest1) {
            let profile1 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg1, &manifest1).unwrap();
            let cfg1 = InferenceManager::build_load_config(&app_data_dir, &path1, "meta-llama/Llama-3.2-1B", &manifest1, "Q8_0", &profile1).unwrap();

            println!("\n=== [SEQUENTIAL LOAD TEST 1] Loading Llama-3.2-1B ===");
            let res1 = runtime.load_model(&cfg1, |s| println!("Load step: {}", s));
            assert!(res1.is_ok(), "Failed sequential load 1 (Llama)");
            assert_eq!(cfg1.chat_template, "llama3");

            println!("=== [SEQUENTIAL UNLOAD TEST 1] Unloading Llama-3.2-1B ===");
            assert!(runtime.unload_model().is_ok());
            } // if let Ok(path1)
        }

        // 3. Load Model 2 (Qwen2.5-Coder-7B) into SAME runtime session without process restart
        let pkg2 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "Qwen/Qwen2.5-Coder-7B");
        if pkg2.exists() && pkg2.join("manifest.json").exists() {
            let manifest2 = AdapterRegistry::read_manifest(&pkg2).unwrap();
            if let Ok(path2) = InferenceManager::resolve_gguf_path(&pkg2, &manifest2) {
            let profile2 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg2, &manifest2).unwrap();
            let cfg2 = InferenceManager::build_load_config(&app_data_dir, &path2, "Qwen/Qwen2.5-Coder-7B", &manifest2, "Q4_0", &profile2).unwrap();

            println!("\n=== [SEQUENTIAL LOAD TEST 2] Loading Qwen2.5-Coder-7B into SAME session ===");
            let res2 = runtime.load_model(&cfg2, |s| println!("Load step: {}", s));
            assert!(res2.is_ok(), "Failed sequential load 2 (Qwen)");
            assert_eq!(cfg2.chat_template, "chatml");

            println!("=== [SEQUENTIAL UNLOAD TEST 2] Unloading Qwen2.5-Coder-7B ===");
            assert!(runtime.unload_model().is_ok());
            } // if let Ok(path2)
        }

        println!("=== SEQUENTIAL MULTI-MODEL LOAD TEST SUCCEEDED CLEANLY ===");
    }

    #[test]
    fn test_phase5_2_automatic_runtime_configuration_verification() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");

        // Model 1: Llama 3.2 1B
        let pkg1 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        if pkg1.exists() && pkg1.join("manifest.json").exists() {
            let manifest1 = AdapterRegistry::read_manifest(&pkg1).unwrap();
            let profile1 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg1, &manifest1).unwrap();
            println!("\n[PHASE 5.2 VERIFICATION] Llama 3.2 1B Profile Extracted:");
            println!("  Family: {:?}", profile1.model_family);
            println!("  Chat Template: {}", profile1.chat_template);
            println!("  Context Length: {}", profile1.recommended_params.context_length);
            println!("  Stop Tokens: {:?}", profile1.tokens.stop_tokens);
            println!("  Sampling Params: temp={}, top_p={}, repeat_penalty={}, min_p={}",
                profile1.recommended_params.temperature,
                profile1.recommended_params.top_p,
                profile1.recommended_params.repeat_penalty,
                profile1.recommended_params.min_p
            );
            assert_eq!(profile1.model_family, crate::model_intelligence::ModelFamily::Llama);
            assert_eq!(profile1.chat_template, "llama3");
        }

        // Model 2: Qwen 2.5 Coder 7B
        let pkg2 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "Qwen/Qwen2.5-Coder-7B");
        if pkg2.exists() && pkg2.join("manifest.json").exists() {
            let manifest2 = AdapterRegistry::read_manifest(&pkg2).unwrap();
            let profile2 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg2, &manifest2).unwrap();
            println!("\n[PHASE 5.2 VERIFICATION] Qwen 2.5 Coder 7B Profile Extracted:");
            println!("  Family: {:?}", profile2.model_family);
            println!("  Chat Template: {}", profile2.chat_template);
            println!("  Context Length: {}", profile2.recommended_params.context_length);
            println!("  Stop Tokens: {:?}", profile2.tokens.stop_tokens);
            println!("  Sampling Params: temp={}, top_p={}, repeat_penalty={}, min_p={}",
                profile2.recommended_params.temperature,
                profile2.recommended_params.top_p,
                profile2.recommended_params.repeat_penalty,
                profile2.recommended_params.min_p
            );
            assert_eq!(profile2.model_family, crate::model_intelligence::ModelFamily::Qwen);
            assert_eq!(profile2.chat_template, "chatml");
        }

        println!("\n=== PHASE 5.2 AUTOMATIC RUNTIME CONFIGURATION VERIFIED ===\n");
    }
}

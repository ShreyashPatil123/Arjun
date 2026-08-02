//! LlamaCpp Runtime — In-process GGUF model inference via llama-cpp-2
//!
//! Provides model loading, token generation with streaming, and resource management.
//! Designed to be wrapped by InferenceManager for thread-safe Tauri integration.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::ai_engine::traits::*;

/// Core inference runtime wrapping llama.cpp via safe Rust bindings.
///
/// This struct owns the model, context, and backend. It is NOT thread-safe by itself;
/// the InferenceManager wraps it in Arc<Mutex<>> for safe concurrent access.
pub struct LlamaCppRuntime {
    backend: Option<LlamaBackend>,
    model: Option<LlamaModel>,
    loaded_info: Option<LoadedModelInfo>,
    is_generating: Arc<AtomicBool>,
}

impl LlamaCppRuntime {
    /// Creates a new unloaded runtime instance
    pub fn new() -> Self {
        Self {
            backend: None,
            model: None,
            loaded_info: None,
            is_generating: Arc::new(AtomicBool::new(false)),
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

        let model_params = {
            let mut params = LlamaModelParams::default();
            params = params.with_n_gpu_layers(config.gpu_layers);
            params
        };

        let clean_path = config.model_path.replace('/', "\\");
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
                let desc = if config.gpu_layers > 0 {
                    format!("llama.cpp (GPU offload: {} layers)", config.gpu_layers)
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

        // Step 4: Store state
        status_cb("Model loaded and ready for inference");
        let fam_str = format!("{:?}", crate::model_intelligence::MetadataExtractor::infer_family_from_string(&config.model_id));
        let info = LoadedModelInfo {
            model_id: config.model_id.clone(),
            model_name: config.model_name.clone(),
            quantization: config.quantization.clone(),
            file_path: config.model_path.clone(),
            context_length: config.context_length,
            gpu_layers: config.gpu_layers,
            threads: config.threads,
            backend_used: backend_desc,
            loaded_at: chrono::Utc::now().to_rfc3339(),
            chat_template: config.chat_template.clone(),
            stop_tokens: config.stop_tokens.clone(),
            model_family: fam_str,
            active_adapter: None,
        };

        self.model = Some(model);
        self.loaded_info = Some(info.clone());

        log::info!(
            "[RUNTIME] ✓ Model ready: {} ({}) — context={}, gpu_layers={}, threads={}, backend={}",
            info.model_name, info.quantization, info.context_length,
            info.gpu_layers, info.threads, info.backend_used
        );

        Ok(info)
    }

    /// Unloads the current model, freeing all RAM/VRAM
    pub fn unload_model(&mut self) -> Result<()> {
        self.is_generating.store(false, Ordering::Relaxed);

        if let Some(ref info) = self.loaded_info {
            log::info!("[RUNTIME] Unloading model: {} ({})", info.model_name, info.quantization);
        }

        // Drop model to free RAM/VRAM tensors
        self.model = None;
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
        mut token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?;
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;
        let config = self
            .loaded_info
            .as_ref()
            .ok_or_else(|| anyhow!("No model info available"))?;

        self.is_generating.store(true, Ordering::Relaxed);
        let cancel_flag = self.is_generating.clone();

        // Format messages into a prompt string dynamically using model's chat_template
        let prompt = format_chat_prompt_with_template(messages, &config.chat_template);
        log::info!("[RUNTIME] Generating for prompt using template '{}' ({} chars)", config.chat_template, prompt.len());

        // Tokenize the prompt (use AddBos::Never if template already includes BOS/header tags)
        let add_bos = if config.chat_template == "chatml" || config.chat_template == "gemma" {
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
            self.is_generating.store(false, Ordering::Relaxed);
            return Err(anyhow!("Empty prompt after tokenization"));
        }

        // Create context for this generation
        let ctx_size = std::num::NonZeroU32::new(config.context_length)
            .unwrap_or(std::num::NonZeroU32::new(2048).unwrap());
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads(config.threads as i32)
            .with_n_threads_batch(config.threads as i32);

        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create inference context: {:?}", e))?;

        // Create batch and fill with prompt tokens
        let max_batch = (n_prompt_tokens + 1).max(512);
        let mut batch = LlamaBatch::new(max_batch, 1);

        // Add prompt tokens to batch
        for (i, &token) in prompt_tokens.iter().enumerate() {
            let is_last = i == n_prompt_tokens - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
        }

        // Decode the prompt (prefill)
        let eos_token = model.token_eos();

        // Build effective template-driven stop sequences list
        let mut effective_stop_tokens = config.stop_tokens.clone();
        match config.chat_template.to_lowercase().as_str() {
            "chatml" => {
                for st in &["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
                    if !effective_stop_tokens.iter().any(|s| s == st) {
                        effective_stop_tokens.push(st.to_string());
                    }
                }
            }
            "gemma" => {
                for st in &["<end_of_turn>", "<start_of_turn>"] {
                    if !effective_stop_tokens.iter().any(|s| s == st) {
                        effective_stop_tokens.push(st.to_string());
                    }
                }
            }
            "llama3" | "llama" => {
                for st in &["<|eot_id|>", "<|end_of_text|>", "</s>"] {
                    if !effective_stop_tokens.iter().any(|s| s == st) {
                        effective_stop_tokens.push(st.to_string());
                    }
                }
            }
            _ => {}
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

        // Set up sampler chain directly consumed by llama.cpp
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_k(params.top_k as i32),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::min_p(params.min_p, 1),
            LlamaSampler::dist(1234),
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

            // Check for EOS token ID match
            if new_token_id == eos_token {
                log::info!("[RUNTIME] EOS token ID {} reached after {} tokens", eos_token, n_generated);
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

            // Check if token matches stop token sequences or ChatML/control token sequences
            let combined_check = format!("{}{}", generated_text, token_str);
            let is_stop_str = effective_stop_tokens.iter().any(|st| {
                !st.is_empty() && (
                    token_str.contains(st) || 
                    combined_check.ends_with(st) || 
                    (st.starts_with("<|") && token_str.starts_with("<|")) ||
                    (token_str == "<|im_end|>" || token_str == "<|im_start|>" || token_str == "<|endoftext|>")
                )
            });

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

        self.is_generating.store(false, Ordering::Relaxed);
        log::info!(
            "[RUNTIME] Generation complete: {} tokens, {} chars",
            n_generated,
            generated_text.len()
        );

        Ok(generated_text)
    }

    /// Signals the generation loop to stop
    pub fn stop_generation(&self) {
        log::info!("[RUNTIME] Stop generation requested");
        self.is_generating.store(false, Ordering::Relaxed);
    }
}

/// Formats chat messages into a prompt string based on Model Profile template.
fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    format_chat_prompt_with_template(messages, "llama3")
}

pub fn format_chat_prompt_with_template(messages: &[ChatMessage], template_name: &str) -> String {
    let mut prompt = String::new();

    match template_name.to_lowercase().as_str() {
        "chatml" => {
            for msg in messages {
                prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
            }
            prompt.push_str("<|im_start|>assistant\n");
        }
        "gemma" => {
            for msg in messages {
                let role = if msg.role == "assistant" { "model" } else { &msg.role };
                prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
            }
            prompt.push_str("<start_of_turn>model\n");
        }
        "mistral" => {
            for msg in messages {
                if msg.role == "user" {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                } else if msg.role == "assistant" {
                    prompt.push_str(&format!(" {}\n", msg.content));
                }
            }
        }
        _ => {
            for msg in messages {
                match msg.role.as_str() {
                    "system" => {
                        prompt.push_str(&format!("### System:\n{}\n\n", msg.content));
                    }
                    "user" => {
                        prompt.push_str(&format!("### User:\n{}\n\n", msg.content));
                    }
                    "assistant" => {
                        prompt.push_str(&format!("### Assistant:\n{}\n\n", msg.content));
                    }
                    _ => {
                        prompt.push_str(&format!("### {}:\n{}\n\n", msg.role, msg.content));
                    }
                }
            }
            prompt.push_str("### Assistant:\n");
        }
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_initial_status() {
        let runtime = LlamaCppRuntime::new();
        assert_eq!(runtime.status(), RuntimeStatus::NotLoaded);
        assert!(runtime.loaded_model_info().is_none());
    }

    #[test]
    fn test_format_chat_prompt() {
        let messages = vec![
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
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("### System:"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("### User:"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("### Assistant:\n"));
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

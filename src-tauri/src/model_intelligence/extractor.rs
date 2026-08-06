//! Source-Driven Metadata Extractor
//!
//! Gathers metadata from downloaded package sources in strict priority order:
//! GGUF Headers -> generation_config.json -> tokenizer_config.json -> config.json -> chat_template -> Model Card -> Fallback Defaults.

use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::adapter_manager::ModelPackageManifest;
use crate::model_intelligence::profile::*;

pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extracts or updates a `ModelProfile` for the given model package directory in strict priority order
    pub fn build_profile_from_package(
        package_dir: &Path,
        manifest: &ModelPackageManifest,
    ) -> Result<ModelProfile> {
        let package_id = &manifest.package_id;
        let model_id = &manifest.base_model.model_id;
        let model_name = &manifest.base_model.model_name;

        let mut profile = ModelProfile::new(package_id, model_id, model_name);
        let mut provenance = MetadataProvenance::default();

        // 1. Inspect GGUF file header metadata if present
        if let Ok(gguf_file) = Self::find_gguf_file(package_dir, manifest) {
            if let Ok(gguf_info) = Self::inspect_gguf_metadata(&gguf_file) {
                // Only claim GGUF provenance if something was genuinely read, so
                // `sourceSummary` stays a truthful record of where values came
                // from. The runtime sets this flag for real once llama.cpp has
                // parsed the container.
                provenance.gguf_metadata_extracted = gguf_info.family.is_some()
                    || gguf_info.architecture.is_some()
                    || gguf_info.context_length.is_some()
                    || gguf_info.eos_token.is_some()
                    || gguf_info.bos_token.is_some()
                    || !gguf_info.stop_tokens.is_empty();
                if let Some(fam) = gguf_info.family {
                    profile.model_family = fam;
                }
                if let Some(arch) = gguf_info.architecture {
                    profile.architecture = arch;
                }
                if let Some(ctx) = gguf_info.context_length {
                    profile.recommended_params.context_length = ctx;
                }
                if let Some(eos) = gguf_info.eos_token {
                    profile.tokens.eos_token = Some(eos);
                }
                if let Some(bos) = gguf_info.bos_token {
                    profile.tokens.bos_token = Some(bos);
                }
                if !gguf_info.stop_tokens.is_empty() {
                    profile.tokens.stop_tokens = gguf_info.stop_tokens;
                }
            }
        }

        // 2. Read generation_config.json if downloaded
        let gen_config_path = package_dir.join("generation_config.json");
        if gen_config_path.exists() {
            if let Ok(content) = fs::read_to_string(&gen_config_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                    provenance.generation_config_extracted = true;
                    if let Some(temp) = v.get("temperature").and_then(|t| t.as_f64()) {
                        profile.recommended_params.temperature = temp as f32;
                    }
                    if let Some(top_p) = v.get("top_p").and_then(|t| t.as_f64()) {
                        profile.recommended_params.top_p = top_p as f32;
                    }
                    if let Some(top_k) = v.get("top_k").and_then(|t| t.as_u64()) {
                        profile.recommended_params.top_k = top_k as u32;
                    }
                    if let Some(max_tokens) = v.get("max_new_tokens").and_then(|t| t.as_u64()) {
                        profile.recommended_params.max_tokens = max_tokens as u32;
                    }
                }
            }
        }

        // 3. Read tokenizer_config.json if downloaded
        let tok_config_path = package_dir.join("tokenizer_config.json");
        if tok_config_path.exists() {
            if let Ok(content) = fs::read_to_string(&tok_config_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                    provenance.tokenizer_config_extracted = true;
                    if let Some(tpl) = v.get("chat_template").and_then(|t| t.as_str()) {
                        profile.chat_template = tpl.to_string();
                    }
                    if let Some(bos) = v.get("bos_token").and_then(|t| t.as_str()) {
                        profile.tokens.bos_token = Some(bos.to_string());
                    }
                }
            }
        }

        // 3.5 Read tokenizer.json if downloaded
        let tok_json_path = package_dir.join("tokenizer.json");
        if tok_json_path.exists() {
            if let Ok(content) = fs::read_to_string(&tok_json_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                    provenance.tokenizer_json_extracted = true;
                    if let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) {
                        for tok in added {
                            if let Some(content) = tok.get("content").and_then(|c| c.as_str()) {
                                if !profile.tokens.stop_tokens.contains(&content.to_string()) && (content.contains("<|") || content.contains("</s") || content.contains("<end")) {
                                    profile.tokens.stop_tokens.push(content.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. Read config.json if downloaded
        let config_path = package_dir.join("config.json");
        if config_path.exists() {
            if let Ok(content) = fs::read_to_string(&config_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                    provenance.config_extracted = true;
                    if let Some(model_type) = v.get("model_type").and_then(|t| t.as_str()) {
                        profile.architecture = model_type.to_string();
                        profile.model_family = Self::infer_family_from_string(model_type);
                    }
                    if let Some(max_pos) = v.get("max_position_embeddings").and_then(|t| t.as_u64()) {
                        profile.recommended_params.context_length = max_pos as u32;
                    }
                }
            }
        }

        // 5. Read chat_template file if present
        let chat_tpl_path = package_dir.join("chat_template.jinja");
        if chat_tpl_path.exists() {
            if let Ok(tpl) = fs::read_to_string(&chat_tpl_path) {
                profile.chat_template = tpl;
            }
        }

        // 6. Infer model family and template if not explicitly set
        if profile.model_family == ModelFamily::Generic || profile.model_family == ModelFamily::Llama {
            profile.model_family = Self::infer_family_from_string(model_id);
        }

        profile.system_prompt_format = Self::default_system_prompt_for_family(&profile.model_family);
        profile.system_prompt_format = Self::default_system_prompt_for_family(&profile.model_family);
        if profile.chat_template.trim().is_empty() {
            profile.chat_template = Self::default_template_for_family(&profile.model_family);
        }

        // 7. Update capabilities dynamically
        let is_qwen = profile.model_family == ModelFamily::Qwen;
        let is_llama3 = profile.model_family == ModelFamily::Llama;
        let is_deepseek = profile.model_family == ModelFamily::DeepSeek;

        profile.capability_registry.set_capability("coding", true, 1.0, "Code generation and refactoring");
        profile.capability_registry.set_capability("reasoning", is_deepseek || is_llama3 || is_qwen, 0.95, "Logic & reasoning");
        profile.capability_registry.set_capability("mathematics", true, 0.9, "Math problems");
        profile.capability_registry.set_capability("tool_calling", is_qwen || is_llama3, 0.9, "Function execution");
        profile.capability_registry.set_capability("research", true, 0.9, "Research synthesis");

        provenance.source_summary = format!(
            "Sources used: GGUF={:?}, GenConfig={:?}, TokConfig={:?}, Config={:?}",
            provenance.gguf_metadata_extracted,
            provenance.generation_config_extracted,
            provenance.tokenizer_config_extracted,
            provenance.config_extracted
        );

        profile.provenance = provenance;
        Ok(profile)
    }

    fn find_gguf_file(package_dir: &Path, manifest: &ModelPackageManifest) -> Result<std::path::PathBuf> {
        let path = package_dir.join(&manifest.base_model.file_path);
        if path.exists() && path.is_file() {
            return Ok(path);
        }
        let base_dir = package_dir.join("base");
        if base_dir.exists() {
            if let Ok(entries) = fs::read_dir(&base_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension() == Some(std::ffi::OsStr::new("gguf")) {
                        return Ok(p);
                    }
                }
            }
        }
        Err(anyhow::anyhow!("No GGUF file found"))
    }

    fn inspect_gguf_metadata(path: &Path) -> Result<GgufExtractedInfo> {
        let meta = fs::metadata(path)?;
        if meta.len() < 1024 {
            return Err(anyhow::anyhow!("File too small"));
        }

        // Nothing here actually parses the GGUF container — doing so means
        // walking the full key/value header, which llama.cpp already does when
        // the model is loaded. The runtime calls `apply_runtime_metadata` at
        // that point to fill in architecture and tokens from the real thing.
        //
        // So this reports no findings rather than inventing any. It previously
        // returned Llama-3 stop tokens for *every* model, which is how a Gemma
        // model came to be profiled with `<|eot_id|>`.
        Ok(GgufExtractedInfo {
            family: None,
            architecture: None,
            context_length: None,
            eos_token: None,
            bos_token: None,
            stop_tokens: Vec::new(),
        })
    }

    pub fn infer_family_from_string(s: &str) -> ModelFamily {
        let lower = s.to_lowercase();
        if lower.contains("qwen") {
            ModelFamily::Qwen
        } else if lower.contains("gemma") {
            ModelFamily::Gemma
        } else if lower.contains("mistral") {
            ModelFamily::Mistral
        } else if lower.contains("mixtral") {
            ModelFamily::Mixtral
        } else if lower.contains("phi") {
            ModelFamily::Phi
        } else if lower.contains("deepseek") {
            ModelFamily::DeepSeek
        } else if lower.contains("command") {
            ModelFamily::CommandR
        } else if lower.contains("starcoder") {
            ModelFamily::Starcoder
        } else if lower.contains("glm") {
            ModelFamily::GLM
        } else if lower.contains("yi-") || lower.ends_with("-yi") {
            ModelFamily::Yi
        } else if lower.contains("baichuan") {
            ModelFamily::Baichuan
        } else if lower.contains("falcon") {
            ModelFamily::Falcon
        } else if lower.contains("granite") {
            ModelFamily::Granite
        } else if lower.contains("internlm") {
            ModelFamily::InternLM
        } else if lower.contains("smollm") {
            ModelFamily::SmolLM
        } else if lower.contains("tinyllama") {
            ModelFamily::TinyLlama
        } else if lower.contains("stablelm") {
            ModelFamily::StableLM
        } else if lower.contains("openchat") {
            ModelFamily::OpenChat
        } else if lower.contains("codellama") {
            ModelFamily::CodeLlama
        } else if lower.contains("llama") {
            ModelFamily::Llama
        } else {
            ModelFamily::Generic
        }
    }

    fn default_system_prompt_for_family(family: &ModelFamily) -> String {
        match family {
            ModelFamily::Qwen => "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string(),
            ModelFamily::DeepSeek => "You are a helpful AI assistant developed by DeepSeek.".to_string(),
            ModelFamily::Gemma => "You are a helpful assistant.".to_string(),
            ModelFamily::Mistral | ModelFamily::Mixtral => "You are a helpful AI assistant.".to_string(),
            _ => "You are a helpful, respectful, and honest assistant.".to_string(),
        }
    }

    fn default_template_for_family(family: &ModelFamily) -> String {
        match family {
            ModelFamily::Qwen | ModelFamily::DeepSeek | ModelFamily::GLM | ModelFamily::Yi | ModelFamily::Baichuan => "chatml".to_string(),
            ModelFamily::Gemma => "gemma".to_string(),
            ModelFamily::Mistral | ModelFamily::Mixtral => "mistral".to_string(),
            _ => "llama3".to_string(),
        }
    }
}

pub struct GgufExtractedInfo {
    pub family: Option<ModelFamily>,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub eos_token: Option<String>,
    pub bos_token: Option<String>,
    pub stop_tokens: Vec<String>,
}

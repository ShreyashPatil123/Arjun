//! HuggingFace GGUF Artifact Resolver
//!
//! Resolves exact GGUF file artifacts, download URLs, and sizes from HuggingFace Hub.

use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedArtifact {
    pub repo_id: String,
    pub file_name: String,
    pub download_url: String,
    pub size_bytes: u64,
    pub quantization: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfTreeItem {
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsInfo>,
}

#[derive(Debug, Deserialize)]
struct HfLfsInfo {
    oid: String,
    size: u64,
}

/// Maps base model IDs from catalog to canonical Hugging Face GGUF repositories
pub fn resolve_gguf_repo(model_id: &str) -> String {
    if model_id.ends_with("-GGUF") || model_id.contains("/GGUF-") {
        return model_id.to_string();
    }

    match model_id {
        "meta-llama/Llama-3.2-1B" => "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
        "meta-llama/Llama-3.2-3B" => "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
        "meta-llama/Llama-3.1-8B" => "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-3B" => "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-7B" => "Qwen/Qwen2.5-7B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-14B" => "Qwen/Qwen2.5-14B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-32B" => "Qwen/Qwen2.5-32B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-Coder-7B" => "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
        "mistralai/Mistral-7B-v0.3" => "bartowski/Mistral-7B-Instruct-v0.3-GGUF".to_string(),
        "mistralai/Mixtral-8x7B-v0.1" => "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string(),
        "microsoft/phi-4" => "bartowski/phi-4-GGUF".to_string(),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" => "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF".to_string(),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" => "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF".to_string(),
        "google/gemma-2-2b" => "bartowski/gemma-2-2b-it-GGUF".to_string(),
        "google/gemma-2-9b" => "bartowski/gemma-2-9b-it-GGUF".to_string(),
        "google/gemma-2-27b" => "bartowski/gemma-2-27b-it-GGUF".to_string(),
        "codellama/CodeLlama-7b-hf" => "TheBloke/CodeLlama-7B-GGUF".to_string(),
        "codellama/CodeLlama-13b-hf" => "TheBloke/CodeLlama-13B-GGUF".to_string(),
        other => {
            if other.contains('/') {
                format!("{}-GGUF", other)
            } else {
                format!("bartowski/{}-GGUF", other)
            }
        }
    }
}

/// Resolves exact downloadable GGUF artifact for a given model and quantization
pub async fn resolve_artifact(model_id: &str, quantization: &str, hf_token: Option<&str>) -> Result<ResolvedArtifact> {
    let repo_id = resolve_gguf_repo(model_id);
    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);

    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0 (Windows; x64)")
        .build()?;

    let mut req = client.get(&api_url);
    if let Some(token) = hf_token {
        if !token.trim().is_empty() {
            req = req.header("Authorization", format!("Bearer {}", token.trim()));
        }
    }

    let quant_lower = quantization.to_lowercase();
    let quant_clean = quant_lower.replace('_', "");

    let resp = req.send().await;
    if let Ok(res) = resp {
        if res.status().is_success() {
            if let Ok(items) = res.json::<Vec<HfTreeItem>>().await {
                for item in items {
                    let path_lower = item.path.to_lowercase();
                    if path_lower.ends_with(".gguf") {
                        let path_clean = path_lower.replace('_', "");
                        if path_clean.contains(&quant_clean) || path_lower.contains(&quant_lower) {
                            let size = item.lfs.as_ref().map(|l| l.size).or(item.size).unwrap_or(0);
                            let sha256 = item.lfs.map(|l| l.oid);
                            let download_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, item.path);
                            return Ok(ResolvedArtifact {
                                repo_id,
                                file_name: item.path,
                                download_url,
                                size_bytes: size,
                                quantization: quantization.to_string(),
                                sha256,
                            });
                        }
                    }
                }
            }
        }
    }

    // Fallback: construct canonical Hugging Face URL if direct API tree search timed out or hit offline mode
    let file_name = format!("{}-{}.gguf", repo_id.split('/').last().unwrap_or("model"), quantization);
    let download_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, file_name);

    Ok(ResolvedArtifact {
        repo_id,
        file_name,
        download_url,
        size_bytes: 0, // 0 = will be populated via HTTP HEAD request Content-Length
        quantization: quantization.to_string(),
        sha256: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_gguf_repo_mapping() {
        assert_eq!(resolve_gguf_repo("meta-llama/Llama-3.2-1B"), "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(resolve_gguf_repo("Qwen/Qwen2.5-Coder-7B"), "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
    }

    #[tokio::test]
    async fn test_real_hf_artifact_resolution() {
        let artifact = resolve_artifact("meta-llama/Llama-3.2-1B", "Q8_0", None).await;
        assert!(artifact.is_ok(), "HuggingFace API artifact resolution must succeed");
        let art = artifact.unwrap();
        println!("\n=== REAL HF ARTIFACT RESOLUTION RESULT ===");
        println!("Repo ID: {}", art.repo_id);
        println!("File Name: {}", art.file_name);
        println!("Download URL: {}", art.download_url);
        println!("Size: {:.2} MB ({} bytes)", art.size_bytes as f64 / 1_048_576.0, art.size_bytes);
        println!("==========================================\n");
        assert!(art.download_url.starts_with("https://huggingface.co/"));
        assert!(art.file_name.to_lowercase().contains("q8_0"));
        assert!(art.size_bytes > 0, "Artifact size must be > 0 bytes from HF API");
    }
}

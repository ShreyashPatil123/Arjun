//! Local AI Runtimes (Ollama, vLLM, etc.) health and status detector

use crate::system_analyzer::process_utils::{create_hidden_command, run_command_with_timeout};
use crate::system_analyzer::traits::AIRuntimeInfo;
use serde::Deserialize;
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Option<Vec<OllamaModelItem>>,
}

#[derive(Deserialize)]
struct OllamaModelItem {
    name: String,
}

#[derive(Deserialize)]
struct OllamaVersionResponse {
    version: String,
}

/// Detects local AI runtimes like Ollama
pub fn detect_ai_runtimes() -> Vec<AIRuntimeInfo> {
    let mut runtimes = Vec::new();
    let ollama_info = check_ollama();
    runtimes.push(ollama_info);

    runtimes
}

fn check_ollama() -> AIRuntimeInfo {
    let endpoint = "http://127.0.0.1:11434".to_string();
    let addr: Option<SocketAddr> = "127.0.0.1:11434".parse().ok();

    // Check if port 11434 is listening
    let is_listening = if let Some(a) = addr {
        std::net::TcpStream::connect_timeout(&a, Duration::from_millis(500)).is_ok()
    } else {
        false
    };

    if is_listening {
        let (version, models) = fetch_ollama_api_info(&endpoint);
        return AIRuntimeInfo {
            name: "Ollama".to_string(),
            status: "running".to_string(),
            version: Some(version),
            endpoint: Some(endpoint),
            models_available: models,
        };
    }

    // Port not listening, check if ollama CLI binary exists
    let mut cmd = create_hidden_command("ollama");
    cmd.arg("--version");
    if let Ok(output) = run_command_with_timeout(cmd, Duration::from_secs(2)) {
        if output.status.success() || !output.stdout.is_empty() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let version = stdout.lines().next().unwrap_or("Ollama").trim().to_string();

            return AIRuntimeInfo {
                name: "Ollama".to_string(),
                status: "stopped".to_string(),
                version: Some(version),
                endpoint: Some(endpoint),
                models_available: Vec::new(),
            };
        }
    }

    AIRuntimeInfo {
        name: "Ollama".to_string(),
        status: "not_installed".to_string(),
        version: None,
        endpoint: None,
        models_available: Vec::new(),
    }
}

fn fetch_ollama_api_info(endpoint: &str) -> (String, Vec<String>) {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return ("Unknown".to_string(), Vec::new()),
    };

    let version_url = format!("{}/api/version", endpoint);
    let mut version = "Unknown".to_string();
    if let Ok(resp) = client.get(&version_url).send() {
        if let Ok(v_data) = resp.json::<OllamaVersionResponse>() {
            version = v_data.version;
        }
    }

    let tags_url = format!("{}/api/tags", endpoint);
    let mut models = Vec::new();
    if let Ok(resp) = client.get(&tags_url).send() {
        if let Ok(tags_data) = resp.json::<OllamaTagsResponse>() {
            if let Some(m_list) = tags_data.models {
                models = m_list.into_iter().map(|m| m.name).collect();
            }
        }
    }

    (version, models)
}

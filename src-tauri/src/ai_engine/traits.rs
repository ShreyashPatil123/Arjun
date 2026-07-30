//! AI engine traits

use anyhow::Result;

pub enum AIBackendType { Ollama, LlamaCpp, VLLM, Custom(String) }

pub struct ChatMessage { pub role: String, pub content: String, pub timestamp: String }
pub struct ChatResponse { pub message: String, pub model: String, pub tokens_used: u32, pub generation_time_ms: u64 }
pub struct StreamChunk { pub text: String, pub is_final: bool, pub tokens_generated: Option<u32> }
pub struct ModelLoadConfig { pub model_path: String, pub context_length: u32, pub gpu_layers: u32, pub threads: u32 }

pub trait AIBackend: Send + Sync {
    fn name(&self) -> &'static str { "Unknown" }
    fn backend_type(&self) -> AIBackendType { AIBackendType::Ollama }
    fn load_model(&self, _config: ModelLoadConfig) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unload_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn is_model_loaded(&self, _model_id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat_stream(&self, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_loaded_models(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_model_info(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait ContextManager: Send + Sync {
    fn add_message(&self, _message: ChatMessage) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_context(&self) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn clear_context(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn trim_context(&self, _max_tokens: u32) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_token_count(&self, _messages: Vec<ChatMessage>) -> Result<u32> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait MemoryManager: Send + Sync {
    fn save_conversation(&self, _id: &str, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn load_conversation(&self, _id: &str) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_conversations(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn delete_conversation(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

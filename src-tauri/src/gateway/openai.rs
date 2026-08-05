//! OpenAI-compatible protocol (`/v1/chat/completions`).
//!
//! Spoken by opencode, openclaw, Cursor, and Continue.dev.
//!
//! The `model` field is accepted and echoed back but does not select anything —
//! Sarathi always serves whichever model the desktop app has loaded. Clients
//! hardcode names like `gpt-4o` that mean nothing here, and failing those
//! requests would break every tool out of the box.

use serde::{Deserialize, Serialize};

use crate::ai_engine::traits::{ChatMessage, GenerationParams};

// ─── Request ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Newer clients send this instead of `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    /// Usually a string, but the spec also allows an array of content parts.
    #[serde(default)]
    pub content: Option<MessageContent>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(default)]
    pub text: Option<String>,
}

impl MessageContent {
    /// Flattens to plain text. Non-text parts (images) are dropped — the local
    /// GGUF path has no vision support, and silently ignoring them beats
    /// rejecting an otherwise valid request.
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl ChatCompletionRequest {
    /// Converts to Sarathi's internal message list.
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.as_ref().map(|c| c.to_text()).unwrap_or_default(),
                timestamp: None,
            })
            .collect()
    }

    /// Layers client-supplied sampling over Sarathi's defaults.
    ///
    /// Only fields the client actually sent are applied, so omitted values keep
    /// the model's configured defaults rather than snapping to OpenAI's.
    pub fn to_generation_params(&self) -> GenerationParams {
        let base = GenerationParams::default();
        GenerationParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            max_tokens: self
                .max_tokens
                .or(self.max_completion_tokens)
                .unwrap_or(base.max_tokens),
            repeat_penalty: self
                .frequency_penalty
                .map(|f| 1.0 + f.clamp(0.0, 1.0))
                .unwrap_or(base.repeat_penalty),
            top_k: base.top_k,
            min_p: base.min_p,
            mirostat: base.mirostat,
        }
    }
}

// ─── Response ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl ChatCompletionResponse {
    pub fn new(id: String, created: u64, model: String, text: String, finish_reason: &str, completion_tokens: u32) -> Self {
        Self {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage { role: "assistant", content: text },
                finish_reason: finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens,
                total_tokens: completion_tokens,
            },
        }
    }
}

// ─── Streaming chunks ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionChunk {
    /// First chunk: announces the assistant role, carries no text.
    pub fn opening(id: &str, created: u64, model: &str) -> Self {
        Self::with_delta(id, created, model, Delta { role: Some("assistant"), content: None }, None)
    }

    pub fn text(id: &str, created: u64, model: &str, text: String) -> Self {
        Self::with_delta(id, created, model, Delta { role: None, content: Some(text) }, None)
    }

    pub fn closing(id: &str, created: u64, model: &str, finish_reason: &str) -> Self {
        Self::with_delta(id, created, model, Delta::default(), Some(finish_reason.to_string()))
    }

    fn with_delta(id: &str, created: u64, model: &str, delta: Delta, finish_reason: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
        }
    }
}

// ─── Model listing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

impl ModelList {
    pub fn single(model_id: Option<String>, created: u64) -> Self {
        let data = match model_id {
            Some(id) => vec![ModelEntry { id, object: "model", created, owned_by: "sarathi" }],
            None => vec![],
        };
        Self { object: "list", data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("should parse")
    }

    #[test]
    fn parses_a_typical_request() {
        let req = parse(
            r#"{"model":"gpt-4o","messages":[
                {"role":"system","content":"Be brief."},
                {"role":"user","content":"Hello"}],
               "stream":true,"temperature":0.3}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].content, "Hello");
        assert!(req.stream);
        assert_eq!(req.to_generation_params().temperature, 0.3);
    }

    #[test]
    fn omitted_sampling_falls_back_to_sarathi_defaults() {
        let req = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        let params = req.to_generation_params();
        let base = GenerationParams::default();

        assert_eq!(params.temperature, base.temperature);
        assert_eq!(params.top_p, base.top_p);
        assert_eq!(params.max_tokens, base.max_tokens);
        assert!(!req.stream, "stream must default to false");
    }

    #[test]
    fn accepts_array_style_content() {
        // Newer clients send content as an array of typed parts.
        let req = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"line one"},
                {"type":"text","text":"line two"}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "line one\nline two");
    }

    #[test]
    fn image_parts_are_dropped_rather_than_failing_the_request() {
        let req = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"describe"},
                {"type":"image_url","image_url":{"url":"data:..."}}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "describe");
    }

    #[test]
    fn max_completion_tokens_is_honoured_as_an_alias() {
        let req = parse(r#"{"messages":[],"max_completion_tokens":123}"#);
        assert_eq!(req.to_generation_params().max_tokens, 123);
    }

    #[test]
    fn unknown_fields_do_not_break_parsing() {
        // Clients send plenty we do not model; rejecting them would be fatal.
        let req = parse(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],
                "tools":[],"tool_choice":"auto","seed":42,"user":"abc","n":1}"#,
        );
        assert_eq!(req.to_chat_messages().len(), 1);
    }

    #[test]
    fn streaming_chunks_have_the_right_shape() {
        let opening = ChatCompletionChunk::opening("id1", 1, "m");
        let json = serde_json::to_string(&opening).unwrap();
        assert!(json.contains(r#""object":"chat.completion.chunk""#));
        assert!(json.contains(r#""role":"assistant""#));
        // No content key on the opening chunk.
        assert!(!json.contains(r#""content""#));

        let text = ChatCompletionChunk::text("id1", 1, "m", "Hi".into());
        assert!(serde_json::to_string(&text).unwrap().contains(r#""content":"Hi""#));

        let closing = ChatCompletionChunk::closing("id1", 1, "m", "stop");
        assert!(serde_json::to_string(&closing).unwrap().contains(r#""finish_reason":"stop""#));
    }

    #[test]
    fn model_list_is_empty_when_nothing_is_loaded() {
        assert!(ModelList::single(None, 0).data.is_empty());
        assert_eq!(ModelList::single(Some("qwen".into()), 0).data.len(), 1);
    }
}

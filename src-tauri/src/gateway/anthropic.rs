//! Anthropic Messages protocol (`/v1/messages`).
//!
//! Spoken by Claude Code, which is pointed here with `ANTHROPIC_BASE_URL`.
//!
//! Three differences from the OpenAI shape drive this module:
//!
//! 1. `system` is a **top-level field**, not a message with `role: "system"`.
//! 2. Message `content` is commonly an **array of typed blocks**, not a string.
//! 3. Streaming uses **named SSE events** in a required order, not one repeated
//!    chunk type. Clients parse by event name, so the sequence must be exact:
//!    `message_start` → `content_block_start` → `content_block_delta`* →
//!    `content_block_stop` → `message_delta` → `message_stop`.

use serde::{Deserialize, Serialize};

use crate::ai_engine::traits::{ChatMessage, GenerationParams};

// ─── Request ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    /// Top-level system prompt. String or array of blocks.
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl SystemPrompt {
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(b) => blocks_to_text(b),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type", default)]
    pub block_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

/// Concatenates the text of `text` blocks, skipping images and tool blocks.
fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter(|b| b.block_type.as_deref().unwrap_or("text") == "text")
        .filter_map(|b| b.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n")
}

impl MessageContent {
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(b) => blocks_to_text(b),
        }
    }
}

impl MessagesRequest {
    /// Flattens into Sarathi's internal list, hoisting `system` to the front as
    /// a system-role message so the rest of the pipeline sees a uniform shape.
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);

        if let Some(system) = &self.system {
            let text = system.to_text();
            if !text.trim().is_empty() {
                out.push(ChatMessage {
                    role: "system".to_string(),
                    content: text,
                    timestamp: None,
                });
            }
        }

        out.extend(self.messages.iter().map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.to_text(),
            timestamp: None,
        }));

        out
    }

    /// `max_tokens` is required by the Anthropic spec, so unlike the OpenAI path
    /// it is normally present and authoritative.
    pub fn to_generation_params(&self) -> GenerationParams {
        let base = GenerationParams::default();
        GenerationParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: self.max_tokens.unwrap_or(base.max_tokens),
            min_p: base.min_p,
            repeat_penalty: base.repeat_penalty,
            mirostat: base.mirostat,
        }
    }
}

// ─── Non-streaming response ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<TextBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl MessagesResponse {
    pub fn new(id: String, model: String, text: String, stop_reason: &str, output_tokens: u32) -> Self {
        Self {
            id,
            kind: "message",
            role: "assistant",
            model,
            content: vec![TextBlock { kind: "text", text }],
            stop_reason: stop_reason.to_string(),
            stop_sequence: None,
            usage: Usage { input_tokens: 0, output_tokens },
        }
    }
}

/// Maps Sarathi's finish reasons onto Anthropic's vocabulary.
pub fn map_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "max_tokens",
        "cancelled" => "stop_sequence",
        _ => "end_turn",
    }
}

// ─── Streaming events ───────────────────────────────────────────────────────

/// Builds the SSE event bodies. Each returns `(event_name, json_payload)`.
pub struct StreamEvents;

impl StreamEvents {
    pub fn message_start(id: &str, model: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        ("message_start", payload.to_string())
    }

    pub fn content_block_start() -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" }
        });
        ("content_block_start", payload.to_string())
    }

    pub fn content_block_delta(text: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": text }
        });
        ("content_block_delta", payload.to_string())
    }

    pub fn content_block_stop() -> (&'static str, String) {
        let payload = serde_json::json!({ "type": "content_block_stop", "index": 0 });
        ("content_block_stop", payload.to_string())
    }

    pub fn message_delta(stop_reason: &str, output_tokens: u32) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": output_tokens }
        });
        ("message_delta", payload.to_string())
    }

    pub fn message_stop() -> (&'static str, String) {
        let payload = serde_json::json!({ "type": "message_stop" });
        ("message_stop", payload.to_string())
    }

    /// Mid-stream failure. Clients surface this instead of hanging.
    pub fn error(message: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "error",
            "error": { "type": "api_error", "message": message }
        });
        ("error", payload.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> MessagesRequest {
        serde_json::from_str(json).expect("should parse")
    }

    #[test]
    fn top_level_system_becomes_a_leading_system_message() {
        let req = parse(
            r#"{"model":"claude-sonnet-4-5","max_tokens":1024,
                "system":"You are a helpful assistant.",
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "You are a helpful assistant.");
        assert_eq!(msgs[1].role, "user");
    }

    #[test]
    fn block_style_content_is_flattened() {
        // Claude Code sends content as blocks in normal operation.
        let req = parse(
            r#"{"max_tokens":100,"messages":[{"role":"user","content":[
                {"type":"text","text":"first"},
                {"type":"text","text":"second"}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "first\nsecond");
    }

    #[test]
    fn non_text_blocks_are_skipped() {
        let req = parse(
            r#"{"max_tokens":100,"messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image","source":{"type":"base64","data":"..."}}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "look");
    }

    #[test]
    fn a_block_style_system_prompt_is_supported() {
        let req = parse(
            r#"{"max_tokens":100,
                "system":[{"type":"text","text":"Rule one"},{"type":"text","text":"Rule two"}],
                "messages":[{"role":"user","content":"go"}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "Rule one\nRule two");
    }

    #[test]
    fn an_empty_system_prompt_adds_no_message() {
        let req = parse(r#"{"max_tokens":100,"system":"   ","messages":[{"role":"user","content":"hi"}]}"#);
        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn client_sampling_is_honoured() {
        let req = parse(
            r#"{"max_tokens":512,"temperature":0.2,"top_p":0.5,"top_k":10,
                "messages":[{"role":"user","content":"hi"}]}"#,
        );
        let p = req.to_generation_params();
        assert_eq!(p.max_tokens, 512);
        assert_eq!(p.temperature, 0.2);
        assert_eq!(p.top_k, 10);
    }

    #[test]
    fn unknown_fields_do_not_break_parsing() {
        let req = parse(
            r#"{"model":"claude-sonnet-4-5","max_tokens":1024,
                "messages":[{"role":"user","content":"hi"}],
                "tools":[],"metadata":{"user_id":"x"},"stop_sequences":["\n"]}"#,
        );
        assert_eq!(req.to_chat_messages().len(), 1);
    }

    #[test]
    fn stop_reasons_map_to_anthropic_vocabulary() {
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("cancelled"), "stop_sequence");
        assert_eq!(map_stop_reason("anything else"), "end_turn");
    }

    #[test]
    fn stream_events_carry_their_declared_type() {
        // Clients dispatch on the event name, and validate the inner "type".
        let (name, body) = StreamEvents::message_start("msg_1", "qwen");
        assert_eq!(name, "message_start");
        assert!(body.contains(r#""type":"message_start""#));
        assert!(body.contains(r#""role":"assistant""#));

        let (name, body) = StreamEvents::content_block_delta("Hello");
        assert_eq!(name, "content_block_delta");
        assert!(body.contains(r#""type":"text_delta""#));
        assert!(body.contains(r#""text":"Hello""#));

        let (name, body) = StreamEvents::message_delta("end_turn", 7);
        assert_eq!(name, "message_delta");
        assert!(body.contains(r#""stop_reason":"end_turn""#));
        assert!(body.contains(r#""output_tokens":7"#));

        assert_eq!(StreamEvents::message_stop().0, "message_stop");
        assert_eq!(StreamEvents::content_block_stop().0, "content_block_stop");
        assert_eq!(StreamEvents::error("boom").0, "error");
    }

    #[test]
    fn delta_text_is_json_escaped() {
        // Newlines and quotes must not corrupt the SSE payload.
        let (_, body) = StreamEvents::content_block_delta("line\n\"quoted\"");
        assert!(body.contains(r#"\n"#));
        assert!(body.contains(r#"\""#));
        // Round-trips as valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["delta"]["text"], "line\n\"quoted\"");
    }
}

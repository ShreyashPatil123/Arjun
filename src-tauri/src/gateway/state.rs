//! Gateway configuration and live statistics.
//!
//! The stats here are what the dashboard renders. They exist to answer one
//! question quickly when a tool misbehaves: *did the request ever reach
//! Sarathi at all?* Without that, users cannot tell a broken client config from
//! a broken model.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::ai_engine::manager::InferenceManager;
use crate::ai_engine::scheduler::GenerationScheduler;

/// User-controlled gateway settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayConfig {
    pub enabled: bool,
    pub port: u16,
    /// Apply intent routing and capability profiles to gateway traffic.
    ///
    /// Off by default: external tools send their own system prompts and
    /// sampling, and overriding those silently can break structured output they
    /// depend on.
    pub apply_capabilities: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: super::DEFAULT_PORT,
            apply_capabilities: false,
        }
    }
}

/// One connected tool, as shown on the dashboard.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientActivity {
    /// Best-effort identity from the User-Agent header.
    pub client: String,
    /// Which surface it used: `openai` or `anthropic`.
    pub protocol: String,
    pub requests: u64,
    /// Unix milliseconds of the most recent request.
    pub last_seen_ms: u64,
}

/// Snapshot for the dashboard.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayStats {
    pub running: bool,
    pub port: u16,
    pub total_requests: u64,
    pub active_requests: u64,
    pub queue_depth: usize,
    pub clients: Vec<ClientActivity>,
}

/// Shared state handed to every request handler.
pub struct GatewayState {
    pub scheduler: Arc<GenerationScheduler>,
    pub inference: Arc<InferenceManager>,
    pub config: Mutex<GatewayConfig>,
    total_requests: AtomicU64,
    active_requests: AtomicU64,
    clients: Mutex<HashMap<String, ClientActivity>>,
}

impl GatewayState {
    pub fn new(
        scheduler: Arc<GenerationScheduler>,
        inference: Arc<InferenceManager>,
        config: GatewayConfig,
    ) -> Self {
        Self {
            scheduler,
            inference,
            config: Mutex::new(config),
            total_requests: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            clients: Mutex::new(HashMap::new()),
        }
    }

    pub fn apply_capabilities(&self) -> bool {
        self.config
            .lock()
            .map(|c| c.apply_capabilities)
            .unwrap_or(false)
    }

    pub fn port(&self) -> u16 {
        self.config.lock().map(|c| c.port).unwrap_or(super::DEFAULT_PORT)
    }

    /// Records the port the gateway actually bound.
    ///
    /// The configured value is a request, not a guarantee — it can be taken.
    /// Everything downstream (the dashboard, and the address handed to a
    /// launched tool) has to use what was really bound, or it advertises an
    /// endpoint nothing is listening on.
    pub fn set_port(&self, port: u16) {
        if let Ok(mut config) = self.config.lock() {
            config.port = port;
        }
    }

    /// Records an incoming request. Call once per request, before generation.
    pub fn record_request(&self, client: &str, protocol: &str, now_ms: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut map) = self.clients.lock() {
            let key = format!("{client}|{protocol}");
            let entry = map.entry(key).or_insert_with(|| ClientActivity {
                client: client.to_string(),
                protocol: protocol.to_string(),
                requests: 0,
                last_seen_ms: now_ms,
            });
            entry.requests += 1;
            entry.last_seen_ms = now_ms;
        }
    }

    /// Marks a request finished, however it ended.
    pub fn finish_request(&self) {
        // Saturating: a double-finish must not wrap to u64::MAX and make the
        // dashboard claim billions of in-flight requests.
        let _ = self
            .active_requests
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)));
    }

    pub fn stats(&self, running: bool) -> GatewayStats {
        let mut clients: Vec<ClientActivity> = self
            .clients
            .lock()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default();
        clients.sort_by(|a, b| b.last_seen_ms.cmp(&a.last_seen_ms));

        GatewayStats {
            running,
            port: self.port(),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            queue_depth: self.scheduler.queue_depth(),
            clients,
        }
    }
}

/// Shortens a User-Agent into something readable on the dashboard.
///
/// Falls back to "unknown" rather than an empty label, so an unidentified
/// client still shows up as *something* — the point of the list is proving a
/// request arrived at all.
pub fn client_label(user_agent: Option<&str>) -> String {
    let raw = match user_agent {
        Some(ua) if !ua.trim().is_empty() => ua.trim(),
        _ => return "unknown".to_string(),
    };
    // "claude-cli/1.2.3 (external)" -> "claude-cli"
    let head = raw.split_whitespace().next().unwrap_or(raw);
    let name = head.split('/').next().unwrap_or(head);
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name.chars().take(40).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_are_off_by_default() {
        let c = GatewayConfig::default();
        assert!(!c.apply_capabilities, "must not interfere with client prompts by default");
        assert_eq!(c.port, super::super::DEFAULT_PORT);
        assert_ne!(c.port, 11434, "must not collide with Ollama");
    }

    #[test]
    fn user_agents_shorten_to_readable_names() {
        assert_eq!(client_label(Some("claude-cli/1.2.3 (external, cli)")), "claude-cli");
        assert_eq!(client_label(Some("opencode/0.1")), "opencode");
        assert_eq!(client_label(Some("curl/8.4.0")), "curl");
        assert_eq!(client_label(None), "unknown");
        assert_eq!(client_label(Some("   ")), "unknown");
    }

    #[test]
    fn a_very_long_user_agent_is_truncated() {
        let long = "x".repeat(500);
        assert!(client_label(Some(&long)).len() <= 40);
    }

    fn state() -> GatewayState {
        let inference = Arc::new(InferenceManager::new());
        let scheduler = Arc::new(GenerationScheduler::start(inference.clone()));
        GatewayState::new(scheduler, inference, GatewayConfig::default())
    }

    #[test]
    fn requests_are_counted_per_client() {
        let s = state();
        s.record_request("claude-cli", "anthropic", 1000);
        s.record_request("claude-cli", "anthropic", 2000);
        s.record_request("opencode", "openai", 1500);

        let stats = s.stats(true);
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.active_requests, 3);
        assert_eq!(stats.clients.len(), 2);

        // Most recently seen first.
        assert_eq!(stats.clients[0].client, "claude-cli");
        assert_eq!(stats.clients[0].requests, 2);
    }

    #[test]
    fn the_same_client_on_two_protocols_is_tracked_separately() {
        let s = state();
        s.record_request("curl", "openai", 1);
        s.record_request("curl", "anthropic", 2);
        assert_eq!(s.stats(true).clients.len(), 2);
    }

    #[test]
    fn finishing_more_than_starting_cannot_underflow() {
        let s = state();
        s.record_request("x", "openai", 1);
        s.finish_request();
        s.finish_request(); // spurious
        s.finish_request();

        assert_eq!(s.stats(true).active_requests, 0, "must clamp at zero, not wrap");
    }
}

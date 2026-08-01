//! LoRA Adapter State Machine
//!
//! Provides strict state transitions, logging, and terminal READY state enforcement.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AdapterState {
    NotFound,
    Searching,
    Found,
    Downloading,
    Verifying,
    Installing,
    Registering,
    Ready,
    Failed,
    Unavailable,
}

impl fmt::Display for AdapterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterState::NotFound => write!(f, "NOT_FOUND"),
            AdapterState::Searching => write!(f, "SEARCHING"),
            AdapterState::Found => write!(f, "FOUND"),
            AdapterState::Downloading => write!(f, "DOWNLOADING"),
            AdapterState::Verifying => write!(f, "VERIFYING"),
            AdapterState::Installing => write!(f, "INSTALLING"),
            AdapterState::Registering => write!(f, "REGISTERING"),
            AdapterState::Ready => write!(f, "READY"),
            AdapterState::Failed => write!(f, "FAILED"),
            AdapterState::Unavailable => write!(f, "UNAVAILABLE"),
        }
    }
}

impl AdapterState {
    pub fn from_str_lenient(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "READY" | "INSTALLED" => AdapterState::Ready,
            "NOT_FOUND" | "NOTFOUND" => AdapterState::NotFound,
            "SEARCHING" | "RESOLVING" => AdapterState::Searching,
            "FOUND" => AdapterState::Found,
            "DOWNLOADING" => AdapterState::Downloading,
            "VERIFYING" => AdapterState::Verifying,
            "INSTALLING" => AdapterState::Installing,
            "REGISTERING" => AdapterState::Registering,
            "FAILED" => AdapterState::Failed,
            "UNAVAILABLE" => AdapterState::Unavailable,
            _ => AdapterState::NotFound,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, AdapterState::Ready)
    }
}

/// Structured transition logging:
/// Capability | Old State | New State | Reason | Source | Timestamp
pub fn log_adapter_transition(
    capability: &str,
    old_state: &AdapterState,
    new_state: &AdapterState,
    reason: &str,
    source: &str,
) {
    let timestamp = chrono::Utc::now().to_rfc3339();
    log::info!(
        "[ADAPTER_STATE_TRANSITION] Capability: {} | Old State: {} | New State: {} | Reason: {} | Source: {} | Timestamp: {}",
        capability, old_state, new_state, reason, source, timestamp
    );
}

/// Validates whether a state transition is allowed under single source of truth rules.
/// Returns true if allowed, false if blocked.
pub fn validate_transition(
    capability: &str,
    old_state: &AdapterState,
    new_state: &AdapterState,
    is_user_action: bool,
    source: &str,
) -> bool {
    if *old_state == AdapterState::Ready && *new_state != AdapterState::Ready && !is_user_action {
        log::warn!(
            "[ADAPTER_STATE_TRANSITION_BLOCKED] Capability: {} | Blocked automatic transition from READY to {} | Source: {}",
            capability, new_state, source
        );
        return false;
    }

    log_adapter_transition(capability, old_state, new_state, "Transition allowed", source);
    true
}

//! Internal publish/subscribe event system

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;
use tokio::sync::broadcast::{self, Receiver, Sender};
use chrono::{DateTime, Utc};

/// All possible events in the Sarathi system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SarathiEvent {
    ApplicationStarted,
    ConfigChanged,
    ThemeChanged,
    SystemAnalyzed,
    SystemAnalysisStarted,
    SystemAnalysisProgress,
    SystemAnalysisCompleted,
    SystemAnalysisFailed,
    HardwareProfileUpdated,
    ModelRecommended,
    DownloadStarted,
    DownloadProgress,
    DownloadCompleted,
    DownloadFailed,
    ModelInstalled,
    ModelUninstalled,
    AIInitialized,
    AIStopped,
    LoRALoaded,
    LoRASwitched,
    LoRAUnloaded,
    PluginLoaded,
    PluginUnloaded,
    Error,
}

/// Payload for a broadcasted event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayload {
    pub event_type: SarathiEvent,
    pub timestamp: DateTime<Utc>,
    pub data: Option<Value>,
}

/// Event bus using tokio's broadcast channel
pub struct EventBus {
    sender: Sender<EventPayload>,
}

impl EventBus {
    /// Creates a new EventBus
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1024);
        Self { sender }
    }

    /// Subscribes to the event bus
    pub fn subscribe(&self) -> Receiver<EventPayload> {
        self.sender.subscribe()
    }

    /// Publishes an event to all subscribers
    pub fn publish(&self, event_type: SarathiEvent, data: Option<Value>) {
        let payload = EventPayload {
            event_type,
            timestamp: Utc::now(),
            data,
        };
        let _ = self.sender.send(payload);
    }
}

static EVENT_BUS: OnceLock<EventBus> = OnceLock::new();

/// Gets the global event bus
pub fn get_event_bus() -> &'static EventBus {
    EVENT_BUS.get_or_init(|| EventBus::new())
}

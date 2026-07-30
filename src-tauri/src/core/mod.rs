//! Sarathi Core Module
//! The central coordinator for the application.

pub mod app_state;
pub mod event_bus;
pub mod module_manager;
pub mod service_registry;

use app_state::AppState;
use event_bus::EventBus;
use module_manager::ModuleManager;
use service_registry::ServiceRegistry;

/// The central coordinator struct.
pub struct SarathiCore {
    /// Global application state manager
    pub state: AppState,
    /// Internal event system
    pub events: EventBus,
    /// Manager for dynamic modules
    pub modules: ModuleManager,
    /// Registry for services
    pub services: ServiceRegistry,
}

impl SarathiCore {
    /// Creates a new instance of SarathiCore
    pub fn new() -> Self {
        Self {
            state: AppState::new(),
            events: EventBus::new(),
            modules: ModuleManager::new(),
            services: ServiceRegistry::new(),
        }
    }
}

/// Initializes the core on app startup
pub fn init() -> SarathiCore {
    SarathiCore::new()
}

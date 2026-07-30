//! Service registry skeleton

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Base trait for all services
pub trait Service: Send + Sync {
    /// The name of the service
    fn name(&self) -> &'static str;
    
    /// The version of the service
    fn version(&self) -> &'static str;
    
    /// Whether the service is ready
    fn is_ready(&self) -> bool;
}

/// Registry for managing services
pub struct ServiceRegistry {
    services: Arc<Mutex<HashMap<String, Box<dyn Service>>>>,
}

impl ServiceRegistry {
    /// Creates a new ServiceRegistry
    pub fn new() -> Self {
        Self {
            services: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Registers a new service
    pub fn register(&self, id: String, service: Box<dyn Service>) {
        let mut services = self.services.lock().unwrap();
        services.insert(id, service);
    }

    /// Lists all registered service IDs
    pub fn list(&self) -> Vec<String> {
        let services = self.services.lock().unwrap();
        services.keys().cloned().collect()
    }

    /// Checks if a service is registered
    pub fn is_registered(&self, id: &str) -> bool {
        let services = self.services.lock().unwrap();
        services.contains_key(id)
    }
}

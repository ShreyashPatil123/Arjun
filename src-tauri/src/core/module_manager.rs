//! Module manager skeleton

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Represents the status of a module
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleStatus {
    Registered,
    Initializing,
    Ready,
    Error,
    Disabled,
}

/// Information about a registered module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub status: ModuleStatus,
    pub dependencies: Vec<String>,
}

/// Manages dynamic modules in the application
pub struct ModuleManager {
    modules: Arc<Mutex<HashMap<String, ModuleInfo>>>,
}

impl ModuleManager {
    /// Creates a new ModuleManager
    pub fn new() -> Self {
        Self {
            modules: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Registers a new module
    pub fn register_module(&self, info: ModuleInfo) {
        let mut modules = self.modules.lock().unwrap();
        modules.insert(info.id.clone(), info);
    }

    /// Gets information about a specific module
    pub fn get_module(&self, id: &str) -> Option<ModuleInfo> {
        let modules = self.modules.lock().unwrap();
        modules.get(id).cloned()
    }

    /// Lists all registered modules
    pub fn list_modules(&self) -> Vec<ModuleInfo> {
        let modules = self.modules.lock().unwrap();
        modules.values().cloned().collect()
    }

    /// Updates a module's status
    pub fn set_module_status(&self, id: &str, status: ModuleStatus) {
        let mut modules = self.modules.lock().unwrap();
        if let Some(module) = modules.get_mut(id) {
            module.status = status;
        }
    }
}

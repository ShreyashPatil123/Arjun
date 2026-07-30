//! Plugin traits

use anyhow::Result;

pub struct PluginInfo { pub id: String, pub name: String, pub version: String, pub author: String, pub description: String }
pub enum PluginStatus { Loaded, Active, Disabled, Error }

pub trait Plugin: Send + Sync {
    fn info(&self) -> PluginInfo;
    fn init(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn start(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn stop(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn is_running(&self) -> bool { false }
}

pub trait PluginManager: Send + Sync {
    fn load_plugin(&self, _path: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unload_plugin(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn enable_plugin(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn disable_plugin(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_plugins(&self) -> Result<Vec<PluginInfo>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_plugin(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait PluginRegistry: Send + Sync {
    fn register(&self, _plugin: Box<dyn Plugin>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unregister(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list(&self) -> Result<Vec<PluginInfo>> { Err(anyhow::anyhow!("Not yet implemented")) }
}

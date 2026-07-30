//! Installer traits

use anyhow::Result;

pub enum InstallTarget { Ollama, LlamaCpp, CudaToolkit, Model, LoRA, Runtime(String) }
pub enum InstallStatus { Checking, Downloading, Installing, Configuring, Completed, Failed }

pub struct InstallResult { pub target: String, pub success: bool, pub version: Option<String>, pub path: Option<String>, pub error: Option<String> }

pub trait InstallerService: Send + Sync {
    fn check_installed(&self, _target: InstallTarget) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn install(&self, _target: InstallTarget) -> Result<InstallResult> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn uninstall(&self, _target: InstallTarget) -> Result<InstallResult> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_version(&self, _target: InstallTarget) -> Result<String> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn verify_installation(&self, _target: InstallTarget) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
}

//! Manual hardware profile overrides manager

use crate::system_analyzer::normalization::compute_ai_capabilities;
use crate::system_analyzer::traits::{
    AIRuntimeInfo, CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, OsInfo, SoftwareEnvironment,
    StorageInfo,
};
use crate::system_analyzer::validation::validate_profile;
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde_json::Value;

/// Applies a manual override to a target field in the HardwareProfile
pub fn apply_hardware_override(
    profile: &mut HardwareProfile,
    field_path: &str,
    value: Value,
) -> Result<()> {
    match field_path.to_lowercase().as_str() {
        "cpu" => {
            let cpu: CpuInfo = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize CpuInfo override: {}", e))?;
            profile.cpu.overridden = Some(cpu);
            profile.cpu.is_overridden = true;
        }
        "gpus" => {
            let gpus: Vec<GpuInfo> = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize Vec<GpuInfo> override: {}", e))?;
            profile.gpus.overridden = Some(gpus);
            profile.gpus.is_overridden = true;
        }
        "memory" => {
            let memory: MemoryInfo = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize MemoryInfo override: {}", e))?;
            profile.memory.overridden = Some(memory);
            profile.memory.is_overridden = true;
        }
        "storage" => {
            let storage: Vec<StorageInfo> = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize Vec<StorageInfo> override: {}", e))?;
            profile.storage.overridden = Some(storage);
            profile.storage.is_overridden = true;
        }
        "os" => {
            let os: OsInfo = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize OsInfo override: {}", e))?;
            profile.os.overridden = Some(os);
            profile.os.is_overridden = true;
        }
        "software" => {
            let software: SoftwareEnvironment = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize SoftwareEnvironment override: {}", e))?;
            profile.software.overridden = Some(software);
            profile.software.is_overridden = true;
        }
        "ai_runtimes" => {
            let runtimes: Vec<AIRuntimeInfo> = serde_json::from_value(value)
                .map_err(|e| anyhow!("Failed to deserialize Vec<AIRuntimeInfo> override: {}", e))?;
            profile.ai_runtimes.overridden = Some(runtimes);
            profile.ai_runtimes.is_overridden = true;
        }
        _ => return Err(anyhow!("Unknown override field path: {}", field_path)),
    }

    recompute_profile_dependents(profile);
    Ok(())
}

/// Reverts a manual override for the specified field back to detected values
pub fn revert_hardware_override(profile: &mut HardwareProfile, field_path: &str) -> Result<()> {
    match field_path.to_lowercase().as_str() {
        "cpu" => {
            profile.cpu.overridden = None;
            profile.cpu.is_overridden = false;
        }
        "gpus" => {
            profile.gpus.overridden = None;
            profile.gpus.is_overridden = false;
        }
        "memory" => {
            profile.memory.overridden = None;
            profile.memory.is_overridden = false;
        }
        "storage" => {
            profile.storage.overridden = None;
            profile.storage.is_overridden = false;
        }
        "os" => {
            profile.os.overridden = None;
            profile.os.is_overridden = false;
        }
        "software" => {
            profile.software.overridden = None;
            profile.software.is_overridden = false;
        }
        "ai_runtimes" => {
            profile.ai_runtimes.overridden = None;
            profile.ai_runtimes.is_overridden = false;
        }
        _ => return Err(anyhow!("Unknown override field path: {}", field_path)),
    }

    recompute_profile_dependents(profile);
    Ok(())
}

fn recompute_profile_dependents(profile: &mut HardwareProfile) {
    profile.profile_updated_at = Utc::now().to_rfc3339();
    profile.ai_capabilities = compute_ai_capabilities(
        profile.cpu.current(),
        profile.gpus.current(),
        profile.memory.current(),
    );
    profile.validation = validate_profile(profile);
}

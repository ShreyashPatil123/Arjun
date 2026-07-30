//! System Analyzer Module (Phase 2)
//! Responsible for detecting hardware capabilities (CPU, GPU, RAM, OS, Storage, Software, AI Runtimes)
//! and computing AI model readiness profiles.

pub mod ai_runtime_collector;
pub mod cpu_collector;
pub mod gpu_collector;
pub mod memory_collector;
pub mod normalization;
pub mod os_collector;
pub mod overrides;
pub mod path_collector;
pub mod process_utils;
pub mod software_collector;
pub mod storage_collector;
pub mod traits;
pub mod validation;

pub use traits::*;

use crate::core::event_bus::{get_event_bus, SarathiEvent};
use anyhow::{anyhow, Result};
use serde_json::json;
use std::sync::{Arc, Mutex, OnceLock};

/// Central manager for system analysis and hardware profile management
pub struct SystemAnalyzerManager {
    profile: Arc<Mutex<Option<HardwareProfile>>>,
}

impl SystemAnalyzerManager {
    /// Creates a new `SystemAnalyzerManager` instance
    pub fn new() -> Self {
        Self {
            profile: Arc::new(Mutex::new(None)),
        }
    }

    /// Performs full system detection and updates stored profile & publishes events
    pub fn analyze_system(&self) -> Result<HardwareProfile> {
        let event_bus = get_event_bus();
        event_bus.publish(SarathiEvent::SystemAnalysisStarted, None);

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting CPU", "progress": 15 })));
        let cpu = self.detect_cpu()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting GPU", "progress": 30 })));
        let gpus = self.detect_gpus()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting Memory", "progress": 45 })));
        let memory = self.detect_memory()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting Storage", "progress": 60 })));
        let storage = self.detect_storage()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting Operating System", "progress": 75 })));
        let os = self.detect_os()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Detecting Installed Software", "progress": 90 })));
        let software = self.detect_software()?;
        let ai_runtimes = self.detect_ai_runtimes()?;
        let paths = self.detect_paths()?;

        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Building Hardware Profile", "progress": 98 })));
        let profile = self.normalize(cpu, gpus, memory, storage, os, software, ai_runtimes, paths)?;

        {
            let mut lock = self.profile.lock().unwrap();
            *lock = Some(profile.clone());
        }

        let json_val = serde_json::to_value(&profile).ok();
        event_bus.publish(SarathiEvent::SystemAnalysisProgress, Some(json!({ "step": "Complete", "progress": 100 })));
        event_bus.publish(SarathiEvent::SystemAnalysisCompleted, json_val.clone());
        event_bus.publish(SarathiEvent::HardwareProfileUpdated, json_val);

        Ok(profile)
    }

    /// Retrieves a copy of the current cached hardware profile
    pub fn get_profile(&self) -> Option<HardwareProfile> {
        self.profile.lock().unwrap().clone()
    }

    /// Overrides a specific field in the hardware profile
    pub fn override_value(
        &self,
        field_path: &str,
        value: serde_json::Value,
    ) -> Result<HardwareProfile> {
        let mut lock = self.profile.lock().unwrap();
        if let Some(ref mut profile) = *lock {
            overrides::apply_hardware_override(profile, field_path, value)?;
            let updated_profile = profile.clone();

            let json_val = serde_json::to_value(&updated_profile).ok();
            get_event_bus().publish(SarathiEvent::HardwareProfileUpdated, json_val);

            Ok(updated_profile)
        } else {
            Err(anyhow!("No hardware profile available to override"))
        }
    }

    /// Reverts an override on a target field back to detected values
    pub fn revert_override(&self, field_path: &str) -> Result<HardwareProfile> {
        let mut lock = self.profile.lock().unwrap();
        if let Some(ref mut profile) = *lock {
            overrides::revert_hardware_override(profile, field_path)?;
            let updated_profile = profile.clone();

            let json_val = serde_json::to_value(&updated_profile).ok();
            get_event_bus().publish(SarathiEvent::HardwareProfileUpdated, json_val);

            Ok(updated_profile)
        } else {
            Err(anyhow!("No hardware profile available to revert"))
        }
    }
}

impl Default for SystemAnalyzerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemAnalyzer for SystemAnalyzerManager {
    fn collect_all(&self) -> Result<HardwareProfile> {
        let cpu = self.detect_cpu()?;
        let gpus = self.detect_gpus()?;
        let memory = self.detect_memory()?;
        let storage = self.detect_storage()?;
        let os = self.detect_os()?;
        let software = self.detect_software()?;
        let ai_runtimes = self.detect_ai_runtimes()?;
        let paths = self.detect_paths()?;

        self.normalize(cpu, gpus, memory, storage, os, software, ai_runtimes, paths)
    }

    fn detect_cpu(&self) -> Result<CpuInfo> {
        Ok(cpu_collector::detect_cpu())
    }

    fn detect_gpus(&self) -> Result<Vec<GpuInfo>> {
        Ok(gpu_collector::detect_gpus())
    }

    fn detect_memory(&self) -> Result<MemoryInfo> {
        Ok(memory_collector::detect_memory())
    }

    fn detect_storage(&self) -> Result<Vec<StorageInfo>> {
        Ok(storage_collector::detect_storage())
    }

    fn detect_os(&self) -> Result<OsInfo> {
        Ok(os_collector::detect_os())
    }

    fn detect_software(&self) -> Result<SoftwareEnvironment> {
        Ok(software_collector::detect_software())
    }

    fn detect_ai_runtimes(&self) -> Result<Vec<AIRuntimeInfo>> {
        Ok(ai_runtime_collector::detect_ai_runtimes())
    }

    fn detect_paths(&self) -> Result<SystemPaths> {
        Ok(path_collector::detect_paths())
    }

    fn normalize(
        &self,
        raw_cpu: CpuInfo,
        raw_gpus: Vec<GpuInfo>,
        raw_memory: MemoryInfo,
        raw_storage: Vec<StorageInfo>,
        raw_os: OsInfo,
        raw_software: SoftwareEnvironment,
        raw_ai_runtimes: Vec<AIRuntimeInfo>,
        raw_paths: SystemPaths,
    ) -> Result<HardwareProfile> {
        Ok(normalization::normalize_profile(
            raw_cpu,
            raw_gpus,
            raw_memory,
            raw_storage,
            raw_os,
            raw_software,
            raw_ai_runtimes,
            raw_paths,
        ))
    }

    fn validate(&self, profile: &HardwareProfile) -> Result<SystemValidationResult> {
        Ok(validation::validate_profile(profile))
    }
}

static SYSTEM_ANALYZER_MANAGER: OnceLock<SystemAnalyzerManager> = OnceLock::new();

/// Returns a static reference to the global `SystemAnalyzerManager` instance
pub fn get_system_analyzer_manager() -> &'static SystemAnalyzerManager {
    SYSTEM_ANALYZER_MANAGER.get_or_init(SystemAnalyzerManager::new)
}

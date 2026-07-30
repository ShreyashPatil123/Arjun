//! System analyzer traits

use anyhow::Result;

pub struct CpuInfo { pub name: String, pub cores: u32, pub threads: u32, pub frequency: f32, pub architecture: String }
pub struct GpuInfo { pub name: String, pub vendor: String, pub vram_bytes: u64, pub cuda_cores: Option<u32>, pub driver_version: Option<String>, pub cuda_version: Option<String> }
pub struct MemoryInfo { pub total_bytes: u64, pub available_bytes: u64, pub used_bytes: u64 }
pub struct StorageInfo { pub total_bytes: u64, pub available_bytes: u64, pub path: String }
pub struct CudaInfo { pub available: bool, pub version: Option<String>, pub devices_count: u32 }
pub struct OsInfo { pub name: String, pub version: String, pub arch: String }
pub struct NetworkInfo { pub connected: bool, pub speed_mbps: Option<f32> }
pub struct HardwareProfile {
    pub cpu: CpuInfo, pub gpus: Vec<GpuInfo>, pub memory: MemoryInfo, pub storage: StorageInfo,
    pub cuda: CudaInfo, pub os: OsInfo, pub network: NetworkInfo, pub profile_created_at: String,
}

pub trait SystemAnalyzer: Send + Sync {
    fn detect_cpu(&self) -> Result<CpuInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_gpus(&self) -> Result<Vec<GpuInfo>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_memory(&self) -> Result<MemoryInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_storage(&self, _path: &str) -> Result<StorageInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_cuda(&self) -> Result<CudaInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_os(&self) -> Result<OsInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn detect_network(&self) -> Result<NetworkInfo> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn create_hardware_profile(&self) -> Result<HardwareProfile> { Err(anyhow::anyhow!("Not yet implemented")) }
}

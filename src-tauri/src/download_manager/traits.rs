//! Download manager traits

use anyhow::Result;

pub enum DownloadStatus { Queued, Downloading, Paused, Completed, Failed, Cancelled, Verifying }

pub struct DownloadTask {
    pub id: String, pub url: String, pub destination: String, pub total_bytes: u64,
    pub downloaded_bytes: u64, pub status: DownloadStatus, pub checksum: Option<String>,
    pub error: Option<String>, pub created_at: String, pub updated_at: String,
}

pub struct DownloadProgress {
    pub task_id: String, pub downloaded_bytes: u64, pub total_bytes: u64,
    pub speed_bps: f64, pub eta_seconds: Option<u64>,
}

pub trait DownloadManager: Send + Sync {
    fn enqueue(&self, _url: &str, _destination: &str) -> Result<String> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn pause(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn resume(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn cancel(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn retry(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_progress(&self, _id: &str) -> Result<DownloadProgress> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_task(&self, _id: &str) -> Result<DownloadTask> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_tasks(&self) -> Result<Vec<DownloadTask>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn clear_completed(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

//! Resumable base-model download manager.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use tauri::Emitter;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;

use crate::download_manager::traits::*;
use crate::model_package::{BaseManifestInfo, ModelPackageManifest, ModelPackageRegistry};
use crate::model_providers::huggingface::resolver;

const MAX_DOWNLOAD_ATTEMPTS: u32 = 5;

enum StreamOutcome {
    Finished { downloaded: u64, expected: u64 },
    Interrupted,
}

enum StreamError {
    Transient(anyhow::Error),
    Fatal(anyhow::Error),
}

impl StreamError {
    fn transient(error: impl Into<anyhow::Error>) -> Self {
        Self::Transient(error.into())
    }
}

fn content_range_start(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .strip_prefix("bytes ")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

pub struct DownloadManager {
    tasks: Arc<Mutex<HashMap<String, DownloadTask>>>,
    cancel_senders: Arc<Mutex<HashMap<String, watch::Sender<bool>>>>,
}

impl DownloadManager {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            cancel_senders: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn check_disk_space(target_path: &Path, required_bytes: u64) -> Result<()> {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let available = disks
            .iter()
            .filter(|disk| target_path.starts_with(disk.mount_point()))
            .max_by_key(|disk| disk.mount_point().as_os_str().len())
            .map(|disk| disk.available_space())
            .unwrap_or(u64::MAX);

        if available < required_bytes {
            return Err(anyhow!(
                "Insufficient disk space: need {} bytes, have {} bytes",
                required_bytes,
                available
            ));
        }
        Ok(())
    }

    pub fn get_model_storage_dir(
        app_data_dir: &Path,
        provider_id: &str,
        model_id: &str,
        _quantization: &str,
    ) -> PathBuf {
        ModelPackageRegistry::resolve_package_dir(app_data_dir, provider_id, model_id).join("base")
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn start_download(
        &self,
        app_handle: tauri::AppHandle,
        app_data_dir: PathBuf,
        model_id: String,
        model_name: String,
        provider_id: String,
        quantization: String,
        format: String,
        backend: String,
        hf_token: Option<String>,
    ) -> Result<String> {
        let task_id = format!(
            "dl_{}_{}",
            model_id.replace('/', "_"),
            quantization.to_lowercase()
        );

        if self.tasks.lock().unwrap().get(&task_id).is_some_and(|task| {
            matches!(
                task.status,
                DownloadStatus::Downloading
                    | DownloadStatus::Resolving
                    | DownloadStatus::Verifying
                    | DownloadStatus::Queued
            )
        }) {
            return Ok(task_id);
        }

        let storage_dir = Self::get_model_storage_dir(
            &app_data_dir,
            &provider_id,
            &model_id,
            &quantization,
        );
        tokio::fs::create_dir_all(&storage_dir).await?;

        let mut task = DownloadTask {
            id: task_id.clone(),
            model_id: model_id.clone(),
            model_name: model_name.clone(),
            provider_id: provider_id.clone(),
            quantization: quantization.clone(),
            format,
            backend,
            url: String::new(),
            destination_path: String::new(),
            temp_path: String::new(),
            total_bytes: 0,
            downloaded_bytes: 0,
            status: DownloadStatus::Resolving,
            speed_bps: 0.0,
            eta_seconds: None,
            checksum: None,
            error: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        self.tasks
            .lock()
            .unwrap()
            .insert(task_id.clone(), task.clone());
        self.broadcast_progress(&app_handle, &task);

        let artifact = resolver::resolve_artifact(&model_id, &quantization, hf_token.as_deref())
            .await
            .map_err(|error| {
                self.fail_task(&app_handle, &task_id, error.to_string());
                error
            })?;
        let file_name = artifact
            .file_name
            .rsplit('/')
            .next()
            .unwrap_or(&artifact.file_name)
            .to_string();
        let destination_path = storage_dir.join(&file_name);
        let temp_path = storage_dir.join(format!("{file_name}.part"));

        if destination_path.is_file() {
            let size = tokio::fs::metadata(&destination_path).await?.len();
            if artifact.size_bytes > 0 && size == artifact.size_bytes {
                Self::write_package_manifest(
                    &app_data_dir,
                    &provider_id,
                    &model_id,
                    &model_name,
                    &quantization,
                    &file_name,
                    size,
                    artifact.sha256.clone(),
                )?;
                task.url = artifact.download_url;
                task.destination_path = destination_path.to_string_lossy().to_string();
                task.temp_path = temp_path.to_string_lossy().to_string();
                task.total_bytes = size;
                task.downloaded_bytes = size;
                task.status = DownloadStatus::Completed;
                task.eta_seconds = Some(0);
                self.tasks
                    .lock()
                    .unwrap()
                    .insert(task_id.clone(), task.clone());
                self.broadcast_progress(&app_handle, &task);
                return Ok(task_id);
            }
        }

        let existing_bytes = tokio::fs::metadata(&temp_path)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        Self::check_disk_space(
            &storage_dir,
            artifact.size_bytes.saturating_sub(existing_bytes),
        )?;

        task.url = artifact.download_url.clone();
        task.destination_path = destination_path.to_string_lossy().to_string();
        task.temp_path = temp_path.to_string_lossy().to_string();
        task.total_bytes = artifact.size_bytes;
        task.downloaded_bytes = existing_bytes;
        task.status = DownloadStatus::Queued;
        task.checksum = artifact.sha256.clone();
        task.updated_at = chrono::Utc::now().to_rfc3339();
        self.tasks
            .lock()
            .unwrap()
            .insert(task_id.clone(), task.clone());
        self.broadcast_progress(&app_handle, &task);

        let (cancel_tx, cancel_rx) = watch::channel(false);
        self.cancel_senders
            .lock()
            .unwrap()
            .insert(task_id.clone(), cancel_tx);

        let tasks = self.tasks.clone();
        let cancel_senders = self.cancel_senders.clone();
        let spawned_task_id = task_id.clone();
        tokio::spawn(async move {
            let result = Self::run_download_loop(
                &app_handle,
                &spawned_task_id,
                &artifact.download_url,
                &temp_path,
                &destination_path,
                artifact.size_bytes,
                existing_bytes,
                hf_token,
                artifact.sha256.clone(),
                cancel_rx,
                tasks.clone(),
            )
            .await;

            if result.is_ok() && destination_path.is_file() {
                let size = tokio::fs::metadata(&destination_path)
                    .await
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                if let Err(error) = Self::write_package_manifest(
                    &app_data_dir,
                    &provider_id,
                    &model_id,
                    &model_name,
                    &quantization,
                    &file_name,
                    size,
                    artifact.sha256,
                ) {
                    Self::set_failed(&app_handle, &tasks, &spawned_task_id, error.to_string());
                }
            } else if let Err(error) = result {
                Self::set_failed(&app_handle, &tasks, &spawned_task_id, error.to_string());
            }

            cancel_senders.lock().unwrap().remove(&spawned_task_id);
        });

        Ok(task_id)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_download_loop(
        app_handle: &tauri::AppHandle,
        task_id: &str,
        url: &str,
        temp_path: &Path,
        destination_path: &Path,
        expected_total_bytes: u64,
        initial_bytes: u64,
        hf_token: Option<String>,
        expected_sha256: Option<String>,
        cancel_rx: watch::Receiver<bool>,
        tasks: Arc<Mutex<HashMap<String, DownloadTask>>>,
    ) -> Result<()> {
        let client = crate::sovereignty::global_broker().authorize(url)?;
        let mut expected_total = expected_total_bytes;
        let mut resume_from = initial_bytes;

        let downloaded = 'attempts: loop {
            let mut last_error = None;
            for attempt in 1..=MAX_DOWNLOAD_ATTEMPTS {
                match Self::stream_once(
                    app_handle,
                    &client,
                    task_id,
                    url,
                    temp_path,
                    expected_total,
                    resume_from,
                    hf_token.as_deref(),
                    &cancel_rx,
                    &tasks,
                )
                .await
                {
                    Ok(StreamOutcome::Interrupted) => return Ok(()),
                    Ok(StreamOutcome::Finished {
                        downloaded,
                        expected,
                    }) => {
                        if expected > 0 {
                            expected_total = expected;
                        }
                        break 'attempts downloaded;
                    }
                    Err(StreamError::Fatal(error)) => return Err(error),
                    Err(StreamError::Transient(error)) => {
                        last_error = Some(error);
                        if attempt < MAX_DOWNLOAD_ATTEMPTS {
                            resume_from = tokio::fs::metadata(temp_path)
                                .await
                                .map(|metadata| metadata.len())
                                .unwrap_or(0);
                            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt.min(4))))
                                .await;
                        }
                    }
                }
            }
            return Err(anyhow!(
                "Download failed after {} attempts: {}",
                MAX_DOWNLOAD_ATTEMPTS,
                last_error
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "unknown transport failure".to_string())
            ));
        };

        {
            if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
                task.downloaded_bytes = downloaded;
                task.status = DownloadStatus::Verifying;
                task.speed_bps = 0.0;
                task.updated_at = chrono::Utc::now().to_rfc3339();
                let _ = app_handle.emit("download:progress", Self::make_payload(task));
            }

            let final_size = tokio::fs::metadata(temp_path).await?.len();
            if expected_total > 0 && final_size != expected_total {
                return Err(anyhow!(
                    "Integrity verification failed: expected {} bytes, got {}",
                    expected_total,
                    final_size
                ));
            }
            if expected_total == 0 && expected_sha256.is_none() {
                return Err(anyhow!(
                    "Cannot verify download because the server reported neither size nor checksum"
                ));
            }
            if let Some(expected_hash) = expected_sha256 {
                let actual_hash = Self::hash_file_sha256(temp_path).await?;
                if !actual_hash.eq_ignore_ascii_case(&expected_hash) {
                    let _ = tokio::fs::remove_file(temp_path).await;
                    return Err(anyhow!(
                        "Checksum verification failed: expected {}, got {}",
                        expected_hash,
                        actual_hash
                    ));
                }
            }

            if destination_path.exists() {
                tokio::fs::remove_file(destination_path).await?;
            }
            tokio::fs::rename(temp_path, destination_path).await?;
            if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
                task.downloaded_bytes = final_size;
                task.total_bytes = final_size;
                task.status = DownloadStatus::Completed;
                task.speed_bps = 0.0;
                task.eta_seconds = Some(0);
                task.updated_at = chrono::Utc::now().to_rfc3339();
                let _ = app_handle.emit("download:progress", Self::make_payload(task));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_once(
        app_handle: &tauri::AppHandle,
        client: &reqwest::Client,
        task_id: &str,
        url: &str,
        temp_path: &Path,
        expected_total_bytes: u64,
        resume_from: u64,
        hf_token: Option<&str>,
        cancel_rx: &watch::Receiver<bool>,
        tasks: &Arc<Mutex<HashMap<String, DownloadTask>>>,
    ) -> std::result::Result<StreamOutcome, StreamError> {
        let mut request = client.get(url);
        if let Some(token) = hf_token.filter(|token| !token.trim().is_empty()) {
            request = request.header("Authorization", format!("Bearer {}", token.trim()));
        }
        if resume_from > 0 {
            request = request.header("Range", format!("bytes={resume_from}-"));
        }

        let response = request.send().await.map_err(StreamError::transient)?;
        let status = response.status();
        if status.is_client_error() {
            return Err(StreamError::Fatal(anyhow!("HTTP error response: {status}")));
        }
        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(StreamError::transient(anyhow!(
                "HTTP error response: {status}"
            )));
        }

        let range_was_honoured = status == reqwest::StatusCode::PARTIAL_CONTENT
            && content_range_start(response.headers()) == Some(resume_from);
        let start_offset = if resume_from > 0 && !range_was_honoured {
            0
        } else {
            resume_from
        };
        let expected_total = response
            .content_length()
            .map(|length| start_offset + length)
            .unwrap_or(expected_total_bytes);

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(start_offset > 0)
            .truncate(start_offset == 0)
            .open(temp_path)
            .await
            .map_err(|error| StreamError::Fatal(error.into()))?;
        let mut stream = response.bytes_stream();
        let mut downloaded = start_offset;
        let mut sample_time = Instant::now();
        let mut sample_bytes = downloaded;

        while let Some(chunk) = stream.next().await {
            if *cancel_rx.borrow() {
                let _ = file.flush().await;
                return Ok(StreamOutcome::Interrupted);
            }
            let chunk = chunk.map_err(StreamError::transient)?;
            file.write_all(&chunk)
                .await
                .map_err(|error| StreamError::Fatal(error.into()))?;
            downloaded += chunk.len() as u64;

            let now = Instant::now();
            let elapsed = now.duration_since(sample_time).as_secs_f64();
            if elapsed >= 0.25 {
                let speed = downloaded.saturating_sub(sample_bytes) as f64 / elapsed;
                if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
                    task.downloaded_bytes = downloaded;
                    task.total_bytes = expected_total;
                    task.speed_bps = speed;
                    task.eta_seconds = (speed > 0.0)
                        .then(|| (expected_total.saturating_sub(downloaded) as f64 / speed) as u64);
                    task.status = DownloadStatus::Downloading;
                    task.updated_at = chrono::Utc::now().to_rfc3339();
                    let _ = app_handle.emit("download:progress", Self::make_payload(task));
                }
                sample_time = now;
                sample_bytes = downloaded;
            }
        }

        file.flush()
            .await
            .map_err(|error| StreamError::Fatal(error.into()))?;
        if expected_total > 0 && downloaded < expected_total {
            return Err(StreamError::transient(anyhow!(
                "Connection closed after {} of {} bytes",
                downloaded,
                expected_total
            )));
        }
        Ok(StreamOutcome::Finished {
            downloaded,
            expected: expected_total,
        })
    }

    async fn hash_file_sha256(path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};
        let mut file = tokio::fs::File::open(path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    #[allow(clippy::too_many_arguments)]
    fn write_package_manifest(
        app_data_dir: &Path,
        provider_id: &str,
        model_id: &str,
        model_name: &str,
        quantization: &str,
        file_name: &str,
        size_bytes: u64,
        checksum: Option<String>,
    ) -> Result<()> {
        let package_dir =
            ModelPackageRegistry::resolve_package_dir(app_data_dir, provider_id, model_id);
        let previous = ModelPackageRegistry::read_manifest(&package_dir).ok();
        let timestamp = chrono::Utc::now().to_rfc3339();
        let manifest = ModelPackageManifest {
            package_id: format!("{model_id}::{quantization}::llama.cpp"),
            provider_id: provider_id.to_string(),
            base_model: BaseManifestInfo {
                model_id: model_id.to_string(),
                model_name: model_name.to_string(),
                quantization: quantization.to_string(),
                file_path: format!("base/{file_name}"),
                size_bytes,
                checksum,
            },
            created_at: previous
                .map(|manifest| manifest.created_at)
                .unwrap_or_else(|| timestamp.clone()),
            updated_at: timestamp,
        };
        ModelPackageRegistry::write_manifest(&package_dir, &manifest)?;
        let _ = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(
            &package_dir,
            &manifest,
        );
        Ok(())
    }

    pub fn pause_download(&self, task_id: &str) -> Result<()> {
        if let Some(sender) = self.cancel_senders.lock().unwrap().get(task_id) {
            let _ = sender.send(true);
        }
        if let Some(task) = self.tasks.lock().unwrap().get_mut(task_id) {
            task.status = DownloadStatus::Paused;
            task.speed_bps = 0.0;
            task.updated_at = chrono::Utc::now().to_rfc3339();
        }
        Ok(())
    }

    pub fn cancel_download(&self, app_handle: &tauri::AppHandle, task_id: &str) -> Result<()> {
        if let Some(sender) = self.cancel_senders.lock().unwrap().remove(task_id) {
            let _ = sender.send(true);
        }
        if let Some(mut task) = self.tasks.lock().unwrap().remove(task_id) {
            task.status = DownloadStatus::Cancelled;
            self.broadcast_progress(app_handle, &task);
            let temp_path = PathBuf::from(task.temp_path);
            tokio::spawn(async move {
                for _ in 0..10 {
                    if !temp_path.exists() || tokio::fs::remove_file(&temp_path).await.is_ok() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            });
        }
        Ok(())
    }

    pub async fn resume_download(
        &self,
        app_handle: tauri::AppHandle,
        app_data_dir: PathBuf,
        task_id: &str,
        hf_token: Option<String>,
    ) -> Result<String> {
        let task = self
            .get_task(task_id)
            .ok_or_else(|| anyhow!("No download task named '{task_id}' to resume"))?;
        self.start_download(
            app_handle,
            app_data_dir,
            task.model_id,
            task.model_name,
            task.provider_id,
            task.quantization,
            task.format,
            task.backend,
            hf_token,
        )
        .await
    }

    pub fn get_task(&self, task_id: &str) -> Option<DownloadTask> {
        self.tasks.lock().unwrap().get(task_id).cloned()
    }

    pub fn list_tasks(&self) -> Vec<DownloadTask> {
        self.tasks.lock().unwrap().values().cloned().collect()
    }

    fn broadcast_progress(&self, app_handle: &tauri::AppHandle, task: &DownloadTask) {
        let _ = app_handle.emit("download:progress", Self::make_payload(task));
    }

    fn fail_task(&self, app_handle: &tauri::AppHandle, task_id: &str, error: String) {
        Self::set_failed(app_handle, &self.tasks, task_id, error);
    }

    fn set_failed(
        app_handle: &tauri::AppHandle,
        tasks: &Arc<Mutex<HashMap<String, DownloadTask>>>,
        task_id: &str,
        error: String,
    ) {
        if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
            task.status = DownloadStatus::Failed;
            task.speed_bps = 0.0;
            task.error = Some(error);
            task.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = app_handle.emit("download:progress", Self::make_payload(task));
        }
    }

    fn make_payload(task: &DownloadTask) -> DownloadProgressPayload {
        let speed = if task.speed_bps.is_finite() && task.speed_bps > 0.0 {
            task.speed_bps
        } else {
            0.0
        };
        let progress_percent = if task.total_bytes > 0 {
            ((task.downloaded_bytes as f64 / task.total_bytes as f64) * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let speed_formatted = match task.status {
            DownloadStatus::Resolving => "Resolving...".to_string(),
            DownloadStatus::Verifying => "Verifying integrity...".to_string(),
            DownloadStatus::Queued => "Starting...".to_string(),
            DownloadStatus::Paused => "Paused".to_string(),
            DownloadStatus::Completed => "Completed".to_string(),
            DownloadStatus::Failed => "Failed".to_string(),
            DownloadStatus::Cancelled => "Cancelled".to_string(),
            DownloadStatus::Downloading if speed >= 1_048_576.0 => {
                format!("{:.1} MB/s", speed / 1_048_576.0)
            }
            DownloadStatus::Downloading if speed >= 1024.0 => {
                format!("{:.0} KB/s", speed / 1024.0)
            }
            DownloadStatus::Downloading => format!("{speed:.0} B/s"),
        };
        DownloadProgressPayload {
            task_id: task.id.clone(),
            model_id: task.model_id.clone(),
            quantization: task.quantization.clone(),
            downloaded_bytes: task.downloaded_bytes,
            total_bytes: task.total_bytes,
            progress_percent,
            speed_bps: speed,
            speed_formatted,
            eta_seconds: task.eta_seconds,
            status: task.status.clone(),
            error: task.error.clone(),
            package_id: Some(task.model_id.replace('/', "_")),
        }
    }
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_start_reads_offset() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 1024-2047/4096".parse().unwrap(),
        );
        assert_eq!(content_range_start(&headers), Some(1024));
    }

    #[test]
    fn storage_path_contains_only_model_package_and_base_directory() {
        let path = DownloadManager::get_model_storage_dir(
            Path::new("C:/data"),
            "huggingface",
            "owner/model",
            "Q4_0",
        );
        assert_eq!(
            path,
            Path::new("C:/data/models/huggingface/owner_model/base")
        );
    }

    #[tokio::test]
    async fn hashes_file_in_streaming_blocks() {
        let path = std::env::temp_dir().join(format!(
            "arjun-download-hash-{}.bin",
            std::process::id()
        ));
        tokio::fs::write(&path, b"abc").await.unwrap();
        let digest = DownloadManager::hash_file_sha256(&path).await.unwrap();
        let _ = tokio::fs::remove_file(path).await;
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

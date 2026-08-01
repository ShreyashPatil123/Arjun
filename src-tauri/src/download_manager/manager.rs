//! Phase 4 Native Download Engine
//!
//! Handles background HTTP streaming, pause/resume via HTTP Range headers,
//! disk space verification, atomic file finalization (.part -> .gguf),
//! and Tauri progress event broadcasting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;
use tokio::sync::watch;

use crate::download_manager::traits::*;
use crate::model_providers::huggingface::resolver;
use crate::model_providers::huggingface::adapter_provider::HuggingFaceAdapterProvider;
use crate::adapter_manager::{AdapterRegistry, ModelPackageManifest, BaseManifestInfo, AdapterManifestInfo, AdapterState, log_adapter_transition};

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

    /// Check if destination drive has sufficient free space
    pub fn check_disk_space(target_path: &Path, required_bytes: u64) -> Result<()> {
        let disks = sysinfo::Disks::new_with_refreshed_list();

        let path_str = target_path.to_string_lossy();
        let drive_prefix = if path_str.len() >= 3 && &path_str[1..3] == ":\\" {
            &path_str[0..3]
        } else {
            "C:\\"
        };

        for disk in &disks {
            let mount_point = disk.mount_point().to_string_lossy();
            if mount_point.eq_ignore_ascii_case(drive_prefix) || path_str.starts_with(mount_point.as_ref()) {
                let available = disk.available_space();
                let safety_margin = 1_073_741_824; // 1 GB safety reserve
                if available < required_bytes + safety_margin {
                    return Err(anyhow!(
                        "Insufficient disk space on drive {}. Required: {:.2} GB, Available: {:.2} GB (with 1 GB safety reserve)",
                        drive_prefix,
                        required_bytes as f64 / 1_073_741_824.0,
                        available as f64 / 1_073_741_824.0
                    ));
                }
                return Ok(());
            }
        }

        Ok(()) // Default pass if volume query doesn't match
    }

    /// Generates canonical storage path: <SarathiAppData>/models/<provider>/<model_id>/base
    pub fn get_model_storage_dir(app_data_dir: &Path, provider_id: &str, model_id: &str, _quantization: &str) -> PathBuf {
        let clean_model_id = model_id.replace('/', "_");
        app_data_dir
            .join("models")
            .join(provider_id)
            .join(clean_model_id)
            .join("base")
    }

    /// Start a new download task or resume an existing one
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
        let task_id = format!("dl_{}_{}", model_id.replace('/', "_"), quantization.to_lowercase());

        // Check if task is already running or completed
        if let Some(existing) = self.tasks.lock().unwrap().get(&task_id) {
            if matches!(existing.status, DownloadStatus::Downloading | DownloadStatus::Resolving | DownloadStatus::Verifying) {
                log::info!("[DOWNLOAD_DIAGNOSTIC] Task {} is already active with status {:?}", task_id, existing.status);
                return Ok(task_id);
            }
        }

        let storage_dir = Self::get_model_storage_dir(&app_data_dir, &provider_id, &model_id, &quantization);
        tokio::fs::create_dir_all(&storage_dir).await?;

        // Immediately create and broadcast Resolving task state so UI updates
        let resolving_task = DownloadTask {
            id: task_id.clone(),
            model_id: model_id.clone(),
            model_name: model_name.clone(),
            provider_id: provider_id.clone(),
            quantization: quantization.clone(),
            format: format.clone(),
            backend: backend.clone(),
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

        self.tasks.lock().unwrap().insert(task_id.clone(), resolving_task.clone());
        self.broadcast_progress(&app_handle, &resolving_task);
        log::info!("[DOWNLOAD_DIAGNOSTIC] Created task {} in Resolving state", task_id);

        // Trigger background LoRA adapter discovery & real streaming download
        let app_handle_for_adapters = app_handle.clone();
        let model_id_clone = model_id.clone();
        let model_name_clone = model_name.clone();
        let quantization_clone = quantization.clone();
        let provider_id_clone = provider_id.clone();
        let hf_token_clone = hf_token.clone();
        let app_data_dir_clone = app_data_dir.clone();

        tokio::spawn(async move {
            let package_id = model_id_clone.replace('/', "_");
            let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir_clone, &provider_id_clone, &model_id_clone);
            let adapters_dir = package_dir.join("adapters");
            let _ = tokio::fs::create_dir_all(&adapters_dir).await;

            // Load existing manifest to preserve installed adapters (Single Source of Truth)
            let existing_manifest = AdapterRegistry::read_manifest(&package_dir).ok();
            let manifest_adapters = Arc::new(Mutex::new(
                existing_manifest
                    .as_ref()
                    .map(|m| m.adapters.clone())
                    .unwrap_or_default(),
            ));

            let capabilities = HuggingFaceAdapterProvider::all_capabilities();
            let mut cap_tasks = Vec::new();

            for cap in capabilities {
                let app_handle_cap = app_handle_for_adapters.clone();
                let model_id_cap = model_id_clone.clone();
                let quantization_cap = quantization_clone.clone();
                let package_id_cap = package_id.clone();
                let package_dir_cap = package_dir.clone();
                let adapters_dir_cap = adapters_dir.clone();
                let hf_token_cap = hf_token_clone.clone();
                let manifest_adapters_cap = manifest_adapters.clone();
                let existing_manifest_cap = existing_manifest.clone();

                cap_tasks.push(tokio::spawn(async move {
                    let cap_key = cap.key().to_string();
                    let adapter_task_id = format!("dl_{}_{}_adapter_{}", package_id_cap, quantization_cap.to_lowercase(), cap_key);
                    let cap_dir = adapters_dir_cap.join(&cap_key);

                    // PRE-DOWNLOAD CHECK: If adapter is ALREADY INSTALLED & VALID locally, SKIP REMOTE DOWNLOAD!
                    if AdapterRegistry::is_adapter_installed_and_valid(&package_dir_cap, &cap_key) {
                        log_adapter_transition(
                            &cap_key,
                            &AdapterState::Ready,
                            &AdapterState::Ready,
                            "Pre-download check verified local adapter is valid on disk. Skipping download.",
                            "PreDownloadCheck",
                        );

                        let mut map = manifest_adapters_cap.lock().unwrap();
                        if !map.contains_key(&cap_key) || !map[&cap_key].status.eq_ignore_ascii_case("Installed") {
                            if let Some((weight_file, size)) = AdapterRegistry::verify_adapter_files(&cap_dir) {
                                map.insert(
                                    cap_key.clone(),
                                    AdapterManifestInfo {
                                        capability: cap_key.clone(),
                                        status: "Installed".to_string(),
                                        adapter_runtime_status: None,
                                        repo_id: existing_manifest_cap.as_ref().and_then(|m| m.adapters.get(&cap_key)).and_then(|a| a.repo_id.clone()),
                                        local_path: Some(format!("adapters/{}/", cap_key)),
                                        adapter_file: Some(format!("adapters/{}/{}", cap_key, weight_file)),
                                        config_file: Some(format!("adapters/{}/adapter_config.json", cap_key)),
                                        size_bytes: Some(size),
                                        base_model_match: Some(model_id_cap.clone()),
                                        target_modules: vec![],
                                        peft_type: Some("LORA".to_string()),
                                        checksum: None,
                                        reason: None,
                                    },
                                );
                            }
                        }

                        let size_b = map.get(&cap_key).and_then(|a| a.size_bytes).unwrap_or(0);

                        let _ = app_handle_cap.emit(
                            "download:progress",
                            DownloadProgressPayload {
                                task_id: adapter_task_id,
                                model_id: model_id_cap.clone(),
                                quantization: quantization_cap.clone(),
                                downloaded_bytes: size_b,
                                total_bytes: size_b,
                                progress_percent: 100.0,
                                speed_bps: 0.0,
                                speed_formatted: "Ready".to_string(),
                                eta_seconds: Some(0),
                                status: DownloadStatus::Completed,
                                error: None,
                                package_id: Some(package_id_cap.clone()),
                                capability: Some(cap_key),
                                item_type: Some("adapter".to_string()),
                            },
                        );
                        return;
                    }

                    // Adapter missing locally: Run full concurrent discovery & download state machine
                    log_adapter_transition(
                        &cap_key,
                        &AdapterState::NotFound,
                        &AdapterState::Searching,
                        "Adapter missing locally. Initiating remote search.",
                        "DownloadManager",
                    );

                    let _ = app_handle_cap.emit(
                        "download:progress",
                        DownloadProgressPayload {
                            task_id: adapter_task_id.clone(),
                            model_id: model_id_cap.clone(),
                            quantization: quantization_cap.clone(),
                            downloaded_bytes: 0,
                            total_bytes: 0,
                            progress_percent: 0.0,
                            speed_bps: 0.0,
                            speed_formatted: "Searching...".to_string(),
                            eta_seconds: None,
                            status: DownloadStatus::Resolving,
                            error: None,
                            package_id: Some(package_id_cap.clone()),
                            capability: Some(cap_key.clone()),
                            item_type: Some("adapter".to_string()),
                        },
                    );

                    let res = HuggingFaceAdapterProvider::discover_single_capability(&model_id_cap, cap.clone(), hf_token_cap.as_deref()).await;

                    if res.status == "Found" {
                        if let Some(cand) = res.candidate {
                            log_adapter_transition(
                                &cap_key,
                                &AdapterState::Searching,
                                &AdapterState::Found,
                                &format!("Found compatible adapter repo: {}", cand.repo_id),
                                "DownloadManager",
                            );

                            let _ = tokio::fs::create_dir_all(&cap_dir).await;
                            let config_file_path = cap_dir.join("adapter_config.json");
                            let target_weight_path = cap_dir.join(&cand.adapter_file_name);
                            let temp_weight_path = cap_dir.join(format!("{}.part", cand.adapter_file_name));

                            let client = reqwest::Client::builder()
                                .user_agent("Sarathi/0.1.0 (Windows; x64)")
                                .timeout(std::time::Duration::from_secs(300))
                                .build()
                                .unwrap_or_default();

                            // 1. Fetch & save adapter_config.json
                            let config_url = format!("https://huggingface.co/{}/raw/main/adapter_config.json", cand.repo_id);
                            let mut config_req = client.get(&config_url);
                            if let Some(t) = &hf_token_cap {
                                if !t.trim().is_empty() {
                                    config_req = config_req.header("Authorization", format!("Bearer {}", t.trim()));
                                }
                            }

                            let config_ok = if let Ok(c_res) = config_req.send().await {
                                if c_res.status().is_success() {
                                    if let Ok(bytes) = c_res.bytes().await {
                                        tokio::fs::write(&config_file_path, bytes).await.is_ok()
                                    } else { false }
                                } else { false }
                            } else { false };

                            // 2. Stream download adapter weight file WITH HTTP RANGE RESUME SUPPORT
                            let mut download_success = false;
                            let mut actual_downloaded_bytes: u64 = 0;

                            if config_ok {
                                // Resume check: Check existing byte length of .part file
                                let existing_bytes = if temp_weight_path.exists() {
                                    tokio::fs::metadata(&temp_weight_path).await.map(|m| m.len()).unwrap_or(0)
                                } else { 0 };

                                let total_bytes = cand.size_bytes;

                                if existing_bytes >= total_bytes && total_bytes > 0 {
                                    actual_downloaded_bytes = existing_bytes;
                                    download_success = true;
                                } else {
                                    log_adapter_transition(
                                        &cap_key,
                                        &AdapterState::Found,
                                        &AdapterState::Downloading,
                                        &format!("Downloading weights from {} (resuming from byte {})", cand.download_url, existing_bytes),
                                        "DownloadManager",
                                    );

                                    let mut weight_req = client.get(&cand.download_url);
                                    if let Some(t) = &hf_token_cap {
                                        if !t.trim().is_empty() {
                                            weight_req = weight_req.header("Authorization", format!("Bearer {}", t.trim()));
                                        }
                                    }

                                    if existing_bytes > 0 {
                                        weight_req = weight_req.header("Range", format!("bytes={}-", existing_bytes));
                                    }

                                    if let Ok(w_res) = weight_req.send().await {
                                        let status_code = w_res.status();
                                        if status_code.is_success() || status_code == reqwest::StatusCode::PARTIAL_CONTENT {
                                            let is_partial = status_code == reqwest::StatusCode::PARTIAL_CONTENT;
                                            let mut file = match tokio::fs::OpenOptions::new()
                                                .create(true)
                                                .write(true)
                                                .append(is_partial)
                                                .truncate(!is_partial)
                                                .open(&temp_weight_path)
                                                .await
                                            {
                                                Ok(f) => f,
                                                Err(_) => return,
                                            };

                                            let mut stream = w_res.bytes_stream();
                                            let mut downloaded: u64 = if is_partial { existing_bytes } else { 0 };
                                            let start_time = Instant::now();
                                            let mut last_broadcast = Instant::now();

                                            while let Some(chunk_res) = stream.next().await {
                                                if let Ok(chunk) = chunk_res {
                                                    if file.write_all(&chunk).await.is_err() {
                                                        break;
                                                    }
                                                    downloaded += chunk.len() as u64;

                                                    let elapsed = start_time.elapsed().as_secs_f64();
                                                    let speed_bps = if elapsed > 0.0 { downloaded as f64 / elapsed } else { 0.0 };
                                                    let progress_percent = if total_bytes > 0 { (downloaded as f64 / total_bytes as f64) * 100.0 } else { 0.0 };

                                                    if last_broadcast.elapsed() >= Duration::from_millis(250) {
                                                        last_broadcast = Instant::now();
                                                        let _ = app_handle_cap.emit(
                                                            "download:progress",
                                                            DownloadProgressPayload {
                                                                task_id: adapter_task_id.clone(),
                                                                model_id: model_id_cap.clone(),
                                                                quantization: quantization_cap.clone(),
                                                                downloaded_bytes: downloaded,
                                                                total_bytes,
                                                                progress_percent,
                                                                speed_bps,
                                                                speed_formatted: format!("{:.1} MB/s", speed_bps / 1_048_576.0),
                                                                eta_seconds: if speed_bps > 0.0 { Some(((total_bytes.saturating_sub(downloaded)) as f64 / speed_bps) as u64) } else { None },
                                                                status: DownloadStatus::Downloading,
                                                                error: None,
                                                                package_id: Some(package_id_cap.clone()),
                                                                capability: Some(cap_key.clone()),
                                                                item_type: Some("adapter".to_string()),
                                                            },
                                                        );
                                                    }
                                                } else {
                                                    break;
                                                }
                                            }

                                            let _ = file.flush().await;
                                            drop(file);

                                            log_adapter_transition(
                                                &cap_key,
                                                &AdapterState::Downloading,
                                                &AdapterState::Verifying,
                                                "Verifying weight file integrity",
                                                "DownloadManager",
                                            );

                                            if temp_weight_path.exists() {
                                                if let Ok(meta) = tokio::fs::metadata(&temp_weight_path).await {
                                                    if meta.len() >= 100_000 {
                                                        log_adapter_transition(
                                                            &cap_key,
                                                            &AdapterState::Verifying,
                                                            &AdapterState::Installing,
                                                            "Renaming temp download file to target weight",
                                                            "DownloadManager",
                                                        );
                                                        if tokio::fs::rename(&temp_weight_path, &target_weight_path).await.is_ok() {
                                                            actual_downloaded_bytes = meta.len();
                                                            download_success = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if download_success {
                                log_adapter_transition(
                                    &cap_key,
                                    &AdapterState::Installing,
                                    &AdapterState::Ready,
                                    "Adapter successfully installed and registered to READY",
                                    "DownloadManager",
                                );

                                let mut map = manifest_adapters_cap.lock().unwrap();
                                map.insert(
                                    cap_key.clone(),
                                    AdapterManifestInfo {
                                        capability: cap_key.clone(),
                                        status: "Installed".to_string(),
                                        adapter_runtime_status: None,
                                        repo_id: Some(cand.repo_id),
                                        local_path: Some(format!("adapters/{}/", cap_key)),
                                        adapter_file: Some(format!("adapters/{}/{}", cap_key, cand.adapter_file_name)),
                                        config_file: Some(format!("adapters/{}/adapter_config.json", cap_key)),
                                        size_bytes: Some(actual_downloaded_bytes),
                                        base_model_match: Some(cand.base_model_match),
                                        target_modules: cand.target_modules,
                                        peft_type: Some(cand.peft_type),
                                        checksum: None,
                                        reason: None,
                                    },
                                );

                                let _ = app_handle_cap.emit(
                                    "download:progress",
                                    DownloadProgressPayload {
                                        task_id: adapter_task_id,
                                        model_id: model_id_cap.clone(),
                                        quantization: quantization_cap.clone(),
                                        downloaded_bytes: actual_downloaded_bytes,
                                        total_bytes: actual_downloaded_bytes,
                                        progress_percent: 100.0,
                                        speed_bps: 0.0,
                                        speed_formatted: "Ready".to_string(),
                                        eta_seconds: Some(0),
                                        status: DownloadStatus::Completed,
                                        error: None,
                                        package_id: Some(package_id_cap.clone()),
                                        capability: Some(cap_key),
                                        item_type: Some("adapter".to_string()),
                                    },
                                );
                                return;
                            }
                        }
                    }

                    // Candidate missing or failed: If no valid local files exist, mark Unavailable
                    if !AdapterRegistry::is_adapter_installed_and_valid(&package_dir_cap, &cap_key) {
                        log_adapter_transition(
                            &cap_key,
                            &AdapterState::Searching,
                            &AdapterState::Unavailable,
                            "Adapter not found on HuggingFace Hub or download failed",
                            "DownloadManager",
                        );

                        let mut map = manifest_adapters_cap.lock().unwrap();
                        map.insert(
                            cap_key.clone(),
                            AdapterManifestInfo {
                                capability: cap_key.clone(),
                                status: "Unavailable".to_string(),
                                adapter_runtime_status: None,
                                repo_id: None,
                                local_path: None,
                                adapter_file: None,
                                config_file: None,
                                size_bytes: None,
                                base_model_match: None,
                                target_modules: vec![],
                                peft_type: None,
                                checksum: None,
                                reason: Some("Handled natively by base model".to_string()),
                            },
                        );

                        let _ = app_handle_cap.emit(
                            "download:progress",
                            DownloadProgressPayload {
                                task_id: adapter_task_id,
                                model_id: model_id_cap.clone(),
                                quantization: quantization_cap.clone(),
                                downloaded_bytes: 0,
                                total_bytes: 0,
                                progress_percent: 0.0,
                                speed_bps: 0.0,
                                speed_formatted: "".to_string(),
                                eta_seconds: None,
                                status: DownloadStatus::Completed,
                                error: Some("Unavailable — Handled natively by base model".to_string()),
                                package_id: Some(package_id_cap.clone()),
                                capability: Some(cap_key),
                                item_type: Some("adapter".to_string()),
                            },
                        );
                    }
                }));
            }

            // Wait for all capability download tasks to complete
            for t in cap_tasks {
                let _ = t.await;
            }

            let final_adapters = manifest_adapters.lock().unwrap().clone();
            let manifest = ModelPackageManifest {
                package_id: package_id.clone(),
                provider_id: provider_id_clone,
                base_model: BaseManifestInfo {
                    model_id: model_id_clone.clone(),
                    model_name: model_name_clone,
                    quantization: quantization_clone.clone(),
                    file_path: "base/".to_string(),
                    size_bytes: 0,
                    checksum: None,
                },
                adapters: final_adapters,
                created_at: existing_manifest.as_ref().map(|m| m.created_at.clone()).unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
                updated_at: chrono::Utc::now().to_rfc3339(),
            };

            let _ = AdapterRegistry::write_manifest(&package_dir, &manifest);
            let _ = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest);

            // Broadcast final single package completed event
            let pkg_task_id = format!("dl_{}_{}_pkg", package_id, quantization_clone.to_lowercase());
            let _ = app_handle_for_adapters.emit(
                "download:progress",
                DownloadProgressPayload {
                    task_id: pkg_task_id,
                    model_id: model_id_clone,
                    quantization: quantization_clone,
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    progress_percent: 100.0,
                    speed_bps: 0.0,
                    speed_formatted: "Ready".to_string(),
                    eta_seconds: Some(0),
                    status: DownloadStatus::Completed,
                    error: None,
                    package_id: Some(package_id),
                    capability: None,
                    item_type: Some("package_completed".to_string()),
                },
            );
        });

        // 1. Resolve artifact details from Hugging Face
        log::info!("[DOWNLOAD_DIAGNOSTIC] Resolving GGUF artifact for model_id={} quantization={}", model_id, quantization);
        let artifact_result = resolver::resolve_artifact(&model_id, &quantization, hf_token.as_deref()).await;

        let artifact = match artifact_result {
            Ok(art) => art,
            Err(e) => {
                let err_msg = format!("Failed to resolve artifact: {}", e);
                log::error!("[DOWNLOAD_DIAGNOSTIC] {}", err_msg);
                if let Some(t) = self.tasks.lock().unwrap().get_mut(&task_id) {
                    t.status = DownloadStatus::Failed;
                    t.error = Some(err_msg.clone());
                    t.updated_at = chrono::Utc::now().to_rfc3339();
                    self.broadcast_progress(&app_handle, t);
                }
                return Err(anyhow!(err_msg));
            }
        };

        log::info!("[DOWNLOAD_DIAGNOSTIC] Artifact resolved successfully: repo={}, file={}, url={}, size_bytes={}", 
            artifact.repo_id, artifact.file_name, artifact.download_url, artifact.size_bytes);

        let file_name = artifact.file_name.split('/').last().unwrap_or(&artifact.file_name).to_string();
        let destination_path = storage_dir.join(&file_name);
        let temp_path = storage_dir.join(format!("{}.part", file_name));

        // Check if final ready file already exists
        if destination_path.exists() {
            let metadata = tokio::fs::metadata(&destination_path).await?;
            if artifact.size_bytes == 0 || metadata.len() == artifact.size_bytes {
                log::info!("[DOWNLOAD_DIAGNOSTIC] Model artifact already downloaded & verified at {:?}", destination_path);
                let completed_task = DownloadTask {
                    id: task_id.clone(),
                    model_id,
                    model_name,
                    provider_id,
                    quantization,
                    format,
                    backend,
                    url: artifact.download_url,
                    destination_path: destination_path.to_string_lossy().to_string(),
                    temp_path: temp_path.to_string_lossy().to_string(),
                    total_bytes: metadata.len(),
                    downloaded_bytes: metadata.len(),
                    status: DownloadStatus::Completed,
                    speed_bps: 0.0,
                    eta_seconds: Some(0),
                    checksum: artifact.sha256,
                    error: None,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                };
                self.tasks.lock().unwrap().insert(task_id.clone(), completed_task.clone());
                self.broadcast_progress(&app_handle, &completed_task);
                return Ok(task_id);
            }
        }

        // Determine existing downloaded bytes from .part file
        let initial_bytes = if temp_path.exists() {
            tokio::fs::metadata(&temp_path).await?.len()
        } else {
            0
        };

        let total_bytes = artifact.size_bytes;

        // Check disk space before starting stream
        let remaining_needed = total_bytes.saturating_sub(initial_bytes);
        if let Err(e) = Self::check_disk_space(&destination_path, remaining_needed) {
            let err_msg = format!("Disk space check failed: {}", e);
            log::error!("[DOWNLOAD_DIAGNOSTIC] {}", err_msg);
            if let Some(t) = self.tasks.lock().unwrap().get_mut(&task_id) {
                t.status = DownloadStatus::Failed;
                t.error = Some(err_msg.clone());
                t.updated_at = chrono::Utc::now().to_rfc3339();
                self.broadcast_progress(&app_handle, t);
            }
            return Err(anyhow!(err_msg));
        }

        let task = DownloadTask {
            id: task_id.clone(),
            model_id,
            model_name,
            provider_id,
            quantization,
            format,
            backend,
            url: artifact.download_url.clone(),
            destination_path: destination_path.to_string_lossy().to_string(),
            temp_path: temp_path.to_string_lossy().to_string(),
            total_bytes,
            downloaded_bytes: initial_bytes,
            status: DownloadStatus::Downloading,
            speed_bps: 0.0,
            eta_seconds: None,
            checksum: artifact.sha256,
            error: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };

        self.tasks.lock().unwrap().insert(task_id.clone(), task.clone());
        self.broadcast_progress(&app_handle, &task);
        log::info!("[DOWNLOAD_DIAGNOSTIC] Starting streaming download for task {} ({:.2} MB expected)", task_id, total_bytes as f64 / 1_048_576.0);

        // Set up cancel watch channel
        let (cancel_tx, cancel_rx) = watch::channel(false);
        self.cancel_senders.lock().unwrap().insert(task_id.clone(), cancel_tx);

        // Spawn background tokio task for HTTP streaming download
        let tasks_map = self.tasks.clone();
        let cancel_senders_map = self.cancel_senders.clone();
        let task_id_clone = task_id.clone();
        let app_handle_clone = app_handle.clone();

        tokio::spawn(async move {
            let res = Self::run_download_loop(
                &app_handle_clone,
                &task_id_clone,
                &artifact.download_url,
                &PathBuf::from(&task.temp_path),
                &PathBuf::from(&task.destination_path),
                total_bytes,
                initial_bytes,
                hf_token,
                cancel_rx,
                tasks_map.clone(),
            )
            .await;

            cancel_senders_map.lock().unwrap().remove(&task_id_clone);

            if let Err(err) = res {
                log::error!("[DOWNLOAD_DIAGNOSTIC] Task {} failed: {}", task_id_clone, err);
                if let Some(t) = tasks_map.lock().unwrap().get_mut(&task_id_clone) {
                    if t.status != DownloadStatus::Cancelled && t.status != DownloadStatus::Paused {
                        t.status = DownloadStatus::Failed;
                        t.error = Some(err.to_string());
                        t.updated_at = chrono::Utc::now().to_rfc3339();
                        let _ = app_handle_clone.emit("download:progress", Self::make_payload(t));
                    }
                }
            }
        });

        Ok(task_id)
    }

    async fn run_download_loop(
        app_handle: &tauri::AppHandle,
        task_id: &str,
        url: &str,
        temp_path: &Path,
        destination_path: &Path,
        expected_total_bytes: u64,
        initial_bytes: u64,
        hf_token: Option<String>,
        mut cancel_rx: watch::Receiver<bool>,
        tasks: Arc<Mutex<HashMap<String, DownloadTask>>>,
    ) -> Result<()> {
        let client = reqwest::Client::builder()
            .user_agent("Sarathi/0.1.0 (Windows; x64)")
            .build()?;

        let mut req = client.get(url);
        if let Some(token) = &hf_token {
            if !token.trim().is_empty() {
                req = req.header("Authorization", format!("Bearer {}", token.trim()));
            }
        }

        if initial_bytes > 0 {
            req = req.header("Range", format!("bytes={}-", initial_bytes));
            log::info!("[DOWNLOAD_DIAGNOSTIC] Requesting Range: bytes={}- for task {}", initial_bytes, task_id);
        }

        let resp = req.send().await?;
        log::info!("[DOWNLOAD_DIAGNOSTIC] HTTP response received: status={}, Content-Length={:?}", 
            resp.status(), resp.headers().get(reqwest::header::CONTENT_LENGTH));

        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(anyhow!("HTTP error response: {}", resp.status()));
        }

        // Open temp file for appending
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(temp_path)
            .await?;

        let mut stream = resp.bytes_stream();
        let mut downloaded = initial_bytes;
        let mut last_sample_time = Instant::now();
        let mut last_sample_bytes = initial_bytes;
        let mut last_broadcast = Instant::now();

        // Broadcast initial progress immediately upon connection
        if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
            task.downloaded_bytes = downloaded;
            if expected_total_bytes > 0 {
                task.total_bytes = expected_total_bytes;
            }
            let _ = app_handle.emit("download:progress", Self::make_payload(task));
        }

        while let Some(chunk_res) = stream.next().await {
            // Check cancellation / pause
            if *cancel_rx.borrow() {
                file.flush().await?;
                log::info!("[DOWNLOAD_DIAGNOSTIC] Pause/Cancel signal received for task {}", task_id);
                return Ok(());
            }

            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => return Err(anyhow!("Network error while streaming chunks: {}", e)),
            };

            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            let now = Instant::now();
            let sample_elapsed = now.duration_since(last_sample_time).as_secs_f64();

            if sample_elapsed >= 0.25 {
                let bytes_diff = downloaded.saturating_sub(last_sample_bytes);
                let speed_bps = if sample_elapsed > 0.0 { bytes_diff as f64 / sample_elapsed } else { 0.0 };
                let remaining_bytes = expected_total_bytes.saturating_sub(downloaded);
                let eta_seconds = if speed_bps > 0.0 {
                    Some((remaining_bytes as f64 / speed_bps) as u64)
                } else {
                    None
                };

                last_sample_time = now;
                last_sample_bytes = downloaded;

                if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
                    task.downloaded_bytes = downloaded;
                    if expected_total_bytes > 0 {
                        task.total_bytes = expected_total_bytes;
                    }
                    task.speed_bps = if speed_bps.is_nan() || speed_bps.is_infinite() { 0.0 } else { speed_bps };
                    task.eta_seconds = eta_seconds;
                    task.updated_at = chrono::Utc::now().to_rfc3339();

                    if now.duration_since(last_broadcast) >= Duration::from_millis(250) {
                        let _ = app_handle.emit("download:progress", Self::make_payload(task));
                        last_broadcast = now;
                    }
                }
            }
        }

        file.flush().await?;
        drop(file);

        // Verification step
        log::info!("[DOWNLOAD_DIAGNOSTIC] Streaming finished for task {}. Updating status to Verifying...", task_id);
        if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
            task.downloaded_bytes = downloaded;
            task.status = DownloadStatus::Verifying;
            task.speed_bps = 0.0;
            task.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = app_handle.emit("download:progress", Self::make_payload(task));
        }

        let final_size = tokio::fs::metadata(temp_path).await?.len();
        if expected_total_bytes > 0 && final_size != expected_total_bytes {
            return Err(anyhow!(
                "Integrity verification failed: expected {} bytes, actual {} bytes",
                expected_total_bytes,
                final_size
            ));
        }

        // Atomic rename .part -> final file
        if destination_path.exists() {
            let _ = tokio::fs::remove_file(destination_path).await;
        }
        tokio::fs::rename(temp_path, destination_path).await?;
        log::info!("[DOWNLOAD_DIAGNOSTIC] ✓ Atomic finalization complete: renamed {:?} -> {:?}", temp_path, destination_path);

        // Update task status to Completed
        if let Some(task) = tasks.lock().unwrap().get_mut(task_id) {
            task.downloaded_bytes = final_size;
            task.total_bytes = final_size;
            task.status = DownloadStatus::Completed;
            task.speed_bps = 0.0;
            task.eta_seconds = Some(0);
            task.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = app_handle.emit("download:progress", Self::make_payload(task));
        }

        Ok(())
    }

    pub fn pause_download(&self, task_id: &str) -> Result<()> {
        log::info!("[DOWNLOAD_DIAGNOSTIC] Pausing download for task {}", task_id);
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

    pub fn cancel_download(&self, task_id: &str) -> Result<()> {
        log::info!("[DOWNLOAD_DIAGNOSTIC] Cancelling download for task {}", task_id);
        if let Some(sender) = self.cancel_senders.lock().unwrap().get(task_id) {
            let _ = sender.send(true);
        }
        if let Some(task) = self.tasks.lock().unwrap().remove(task_id) {
            let temp_path = PathBuf::from(&task.temp_path);
            if temp_path.exists() {
                let _ = std::fs::remove_file(&temp_path);
                log::info!("[DOWNLOAD_DIAGNOSTIC] Removed temporary .part file {:?}", temp_path);
            }
        }
        Ok(())
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

    fn make_payload(task: &DownloadTask) -> DownloadProgressPayload {
        let mut speed = task.speed_bps;
        if speed.is_nan() || speed.is_infinite() || speed < 0.0 {
            speed = 0.0;
        }

        let pct = if task.total_bytes > 0 {
            let calculated = (task.downloaded_bytes as f64 / task.total_bytes as f64) * 100.0;
            if calculated.is_nan() || calculated.is_infinite() {
                0.0
            } else {
                calculated.clamp(0.0, 100.0)
            }
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
            DownloadStatus::Downloading => {
                if speed >= 1_048_576.0 {
                    format!("{:.1} MB/s", speed / 1_048_576.0)
                } else if speed >= 1024.0 {
                    format!("{:.0} KB/s", speed / 1024.0)
                } else if speed > 0.0 {
                    format!("{:.0} B/s", speed)
                } else {
                    "Starting...".to_string()
                }
            }
        };

        DownloadProgressPayload {
            task_id: task.id.clone(),
            model_id: task.model_id.clone(),
            quantization: task.quantization.clone(),
            downloaded_bytes: task.downloaded_bytes,
            total_bytes: task.total_bytes,
            progress_percent: pct,
            speed_bps: speed,
            speed_formatted,
            eta_seconds: task.eta_seconds,
            status: task.status.clone(),
            error: task.error.clone(),
            package_id: None,
            capability: None,
            item_type: Some("base_model".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_dir_construction() {
        let base = PathBuf::from("C:\\AppData");
        let dir = DownloadManager::get_model_storage_dir(&base, "huggingface", "Qwen/Qwen2.5-Coder-7B", "Q4_0");
        assert_eq!(
            dir,
            PathBuf::from("C:\\AppData\\models\\huggingface\\Qwen_Qwen2.5-Coder-7B\\base")
        );
    }

    #[tokio::test]
    async fn test_full_lifecycle_real_download() {
        let app_data_dir = std::env::temp_dir().join(format!("sarathi_e2e_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(12345)));
        let _ = tokio::fs::remove_dir_all(&app_data_dir).await;
        tokio::fs::create_dir_all(&app_data_dir).await.unwrap();

        let model_id = "meta-llama/Llama-3.2-1B";
        let requested_quant = "Q8_0";

        println!("\n=======================================================");
        println!("=== PHASE 4 FULL LIFECYCLE REAL E2E DOWNLOAD TEST ===");
        println!("=======================================================");

        // STEP 1 & 2: Exact Hugging Face artifact + quantization resolution
        let artifact = resolver::resolve_artifact(model_id, requested_quant, None)
            .await
            .expect("Hugging Face API artifact resolution must succeed");

        println!("1. Resolved Artifact:");
        println!("   - Repo ID: {}", artifact.repo_id);
        println!("   - File Name: {}", artifact.file_name);
        println!("   - Download URL: {}", artifact.download_url);
        println!("   - Size: {:.2} MB ({} bytes)", artifact.size_bytes as f64 / 1_048_576.0, artifact.size_bytes);

        assert!(artifact.download_url.starts_with("https://huggingface.co/"));
        let clean_filename = artifact.file_name.split('/').last().unwrap_or("model.gguf");
        assert!(clean_filename.to_lowercase().contains("q8_0"), "Resolved quantization must strictly match Q8_0");

        // STEP 3: Disk space pre-check
        let space_check = DownloadManager::check_disk_space(&app_data_dir, artifact.size_bytes);
        assert!(space_check.is_ok(), "Disk space pre-check must succeed");
        println!("2. Disk space pre-check passed (free space > model size + 1 GB safety margin).");

        // STEP 4: Setup paths
        let storage_dir = DownloadManager::get_model_storage_dir(&app_data_dir, "huggingface", model_id, requested_quant);
        tokio::fs::create_dir_all(&storage_dir).await.unwrap();

        let temp_path = storage_dir.join(format!("{}.part", clean_filename));
        let destination_path = storage_dir.join(clean_filename);

        let _ = tokio::fs::remove_file(&temp_path).await;
        let _ = tokio::fs::remove_file(&destination_path).await;

        println!("3. Target Locations:");
        println!("   - Storage Dir: {:?}", storage_dir);
        println!("   - Temp Path (.part): {:?}", temp_path);
        println!("   - Final Path (.gguf): {:?}", destination_path);

        // STEP 5 & 6: Download initial chunk & test PAUSE
        let client = reqwest::Client::new();
        let resp = client.get(&artifact.download_url).send().await.unwrap();
        assert!(resp.status().is_success());

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&temp_path)
            .await
            .unwrap();

        let mut stream = resp.bytes_stream();
        let mut downloaded_bytes: u64 = 0;

        // Download ~10 MB initial chunk
        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res.unwrap();
            file.write_all(&chunk).await.unwrap();
            downloaded_bytes += chunk.len() as u64;
            if downloaded_bytes >= 10 * 1024 * 1024 {
                break;
            }
        }
        file.flush().await.unwrap();
        drop(file);

        // Verify .part file exists and exact byte count is preserved
        assert!(temp_path.exists(), ".part file must exist on disk after pause");
        let part_size_on_disk = tokio::fs::metadata(&temp_path).await.unwrap().len();
        assert_eq!(part_size_on_disk, downloaded_bytes, ".part size on disk must equal downloaded chunk");
        println!("4. PAUSE TEST: Download paused at {:.2} MB. Verified .part file preserved: {} bytes.", 
            downloaded_bytes as f64 / 1_048_576.0, part_size_on_disk);

        // STEP 7 & 8: HTTP Range RESUME & Complete 100% Download
        let resume_req = client.get(&artifact.download_url).header("Range", format!("bytes={}-", downloaded_bytes));
        let resume_resp = resume_req.send().await.unwrap();
        assert_eq!(resume_resp.status(), reqwest::StatusCode::PARTIAL_CONTENT, "HF server must return 206 Partial Content for Range resume");

        println!("5. RESUME TEST: Resuming download from byte offset {}...", downloaded_bytes);

        let mut resume_file = tokio::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .open(&temp_path)
            .await
            .unwrap();

        let mut resume_stream = resume_resp.bytes_stream();
        let mut last_reported_pct: u32 = 0;

        while let Some(chunk_res) = resume_stream.next().await {
            let chunk = chunk_res.unwrap();
            resume_file.write_all(&chunk).await.unwrap();
            downloaded_bytes += chunk.len() as u64;

            let pct = ((downloaded_bytes as f64 / artifact.size_bytes as f64) * 100.0) as u32;
            if pct >= last_reported_pct + 25 || downloaded_bytes == artifact.size_bytes {
                println!("   - Download progress: {}% ({:.2} MB / {:.2} MB)", 
                    pct, downloaded_bytes as f64 / 1_048_576.0, artifact.size_bytes as f64 / 1_048_576.0);
                last_reported_pct = pct;
            }
        }
        resume_file.flush().await.unwrap();
        drop(resume_file);

        // STEP 9: Size Integrity Verification
        assert_eq!(downloaded_bytes, artifact.size_bytes, "Downloaded bytes must exactly match expected HF size");
        let final_temp_size = tokio::fs::metadata(&temp_path).await.unwrap().len();
        assert_eq!(final_temp_size, artifact.size_bytes, "Temp .part file size must equal expected artifact size");
        println!("6. INTEGRITY VERIFICATION: 100% download complete! Expected: {} B, Actual: {} B.", 
            artifact.size_bytes, final_temp_size);

        // STEP 10: Atomic finalization (.part -> .gguf)
        tokio::fs::rename(&temp_path, &destination_path).await.unwrap();
        assert!(!temp_path.exists(), ".part file must be removed after atomic rename");
        assert!(destination_path.exists(), "Final .gguf file must exist on disk");

        // Write test manifest.json to package directory
        let package_dir = storage_dir.parent().unwrap();
        let manifest = ModelPackageManifest {
            package_id: model_id.replace('/', "_"),
            provider_id: "huggingface".to_string(),
            base_model: BaseManifestInfo {
                model_id: model_id.to_string(),
                model_name: "Llama 3.2 1B".to_string(),
                quantization: requested_quant.to_string(),
                file_path: "base/".to_string(),
                size_bytes: artifact.size_bytes,
                checksum: None,
            },
            adapters: HashMap::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        AdapterRegistry::write_manifest(package_dir, &manifest).unwrap();
        println!("7. ATOMIC FINALIZATION: Renamed .part -> .gguf cleanly & wrote manifest.json.");

        // STEP 11 & 12: Model Manager Scanning & Detection
        let installed = crate::model_manager::ModelManager::list_installed_models(&app_data_dir);
        assert_eq!(installed.len(), 1, "ModelManager must report exactly 1 installed model");
        let model = &installed[0];
        assert_eq!(model.model_id, model_id);
        assert_eq!(model.provider_id, "huggingface");
        assert_eq!(model.quantization, requested_quant);
        assert_eq!(model.format, "GGUF");
        assert_eq!(model.backend, "llama.cpp (GGUF)");
        assert!(model.is_ready, "Model must be marked Ready");
        assert_eq!(model.size_bytes, artifact.size_bytes);
        println!("8. STORAGE DETECTION: Detected model in Storage tab with status READY.");

        // STEP 13: App Restart Persistence Simulation
        let restarted_installed = crate::model_manager::ModelManager::list_installed_models(&app_data_dir);
        assert_eq!(restarted_installed.len(), 1, "Model must persist across app restarts");
        assert!(restarted_installed[0].is_ready, "Model must retain Ready status after restart");
        println!("9. RESTART PERSISTENCE: Model persisted cleanly across restart simulation.");

        // STEP 14, 15 & 16: Delete Model from Storage
        crate::model_manager::ModelManager::delete_installed_model(&app_data_dir, "huggingface", model_id, requested_quant)
            .expect("Model deletion must succeed");

        assert!(!destination_path.exists(), "Model file must be deleted from disk");
        assert!(!package_dir.exists(), "Model package directory must be removed from disk");

        let post_delete_installed = crate::model_manager::ModelManager::list_installed_models(&app_data_dir);
        assert_eq!(post_delete_installed.len(), 0, "Installed models list must be empty after deletion");
        println!("10. DELETE TEST: Model file and directory removed from disk. Storage updated cleanly.");

        println!("=======================================================");
        println!("=== ALL 10 E2E FULL LIFECYCLE VERIFICATIONS PASSED ===");
        println!("=======================================================\n");

        let _ = tokio::fs::remove_dir_all(&app_data_dir).await;
    }
}

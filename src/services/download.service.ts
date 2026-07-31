import { invoke } from '@tauri-apps/api/core';
import { listen, Event } from '@tauri-apps/api/event';
import type { DownloadTask, DownloadProgressPayload, InstalledModel, StorageSummary } from '../types/download';

export async function startModelDownload(params: {
  modelId: string;
  modelName: string;
  providerId: string;
  quantization: string;
  format: string;
  backend: string;
  hfToken?: string;
}): Promise<string> {
  return invoke('start_model_download', {
    modelId: params.modelId,
    modelName: params.modelName,
    providerId: params.providerId || 'huggingface',
    quantization: params.quantization,
    format: params.format || 'GGUF',
    backend: params.backend || 'llama.cpp (GGUF)',
    hfToken: params.hfToken || null,
  });
}

export async function pauseModelDownload(taskId: string): Promise<void> {
  return invoke('pause_model_download', { taskId });
}

export async function cancelModelDownload(taskId: string): Promise<void> {
  return invoke('cancel_model_download', { taskId });
}

export async function getActiveDownloads(): Promise<DownloadTask[]> {
  return invoke('get_active_downloads');
}

export async function getInstalledModels(): Promise<InstalledModel[]> {
  return invoke('get_installed_models');
}

export async function deleteInstalledModel(
  providerId: string,
  modelId: string,
  quantization: string
): Promise<void> {
  return invoke('delete_installed_model', { providerId, modelId, quantization });
}

export async function getStorageSummary(): Promise<StorageSummary> {
  return invoke('get_storage_summary');
}

export async function listenDownloadProgress(
  callback: (payload: DownloadProgressPayload) => void
) {
  return listen<DownloadProgressPayload>('download:progress', (event: Event<DownloadProgressPayload>) => {
    callback(event.payload);
  });
}

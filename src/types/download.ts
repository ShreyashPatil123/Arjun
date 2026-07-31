export type DownloadStatus =
  | 'Resolving'
  | 'Queued'
  | 'Downloading'
  | 'Paused'
  | 'Completed'
  | 'Failed'
  | 'Cancelled'
  | 'Verifying';

export interface DownloadTask {
  id: string;
  modelId: string;
  modelName: string;
  providerId: string;
  quantization: string;
  format: string;
  backend: string;
  url: string;
  destinationPath: string;
  tempPath: string;
  totalBytes: number;
  downloadedBytes: number;
  status: DownloadStatus;
  speedBps: number;
  etaSeconds: number | null;
  checksum: string | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface DownloadProgressPayload {
  taskId: string;
  modelId: string;
  quantization: string;
  downloadedBytes: number;
  totalBytes: number;
  progressPercent: number;
  speedBps: number;
  speedFormatted: string;
  etaSeconds: number | null;
  status: DownloadStatus;
  error: string | null;
}

export interface InstalledModel {
  id: string;
  modelId: string;
  modelName: string;
  providerId: string;
  quantization: string;
  format: string;
  backend: string;
  fileName: string;
  filePath: string;
  sizeBytes: number;
  installedAt: string;
  isReady: boolean;
  checksum: string | null;
}

export interface StorageSummary {
  modelsDirectory: string;
  totalInstalledModels: number;
  totalModelsBytes: number;
  availableDiskSpaceBytes: number;
  totalDiskSpaceBytes: number;
}

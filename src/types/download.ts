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
  packageId?: string;
  capability?: string;
  itemType?: string;
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
  adapters?: Record<string, AdapterManifestInfo>;
}

export interface StorageSummary {
  modelsDirectory: string;
  totalInstalledModels: number;
  totalModelsBytes: number;
  availableDiskSpaceBytes: number;
  totalDiskSpaceBytes: number;
}

export interface AdapterCandidate {
  repoId: string;
  capability: string;
  baseModelMatch: string;
  peftType: string;
  targetModules: string[];
  adapterFileName: string;
  downloadUrl: string;
  sizeBytes: number;
  downloads: number;
  likes: number;
  confidenceScore: number;
}

export interface AdapterSearchResult {
  capability: string;
  status: 'Found' | 'Unavailable' | 'Searching';
  candidate: AdapterCandidate | null;
  reason: string | null;
}

export interface BaseManifestInfo {
  modelId: string;
  modelName: string;
  quantization: string;
  filePath: string;
  sizeBytes: number;
  checksum: string | null;
}

export interface AdapterManifestInfo {
  capability: string;
  status: 'Installed' | 'READY' | 'Unavailable' | 'Failed' | string;
  adapterRuntimeStatus?: 'compatible' | 'requires_conversion' | 'incompatible' | 'not_present' | null;
  repoId: string | null;
  localPath: string | null;
  adapterFile: string | null;
  configFile: string | null;
  sizeBytes: number | null;
  baseModelMatch: string | null;
  targetModules: string[];
  peftType: string | null;
  checksum: string | null;
  reason: string | null;
}

export interface ModelPackageManifest {
  packageId: string;
  providerId: string;
  baseModel: BaseManifestInfo;
  adapters: Record<string, AdapterManifestInfo>;
  createdAt: string;
  updatedAt: string;
}

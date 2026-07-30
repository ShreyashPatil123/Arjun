export interface OverrideValue<T> {
  detected: T;
  overridden: T | null;
  isOverridden: boolean;
}

export interface CpuInfo {
  manufacturer: string;
  model: string;
  architecture: string;
  physicalCores: number;
  logicalProcessors: number;
  baseFrequencyMhz: number;
  boostFrequencyMhz: number;
  cacheL1Kb?: number;
  cacheL2Kb?: number;
  cacheL3Kb?: number;
  virtualizationSupported: boolean;
  simdCapabilities: string[];
}

export interface GpuInfo {
  vendor: string;
  model: string;
  isDedicated: boolean;
  vramTotalBytes: number;
  vramFreeBytes: number;
  driverVersion: string;
  computeCapability?: string;
  cudaSupported: boolean;
  rocmSupported: boolean;
  directxSupported: boolean;
  vulkanSupported: boolean;
  openclSupported: boolean;
}

export interface MemoryInfo {
  totalBytes: number;
  availableBytes: number;
  usedBytes: number;
  memoryType: string;
  speedMts: number;
  totalSlots: number;
  populatedSlots: number;
}

export interface StorageInfo {
  driveName: string;
  mountPoint: string;
  driveType: string;
  totalBytes: number;
  freeBytes: number;
  fileSystem: string;
  isAiStorageReady: boolean;
}

export interface OsInfo {
  name: string;
  edition: string;
  version: string;
  buildNumber: string;
  architecture: string;
  locale: string;
}

export interface SoftwareDetectorInfo {
  name: string;
  installed: boolean;
  version?: string | null;
  path?: string | null;
}

export interface SoftwareEnvironment {
  python: SoftwareDetectorInfo;
  rust: SoftwareDetectorInfo;
  cargo: SoftwareDetectorInfo;
  git: SoftwareDetectorInfo;
  nodejs: SoftwareDetectorInfo;
  npm: SoftwareDetectorInfo;
  pnpm: SoftwareDetectorInfo;
  ollama: SoftwareDetectorInfo;
  cudaToolkit: SoftwareDetectorInfo;
  vcRedistributable: SoftwareDetectorInfo;
  additional?: Record<string, SoftwareDetectorInfo>;
}

export interface AIRuntimeInfo {
  name: string;
  status: 'running' | 'stopped' | 'not_installed' | string;
  version?: string;
  endpoint?: string;
  modelsAvailable?: string[];
}

export interface SystemPaths {
  userHome: string;
  downloads: string;
  documents: string;
  desktop: string;
  appData: string;
  cacheDir: string;
  modelStorageDir: string;
}

export interface AICapabilityProfile {
  maxRecommendedModelSizeBytes: number;
  recommendedQuantizations: string[];
  recommendedContextLength: number;
  preferredInferenceBackend: string;
  multiModelCapable: boolean;
  loraReady: boolean;
  visionReady: boolean;
  embeddingReady: boolean;
  extraCapabilities?: Record<string, boolean | string>;
}

export interface SystemValidationResult {
  isReadyForAi: boolean;
  score: number;
  warnings: string[];
  errors: string[];
  recommendations: string[];
}

export interface HardwareProfile {
  id: string;
  profileCreatedAt: string;
  profileUpdatedAt: string;
  cpu: CpuInfo;
  gpus: GpuInfo[];
  memory: MemoryInfo;
  storage: StorageInfo[];
  os: OsInfo;
  software: SoftwareEnvironment;
  aiRuntimes: AIRuntimeInfo[];
  paths: SystemPaths;
  aiCapabilities: AICapabilityProfile;
  validation: SystemValidationResult;
  overrides?: Record<string, unknown>;
}
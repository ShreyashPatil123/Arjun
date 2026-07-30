import { getBackendService } from './api';
import { HardwareProfile, SystemValidationResult } from '../types/system';

const emptyHardwareProfile: HardwareProfile = {
  id: 'hw-prof-unknown',
  profileCreatedAt: new Date().toISOString(),
  profileUpdatedAt: new Date().toISOString(),
  cpu: {
    manufacturer: 'Unknown',
    model: 'Unknown',
    architecture: 'Unknown',
    physicalCores: 0,
    logicalProcessors: 0,
    baseFrequencyMhz: 0,
    boostFrequencyMhz: 0,
    cacheL1Kb: undefined,
    cacheL2Kb: undefined,
    cacheL3Kb: undefined,
    virtualizationSupported: false,
    simdCapabilities: []
  },
  gpus: [
    {
      vendor: 'Unknown',
      model: 'Unknown',
      isDedicated: false,
      vramTotalBytes: 0,
      vramFreeBytes: 0,
      driverVersion: undefined,
      computeCapability: undefined,
      cudaSupported: false,
      rocmSupported: false,
      directxSupported: false,
      vulkanSupported: false,
      openclSupported: false
    }
  ],
  memory: {
    totalBytes: 0,
    availableBytes: 0,
    usedBytes: 0,
    memoryType: 'Unknown',
    speedMts: undefined,
    totalSlots: undefined,
    populatedSlots: undefined
  },
  storage: [],
  os: {
    name: 'Unknown',
    edition: 'Unknown',
    version: 'Unknown',
    buildNumber: 'Unknown',
    architecture: 'Unknown',
    locale: 'Unknown'
  },
  software: {
    python: { name: 'Python', installed: false, version: undefined, path: undefined },
    rust: { name: 'Rust', installed: false, version: undefined, path: undefined },
    cargo: { name: 'Cargo', installed: false, version: undefined, path: undefined },
    git: { name: 'Git', installed: false, version: undefined, path: undefined },
    nodejs: { name: 'Node.js', installed: false, version: undefined, path: undefined },
    npm: { name: 'npm', installed: false, version: undefined, path: undefined },
    pnpm: { name: 'pnpm', installed: false, version: undefined, path: undefined },
    ollama: { name: 'Ollama', installed: false, version: undefined, path: undefined },
    cudaToolkit: { name: 'CUDA Toolkit', installed: false, version: undefined, path: undefined },
    vcRedistributable: { name: 'VC++ Redistributable', installed: false, version: undefined, path: undefined }
  },
  aiRuntimes: [],
  paths: {
    userHome: 'Unknown',
    downloads: 'Unknown',
    documents: 'Unknown',
    desktop: 'Unknown',
    appData: 'Unknown',
    cacheDir: 'Unknown',
    modelStorageDir: 'Unknown'
  },
  aiCapabilities: {
    maxRecommendedModelSizeBytes: undefined,
    recommendedQuantizations: [],
    recommendedContextLength: undefined,
    preferredInferenceBackend: undefined,
    multiModelCapable: false,
    loraReady: false,
    visionReady: false,
    embeddingReady: false
  },
  validation: {
    isReadyForAi: false,
    score: 0,
    warnings: ['Hardware detection pending or unavailable.'],
    errors: [],
    recommendations: ['Run system analysis to detect hardware capabilities.']
  },
  overrides: {}
};

/**
 * Set field path on an object dynamically
 */
function setNestedField(obj: any, path: string, value: unknown): void {
  const parts = path.split('.');
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (!(part in current) || typeof current[part] !== 'object') {
      current[part] = {};
    }
    current = current[part];
  }
  current[parts[parts.length - 1]] = value;
}

export async function getHardwareProfile(): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile | null>('get_hardware_profile');
    if (profile && profile.cpu && profile.cpu.model !== 'Unknown') {
      return profile;
    }
    // If cached profile is null or un-analyzed, trigger fresh system analysis
    return await analyzeSystem();
  } catch (err) {
    console.debug('Tauri get_hardware_profile invoke fallback to analyzeSystem:', err);
    return await analyzeSystem();
  }
}

export async function analyzeSystem(): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile>('analyze_system');
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri analyze_system invoke failed:', err);
  }
  return { ...emptyHardwareProfile };
}

export async function overrideHardwareValue(fieldPath: string, value: unknown): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile>('override_hardware_value', {
      fieldPath,
      value,
      field_path: fieldPath
    });
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri override_hardware_value invoke fallback:', err);
  }

  const updatedProfile = JSON.parse(JSON.stringify(emptyHardwareProfile)) as HardwareProfile;
  if (!updatedProfile.overrides) updatedProfile.overrides = {};
  updatedProfile.overrides[fieldPath] = value;
  setNestedField(updatedProfile, fieldPath, value);
  updatedProfile.profileUpdatedAt = new Date().toISOString();

  return updatedProfile;
}

export async function revertHardwareOverride(fieldPath: string): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile>('revert_hardware_override', {
      fieldPath,
      value: null,
      field_path: fieldPath
    });
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri revert_hardware_override invoke fallback:', err);
  }

  return { ...emptyHardwareProfile };
}

export async function validateSystem(): Promise<SystemValidationResult> {
  try {
    const validation = await getBackendService().invoke<SystemValidationResult>('validate_system');
    if (validation) return validation;
  } catch (err) {
    console.debug('Tauri validate_system invoke fallback:', err);
  }
  return { ...emptyHardwareProfile.validation };
}
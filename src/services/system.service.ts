import { getBackendService } from './api';
import { HardwareProfile, SystemValidationResult } from '../types/system';

let mockProfileState: HardwareProfile = {
  id: 'hw-prof-default-01',
  profileCreatedAt: new Date().toISOString(),
  profileUpdatedAt: new Date().toISOString(),
  cpu: {
    manufacturer: 'Intel',
    model: '13th Gen Intel(R) Core(TM) i9-13900K',
    architecture: 'x86_64',
    physicalCores: 16,
    logicalProcessors: 24,
    baseFrequencyMhz: 3000,
    boostFrequencyMhz: 5800,
    cacheL1Kb: 768,
    cacheL2Kb: 32768,
    cacheL3Kb: 36864,
    virtualizationSupported: true,
    simdCapabilities: ['AVX', 'AVX2', 'AVX512', 'FMA3', 'SSE4.2', 'NEON']
  },
  gpus: [
    {
      vendor: 'NVIDIA',
      model: 'NVIDIA GeForce RTX 4090',
      isDedicated: true,
      vramTotalBytes: 25769803776, // 24 GB
      vramFreeBytes: 19327352832,  // 18 GB
      driverVersion: '551.86',
      computeCapability: '8.9',
      cudaSupported: true,
      rocmSupported: false,
      directxSupported: true,
      vulkanSupported: true,
      openclSupported: true
    }
  ],
  memory: {
    totalBytes: 68719476736,     // 64 GB
    availableBytes: 45097156608, // 42 GB
    usedBytes: 23622320128,      // 22 GB
    memoryType: 'DDR5',
    speedMts: 6000,
    totalSlots: 4,
    populatedSlots: 2
  },
  storage: [
    {
      driveName: 'System Drive (C:)',
      mountPoint: 'C:\\',
      driveType: 'NVMe SSD',
      totalBytes: 2000398934016,  // 2 TB
      freeBytes: 1288490188800,   // 1.2 TB
      fileSystem: 'NTFS',
      isAiStorageReady: true
    },
    {
      driveName: 'AI Models Storage (D:)',
      mountPoint: 'D:\\',
      driveType: 'NVMe SSD',
      totalBytes: 4000797868032,  // 4 TB
      freeBytes: 3113851289600,   // 2.9 TB
      fileSystem: 'NTFS',
      isAiStorageReady: true
    }
  ],
  os: {
    name: 'Windows',
    edition: 'Windows 11 Pro',
    version: '10.0.22631',
    buildNumber: '22631',
    architecture: 'x86_64',
    locale: 'en-US'
  },
  software: {
    python: { name: 'Python', installed: true, version: '3.11.8', path: 'C:\\Python311\\python.exe' },
    rust: { name: 'Rust', installed: true, version: '1.77.0', path: 'C:\\Users\\User\\.cargo\\bin\\rustc.exe' },
    cargo: { name: 'Cargo', installed: true, version: '1.77.0', path: 'C:\\Users\\User\\.cargo\\bin\\cargo.exe' },
    git: { name: 'Git', installed: true, version: '2.43.0', path: 'C:\\Program Files\\Git\\cmd\\git.exe' },
    nodejs: { name: 'Node.js', installed: true, version: '20.11.1', path: 'C:\\Program Files\\nodejs\\node.exe' },
    npm: { name: 'npm', installed: true, version: '10.2.4', path: 'C:\\Program Files\\nodejs\\npm.cmd' },
    pnpm: { name: 'pnpm', installed: true, version: '8.15.4', path: 'C:\\Users\\User\\AppData\\Roaming\\npm\\pnpm.cmd' },
    ollama: { name: 'Ollama', installed: true, version: '0.1.28', path: 'C:\\Users\\User\\AppData\\Local\\Programs\\Ollama\\ollama.exe' },
    cudaToolkit: { name: 'CUDA Toolkit', installed: true, version: '12.3', path: 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3' },
    vcRedistributable: { name: 'VC++ Redistributable 2015-2022', installed: true, version: '14.38.33130', path: null }
  },
  aiRuntimes: [
    {
      name: 'Ollama Engine',
      status: 'running',
      version: '0.1.28',
      endpoint: 'http://localhost:11434',
      modelsAvailable: ['llama3:8b', 'mistral:7b-instruct', 'nomic-embed-text']
    },
    {
      name: 'vLLM Local Server',
      status: 'stopped',
      version: '0.4.0',
      endpoint: 'http://localhost:8000',
      modelsAvailable: []
    }
  ],
  paths: {
    userHome: 'C:\\Users\\User',
    downloads: 'C:\\Users\\User\\Downloads',
    documents: 'C:\\Users\\User\\Documents',
    desktop: 'C:\\Users\\User\\Desktop',
    appData: 'C:\\Users\\User\\AppData\\Roaming\\Sarathi',
    cacheDir: 'C:\\Users\\User\\AppData\\Local\\Sarathi\\Cache',
    modelStorageDir: 'C:\\Users\\User\\.sarathi\\models'
  },
  aiCapabilities: {
    maxRecommendedModelSizeBytes: 17179869184, // 16 GB (up to 34B Q4 / 13B FP16)
    recommendedQuantizations: ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'FP16'],
    recommendedContextLength: 8192,
    preferredInferenceBackend: 'CUDA (NVIDIA RTX 4090)',
    multiModelCapable: true,
    loraReady: true,
    visionReady: true,
    embeddingReady: true
  },
  validation: {
    isReadyForAi: true,
    score: 92,
    warnings: [
      'VRAM allocation target set to 80% to avoid OS display buffer contention.'
    ],
    errors: [],
    recommendations: [
      'NVMe SSD detected on drive D: with 2.9 TB free. Recommended for storing large GGUF model weights.',
      'NVIDIA CUDA Toolkit 12.3 detected with compute capability 8.9 (FlashAttention-2 enabled).',
      'System memory (64 GB) is sufficient for multi-adapter dynamic LoRA composition.'
    ]
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

export async function getHardwareProfile(): Promise<HardwareProfile | null> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile | null>('get_hardware_profile');
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri get_hardware_profile invoke fallback to mock:', err);
  }
  return { ...mockProfileState };
}

export async function analyzeSystem(): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile>('analyze_system');
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri analyze_system invoke fallback to mock:', err);
  }
  mockProfileState = {
    ...mockProfileState,
    profileUpdatedAt: new Date().toISOString()
  };
  return { ...mockProfileState };
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
    console.debug('Tauri override_hardware_value invoke fallback to mock:', err);
  }

  const updatedProfile = JSON.parse(JSON.stringify(mockProfileState)) as HardwareProfile;
  if (!updatedProfile.overrides) updatedProfile.overrides = {};
  updatedProfile.overrides[fieldPath] = value;
  setNestedField(updatedProfile, fieldPath, value);
  updatedProfile.profileUpdatedAt = new Date().toISOString();
  mockProfileState = updatedProfile;

  return mockProfileState;
}

export async function revertHardwareOverride(fieldPath: string): Promise<HardwareProfile> {
  try {
    const profile = await getBackendService().invoke<HardwareProfile>('revert_hardware_override', {
      fieldPath,
      field_path: fieldPath
    });
    if (profile) return profile;
  } catch (err) {
    console.debug('Tauri revert_hardware_override invoke fallback to mock:', err);
  }

  const updatedProfile = JSON.parse(JSON.stringify(mockProfileState)) as HardwareProfile;
  if (updatedProfile.overrides) {
    delete updatedProfile.overrides[fieldPath];
  }
  updatedProfile.profileUpdatedAt = new Date().toISOString();
  mockProfileState = updatedProfile;

  return mockProfileState;
}

export async function validateSystem(): Promise<SystemValidationResult> {
  try {
    const validation = await getBackendService().invoke<SystemValidationResult>('validate_system');
    if (validation) return validation;
  } catch (err) {
    console.debug('Tauri validate_system invoke fallback to mock:', err);
  }
  return { ...mockProfileState.validation };
}
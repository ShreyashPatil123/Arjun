// Phase 3 & Certified Ecosystem: Model Recommendation & Certification Types

export type FitCategory = 'Recommended' | 'Compatible' | 'MayRun';
export type CertificationTier = 'Certified' | 'Compatible' | 'Experimental';

export interface NumericScores {
  instructionFollowing: number;
  reasoningQuality: number;
  hallucinationRate: number;
  codingAbility: number;
  mathematicalReasoning: number;
  jsonReliability: number;
  toolCallingAccuracy: number;
  memoryEngineCompatibility: number;
  loraAdapterSwitching: number;
  contextWindowRetention: number;
  responseStability: number;
  chatTemplateCorrectness: number;
  bosEosStopTokenCompliance: number;
  reasoningTagLeakageFilter: number;
  streamingParserStability: number;
  runtimeProcessStability: number;
  restartStatePersistence: number;
}

export interface Provenance {
  createdBy: string;
  certifiedBy: string;
  generatedWith: string;
  runnerVersion: string;
  profileHash: string;
  signature: string;
  generatedAt: string;
}

export interface PackageCertification {
  packageId: string;
  modelId: string;
  modelName: string;
  quantLabel: string;
  backend: string;
  tier: CertificationTier;
  confidenceScore: number;
  runtimeProfileId: string;
  numericScores: NumericScores;
  loraCapabilityMatrix: Record<string, CertificationTier>;
  provenance: Provenance;
  quirksAndNotes: string;
}

export interface RuntimeProfile {
  profileId: string;
  name: string;
  pinnedVersions: {
    sarathiVersion: string;
    profileSchemaVersion: string;
    llamacppVersion: string;
    llamacpp2RustVersion: string;
    certificationSpecVersion: string;
  };
  executionConfig: {
    chatTemplate: string;
    stopTokens: string[];
    contextLength: number;
    gpuLayers: number;
    threads: number;
    samplingDefaults: {
      temperature: number;
      topP: number;
      topK: number;
      minP: number;
      repeatPenalty: number;
      maxTokens: number;
    };
  };
}

export interface ModelRecommendation {
  modelId: string;
  modelName: string;
  modelFamily: string;
  providerId: string | null;

  quantization: string;
  quantizationBitsPerWeight: number;
  recommendedContext: number;
  maxPossibleContext: number;
  backend: string;
  runMode: string;

  downloadSizeBytes: number | null;
  estimatedVramBytes: number;
  estimatedRamBytes: number;
  estimatedSharedMemBytes: number;
  estimatedTotalMemoryBytes: number;
  headroomPercent: number;

  fitScore: number;
  category: FitCategory;
  confidence: string;

  explanation: string;
  warnings: string[];

  architecture: string;
  totalParameters: number;
  activeParameters: number | null;

  certification?: PackageCertification | null;

  estimatedTokensPerSec: number | null;
  performanceNote: string;
}
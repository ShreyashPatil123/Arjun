// Phase 3: Model Recommendation Types
// Mirrors Rust model_recommendation::traits structs

export type FitCategory = 'Recommended' | 'Compatible' | 'MayRun';

export interface ModelRecommendation {
  // Identity (Phase 4 uses these to locate the model)
  modelId: string;
  modelName: string;
  modelFamily: string;
  providerId: string | null;

  // Selected configuration (highest quality safe fit)
  quantization: string;
  quantizationBitsPerWeight: number;
  recommendedContext: number;
  maxPossibleContext: number;
  backend: string;
  runMode: string;

  // Resource estimates
  estimatedVramBytes: number;
  estimatedRamBytes: number;
  estimatedSharedMemBytes: number;
  estimatedTotalMemoryBytes: number;
  headroomPercent: number;

  // Scoring
  fitScore: number;
  category: FitCategory;
  confidence: string;

  // Explainability
  explanation: string;
  warnings: string[];

  // Architecture info
  architecture: string;
  totalParameters: number;
  activeParameters: number | null;

  // Performance (Phase 3: always null)
  estimatedTokensPerSec: number | null;
  performanceNote: string;
}
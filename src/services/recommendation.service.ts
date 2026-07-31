// Phase 3: Model Recommendation Service
// Wraps Tauri IPC invoke for model recommendations

import { getBackendService } from './api';
import type { ModelRecommendation } from '../types/recommendation';

/**
 * Fetches model recommendations based on a fresh hardware scan.
 * Returns ranked recommendations sorted by fit_score descending.
 */
export async function getModelRecommendations(): Promise<ModelRecommendation[]> {
  const service = getBackendService();
  return service.invoke<ModelRecommendation[]>('get_model_recommendations');
}

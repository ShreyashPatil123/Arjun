import { invoke } from '@tauri-apps/api/core';
import type { ModelProfile, InferenceParameters, AdapterRouteResult } from '../types/intelligence';

export async function getModelProfile(providerId: string, modelId: string): Promise<ModelProfile> {
  return invoke<ModelProfile>('get_model_profile', { providerId, modelId });
}

export async function updateModelProfile(
  providerId: string,
  modelId: string,
  params: InferenceParameters
): Promise<ModelProfile> {
  return invoke<ModelProfile>('update_model_profile', { providerId, modelId, params });
}

export async function refreshModelProfile(providerId: string, modelId: string): Promise<ModelProfile> {
  return invoke<ModelProfile>('refresh_model_profile', { providerId, modelId });
}

export async function routePromptCapability(
  providerId: string,
  modelId: string,
  prompt: string,
  userOverride?: string
): Promise<AdapterRouteResult> {
  return invoke<AdapterRouteResult>('route_prompt_capability', {
    providerId,
    modelId,
    prompt,
    userOverride,
  });
}

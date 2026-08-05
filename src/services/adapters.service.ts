// Adapter download and management IPC.
//
// Mirrors src-tauri/src/commands/adapters.rs.

import { invoke } from '@tauri-apps/api/core';

export interface InstalledAdapter {
  id: string;
  repoId: string;
  name: string;
  baseModelId: string;
  filePath: string;
  sizeBytes: number;
}

export interface InstalledAdapters {
  adapters: InstalledAdapter[];
  totalBytes: number;
}

export function listInstalledAdapters(
  providerId: string,
  modelId: string
): Promise<InstalledAdapters> {
  return invoke<InstalledAdapters>('list_installed_adapters', { providerId, modelId });
}

/**
 * Downloads an adapter for a model.
 *
 * Rejects PEFT safetensors adapters before fetching anything — llama.cpp cannot
 * load them, so the error explains that rather than leaving a file that fails
 * only when a request needs it.
 */
export function downloadAdapter(
  providerId: string,
  modelId: string,
  adapterRepoId: string
): Promise<InstalledAdapter> {
  return invoke<InstalledAdapter>('download_adapter', { providerId, modelId, adapterRepoId });
}

export function removeAdapter(
  providerId: string,
  modelId: string,
  adapterId: string
): Promise<void> {
  return invoke<void>('remove_adapter', { providerId, modelId, adapterId });
}

/** How sure a claim about an adapter is — see adapter_details.rs. */
export type Confidence = 'stated' | 'suggested';

export interface AdapterEffect {
  skill: string;
  description: string;
  confidence: Confidence;
}

export interface AdapterDetails {
  repoId: string;
  name: string;
  author: string;
  license: string | null;
  downloads: number;
  likes: number;
  sizeBytes: number;
  ggufReady: boolean;
  blockedReason: string | null;
  datasets: string[];
  effects: AdapterEffect[];
  notice: string | null;
}

/**
 * What an adapter changes about the model it is applied to.
 *
 * Every effect is labelled with where it came from — the author's own tags
 * (`stated`) or the repository name (`suggested`) — so a hint is never shown
 * as a fact.
 */
export function getAdapterDetails(repoId: string): Promise<AdapterDetails> {
  return invoke<AdapterDetails>('get_adapter_details', { repoId });
}

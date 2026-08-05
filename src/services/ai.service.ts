import { invoke } from '@tauri-apps/api/core';
import { listen, Event } from '@tauri-apps/api/event';
import type {
  LoadedModelInfo,
  InferenceStatusPayload,
  StreamChunkPayload,
  ChatMessage,
  GenerationParams,
  CapabilityPayload,
} from '../types/ai';

export async function loadInstalledModel(
  providerId: string,
  modelId: string,
  quantization: string
): Promise<LoadedModelInfo> {
  return invoke('load_installed_model', {
    providerId,
    modelId,
    quantization,
  });
}

export async function unloadActiveModel(): Promise<void> {
  return invoke('unload_active_model');
}

export async function getInferenceStatus(): Promise<InferenceStatusPayload> {
  return invoke('get_inference_status');
}

export async function restoreLastSession(): Promise<LoadedModelInfo | null> {
  return invoke<LoadedModelInfo | null>('restore_last_session');
}

/**
 * Sends a turn for generation.
 *
 * The backend classifies intent, applies switch hysteresis, and binds the
 * resulting capability before generating — pass `manualCapability` only to pin
 * a specific one. `'auto'`, `'none'`, and `null` all mean "classify normally".
 */
export async function sendChatMessage(
  messages: ChatMessage[],
  params?: GenerationParams,
  manualCapability?: string | null
): Promise<void> {
  return invoke('send_chat_message', {
    messages,
    params: params || null,
    manualCapability: manualCapability || null,
  });
}

export async function stopChatGeneration(): Promise<void> {
  return invoke('stop_chat_generation');
}

export async function listenInferenceStatus(
  callback: (payload: InferenceStatusPayload) => void
) {
  return listen<InferenceStatusPayload>('inference:status', (event: Event<InferenceStatusPayload>) => {
    callback(event.payload);
  });
}

export async function listenInferenceToken(
  callback: (payload: StreamChunkPayload) => void
) {
  return listen<StreamChunkPayload>('inference:token', (event: Event<StreamChunkPayload>) => {
    callback(event.payload);
  });
}

export async function listenInferenceError(
  callback: (payload: { error: string }) => void
) {
  return listen<{ error: string }>('inference:error', (event: Event<{ error: string }>) => {
    callback(event.payload);
  });
}

/**
 * Subscribes to capability changes.
 *
 * Fires once per turn, after the capability has actually been applied — so the
 * UI reports what the model is doing, not what was merely intended.
 */
export async function listenCapabilityChanged(
  callback: (payload: CapabilityPayload) => void
) {
  return listen<CapabilityPayload>('capability:changed', (event: Event<CapabilityPayload>) => {
    callback(event.payload);
  });
}

// Backward compatibility aliases for SDK client
export async function getAIStatus() {
  const status = await getInferenceStatus();
  return status.status;
}

export async function loadModel(providerId: string, modelId?: string, quantization?: string) {
  if (!modelId || !quantization) {
    throw new Error('Model ID and quantization required');
  }
  return loadInstalledModel(providerId, modelId, quantization);
}

export async function unloadModel() {
  return unloadActiveModel();
}

export async function chat(messages?: ChatMessage[]) {
  if (messages && messages.length > 0) {
    await sendChatMessage(messages);
  }
  return { message: 'Generation started' };
}
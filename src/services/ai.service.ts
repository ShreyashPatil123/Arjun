import { invoke } from '@tauri-apps/api/core';
import { listen, Event } from '@tauri-apps/api/event';
import type {
  LoadedModelInfo,
  InferenceStatusPayload,
  StreamChunkPayload,
  ChatMessage,
  GenerationParams,
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

export async function sendChatMessage(
  messages: ChatMessage[],
  params?: GenerationParams
): Promise<void> {
  return invoke('send_chat_message', {
    messages,
    params: params || null,
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
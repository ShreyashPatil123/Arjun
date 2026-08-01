export type RuntimeStatus =
  | 'NotLoaded'
  | 'Loading'
  | 'Ready'
  | 'Generating'
  | 'Unloading'
  | 'Error';

export interface LoadedModelInfo {
  modelId: string;
  modelName: string;
  quantization: string;
  filePath: string;
  contextLength: number;
  gpuLayers: number;
  threads: number;
  backendUsed: string;
  chatTemplate?: string;
  stopTokens?: string[];
  modelFamily?: string;
  activeAdapter?: string | null;
}

export interface InferenceStatusPayload {
  status: string;
  step?: string | null;
  model?: LoadedModelInfo | null;
  error?: string | null;
}

export interface StreamChunkPayload {
  text: string;
  isFinal: boolean;
  tokensGenerated?: number | null;
  finishReason?: string | null;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface GenerationParams {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  repeatPenalty?: number;
}
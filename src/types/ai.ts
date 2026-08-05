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

/**
 * What a model is for. A model carries several — a small coding model is both
 * `coding` and `small-and-fast`, so either filter surfaces it.
 */
export type ModelCategory =
  | 'lora-adapter'
  | 'coding'
  | 'reasoning'
  | 'math'
  | 'agentic'
  | 'vision'
  | 'mixture-of-experts'
  | 'long-context'
  | 'multilingual'
  | 'small-and-fast'
  | 'general';

/**
 * Display order for the category sidebar.
 *
 * `lora-adapter` leads because it changes what an entry *is* — a patch applied
 * on top of a base model, not something that runs on its own — rather than what
 * it is good at.
 */
export const MODEL_CATEGORIES: readonly ModelCategory[] = [
  'lora-adapter',
  'coding',
  'reasoning',
  'math',
  'agentic',
  'vision',
  'mixture-of-experts',
  'long-context',
  'multilingual',
  'small-and-fast',
  'general',
] as const;

export const CATEGORY_LABELS: Record<ModelCategory, string> = {
  'lora-adapter': 'LoRA adapter',
  coding: 'Coding',
  reasoning: 'Reasoning',
  math: 'Math',
  agentic: 'Agentic',
  vision: 'Vision',
  'mixture-of-experts': 'MoE',
  'long-context': 'Long context',
  multilingual: 'Multilingual',
  'small-and-fast': 'Small & fast',
  general: 'General',
};

/**
 * What an entry is, in terms someone can act on.
 *
 * The distinction that matters: a base model, fine-tune, or quantization can be
 * downloaded and run. A LoRA adapter cannot — it needs a base model underneath.
 */
export type ModelKind = 'base-model' | 'fine-tuned' | 'quantized' | 'lora-adapter';

export const KIND_LABELS: Record<ModelKind, string> = {
  'base-model': 'Base model',
  'fine-tuned': 'Fine-tuned',
  quantized: 'Quantized',
  'lora-adapter': 'LoRA adapter',
};

/** One quantization choice, for the size/quality comparison. */
export interface QuantizationOption {
  label: string;
  sizeBytes: number;
  /** Plain-English quality note, e.g. "Balanced — recommended". */
  qualityNote: string;
  /** Whether this option fits the detected memory budget. */
  fits: boolean;
}

/**
 * Presentation data for a model listing card.
 *
 * Optional fields are genuinely unknown rather than zero — the card omits them
 * instead of showing a placeholder that looks like a measurement.
 */
export interface ModelCard {
  repoId: string;
  /** HuggingFace org or user. */
  publisher: string;
  name: string;
  /** Factual one-liner assembled from metadata, not marketing copy. */
  summary: string;
  categories: ModelCategory[];
  /** Base model, fine-tune, quantization, or adapter. */
  kind: ModelKind;
  /** Plain-language note on whether this can run on its own. */
  kindExplanation: string;
  /** Datasets it was trained on, when named and informative. */
  datasets?: string[];
  license?: string | null;
  downloads: number;
  /** Compact form, e.g. "52M". */
  downloadsLabel: string;
  likes: number;
  /** Compact relative age, e.g. "2mo". Absent when the date is unknown. */
  ageLabel?: string | null;
  isFinetune: boolean;
  /**
   * True when this is a LoRA adapter — a patch for a base model, not something
   * that can be loaded and run on its own.
   */
  isLoraAdapter: boolean;
  /** Parent model, when this is a fine-tune, adapter, or re-quantization. */
  baseModel?: string | null;
  totalParameters?: number | null;
  contextLength?: number | null;
  quantizations: QuantizationOption[];
}

/** How a capability is actually being applied to the model. */
export type CapabilityBackend = 'base' | 'prompt-profile' | 'lora';

/**
 * Emitted by the backend on `capability:changed` *after* the capability has
 * been applied to the prompt, sampler, and adapter binding.
 *
 * This replaces the previous client-side `routePromptCapability` badge, which
 * displayed a routing decision that never reached inference.
 */
export interface CapabilityPayload {
  capability: string;
  displayName: string;
  /** Pre-formatted label, e.g. `Code · lora`. */
  badge: string;
  backend: CapabilityBackend;
  confidence: number;
  switched: boolean;
  /** Why this capability is active (classification / hysteresis / override). */
  reason: string;
  /** Why this backend was chosen, including any degradation from LoRA. */
  backendReason: string;
  adapterPath?: string | null;
  /** Sampling values generation actually used, after capability overrides. */
  effectiveTemperature: number;
  effectiveTopP: number;
  effectiveRepeatPenalty: number;
}
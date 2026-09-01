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
 * Categories describe what a runnable model is good at.
 */
export const MODEL_CATEGORIES: readonly ModelCategory[] = [
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
 * Every listed kind is a runnable model.
 */
export type ModelKind = 'base-model' | 'fine-tuned' | 'quantized';

export const KIND_LABELS: Record<ModelKind, string> = {
  'base-model': 'Base model',
  'fine-tuned': 'Fine-tuned',
  quantized: 'Quantized',
};

/** One quantization choice, for the size/quality comparison. */
export interface QuantizationOption {
  label: string;
  sizeBytes: number;
  /** Plain-English quality note, e.g. "Balanced — recommended". */
  qualityNote: string;
  /** Whether this option fits the detected memory budget. */
  fits: boolean;
  /** True for 1–2 bit quantizations, where output can degrade to nonsense. */
  lowQuality?: boolean;
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
  /** Base model, fine-tune, or quantization. */
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
  /** Parent model, when this is a fine-tune or re-quantization. */
  baseModel?: string | null;
  totalParameters?: number | null;
  contextLength?: number | null;
  quantizations: QuantizationOption[];
  /**
   * True when Sarathi vouches for this one: a publisher whose conversions are
   * dependable, enough real use to have surfaced problems, no reasoning tokens
   * in its output, and a size that runs on this machine.
   */
  recommended: boolean;
  /**
   * True when the model narrates its reasoning (`<think>` and similar) into its
   * replies. Read from the GGUF's own chat template where one was published.
   */
  emitsReasoning: boolean;
  /** ISO-8601 last-modified stamp, for sorting. `ageLabel` is the readable form. */
  lastModified: string;
}

/** How a capability is actually being applied to the model. */
export type CapabilityBackend = 'base' | 'prompt-profile';

/**
 * Emitted by the backend on `capability:changed` *after* the capability has
 * been applied to the prompt and sampler.
 *
 * This keeps the displayed state aligned with the route actually used for
 * generation.
 */
export interface CapabilityPayload {
  capability: string;
  displayName: string;
  /** Pre-formatted label, e.g. `Code · prompt-profile`. */
  badge: string;
  backend: CapabilityBackend;
  confidence: number;
  switched: boolean;
  /** Why this capability is active (classification / hysteresis / override). */
  reason: string;
  /** Why this backend was chosen. */
  backendReason: string;
  /** Sampling values generation actually used, after capability overrides. */
  effectiveTemperature: number;
  effectiveTopP: number;
  effectiveRepeatPenalty: number;
}

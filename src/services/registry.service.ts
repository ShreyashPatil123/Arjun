import { getBackendService } from './api';

export type ModelRuntime = 'llamaCpp' | 'pythonSidecar';

export type ModelRole =
  | 'reasoning'
  | 'coding'
  | 'vision'
  | 'documentOcr'
  | 'embedding'
  | 'rerank';

export type Classification =
  | 'internal'
  | 'processDiagram'
  | 'financial'
  | 'vendorNegotiation'
  | 'unreleasedDesign'
  | 'internalCorrespondence'
  | 'businessStrategy';

export interface ModelEntry {
  id: string;
  name: string;
  version: string;
  license: string;
  sha256: string | null;
  runtime: ModelRuntime;
  roles: ModelRole[];
  quantization: string | null;
  parametersB: number;
  activeParametersB: number | null;
  contextLength: number;
  weightsBytes: number;
  permittedClassifications: Classification[];
  path: string;
  enabled: boolean;
}

/** What the router chose, and every reason that led there. */
export interface RoutingDecision {
  modelId: string;
  modelName: string;
  role: ModelRole;
  /** What the prompt was taken to be asking for. */
  intent: string;
  confidence: number;
  /** True when the preferred model did not fit and something smaller was used. */
  usedFallback: boolean;
  /** Ordered, human-readable. Shown verbatim. */
  reasons: string[];
  gpuPlanSummary: string;
  fullyOnGpu: boolean;
}

export const ROLE_LABELS: Record<ModelRole, string> = {
  reasoning: 'Reasoning',
  coding: 'Coding',
  vision: 'Vision',
  documentOcr: 'Document OCR',
  embedding: 'Embedding',
  rerank: 'Reranking',
};

export const CLASSIFICATION_LABELS: Record<Classification, string> = {
  internal: 'Internal',
  processDiagram: 'P&ID / process diagram',
  financial: 'Financial',
  vendorNegotiation: 'Vendor negotiation',
  unreleasedDesign: 'Unreleased design',
  internalCorrespondence: 'Internal correspondence',
  businessStrategy: 'Business strategy',
};

/** What happened when the routed model was made ready. */
export interface ActivationOutcome {
  modelId: string;
  modelName: string;
  /** True when nothing had to happen — the common case, and free. */
  alreadyResident: boolean;
  /** What was released to make room, if anything. */
  evicted: string | null;
  reason: string;
  tookMs: number;
}

/** A routed and loaded model, ready to run. */
export interface PreparedModel {
  routing: RoutingDecision;
  activation: ActivationOutcome;
}

export interface OrchestratorModelSelection {
  providerId: string;
  modelId: string;
  quantization: string;
}

export const registryService = {
  listModels(): Promise<ModelEntry[]> {
    return getBackendService().invoke<ModelEntry[]>('list_registered_models');
  },

  /** The file an administrator edits to register a model. */
  manifestPath(): Promise<string> {
    return getBackendService().invoke<string>('model_manifest_path');
  },

  /** Exact model variant that will be loaded as the orchestrator on startup. */
  getOrchestratorModel(): Promise<OrchestratorModelSelection> {
    return getBackendService().invoke<OrchestratorModelSelection>('get_orchestrator_model');
  },

  /** Administrator-only: persist any ready installed model as the orchestrator. */
  setOrchestratorModel(
    selection: OrchestratorModelSelection
  ): Promise<OrchestratorModelSelection> {
    return getBackendService().invoke<OrchestratorModelSelection>('set_orchestrator_model', {
      providerId: selection.providerId,
      modelId: selection.modelId,
      quantization: selection.quantization,
    });
  },

  /**
   * Which model would handle this prompt, without running anything.
   * Rejects with an explanation naming what would fix it when nothing is eligible.
   */
  previewRouting(prompt: string, classification?: Classification): Promise<RoutingDecision> {
    return getBackendService().invoke<RoutingDecision>('preview_routing', {
      prompt,
      classification: classification ?? null,
    });
  },

  /**
   * Picks the right model for a prompt and loads it, with no human step.
   *
   * Rejects while another task holds the model, naming the holder — swapping
   * underneath a running task would change models partway through it.
   */
  prepareModelFor(prompt: string, classification?: Classification): Promise<PreparedModel> {
    return getBackendService().invoke<PreparedModel>('prepare_model_for', {
      prompt,
      classification: classification ?? null,
    });
  },

  /** Who currently holds the model, if anyone. */
  residency(): Promise<{ heldBy: string | null }> {
    return getBackendService().invoke('model_residency');
  },
};

import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { getBackendService } from './api';

/** Where a swap narrates itself. One message per stage. */
const ORCHESTRATOR_SWAP_EVENT = 'models://orchestrator';

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

/** One stage of a swap, as it happens. */
export interface OrchestratorSwapStep {
  phase: 'releasing' | 'loading' | 'ready' | 'failed';
  modelId: string;
  /**
   * The model this stage is about. During `releasing` that is the model going
   * away, not the one arriving — the point of showing the stage at all is to
   * say which is which.
   */
  modelName: string;
  /** Present on `failed`, and on nothing else. */
  detail: string | null;
}

/** What choosing an orchestrator did. */
export interface OrchestratorChange {
  /** The coordinates written to the configuration, as the registry states them. */
  selected: OrchestratorModelSelection;
  modelId: string;
  modelName: string;
  /** Models whose servers were stopped to make room. Empty when none ran. */
  released: string[];
  /**
   * Whether the new orchestrator is up and answering. The choice is saved
   * either way: a model that cannot start right now is still the model that
   * was chosen, and the next launch will try it again.
   */
  serving: boolean;
  /** Why it is not serving, when it is not. */
  detail: string | null;
}

/**
 * One model as the library lists it, from the manifest rather than from a
 * downloaded package.
 *
 * The two lists differ, and the difference is the bug this type exists to fix:
 * the Models screen used to show only packages the downloader had installed,
 * so the Unlimited-OCR weights — registered, loaded, and reading documents in
 * chat — appeared nowhere on it.
 */
export interface LibraryModel {
  id: string;
  name: string;
  quantization: string | null;
  roles: string[];
  modalities: string[];
  parametersB: number;
  contextLength: number;
  /** What the manifest says the weights weigh. */
  weightsBytes: number;
  license: string;
  enabled: boolean;
  /**
   * Empty means the model may not be used on any classified material. That is
   * the state everything detected on disk starts in, deliberately.
   */
  permittedClassifications: string[];
  runtime: string;
  path: string;
  projector: string | null;
  /** Whether the weights are where the entry says they are. */
  present: boolean;
  /** Size on disk now, or null when the file is missing. */
  bytesOnDisk: number | null;
}

/** What one detection pass found. */
export interface DetectionReport {
  /**
   * Directories actually walked, so "nothing new" is distinguishable from
   * "nowhere was looked".
   */
  roots: string[];
  filesSeen: number;
  alreadyRegistered: number;
  added: LibraryModel[];
  /** The library as it stands after the pass. */
  models: LibraryModel[];
  /**
   * True when something was added, and so when a restart is needed before the
   * router can use it.
   */
  restartRequired: boolean;
}

export const registryService = {
  listModels(): Promise<ModelEntry[]> {
    return getBackendService().invoke<ModelEntry[]>('list_registered_models');
  },

  /**
   * Every model in the manifest, read from disk.
   *
   * Separate from {@link listModels}, which returns the registry as it was
   * loaded at startup. This one re-reads the file, so a model registered by
   * {@link detectSystemModels} moments ago is listed without a restart.
   */
  listLibraryModels(): Promise<LibraryModel[]> {
    return getBackendService().invoke<LibraryModel[]>('list_library_models');
  },

  /**
   * Walks the machine for weight files and registers the ones the manifest
   * does not list.
   *
   * Adds only. An entry an administrator wrote by hand is never edited or
   * replaced, and everything newly found arrives cleared for no classification
   * until somebody reviews it.
   *
   * `extraRoots` is for a library kept somewhere ARJUN would not think to
   * look; the standard locations are always searched.
   */
  detectSystemModels(extraRoots?: string[]): Promise<DetectionReport> {
    return getBackendService().invoke<DetectionReport>('detect_system_models', {
      extraRoots: extraRoots ?? [],
    });
  },


  /** The file an administrator edits to register a model. */
  manifestPath(): Promise<string> {
    return getBackendService().invoke<string>('model_manifest_path');
  },

  /**
   * Exact model variant that will be loaded as the orchestrator on startup, or
   * `null` when an administrator has not chosen one. No model is hardcoded as a
   * default, so "not chosen yet" is a real state rather than an error.
   */
  getOrchestratorModel(): Promise<OrchestratorModelSelection | null> {
    return getBackendService().invoke<OrchestratorModelSelection | null>('get_orchestrator_model');
  },

  /**
   * Administrator-only: choose the orchestrator and swap to it now.
   *
   * Resolves once the new model is answering, so it takes as long as loading
   * the weights takes. Follow {@link subscribeOrchestratorSwap} while it runs
   * to show which stage it is on rather than an undifferentiated spinner.
   *
   * `selected` comes back as the *registry* spells the coordinates, which is
   * not always how the installed package spells them — the package reads its
   * quantisation off the file name and records "GGUF" when it cannot parse
   * one. Saving the package's spelling is what previously made this setting do
   * nothing at all, so the resolved form is what is returned and stored.
   */
  setOrchestratorModel(selection: OrchestratorModelSelection): Promise<OrchestratorChange> {
    return getBackendService().invoke<OrchestratorChange>('set_orchestrator_model', {
      providerId: selection.providerId,
      modelId: selection.modelId,
      quantization: selection.quantization,
    });
  },

  /**
   * Follows a swap as it happens.
   *
   * Returns the unsubscribe function; call it when the screen unmounts. The
   * backend emits for the life of the session, and a listener left behind
   * keeps a closed component's state setters alive.
   */
  async subscribeOrchestratorSwap(
    onStep: (step: OrchestratorSwapStep) => void
  ): Promise<UnlistenFn> {
    return listen<OrchestratorSwapStep>(ORCHESTRATOR_SWAP_EVENT, ({ payload }) => {
      onStep(payload);
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

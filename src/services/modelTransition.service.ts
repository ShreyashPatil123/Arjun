/**
 * Changing the model an agent runs on, as the frontend sees it.
 *
 * ## Why this is not part of the agent edit form
 *
 * Because saving a form cannot do what a model change has to do. The backend
 * refuses a submitted model binding on `agent_registry_update` outright — see
 * `RegistryError::ModelBindingNeedsTransition` — because moving an agent to a
 * different model has to stop its work at a safe point, settle anything in
 * flight, save its state, prove the new model can actually hold the task, and be
 * able to undo itself.
 *
 * So a screen that edits an agent sends every other field through
 * `agentRegistryService.update` and this one through `begin` below. A form that
 * submitted a changed `models` object is refused with a sentence naming this
 * path, which is a worse experience than not offering it.
 *
 * ## These types are one contract written twice
 *
 * Every shape here has a counterpart in
 * `src-tauri/src/agent_runtime/model_transition.rs` and
 * `src-tauri/src/commands/agents.rs`, and the field names are what
 * `serde(rename_all = "camelCase")` produces from them. A field added on one
 * side and not the other is a runtime `undefined` that typechecks.
 *
 * ## The one thing a surface must not do
 *
 * Treat a missing `outcome` as success, or `pending` as failure. A handoff has
 * five outcomes and they are not two: `pending` means it is still going *or*
 * parked waiting for a person to answer an approval, `rolledBack` means the
 * agent is safely back where it was, and `rollbackFailed` means the agent may be
 * bound to a model nothing is serving and must not be given work until somebody
 * looks. `needsHuman` is the flag for that last one.
 */

import { getBackendService } from './api';

/**
 * How far a handoff goes in proving the destination works.
 *
 * `loadAndVerify` is the only honest setting when there is work to hand over:
 * the run continues on the new model immediately, so "it can probably be served"
 * is not good enough. `planOnly` is for reassigning an idle agent, where loading
 * a large model would evict whatever somebody else's conversation is mid-turn
 * on.
 */
export type VerifyDestination = 'loadAndVerify' | 'planOnly';

/** Where a handoff has got to. The backend's own spelling. */
export type TransitionPhase =
  | 'requested'
  | 'draining'
  | 'awaitingSettlement'
  | 'checkpointed'
  | 'validating'
  | 'loading'
  | 'recompiling'
  | 'commitPending'
  | 'committed'
  | 'rollingBack'
  | 'rolledBack'
  | 'failed'
  | 'rollbackFailed';

/** What it came to, in the words a surface needs. */
export type TransitionOutcome =
  | 'succeeded'
  /** Still going, or parked waiting for a person. Not a failure. */
  | 'pending'
  /** Refused or abandoned. Nothing changed. */
  | 'failed'
  /** Undone. The agent is on the model it started on. */
  | 'rolledBack'
  /** Undone unsuccessfully. Needs a person before the agent is given work. */
  | 'rollbackFailed';

/**
 * Which bytes a model id referred to at the moment of a transition.
 *
 * `basis` matters: `weightsSha256` is a claim about a file, `derived` is a claim
 * about a registry entry, and `unregistered` is neither. Two records whose
 * digests match are the same model only when both say `weightsSha256`.
 */
export interface ModelFingerprint {
  modelId: string;
  digest: string;
  basis: 'weightsSha256' | 'derived' | 'unregistered';
  weightsBytes: number;
  /** What the registry declares. Not what a server was started with. */
  declaredWindow: number;
  roles: string[];
}

/** Where the number a turn is budgeted against came from. */
export type WindowSource = 'serverReported' | 'registryDeclared' | 'notMeasured';

/** What a server's own tokenizer made of a fixed probe string. */
export interface TokenizerProbe {
  probe: string;
  tokens: number;
}

/** The destination as it actually came up. */
export interface ServedBinding {
  modelId: string;
  servedModelId: string;
  /**
   * The window the server was started with — routinely smaller than the model's
   * trained window, because GPU layers are bought by walking it down. Read this
   * rather than `declaredWindow` when showing what a model can hold.
   */
  servedWindow: number;
  windowSource: WindowSource;
  templateId?: string;
  tokenizer?: TokenizerProbe;
  supportsToggledReasoning: boolean;
  healthy: boolean;
}

/** The source revision a handoff was frozen against. */
export interface SourceFreeze {
  /** `-1` means this deployment has no runtime memory graph. */
  graphRevision: number;
  /** `-1` means there was no run to hand over. */
  lastEventSeq: number;
  sourceManifestHash?: string;
  checkpointHash?: string;
  at: string;
}

/** One step of a handoff's history. */
export interface PhaseEntry {
  phase: TransitionPhase;
  at: string;
  detail?: string;
}

/** Which models an agent may use. Mirrors `agentRegistry.service.ts`. */
export interface ModelBinding {
  defaultModelId: string | null;
  fallbackModelIds: string[];
  eligibleModelIds: string[];
}

/** One handoff, as the deployment keeps it. */
export interface TransitionRecord {
  transitionId: string;
  schemaVersion: number;

  /** Never changes across a transition. The point of the whole feature. */
  agentId: string;
  /** Absent for an idle-agent reassignment. */
  runId?: string;
  taskId?: string;
  attemptId?: string;

  fromModel: ModelFingerprint;
  toModel: ModelFingerprint;
  fromDefinitionVersion: number;
  toDefinitionVersion: number;
  fromBinding: ModelBinding;
  toBinding: ModelBinding;

  freeze?: SourceFreeze;
  /** Absent when nothing was loaded — see `planOnly`. */
  served?: ServedBinding;
  sourceManifestHash?: string;
  targetManifestHash?: string;
  portableStateHash?: string;
  artifactHashes?: string[];
  /**
   * Whether the two models demonstrably tokenise differently.
   *
   * `undefined` means one of them could not be asked, which is not the same as
   * `false`. Do not render an unasked question as "unchanged".
   */
  tokenizerChanged?: boolean;

  phase: TransitionPhase;
  history: PhaseEntry[];
  because?: string;
  /** The typed refusal, when a validation produced one. */
  refusal?: { refusal: string } & Record<string, unknown>;
  /** What a person has to settle, when the handoff is parked. */
  awaiting?: string[];

  requestedBy: string;
  reason: string;
  requestedAt: string;
  settledAt?: string;
}

/** One thing a handoff left behind, and why. */
export interface DroppedNote {
  what: string;
  because: string;
}

/** A handoff plus what a screen would otherwise re-derive. */
export interface TransitionView {
  record: TransitionRecord;
  outcome: TransitionOutcome;
  /** One sentence, already in an operator's terms. Render this, not the phase. */
  summary: string;
  /** True when the agent must not be given work until somebody looks. */
  needsHuman: boolean;
  /**
   * What the new model was deliberately not given, present once a handoff
   * commits.
   *
   * Worth showing. The first answer after a model change can legitimately read
   * differently — the previous model's summary of its own transcript does not
   * cross, because it is a summary only that model could read — and somebody
   * noticing that difference deserves this list rather than a bug report.
   */
  notCarried?: DroppedNote[];
}

/** Where an agent stands with respect to its model. */
export interface TransitionStatus {
  agentId: string;
  /** `null` means routing chooses per turn for this agent's role. */
  currentModelId: string | null;
  /** Send this back with a change request; a concurrent edit is then refused. */
  definitionVersion: number;
  /** While this is present the agent cannot be moved again. */
  open: TransitionView | null;
  /** Newest first. */
  history: TransitionView[];
}

/** What to ask for. */
export interface HandoffRequest {
  agentId: string;
  targetModelId: string;
  /** The version the screen was opened at. */
  expectedDefinitionVersion: number;
  reason?: string;
  /** The run to hand over. Omit to reassign an idle agent. */
  runId?: string;
  taskId?: string;
  projectId?: string;
  verify: VerifyDestination;
  /**
   * Normally omitted. The backend then uses the reserves the interrupted turn
   * was actually budgeted with, and refuses if it has none recorded rather than
   * inventing a figure — a guessed reserve would decide whether an operator's
   * correction fits in the new model's window.
   */
  reserves?: { toolSchemas: number; output: number; framing: number };
}

/**
 * Whether this handoff is waiting for a person rather than for the machine.
 *
 * The distinction a spinner cannot express: a handoff at `draining` resolves
 * itself when the current round finishes, and one at `awaitingSettlement` never
 * will until somebody answers an approval or accounts for an interrupted
 * action.
 */
export function isWaitingForSomebody(view: TransitionView): boolean {
  return view.record.phase === 'awaitingSettlement';
}

/** What the person has to settle, if anything. */
export function awaiting(view: TransitionView): string[] {
  return view.record.awaiting ?? [];
}

/**
 * Whether the two models count tokens differently, as a three-way answer.
 *
 * `'unknown'` when one of the servers would not answer a tokenize request,
 * which is the ordinary case for an external endpoint ARJUN did not start.
 * Rendering that as "the same" would claim a measurement nobody made.
 */
export function tokenizerChange(
  view: TransitionView,
): 'changed' | 'same' | 'unknown' {
  const changed = view.record.tokenizerChanged;
  if (changed === undefined || changed === null) return 'unknown';
  return changed ? 'changed' : 'same';
}

/**
 * The window the target actually came up with, and how much to trust it.
 *
 * `null` when nothing was loaded. `measured` is false when the server would not
 * report its own window and the registry's declared figure was used — a
 * difference worth showing, because a slow or refused turn reads back to it.
 */
export function servedWindow(
  view: TransitionView,
): { tokens: number; measured: boolean } | null {
  const served = view.record.served;
  if (!served) return null;
  return {
    tokens: served.servedWindow,
    measured: served.windowSource === 'serverReported',
  };
}

export const modelTransitionService = {
  /**
   * Moves an agent to a different model.
   *
   * Resolves with the record whatever the outcome — including a refusal, which
   * is why the result has to be read rather than assumed. It rejects only when
   * the handoff could not be *opened*: not an administrator, no such agent, a
   * concurrent edit, or another handoff already under way.
   *
   * A result whose `outcome` is `pending` has not finished. When
   * `isWaitingForSomebody` is true a person has to settle what `awaiting` names,
   * and asking again afterwards continues the same handoff; otherwise the run is
   * finishing a round and asking again shortly continues it.
   */
  begin(request: HandoffRequest): Promise<TransitionView> {
    return getBackendService().invoke<TransitionView>(
      'agent_model_transition_begin',
      { request },
    );
  },

  /** Where an agent stands, and what has been tried. */
  status(agentId: string): Promise<TransitionStatus> {
    return getBackendService().invoke<TransitionStatus>(
      'agent_model_transition_status',
      { agentId },
    );
  },

  /**
   * Undoes a handoff that has not settled.
   *
   * Bounded: the binding goes back and the source model is served again if
   * something was released to make room. The work is not unwound — a document
   * written before the failure stays written, and its receipt stays in the run's
   * notes so a resumption does not write it a second time.
   */
  rollback(transitionId: string): Promise<TransitionView> {
    return getBackendService().invoke<TransitionView>(
      'agent_model_transition_rollback',
      { transitionId },
    );
  },

  /**
   * Settles every handoff the process died in the middle of.
   *
   * Runs automatically at start-up; this is here so an operator can ask for it
   * after a crash without restarting. Administrator only, and safe to call
   * twice — a settled handoff is not swept again.
   */
  reconcile(): Promise<TransitionView[]> {
    return getBackendService().invoke<TransitionView[]>(
      'agent_model_transition_reconcile',
    );
  },
};

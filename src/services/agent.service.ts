import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { getBackendService } from './api';

/**
 * Starting, watching and stopping an agent run.
 *
 * A run is not a request/response — it is minutes of work with tool calls in
 * the middle, and an operator who needs to see what is happening while it
 * happens. So the API is in two halves: `start` resolves once with the answer,
 * and `subscribe` streams the lifecycle in between.
 *
 * The loop itself runs in a Node child process built from OpenClaw's
 * `agent-core`; every tool call it wants to make is decided in Rust first. None
 * of that is visible here, and deliberately so: the UI's business is showing
 * work, not deciding what is permitted.
 */

/** Which material this is, so the router only considers models cleared for it. */
export type Classification =
  | 'internal'
  | 'processDiagram'
  | 'financial'
  | 'vendorNegotiation'
  | 'unreleasedDesign'
  | 'internalCorrespondence'
  | 'businessStrategy';

/**
 * What starts a run.
 *
 * Deliberately no model. Which model answers is the backend router's decision —
 * letting the UI name one would make automatic selection optional, which is the
 * opposite of what the product is demonstrating.
 */
export interface StartRunRequest {
  prompt: string;
  classification?: Classification;
  /** Overrides the default instructions. Scripted demonstrations only. */
  systemPrompt?: string;
  /**
   * Echoed back on the run's first event.
   *
   * `start` does not resolve until the run is over, so without this a caller
   * watching the event stream cannot tell its own run's events from another
   * window's until the very end. Naming the run is still the backend's job —
   * this only identifies the stream.
   */
  correlationId?: string;
}

/** Why a model was chosen. Rendered verbatim in the task trace. */
export interface RoutingDecision {
  modelId: string;
  modelName: string;
  role: 'reasoning' | 'coding' | 'vision' | 'documentOcr' | 'embedding' | 'rerank';
  /** What the prompt was taken to be asking for. */
  intent: string;
  confidence: number;
  /** True when the first choice did not fit and something smaller was used. */
  usedFallback: boolean;
  /** Ordered and human-readable. Show these as given; do not summarise them. */
  reasons: string[];
  gpuPlanSummary: string;
  fullyOnGpu: boolean;
}

/** Where the model actually ran. */
export interface Endpoint {
  /** Always loopback. Both runtimes are reached the same way. */
  baseUrl: string;
  servedModelId: string;
  /** True when ARJUN started the server; false when an operator runs it. */
  managed: boolean;
  runtime: 'llamaCpp' | 'pythonSidecar';
}

/**
 * One planned step, and whether the run left behind the evidence for it.
 *
 * `done` is judged against what the run produced — a successful tool call, an
 * answer, a completed check — never against the model's account of itself. A
 * model that says it wrote the document and never called the tool leaves the
 * step unfinished, which is the whole point of the field.
 */
export interface PlanStep {
  ordinal: number;
  /** What the step is for, in the person's terms. */
  intent: string;
  done: boolean;
  /** What would settle this step. Shown on an unfinished one so the gap says
   *  what is missing rather than only that something is. */
  settledBy: string;
}

/** How one tool call ended. */
export type CallOutcome = 'succeeded' | 'failed' | 'refused';

/** One tool call the run made. Arguments are deliberately not carried. */
export interface ToolCallRecord {
  tool: string;
  outcome: CallOutcome;
  /** What the tool reported, trimmed — what the model saw, not a summary. */
  detail: string;
  at: string;
}

/** One thing a person was asked to allow during the run, and what they said. */
export interface ApprovalRecord {
  id: string;
  tool: string;
  target: string;
  arguments: string[];
  consequences: string;
  requestedAt: string;
  /** `approved`, `rejected`, or `pending` for one nobody answered. */
  state: string;
  decidedBy: string | null;
  decidedAt: string | null;
  because: string | null;
}

/**
 * Why a run ended.
 *
 * Mirrors `StopReason` in `orchestrator::plan`, tagged on `reason`. Read
 * `stoppedBecause` for the sentence to show; read this only where the shape
 * itself matters.
 */
export type StopReason =
  | { reason: 'completed' }
  | { reason: 'stepsExhausted'; taken: number; allowed: number }
  | { reason: 'timeExhausted'; allowedSeconds: number }
  | { reason: 'looping'; tool: string; repeats: number }
  | { reason: 'awaitingApproval'; tool: string }
  | { reason: 'failed'; detail: string };

/**
 * The plan a run is held to.
 *
 * Fixed before the model is told anything, and not extendable by it. Rendered
 * as given: the steps are what the run said it would do, and showing an
 * incomplete plan honestly is most of the point of having one.
 */
export interface PlanRecord {
  steps: PlanStep[];
  maxSteps: number;
  maxDurationSeconds: number;
  /** Tool names, exactly as the model would have had to write them. */
  permittedTools: string[];
  repeatLimit: number;
  stepsTaken: number;
  /** Absent while the run is still going. */
  stopReason: StopReason | null;
  /** The stop reason as a sentence, ready to show. */
  stoppedBecause: string;
}

/** How serious a verification finding is. */
export type Severity = 'blocking' | 'advisory';

export interface Finding {
  severity: Severity;
  /** What is wrong, in the words a reviewer would use. */
  detail: string;
  /** The text it is about, so a reviewer can find it. */
  excerpt: string | null;
}

/** Whether an answer may be presented as finished. */
export type Standing =
  | { standing: 'ready' }
  | { standing: 'needsReview'; blocking: number; advisory: number };

/**
 * What the verifier found in the final answer.
 *
 * It does not edit the answer and does not withhold it — it reports whether
 * every claim resolves to a passage the run actually retrieved, and whether
 * every figure matches a calculation the engine actually ran. An answer that
 * fails is shown with its findings attached, because a reviewer who cannot see
 * what the model said cannot judge it.
 */
export interface VerificationReport {
  standing: Standing;
  findings: Finding[];
  citationsResolved: number;
  figuresChecked: number;
}

export type ArtifactKind = 'document' | 'workbook' | 'text';

/** A file the run produced, re-opened and checked rather than taken on trust. */
export interface ArtifactReport {
  /** Relative to the run's working directory — the name the model wrote. */
  name: string;
  path: string;
  kind: ArtifactKind;
  /** The template a document was rendered from, so a re-check asks the same
   *  question. Null for files that have no template. */
  template: string | null;
  bytes: number;
  /** False when the file is missing, empty, will not open, or is incomplete. */
  sound: boolean;
  detail: string;
  problems: string[];
  producedAt: string;
}

export interface RunSummary {
  runId: string;
  text: string;
  turns: number;
  routing: RoutingDecision;
  endpoint: Endpoint;
  plan: PlanRecord;
  /** Absent when the run produced no answer to check. */
  verification: VerificationReport | null;
  artifacts: ArtifactReport[];
}

/** One passage the run stood on, as its `[E3]` marker refers to it. */
export interface EvidenceRecord {
  marker: number;
  citation: string;
  documentName: string;
  page: number;
  excerpt: string;
}

/** One step of a calculation the engine performed. */
export interface CalculationStep {
  description: string;
  result: string;
}

export interface CalculationRecord {
  expression: string;
  inputs: string[];
  steps: CalculationStep[];
  value: number;
  unit: string;
  formatted: string;
  rounding: string;
  /** Always true — the engine computed this, not a model. */
  deterministic: boolean;
}

/** Everything kept about one finished run. */
export interface TaskRecord {
  runId: string;
  prompt: string;
  startedAt: string;
  finishedAt: string;
  durationSeconds: number;
  userId: string;
  routing: RoutingDecision;
  endpoint: Endpoint;
  plan: PlanRecord;
  answer: string;
  turns: number;
  verification: VerificationReport | null;
  artifacts: ArtifactReport[];
  evidence: EvidenceRecord[];
  calculations: CalculationRecord[];
  toolCalls: ToolCallRecord[];
  approvals: ApprovalRecord[];
  /** Set when the run ended badly, in the words shown to the person. */
  failure: string | null;
}

/** A row on the Tasks screen. */
export interface TaskSummary {
  runId: string;
  /** Who ran it. The backend has already filtered the list to what the
   *  signed-in person may read. */
  userId: string;
  prompt: string;
  startedAt: string;
  finishedAt: string;
  durationSeconds: number;
  modelName: string;
  intent: string;
  turns: number;
  artifactCount: number;
  evidenceCount: number;
  toolCallCount: number;
  /** Steps planned but never reached. Non-zero is the signal to look. */
  unfinishedSteps: number;
  approvalsPending: number;
  stoppedBecause: string;
  /** False when it failed, needs review, or produced an unsound file. */
  ready: boolean;
  failure: string | null;
}

/**
 * Lifecycle events from the agent loop.
 *
 * Mirrors `AgentEvent` in `@openclaw/agent-core`, narrowed to what the UI
 * renders. Tool *arguments* are stripped before they leave the backend — they
 * can carry document text and the audit record already holds them under access
 * control, so they do not travel a second path just to be displayed.
 */
export type AgentEvent =
  | { type: 'agent_start' }
  | { type: 'agent_end' }
  | { type: 'turn_start' }
  | { type: 'turn_end' }
  | { type: 'message_start'; message: unknown }
  | { type: 'message_update'; message: unknown }
  | { type: 'message_end'; message: unknown }
  | { type: 'tool_execution_start'; toolCallId: string; toolName: string }
  | { type: 'tool_execution_update'; toolCallId: string; toolName: string }
  | {
      /**
       * Older history was replaced by a summary so the run could continue.
       *
       * Worth showing: the model's answers after this point are grounded in a
       * summary of the earlier turns rather than the turns themselves, and an
       * operator reading the trace should know that.
       */
      type: 'context_compacted';
      tokensBefore: number;
      tokensAfter: number;
      messagesSummarised: number;
    }
  | {
      type: 'tool_execution_end';
      toolCallId: string;
      toolName: string;
      isError: boolean;
      /** False when the gateway refused before the tool ran. */
      executionStarted?: boolean;
    }
  | {
      /**
       * The plan this run is held to, published before the first turn.
       *
       * Emitted by the backend rather than the loop: the plan is fixed before
       * the model is told anything, so it is known before there is any loop
       * activity to report.
       */
      type: 'plan_ready';
      plan: PlanRecord;
      /** Whatever the caller sent on `StartRunRequest`, echoed once. */
      correlationId?: string | null;
    }
  | {
      /** A step spent. Sent after the tool ran, whatever it returned. */
      type: 'plan_step';
      tool: string;
      stepsTaken: number;
      maxSteps: number;
      stepsDone: number;
      stepsPlanned: number;
    }
  | {
      /**
       * The run hit its budget, or went in circles, and will do nothing more.
       *
       * Worth showing the moment it happens: the loop still has to wind down
       * and produce a final answer, and an operator watching a suddenly quiet
       * trace should know it is stopping rather than thinking.
       */
      type: 'plan_stopped';
      reason: string;
      tool: string;
    };

/** One event, tagged with the run it belongs to. */
export interface AgentEventEnvelope {
  runId: string;
  event: AgentEvent;
}

/** Backend event channel. One stream for every run; filter on `runId`. */
const AGENT_EVENT = 'agent://event';

export const agentService = {
  /**
   * Runs one prompt to completion.
   *
   * Resolves with the final answer. Subscribe first if you want to show
   * anything before it settles — a run with tool calls takes a while, and an
   * interface that shows nothing until the end looks broken.
   */
  start(request: StartRunRequest): Promise<RunSummary> {
    return getBackendService().invoke<RunSummary>('agent_start_run', { request });
  },

  /**
   * Stops a run in flight.
   *
   * Resolves `false` when there was nothing to stop, which is an ordinary race
   * rather than a failure — do not surface it as an error.
   */
  abort(runId: string): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_abort_run', { runId });
  },

  /**
   * Corrects a run already in flight, without stopping it.
   *
   * Applied at the next point the loop is safe to interrupt — before an
   * unstarted tool call or the next model turn — never mid-tool. Resolves
   * `false` when the run had already finished, which is an ordinary race and
   * should not be surfaced as an error.
   */
  steer(runId: string, text: string): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_steer_run', { runId, text });
  },

  /**
   * Subscribes to run lifecycle events.
   *
   * Returns the unsubscribe function. Call it on unmount: the backend keeps
   * emitting for the life of the session, and a listener left behind will
   * update a component that is no longer mounted.
   *
   * Pass `runId` to receive only one run's events.
   */
  async subscribe(
    handler: (envelope: AgentEventEnvelope) => void,
    runId?: string,
  ): Promise<UnlistenFn> {
    return listen<AgentEventEnvelope>(AGENT_EVENT, ({ payload }) => {
      if (runId && payload.runId !== runId) return;
      handler(payload);
    });
  },

  /**
   * Whether the agent runtime can start on this machine.
   *
   * Starts it if it is not already running, so this doubles as the "can this
   * deployment run an agent at all" check for the health screen. Rejects with a
   * readable reason when the bundle is missing or Node is absent.
   */
  health(): Promise<{ ready: boolean; pid: number; node: string }> {
    return getBackendService().invoke('agent_runtime_health');
  },

  /**
   * Every task this machine has run, newest first.
   *
   * Read from disk on each call rather than cached: a record is written by the
   * run that produced it, and a list held in memory goes stale the moment a
   * second window runs something.
   */
  history(): Promise<TaskSummary[]> {
    return getBackendService().invoke<TaskSummary[]>('agent_task_history');
  },

  /** One task in full — its plan, routing, evidence, working and artifacts. */
  task(runId: string): Promise<TaskRecord> {
    return getBackendService().invoke<TaskRecord>('agent_task', { runId });
  },

  /**
   * Re-opens a finished task's files and reports what is in them *now*.
   *
   * The saved record says what the check found when the run ended; this says
   * what it finds today. The two disagreeing is worth knowing — a deliverable
   * can be moved, replaced or truncated long after the run that made it.
   */
  taskArtifacts(runId: string): Promise<ArtifactReport[]> {
    return getBackendService().invoke<ArtifactReport[]>('agent_task_artifacts', { runId });
  },

  /**
   * Shows a produced file in the operating system's file manager.
   *
   * Reveals rather than opens: handing a path a model named to the shell to
   * *open* would let that file decide which application runs.
   */
  revealArtifact(runId: string, name: string): Promise<void> {
    return getBackendService().invoke<void>('agent_reveal_artifact', { runId, name });
  },
};

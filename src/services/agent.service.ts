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

export interface RunSummary {
  runId: string;
  text: string;
  turns: number;
  routing: RoutingDecision;
  endpoint: Endpoint;
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
};

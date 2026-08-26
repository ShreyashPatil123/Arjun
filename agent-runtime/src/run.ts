/**
 * One agent run: model turns, tool calls, and the events a person watches.
 *
 * This is the thin layer between OpenClaw's `Agent` and ARJUN's Rust core. It
 * owns three things and deliberately nothing else:
 *
 * - building the model record from what Rust chose (the *routing* decision is
 *   Rust's, in `registry::router` -- this side never picks a model);
 * - installing the authorisation hook so every tool call is put to the gateway;
 * - forwarding lifecycle events so the UI can show work as it happens.
 *
 * Everything that decides anything -- which model, whether a tool may run, what
 * a tool does -- lives on the other side of the wire.
 */

import { Agent, convertToLlm, type AgentEvent } from "@openclaw/agent-core";
import { createLlmRuntime, type Model } from "@openclaw/ai";
import { registerBuiltInApiProviders } from "@openclaw/ai/providers";
import type { RpcPeer } from "./peer.js";
import { RunCompactor } from "./compaction.js";
import { payloadPolicy } from "./providers.js";
import { withToolCallRepair } from "./repair.js";
import { GrantLedger, authorizeToolCall, buildTools } from "./tools.js";

/** What Rust sends with `run.start`. */
export interface RunRequest {
  runId: string;
  prompt: string;
  systemPrompt: string;
  /** The routed model. Chosen by `registry::router` on the Rust side. */
  model: {
    id: string;
    name?: string;
    provider: string;
    /** Endpoint of the local inference server. Must be loopback. */
    baseUrl: string;
    contextWindow?: number;
    maxTokens?: number;
    input?: ("text" | "image")[];
    reasoning?: boolean;
  };
}

export interface RunOutcome {
  runId: string;
  /** Assistant text of the final turn, for callers that want just the answer. */
  text: string;
  turns: number;
  stopReason?: string;
}

/**
 * A model served locally has no price and no vendor.
 *
 * agent-core requires the cost table, and zeros are the truthful entry: the
 * marginal cost of a token on a machine the organisation already owns is not a
 * number this product should invent. Anything non-zero here would show up in
 * run manifests as a fabricated figure.
 */
const LOCAL_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const;

/** Stands in for the credential a loopback inference server does not want. */
const LOCAL_PLACEHOLDER_KEY = "sovereign-local";

/** Loopback-only. Anything else is a routing bug and is refused before a socket opens. */
function assertLoopback(baseUrl: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch (cause) {
    throw new Error(`Model baseUrl is not a URL: ${baseUrl}`, { cause });
  }
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const loopback = host === "localhost" || host === "::1" || /^127\./.test(host);
  if (!loopback) {
    throw new Error(
      `Model endpoint ${url.origin} is not loopback. This runtime only reaches inference servers on this machine.`,
    );
  }
}

function toModel(spec: RunRequest["model"]): Model {
  assertLoopback(spec.baseUrl);
  return {
    id: spec.id,
    name: spec.name ?? spec.id,
    api: "openai-completions",
    provider: spec.provider,
    baseUrl: spec.baseUrl,
    reasoning: spec.reasoning ?? false,
    input: spec.input ?? ["text"],
    cost: LOCAL_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens ?? 4096,
  } as Model;
}

/** A run in flight, so `run.abort` can reach it. */
export interface ActiveRun {
  abort(reason?: unknown): void;
  /**
   * Injects a correction into a run already in flight.
   *
   * The alternative an operator has otherwise is to stop the run and start
   * again, losing every tool result gathered so far. On a task that has already
   * read a 200-page drawing set that is an expensive way to say "use the 2019
   * revision".
   *
   * Applied at the next point the loop is safe to interrupt — before an
   * unstarted tool call or the next model turn — never in the middle of one.
   */
  steer(text: string): void;
}

/**
 * Runs one prompt to completion.
 *
 * Resolves when the agent goes idle. Rejects only for failures that are not the
 * model's to recover from -- a refused tool call is a tool result, not an
 * exception, because the model can read it and try something else.
 */
export async function startRun(
  peer: RpcPeer,
  request: RunRequest,
  register: (run: ActiveRun) => void,
): Promise<RunOutcome> {
  const { runId } = request;
  const ledger = new GrantLedger();
  const runtime = createLlmRuntime();
  registerBuiltInApiProviders(runtime.registry);

  const model = toModel(request.model);
  const tools = buildTools(peer, ledger, runId, request.model.id);

  const compactor = new RunCompactor({
    model,
    runtime,
    apiKey: LOCAL_PLACEHOLDER_KEY,
    onCompacted: (event) =>
      peer.notify("run.event", {
        runId,
        event: { type: "context_compacted", ...event },
      }),
  });

  const agent = new Agent({
    streamFn: withToolCallRepair(
      runtime.streamSimple,
      tools.map((tool) => tool.name),
    ),
    /**
     * The harness converter, not the default.
     *
     * agent-core's default keeps only user, assistant and tool-result messages
     * and silently drops everything else. Two things depend on that not
     * happening, and both fail quietly rather than loudly:
     *
     * - **Compaction.** Its output is a `compactionSummary` message. Dropped,
     *   compaction appears to work while discarding the very thing it produced,
     *   and the model simply loses the earlier history.
     * - **Interrupt recovery.** When an operator stops a run mid-tool, `Agent`
     *   appends a `custom` message saying the previous turn was interrupted and
     *   tools may have partially executed. Dropped, a continuation is never told,
     *   and may repeat a write that already happened.
     */
    convertToLlm,
    transformContext: (messages, signal) => compactor.transform(messages, signal),
    /**
     * Read-only tools run together.
     *
     * A document task typically wants several searches at once. Executing them
     * one at a time makes the operator wait for the sum rather than the slowest,
     * for no safety gain — each call is still authorised individually, and a
     * search cannot affect what another search returns. Anything that writes
     * declares `executionMode: "sequential"` on itself.
     */
    toolExecution: "parallel",
    /**
     * A correction is applied at the next safe point, not queued behind the
     * whole run. `one-at-a-time` so two rapid corrections do not both land
     * before the model has responded to either.
     */
    steeringMode: "one-at-a-time",
    initialState: {
      systemPrompt: request.systemPrompt,
      model,
      tools,
    },
    beforeToolCall: (context) => authorizeToolCall(peer, ledger, runId, context),
    /**
     * A local inference server needs no credential, but the OpenAI client
     * refuses to construct without one. So a placeholder is supplied rather
     * than the transport being special-cased for local providers.
     *
     * It is a constant on purpose: there is no secret here to leak, and reading
     * a real key from the environment would create a path by which one could
     * reach a local endpoint and, from there, a log.
     */
    getApiKey: () => LOCAL_PLACEHOLDER_KEY,
    /**
     * Local-model quirks, applied to every request.
     *
     * Installed unconditionally because the policy is a no-op for models that
     * need nothing — a per-model switch is one an operator would have to
     * remember to set, and forgetting produces an approval note that opens with
     * the model thinking out loud.
     */
    onPayload: payloadPolicy(request.model.reasoning ?? false),
  });

  let turns = 0;
  agent.subscribe((event: AgentEvent) => {
    if (event.type === "turn_end") turns += 1;
    // Best-effort by design: a dropped event costs the operator a progress line,
    // whereas awaiting delivery would let a slow UI stall the run.
    peer.notify("run.event", { runId, event: redactEvent(event) });
  });

  register({
    abort: (reason) => agent.abort(reason),
    steer: (text) =>
      agent.steer({
        role: "user",
        content: [{ type: "text", text }],
        timestamp: Date.now(),
      }),
  });

  try {
    await agent.prompt(request.prompt);
  } finally {
    // A run that ends holding grants means authorisation outlived its call. That
    // is a defect, but clearing is the safe half of handling it either way.
    ledger.clear();
  }

  const messages = agent.state.messages;
  const last = messages[messages.length - 1];
  const text =
    last && last.role === "assistant" && Array.isArray(last.content)
      ? last.content
          .filter((block): block is { type: "text"; text: string } => block.type === "text")
          .map((block) => block.text)
          .join("\n")
      : "";

  return { runId, text, turns, stopReason: agent.state.errorMessage ? "error" : undefined };
}

/**
 * Strips tool *arguments* from the event stream.
 *
 * The UI needs to know a tool ran; it does not need the arguments echoed back
 * over a second channel, and those can carry document text or a file path that
 * the audit record already holds under access control. Sending less is cheaper
 * to defend than sending everything and redacting at the display.
 */
function redactEvent(event: AgentEvent): AgentEvent {
  if (
    event.type === "tool_execution_start" ||
    event.type === "tool_execution_update" ||
    event.type === "tool_execution_end"
  ) {
    return { ...event, args: undefined } as AgentEvent;
  }
  return event;
}

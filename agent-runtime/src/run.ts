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
import { RunCompactor, type PreservedState } from "./compaction.js";
import { ContextLedger } from "./context-ledger.js";
import { WorkingNotes, type WorkingNotesState } from "./working-notes.js";
import { payloadPolicy } from "./providers.js";
import { withToolCallRepair } from "./repair.js";
import { withCallTiming } from "./timing.js";
import { GrantLedger, authorizeToolCall, buildTools, fetchCatalogue } from "./tools.js";
import { observeToolResult } from "./note-taking.js";

/** What Rust sends with `run.start`. */
export interface RunRequest {
  runId: string;
  /**
   * The id of the assistant `Message` row the chat surface reserved for this
   * turn via `agent_append_turn`. Attached to every `message_start`,
   * `message_update`, and `message_end` event so the consumer can route each
   * token to the right cell without filtering by `runId`.
   */
  messageId: string;
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
  /**
   * When this run must stop, as epoch milliseconds.
   *
   * The same instant the Rust side is holding. Sent so the loop can stop
   * *itself* at a point it knows is safe, rather than being killed from outside
   * in the middle of a turn — this side knows where its own safe points are and
   * the other side does not.
   *
   * It is not a second authority. The only thing it can do is end the run
   * earlier than Rust would; every tool call still goes through the gateway,
   * and nothing here decides whether an action is permitted.
   */
  deadlineMs?: number;
  /**
   * Notes carried over from an earlier attempt at this run.
   *
   * Sent when a run is resumed after the process went away. What makes the
   * resumption safe rather than merely faster is `completed`: it names the side
   * effects that already happened, so the model is told not to repeat them
   * instead of rediscovering by doing them twice.
   */
  notes?: Partial<WorkingNotesState>;
  /**
   * State the Rust side owns and this side must carry across compaction
   * unchanged. Refreshed by `run.note`; see {@link PreservedState}.
   */
  preserved?: PreservedState;
}

export interface RunOutcome {
  runId: string;
  /** Assistant text of the final turn, for callers that want just the answer. */
  text: string;
  turns: number;
  stopReason?: string;
  /**
   * The run's notes as they finished.
   *
   * Returned so Rust can persist them with the task record. A run that ends
   * without handing these back is a run whose next attempt starts from nothing,
   * which is the case this whole mechanism exists to remove.
   */
  notes: WorkingNotesState;
  /** Where the context stood at the end. Shown on the task trace. */
  ledger: ReturnType<ContextLedger["snapshot"]>;
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
  /**
   * Updates the state this side must preserve, and the run's notes.
   *
   * Pushed from Rust rather than pulled, because everything in it — the plan,
   * the approvals, the evidence markers — is decided there, and a pull would
   * mean this side asking mid-compaction over a channel that is also carrying
   * the tool call it is compacting around.
   */
  note(update: { preserved?: PreservedState; notes?: Partial<WorkingNotesState> }): void;
  /** The notes as they stand. Read when the run ends. */
  readonly notes: WorkingNotes;
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

  // Seeded from what Rust sent. On a first attempt that is nothing; on a
  // resumption it is the record of what already happened, including the side
  // effects that must not happen twice.
  const notes = WorkingNotes.from(request.notes);
  let preserved: PreservedState = { ...(request.preserved ?? {}) };

  // The notes are kept from what the tools returned rather than from what the
  // model chose to write down. See `note-taking.ts` — the entries that make a
  // resumption safe are exactly the ones a model does not think to record.
  // Deferred discovery: Rust says which tools this run is eligible for, and
  // only those get their schema loaded. The plan that decides it was fixed
  // before the model was told anything, so nothing the model does afterwards
  // can widen this — and a tool it is never shown is one it cannot spend a turn
  // being refused for asking about.
  //
  // A catalogue that could not be fetched comes back empty, which is the
  // failing-closed reading: silence from the gateway is not a list of tools.
  const catalogue = await fetchCatalogue(peer, runId);
  const tools = buildTools(
    peer,
    ledger,
    runId,
    request.model.id,
    (observation) => observeToolResult(notes, observation),
    catalogue.tools,
  );

  const contextLedger = new ContextLedger(request.model.contextWindow ?? 0);
  // Measured once. Neither the system prompt nor the tool catalogue changes
  // during a run, and re-counting them every turn would spend real time
  // counting characters that are identical to last turn's.
  contextLedger.setText("system", request.systemPrompt);
  contextLedger.setText(
    "toolSchema",
    tools.map((tool) => `${tool.name}${tool.description ?? ""}${JSON.stringify(tool.parameters ?? {})}`).join(""),
  );

  const compactor = new RunCompactor({
    model,
    runtime,
    apiKey: LOCAL_PLACEHOLDER_KEY,
    notes,
    ledger: contextLedger,
    // Read at the moment of compaction rather than captured, so a decision
    // taken two turns ago is carried and one taken since the run started is
    // not silently the stale copy.
    preserved: () => ({
      ...preserved,
      evidenceRefs: preserved.evidenceRefs ?? notes.state.evidenceIds,
      unresolvedIssues: preserved.unresolvedIssues ?? notes.state.openQuestions,
      recentFiles: preserved.recentFiles ?? notes.state.artifactIds,
    }),
    onCompacted: (event) =>
      peer.notify("run.event", {
        runId,
        event: { type: "context_compacted", ...event },
      }),
  });

  const agent = new Agent({
    // Timed on the outside of the repair wrapper, so a call the repair layer
    // re-issues is counted as the second call it is. Counting them together
    // would report one very slow model instead of two ordinary ones, which is
    // the distinction the measurement exists to make.
    streamFn: withCallTiming(
      withToolCallRepair(
        runtime.streamSimple,
        tools.map((tool) => tool.name),
      ),
      runId,
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

  /**
   * Stops the run when its deadline passes.
   *
   * `agent.abort` is the same path an operator's stop button takes, so the
   * wind-down is the one already tested: the loop finishes what it is doing,
   * appends the message saying the turn was interrupted, and returns whatever
   * it had. A deadline that killed the process instead would leave a tool call
   * in flight and nobody able to say whether it took effect.
   */
  let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
  if (typeof request.deadlineMs === "number") {
    const remaining = request.deadlineMs - Date.now();
    if (remaining <= 0) {
      // Already past it before the first turn. Refused rather than started, so
      // a run that waited too long in a queue does not spend a model call to
      // discover it has no time left.
      throw new Error(
        "This task's time budget had already expired before the loop started, so nothing was run.",
      );
    }
    deadlineTimer = setTimeout(() => {
      agent.abort("the task reached the time limit its plan allowed");
    }, remaining);
    // Never hold the process open on its own account: if everything else has
    // finished, a pending deadline is not a reason to stay alive.
    deadlineTimer.unref?.();
  }

  let turns = 0;
  // Stateful translator. Without state, llama-server's `text_start` +
  // `text_delta*` + `text_end` triple would each carry the full accumulated
  // text, producing the same answer 2-3 times in a row. The translator
  // tracks which (run, content block) has been sent and only emits a
  // `message_update` for genuinely new text.
  const translator = new MessageTranslator(request.messageId);
  agent.subscribe((event: AgentEvent) => {
    if (event.type === "turn_end") turns += 1;
    // Best-effort by design: a dropped event costs the operator a progress line,
    // whereas awaiting delivery would let a slow UI stall the run.
    //
    // Two-pass forwarding. The OpenClaw `message_*` events carry a different
    // shape than the Arjun chat surface expects, so `Translator` maps them
    // to the wire contract (with the front-end's `messageId` attached).
    // Everything else is forwarded as-is after `redactEvent` strips tool
    // arguments. Both lists are merged so the chat sees a single ordered
    // stream of `run.event` frames.
    const translated = translator.translate(event);
    if (translated.length > 0) {
      for (const wire of translated) {
        peer.notify("run.event", { runId, event: wire });
      }
    } else {
      peer.notify("run.event", { runId, event: redactEvent(event) });
    }
  });

  register({
    abort: (reason) => agent.abort(reason),
    steer: (text) =>
      agent.steer({
        role: "user",
        content: [{ type: "text", text }],
        timestamp: Date.now(),
      }),
    note: (update) => {
      if (update.preserved) preserved = { ...preserved, ...update.preserved };
      if (update.notes) applyNotes(notes, update.notes);
    },
    notes,
  });

  try {
    await agent.prompt(request.prompt);
  } finally {
    if (deadlineTimer) clearTimeout(deadlineTimer);
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

  return {
    runId,
    text,
    turns,
    stopReason: agent.state.errorMessage ? "error" : undefined,
    notes: notes.state,
    ledger: contextLedger.snapshot(),
  };
}

/**
 * Folds an update into the notes, through the setters rather than by assignment.
 *
 * The caps and the de-duplication live in the setters. Assigning the fields
 * directly would let one `run.note` carrying a hundred evidence markers put a
 * hundred markers into a list whose ceiling is sixty-four — which is how a
 * bounded structure quietly stops being bounded.
 */
function applyNotes(notes: WorkingNotes, update: Partial<WorkingNotesState>): void {
  if (typeof update.goal === "string") notes.setGoal(update.goal);
  if (update.stage) notes.atStage(update.stage.ordinal, update.stage.intent);
  if (typeof update.nextAction === "string") notes.setNextAction(update.nextAction);
  for (const decision of update.decisions ?? []) {
    notes.decided(decision.what, decision.because, decision.at);
  }
  for (const id of update.evidenceIds ?? []) notes.sawEvidence(id);
  for (const id of update.calculationIds ?? []) notes.calculated(id);
  for (const id of update.artifactIds ?? []) notes.produced(id);
  for (const question of update.openQuestions ?? []) notes.asked(question);
  for (const effect of update.completed ?? []) {
    notes.didEffect(effect.tool, effect.target, effect.at);
  }
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

/**
 * Wire shape the Arjun chat surface consumes.
 *
 * The TypeScript `AgentEvent` union in `src/services/agent.service.ts` is the
 * source of truth for the contract. We mirror it here as a structural type so
 * a drift on either side fails to type-check rather than failing at runtime.
 *
 * Only the three message-stream events are part of the streaming contract;
 * every other event is forwarded as-is after `redactEvent`.
 */
type WireEvent =
  | { type: "message_start"; messageId: string; role: "assistant" }
  | { type: "message_update"; messageId: string; delta: string }
  | {
      /**
       * The model is reasoning privately, or has stopped.
       *
       * Carries no reasoning. `characters` is the *size* of what the model
       * produced, which is a progress signal in the same way a byte count is,
       * and `elapsedMs` is how long it has been at it. Neither can be turned
       * back into a single token of the thought.
       *
       * This exists because dropping the thinking stream entirely — which is
       * the right thing to do with its *content* — also dropped the only
       * evidence that anything was happening. A reasoning model that thinks
       * for ninety seconds before its first word left the chat surface with
       * nothing to show for a minute and a half.
       */
      type: "model_thinking";
      messageId: string;
      state: "start" | "active" | "end";
      characters: number;
      elapsedMs: number;
    }
  | {
      type: "message_end";
      messageId: string;
      finishReason: "stop" | "length" | "tool_calls" | "content_filter" | "error";
      tokensIn?: number;
      tokensOut?: number;
    };

/**
 * Translates OpenClaw agent events into the wire shape the Arjun chat expects.
 *
 * The chat subscribes to `agent://event` for token-level updates and filters
 * each event on `event.messageId`. OpenClaw's `message_*` events do not carry
 * a `messageId` — they carry the whole `AgentMessage` (or its
 * `assistantMessageEvent` slice) — so without this translation every
 * streaming event is dropped at the consumer and the cell stays on
 * "thinking…". The translation is the single source of contract between the
 * two event worlds and is the only place that needs to change if the wire
 * shape ever evolves.
 *
 * Safety rules:
 *  - Only `text_delta` contributes to the visible answer. `thinking_delta`
 *    is the model's chain-of-thought and is intentionally *not* exposed on
 *    the live channel; the audit record holds it under access control.
 *  - `toolcall_delta` is a wire-format repair artefact, not visible prose,
 *    so it is not forwarded as a `message_update` either.
 *  - The `messageId` is the one the front-end reserved on
 *    `agent_append_turn`; the same id appears on every event in the stream.
 *  - One OpenClaw `message_update` may carry several `assistantMessageEvent`
 *    sub-events, so the translator yields zero or more wire events per input.
 */
export class MessageTranslator {
  private sentTextStart = new Set<number>();
  private sentTextEnd = new Set<number>();
  private sawTextDelta = new Set<number>();
  private seenStart = false;
  private seenEnd = false;
  /** When the current run of private reasoning began, or null if none is open. */
  private thinkingSince: number | null = null;
  /** How many characters of reasoning this block has produced. Never the text. */
  private thinkingChars = 0;
  /** When the last `active` tick went out, so the channel is not flooded. */
  private thinkingTickedAt = 0;
  /**
   * How many of each inner event type the loop delivered.
   *
   * The chat surface can only stream as finely as the events it is given, so
   * when an answer arrives in one lump the question is always the same: did
   * the model not stream, or did something between the model and here glue
   * the pieces back together? A shape count answers it in one log line, and
   * counts are all it holds — never a fragment of what the events carried.
   */
  private readonly shape = new Map<string, number>();

  constructor(
    private readonly messageId: string,
    private readonly now: () => number = Date.now,
  ) {}

  /**
   * Opens or advances the private-reasoning signal.
   *
   * `size` is the length of the delta, and it is the only thing taken from
   * it. The delta itself is not read, not stored, and not passed on.
   */
  private thinking(size: number): WireEvent[] {
    const at = this.now();
    this.thinkingChars += size;
    if (this.thinkingSince === null) {
      this.thinkingSince = at;
      this.thinkingTickedAt = at;
      return [
        {
          type: "model_thinking",
          messageId: this.messageId,
          state: "start",
          characters: this.thinkingChars,
          elapsedMs: 0,
        },
      ];
    }
    // A reasoning model emits deltas as fast as it can decode. Forwarding one
    // event per delta would put thousands of frames through the stdio channel
    // to move a number the person reads once a second. Ticked instead.
    if (at - this.thinkingTickedAt < THINKING_TICK_MS) return [];
    this.thinkingTickedAt = at;
    return [
      {
        type: "model_thinking",
        messageId: this.messageId,
        state: "active",
        characters: this.thinkingChars,
        elapsedMs: at - this.thinkingSince,
      },
    ];
  }

  /**
   * Closes the private-reasoning signal, if one is open.
   *
   * Called on `thinking_end`, but also on the first visible text and on
   * `message_end`: a model that stops reasoning by simply starting to answer
   * never sends `thinking_end`, and without this the surface would show
   * "Thinking" underneath an answer that was already being written.
   */
  private endThinking(): WireEvent[] {
    if (this.thinkingSince === null) return [];
    const elapsed = this.now() - this.thinkingSince;
    const characters = this.thinkingChars;
    this.thinkingSince = null;
    this.thinkingChars = 0;
    return [
      {
        type: "model_thinking",
        messageId: this.messageId,
        state: "end",
        characters,
        elapsedMs: elapsed,
      },
    ];
  }

  translate(event: AgentEvent): WireEvent[] {
    if (event.type === "message_start") {
      if (this.seenStart) return [];
      this.seenStart = true;
      this.seenEnd = false;
      this.sentTextStart.clear();
      this.sentTextEnd.clear();
      this.sawTextDelta.clear();
      this.thinkingSince = null;
      this.thinkingChars = 0;
      return [{ type: "message_start", messageId: this.messageId, role: "assistant" }];
    }

    if (event.type === "message_update") {
      const inner = event.assistantMessageEvent;
      this.shape.set(inner.type, (this.shape.get(inner.type) ?? 0) + 1);

      if (inner.type === "text_delta") {
        const delta = (inner as { delta?: unknown }).delta;
        if (typeof delta !== "string" || delta.length === 0) return [];
        const contentIndex = (inner as { contentIndex?: number }).contentIndex ?? 0;
        this.sawTextDelta.add(contentIndex);
        // Visible text means the reasoning pass is over, whether or not the
        // model bothered to say so.
        const closed = this.endThinking();
        // Once we've sent the full block via text_start or text_end, ignore
        // further deltas for that block — the wire contract is "delta = new
        // text", and the deltas are echoes of the text_start/text_end payload.
        if (this.sentTextStart.has(contentIndex)) return closed;
        if (this.sentTextEnd.has(contentIndex)) return closed;
        return [...closed, { type: "message_update", messageId: this.messageId, delta }];
      }

      if (inner.type === "text_start" || inner.type === "text_end") {
        const partial = (inner as { partial?: { content?: Array<{ type: string; text?: string }> } })
          .partial;
        const block = partial?.content?.[0];
        if (!block || block.type !== "text" || typeof block.text !== "string" || block.text.length === 0) {
          return [];
        }
        const contentIndex = (inner as { contentIndex?: number }).contentIndex ?? 0;

        if (inner.type === "text_start") {
          // If deltas already arrived for this block, suppress the text_start
          // payload: the consumer has the text already.
          if (this.sawTextDelta.has(contentIndex)) {
            this.sentTextStart.add(contentIndex);
            return [];
          }
          if (this.sentTextStart.has(contentIndex)) return [];
          this.sentTextStart.add(contentIndex);
          return [{ type: "message_update", messageId: this.messageId, delta: block.text }];
        }

        // text_end: emit a final delta only if we have not streamed the
        // text via deltas, and we have not already emitted a text_start
        // for the same block (which would itself be a duplicate).
        if (this.sawTextDelta.has(contentIndex)) {
          this.sentTextEnd.add(contentIndex);
          return [];
        }
        if (this.sentTextEnd.has(contentIndex)) return [];
        this.sentTextEnd.add(contentIndex);
        return [{ type: "message_update", messageId: this.messageId, delta: block.text }];
      }

      // Private reasoning. The *content* never leaves this function: no
      // branch below reads `delta`, `content`, or `partial` for anything but
      // its length, and none of them produces a `message_update`. What does
      // go out is that the model is reasoning, how long for, and how much it
      // has produced — the same class of fact as a byte counter, and the only
      // thing that keeps the surface honest during a long reasoning pass.
      if (inner.type === "thinking_start") {
        return this.thinking(0);
      }
      if (inner.type === "thinking_delta") {
        const delta = (inner as { delta?: unknown }).delta;
        return this.thinking(typeof delta === "string" ? delta.length : 0);
      }
      if (inner.type === "thinking_end") {
        return this.endThinking();
      }

      // toolcall_* is a wire-format repair artefact, not visible prose, so it
      // is not forwarded either. The audit record holds the model-side view
      // under access control; the chat surface only ever sees the text.
      return [];
    }

    if (event.type === "message_end") {
      if (this.seenEnd) return [];
      this.seenEnd = true;
      const shape = [...this.shape.entries()]
        .map(([type, count]) => `${type}=${count}`)
        .sort()
        .join(" ");
      process.stderr.write(`[agent-runtime:log] [stream] messageId=${this.messageId} ${shape}
`);
      // A model that reasoned and then stopped without answering still has an
      // open thinking block. Closing it here is what stops the surface
      // spinning on a run that is already over.
      const closed = this.endThinking();
      const message = event.message;
      const stopReason = message.role === "assistant" ? message.stopReason : undefined;
      const usage = message.role === "assistant" ? message.usage : undefined;
      return [
        ...closed,
        {
          type: "message_end",
          messageId: this.messageId,
          finishReason: mapStopReason(stopReason),
          tokensIn: usage?.input,
          tokensOut: usage?.output,
        },
      ];
    }

    return [];
  }
}

/**
 * How often an in-progress reasoning pass reports its size.
 *
 * A second is slower than the model produces and faster than a person reads,
 * which is the whole requirement. Lower would spend frames on a number nobody
 * looked at; higher would make a long pause look like a stall again.
 */
const THINKING_TICK_MS = 1000;

/**
 * Backwards-compatible stateless wrapper. Used by the unit tests; the
 * production path uses {@link MessageTranslator} so a run cannot double-emit.
 */
export function translateForWire(
  event: AgentEvent,
  runId: string,
  messageId: string,
): WireEvent[] {
  void runId;
  const translator = new MessageTranslator(messageId);
  return translator.translate(event);
}

/**
 * Maps OpenClaw's `StopReason` to the chat surface's `finishReason` union.
 *
 * OpenClaw uses `"stop" | "length" | "toolUse" | "error" | "aborted"`. The
 * chat surface accepts `"stop" | "length" | "tool_calls" | "content_filter" |
 * "error"`. We collapse the variants an operator never needs to distinguish
 * into the closest equivalent; a future revision that needs the distinction
 * can extend the chat union.
 */
function mapStopReason(
  reason: unknown,
): "stop" | "length" | "tool_calls" | "content_filter" | "error" {
  if (reason === "length") return "length";
  if (reason === "toolUse") return "tool_calls";
  if (reason === "aborted" || reason === "error") return "error";
  // Default: any unknown future value lands in the safe bucket.
  return "stop";
}

/**
 * Keeping a long run inside the model's context window.
 *
 * ## The failure this prevents
 *
 * A refinery task — read a scanned report, search the manuals, compute
 * something, draft a note — is twenty or more turns, and every tool result adds
 * to the transcript. A local model's window is 8k to 32k tokens, not 200k. Left
 * alone the run does not slow down or degrade; it **stops**, because the
 * inference server refuses a prompt at or over its window. That is a demo
 * failing in front of an audience with an error about token counts.
 *
 * ARJUN's Rust engine has that refusal at `ai_engine/runtime.rs`, and
 * `ContextManager::trim_context` next to it was never implemented. This is what
 * fills that hole, using OpenClaw's compaction rather than a fresh attempt —
 * the parts that are hard to get right are exactly the parts already solved
 * there.
 *
 * ## What is reused, and why each part matters
 *
 * - `estimateTokens` / `estimateContextTokens` — counts an image block as a flat
 *   {@link IMAGE_BLOCK_TOKENS}. ARJUN feeds rendered PDF pages to vision models,
 *   and a character-count heuristic would read a page image as ~0 tokens and
 *   compact far too late.
 * - `findCutPoint` — chooses where to cut so an assistant tool call is never
 *   separated from its tool result. Cutting between them produces a transcript
 *   the provider rejects as malformed, which looks like a bug in the loop.
 * - `generateSummary` — carries the previous summary forward, so compacting
 *   twice refines one summary instead of summarising a summary.
 * - `capCompactionSummary` — bounds the summary, so it cannot itself grow into
 *   the thing that overflows the window.
 *
 * ## Where it runs
 *
 * `Agent.transformContext`, which rewrites the context before each provider
 * request and leaves the stored transcript intact. That split is what we want:
 * the model sees a summary, while the audit record keeps every message.
 */

import {
  createCompactionSummaryMessage,
  DEFAULT_COMPACTION_SETTINGS,
  estimateContextTokens,
  findCutPoint,
  generateSummary,
  shouldCompact,
  type AgentMessage,
  type CompactionSettings,
} from "@openclaw/agent-core";
// Not on the package barrel upstream. Imported through the subpath the vendored
// tsconfig already maps rather than by patching the barrel, which would add a
// conflict to every future re-sync for one symbol.
import { capCompactionSummary } from "@openclaw/agent-core/harness/compaction";
import type { Model } from "@openclaw/ai";
import type { AgentCoreCompletionRuntimeDeps } from "@openclaw/agent-core";

/** What compaction did, for the event stream and the run record. */
export interface CompactionEvent {
  tokensBefore: number;
  tokensAfter: number;
  /** Transcript messages now represented by the summary rather than sent whole. */
  messagesSummarised: number;
}

export interface CompactorOptions {
  model: Model;
  runtime: AgentCoreCompletionRuntimeDeps;
  /** Placeholder credential; a loopback server wants none but the client demands one. */
  apiKey: string;
  /** Called when a compaction happens, so an operator can be told. */
  onCompacted?: (event: CompactionEvent) => void;
  settings?: Partial<CompactionSettings>;
}

/**
 * Settings for a local model, which has far less room than a cloud one.
 *
 * Upstream reserves 16k tokens and keeps 20k of recent context — sensible
 * against a 200k window, and larger than the entire window of a model ARJUN
 * routinely runs. Both are therefore derived from the window rather than fixed:
 * a fifth reserved for the summary request and its output, two fifths kept as
 * recent context. On a 200k window this lands near the upstream numbers; on an
 * 8k one it stays proportionate instead of demanding more than exists.
 */
export function settingsForWindow(contextWindow: number): CompactionSettings {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) {
    return { ...DEFAULT_COMPACTION_SETTINGS, enabled: false };
  }
  return {
    enabled: true,
    reserveTokens: Math.max(512, Math.floor(contextWindow * 0.2)),
    keepRecentTokens: Math.max(512, Math.floor(contextWindow * 0.4)),
  };
}

/**
 * Wraps messages as the session entries `findCutPoint` expects.
 *
 * The ids are positional and exist only for the length of one call. ARJUN does
 * not adopt OpenClaw's session tree as a persistence format — the audit ledger
 * is the record — but the cut-point selection is written against that shape and
 * is the part worth reusing, so the shape is supplied.
 */
function asEntries(messages: AgentMessage[]) {
  return messages.map((message, index) => ({
    type: "message" as const,
    id: `m${index}`,
    parentId: index === 0 ? null : `m${index - 1}`,
    timestamp: new Date(message.timestamp ?? 0).toISOString(),
    message,
  }));
}

/**
 * Compacts one run's context as it grows.
 *
 * Stateful across turns: it remembers the summary produced so far and how much
 * of the transcript that summary already covers, so each compaction extends the
 * previous one rather than starting again.
 */
export class RunCompactor {
  readonly #options: CompactorOptions;
  readonly #settings: CompactionSettings;
  #summary?: string;
  /** Messages the summary stands in for: `messages[0..covered)`. */
  #covered = 0;
  #compactions = 0;

  constructor(options: CompactorOptions) {
    this.#options = options;
    this.#settings = {
      ...settingsForWindow(options.model.contextTokens ?? options.model.contextWindow ?? 0),
      ...options.settings,
    };
  }

  get compactions(): number {
    return this.#compactions;
  }

  /** What the model is shown, given the transcript and any summary so far. */
  #project(messages: AgentMessage[]): AgentMessage[] {
    if (!this.#summary || this.#covered === 0) {
      return messages;
    }
    const summary = createCompactionSummaryMessage(
      this.#summary,
      this.#tokensAt(messages.slice(0, this.#covered)),
      new Date(messages[0]?.timestamp ?? Date.now()).toISOString(),
    ) as unknown as AgentMessage;
    return [summary, ...messages.slice(this.#covered)];
  }

  #tokensAt(messages: AgentMessage[]): number {
    return estimateContextTokens(messages).tokens;
  }

  /**
   * The `transformContext` hook.
   *
   * Measures the *projected* context, not the raw transcript: once a summary
   * exists, the raw transcript stays over the limit forever and measuring it
   * would compact on every single turn.
   */
  async transform(messages: AgentMessage[], signal?: AbortSignal): Promise<AgentMessage[]> {
    const window = this.#options.model.contextTokens ?? this.#options.model.contextWindow ?? 0;
    let projected = this.#project(messages);
    const tokensBefore = this.#tokensAt(projected);

    if (!shouldCompact(tokensBefore, window, this.#settings)) {
      return projected;
    }

    const entries = asEntries(messages);
    const { firstKeptEntryIndex } = findCutPoint(
      entries,
      this.#covered,
      entries.length,
      this.#settings.keepRecentTokens,
    );

    // Nothing new to fold in. Returning the projection unchanged is the honest
    // answer: the request may still be too large, and the provider's own
    // refusal names the real problem better than a summary of nothing would.
    if (firstKeptEntryIndex <= this.#covered) {
      return projected;
    }

    const toSummarise = messages.slice(this.#covered, firstKeptEntryIndex);

    // Two failure shapes, both of which must leave the run alive: a returned
    // error result, and a throw. `generateSummary` propagates whatever the
    // completion function raises, so the transport being down surfaces here as
    // an exception rather than an `err`. Catching only one of the two would
    // mean a model server that dies mid-run takes the task with it — a failure
    // an operator experiences as ARJUN crashing, not as summarisation failing.
    let summary: string | undefined;
    try {
      const result = await generateSummary(
        toSummarise,
        this.#options.model,
        this.#settings.reserveTokens,
        this.#options.apiKey,
        undefined,
        signal,
        undefined,
        this.#summary,
        undefined,
        undefined,
        this.#options.runtime,
      );
      summary = result.ok ? result.value : undefined;
    } catch {
      summary = undefined;
    }

    if (summary === undefined) {
      // The context is returned as it was. If it really is too large the
      // provider says so, which is a clearer error than one about summarisation
      // the operator never asked for.
      return projected;
    }

    this.#summary = capCompactionSummary(summary);
    this.#covered = firstKeptEntryIndex;
    this.#compactions += 1;

    projected = this.#project(messages);
    this.#options.onCompacted?.({
      tokensBefore,
      tokensAfter: this.#tokensAt(projected),
      messagesSummarised: this.#covered,
    });
    return projected;
  }
}

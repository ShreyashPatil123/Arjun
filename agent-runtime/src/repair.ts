/**
 * Recovering a tool call a small model wrote as prose.
 *
 * ## The failure this fixes
 *
 * Quantised 7B-class models are the ones a plant workstation can actually run,
 * and they get the *shape* of a tool call wrong a meaningful fraction of the
 * time: the intent is right, but it arrives as text in the answer rather than
 * in the provider's tool-call field. To the loop that is an assistant turn that
 * called nothing, so the run stops one step early and the operator sees a
 * confident description of a search that never happened.
 *
 * The rate is small per call and compounds across a multi-step task. A dozen
 * steps at a few percent each is closer to a coin toss than to a process.
 *
 * ## Why repair rather than constrain
 *
 * ARJUN's retired Rust orchestrator fought this with GBNF grammars: constrain
 * the sampler so a malformed call cannot be emitted. That works, but it costs a
 * second pass per step, because constraining the *reasoning* turn measurably
 * degrades it — and this product exists for tasks where the thinking matters.
 *
 * OpenClaw's `tool-call-repair` takes the other route: let the model write, then
 * recognise the well-known plain-text shapes and promote them into real tool
 * calls. No extra pass, and it handles the several formats different model
 * families emit. `orchestrator/grammar.rs` is kept, unwired, as the fallback if
 * this proves insufficient on a particular model.
 *
 * ## Where it runs
 *
 * Wrapped around the stream function. The agent loop reads its tool calls from
 * the final assistant message, so repairing that message is the whole job —
 * the events are forwarded untouched, which keeps streamed text arriving as the
 * model produced it.
 */

import {
  createPromotedPlainTextToolCallBlock,
  projectStandalonePlainTextToolCallMessage,
} from "@openclaw/tool-call-repair";
import type { StreamFn } from "@openclaw/agent-core";

/**
 * Stop reasons whose message is eligible for repair.
 *
 * A turn that already made a tool call, or that was aborted or errored, is left
 * alone: promoting text there would invent a second call the model did not ask
 * for, which is a far worse failure than the one being fixed.
 */
const REPAIRABLE_STOP_REASONS: ReadonlySet<unknown> = new Set(["stop", "length"]);

/**
 * Wraps a stream function so plain-text tool calls become real ones.
 *
 * `allowedToolNames` is the run's own catalogue, not every tool that exists —
 * so a model that writes something resembling a call to a tool it was not given
 * is not granted one by the repair. The gateway would refuse it anyway; not
 * manufacturing it means the refusal never has to happen.
 */
export function withToolCallRepair(streamFn: StreamFn, allowedToolNames: string[]): StreamFn {
  const allowed = new Set(allowedToolNames);
  if (allowed.size === 0) {
    return streamFn;
  }

  return ((model: unknown, context: unknown, options: unknown) => {
    const inner = (streamFn as (m: unknown, c: unknown, o: unknown) => never)(
      model,
      context,
      options,
    ) as {
      [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
      result: () => Promise<unknown>;
      push: (event: unknown) => void;
      end: (message?: unknown) => void;
    };

    return {
      ...inner,
      // Bound rather than spread: the iterator and the queue methods must stay
      // attached to the original stream object.
      [Symbol.asyncIterator]: () => inner[Symbol.asyncIterator](),
      push: (event: unknown) => inner.push(event),
      end: (message?: unknown) => inner.end(message),
      async result() {
        const message = await inner.result();
        const projection = projectStandalonePlainTextToolCallMessage({
          message,
          allowedToolNames: allowed,
          createToolCallBlock: createPromotedPlainTextToolCallBlock,
          requireAssistantRole: true,
          allowedStopReasons: REPAIRABLE_STOP_REASONS,
        });
        return projection?.message ?? message;
      },
    };
  }) as StreamFn;
}

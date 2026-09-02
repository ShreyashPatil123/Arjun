/**
 * How long each model call actually takes, measured where it happens.
 *
 * ## Why this exists
 *
 * The durable event log records `run_started` and `turn_ended` and nothing in
 * between. On a measured run those two were 122 seconds apart for an answer
 * of 340 characters, while the same model answering the same question through
 * `llama-server` directly took under six seconds. That gap is inside this
 * process, and no record anywhere could say which part of it.
 *
 * So each model call is timed at the transport: when it was issued, when the
 * first chunk came back, when the last one did, and how many calls a turn
 * made. Those four numbers separate "the model is slow" from "the loop is
 * calling it repeatedly" from "the first token took a minute to arrive" —
 * three very different faults that look identical from outside.
 *
 * ## What is measured, and what is not
 *
 * Timings only. The wrapper never inspects a chunk's content, so nothing the
 * model produces — reasoning included — can reach the log through here.
 */

import type { StreamFn } from "@openclaw/agent-core";

interface TimedStream {
  [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
  result: () => Promise<unknown>;
  push: (event: unknown) => void;
  end: (message?: unknown) => void;
}

/** Writes one measurement line. Stderr, so Rust logs it with the run. */
function report(runId: string, fields: Record<string, number | string>): void {
  const body = Object.entries(fields)
    .map(([key, value]) => `${key}=${value}`)
    .join(" ");
  process.stderr.write(`[agent-runtime:log] [model] run=${runId} ${body}\n`);
}

/**
 * Wraps a stream function so every call it makes is timed.
 *
 * Transparent: the returned object forwards every method to the original, and
 * the iterator is the original's. The only additions are a clock and a line of
 * output when the stream finishes.
 */
export function withCallTiming(
  streamFn: StreamFn,
  runId: string,
  now: () => number = Date.now,
): StreamFn {
  let call = 0;
  return ((model: unknown, context: unknown, options: unknown) => {
    const index = ++call;
    const issuedAt = now();
    const inner = (streamFn as (m: unknown, c: unknown, o: unknown) => never)(
      model,
      context,
      options,
    ) as unknown as TimedStream;

    let firstChunkAt: number | null = null;
    let chunks = 0;

    const iterator = () => {
      const source = inner[Symbol.asyncIterator]();
      return {
        async next() {
          const step = await source.next();
          if (!step.done) {
            chunks += 1;
            if (firstChunkAt === null) firstChunkAt = now();
          }
          return step;
        },
        return: source.return?.bind(source),
        throw: source.throw?.bind(source),
      } as AsyncIterator<unknown>;
    };

    return {
      ...inner,
      [Symbol.asyncIterator]: iterator,
      push: (event: unknown) => inner.push(event),
      end: (message?: unknown) => inner.end(message),
      async result() {
        try {
          return await inner.result();
        } finally {
          const finishedAt = now();
          report(runId, {
            call: index,
            // The number this whole exercise is about: how long the person
            // waited between the loop asking and the model saying anything.
            firstChunkMs: firstChunkAt === null ? -1 : firstChunkAt - issuedAt,
            totalMs: finishedAt - issuedAt,
            chunks,
          });
        }
      },
    } as unknown as ReturnType<StreamFn>;
  }) as unknown as StreamFn;
}

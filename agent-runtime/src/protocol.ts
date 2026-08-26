/**
 * The wire between the Rust core and this runtime.
 *
 * ## Why stdio and not a socket
 *
 * A listening socket is a thing that can be connected to. On an air-gapped
 * workbench the claim being made is not "nothing connected to it" but "there was
 * nothing to connect to", and the cheapest way to be able to say that is to
 * never open one. So the runtime speaks over the pipes its parent already owns:
 * stdin carries requests in, stdout carries requests and results back, and there
 * is no third party that could be either end.
 *
 * That has one consequence worth stating loudly, because violating it is silent
 * and total: **stdout is the channel**. A stray `console.log` anywhere in this
 * process — or in any vendored dependency — injects a line the framing cannot
 * parse and desynchronises the peer. All diagnostics go to stderr. See
 * `installStdoutGuard` in `main.ts`, which enforces this rather than trusting it.
 *
 * ## Framing
 *
 * Newline-delimited JSON, one frame per line. Chosen over a length-prefixed
 * framing because it stays readable in a log or a capture, which matters for a
 * system whose whole point is being auditable by someone who did not write it.
 * JSON strings escape their newlines, so a literal `\n` can only ever be a frame
 * boundary.
 *
 * ## Direction
 *
 * Both ends may open a request; `id` correlates it with its reply. A frame with
 * no `id` is a notification and is never replied to — used for the event stream,
 * which is high-volume and where a reply per frame would double the traffic for
 * nothing.
 */

/** A request awaiting a reply. `id` is unique per originating peer. */
export interface RequestFrame {
  id: string;
  method: string;
  params?: unknown;
}

/** A successful reply. */
export interface ResultFrame {
  id: string;
  result: unknown;
}

/**
 * A failed reply.
 *
 * `code` is a stable machine-readable token; `message` is for a person. The
 * split matters because the peer branches on `code` and shows `message`, and
 * conflating them makes error text load-bearing.
 */
export interface ErrorFrame {
  id: string;
  error: { code: string; message: string };
}

/** A one-way message. Never replied to. */
export interface NotificationFrame {
  method: string;
  params?: unknown;
}

export type Frame = RequestFrame | ResultFrame | ErrorFrame | NotificationFrame;

export function isRequest(frame: Frame): frame is RequestFrame {
  return "id" in frame && "method" in frame;
}

export function isResult(frame: Frame): frame is ResultFrame {
  return "id" in frame && "result" in frame;
}

export function isError(frame: Frame): frame is ErrorFrame {
  return "id" in frame && "error" in frame;
}

export function isNotification(frame: Frame): frame is NotificationFrame {
  return !("id" in frame) && "method" in frame;
}

/** Error codes that cross the wire. Both ends branch on these. */
export const ErrorCode = {
  /** The method name is not one this peer serves. */
  UnknownMethod: "unknown_method",
  /** Params were absent, malformed, or the wrong shape. */
  BadParams: "bad_params",
  /** The gateway refused the call. Carries the refusal text for the model. */
  Refused: "refused",
  /** The tool ran and threw. */
  ToolFailed: "tool_failed",
  /** The peer went away mid-request. */
  PeerClosed: "peer_closed",
  /** The handler threw something that was not one of the above. */
  Internal: "internal",
} as const;

export type ErrorCodeValue = (typeof ErrorCode)[keyof typeof ErrorCode];

/**
 * Splits a byte stream into frames.
 *
 * Stateful because a chunk boundary lands mid-line often enough that treating
 * each chunk as a whole message is a bug that only appears under load. `push`
 * returns whole frames and retains the partial tail for the next call.
 *
 * Malformed lines are surfaced, not skipped. A line this cannot parse means the
 * two ends disagree about the channel, and continuing past that produces
 * confusing failures much later; the caller's job is to treat it as fatal.
 */
export class FrameDecoder {
  #buffer = "";
  readonly #maxLineBytes: number;

  /**
   * @param maxLineBytes Guard against a peer that never sends a newline. Tool
   *   results carry document text, so this is generous — but unbounded means a
   *   single bad frame can exhaust memory, which is a worse failure than a
   *   refused one.
   */
  constructor(maxLineBytes = 64 * 1024 * 1024) {
    this.#maxLineBytes = maxLineBytes;
  }

  push(chunk: string): Frame[] {
    this.#buffer += chunk;
    if (this.#buffer.length > this.#maxLineBytes) {
      const overrun = this.#buffer.length;
      // Drop it: keeping the buffer would make every subsequent push throw.
      this.#buffer = "";
      throw new Error(
        `Frame exceeded ${this.#maxLineBytes} bytes without a newline (${overrun} buffered). Channel desynchronised.`,
      );
    }

    const frames: Frame[] = [];
    let newline = this.#buffer.indexOf("\n");
    while (newline !== -1) {
      const line = this.#buffer.slice(0, newline).trim();
      this.#buffer = this.#buffer.slice(newline + 1);
      if (line.length > 0) {
        frames.push(parseFrame(line));
      }
      newline = this.#buffer.indexOf("\n");
    }
    return frames;
  }

  /** Bytes held back awaiting a newline. Non-zero at EOF means a truncated frame. */
  get pending(): number {
    return this.#buffer.length;
  }
}

function parseFrame(line: string): Frame {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (cause) {
    throw new Error(`Malformed frame: ${line.slice(0, 200)}`, { cause });
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`Frame must be a JSON object, got ${Array.isArray(parsed) ? "array" : typeof parsed}`);
  }
  const frame = parsed as Frame;
  if (!isRequest(frame) && !isResult(frame) && !isError(frame) && !isNotification(frame)) {
    throw new Error(`Frame matches no known shape: ${line.slice(0, 200)}`);
  }
  return frame;
}

/** Serialises one frame to a line, newline included. */
export function encodeFrame(frame: Frame): string {
  return `${JSON.stringify(frame)}\n`;
}

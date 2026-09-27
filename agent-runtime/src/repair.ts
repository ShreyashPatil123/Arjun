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
        return projection?.message ?? promoteStandaloneJsonToolCalls(message, allowed) ?? message;
      },
    };
  }) as StreamFn;
}

/**
 * The shape the vendored repair does not know: a JSON object naming the tool.
 *
 * Qwen-family models are trained to wrap a call as
 * `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, and the chat template
 * says so. The small Qwen2.5-Coder models often write the same object in a
 * ```` ```json ```` fence instead, or bare. llama-server then parses nothing, the
 * loop sees an answer that called no tool, and the run ends having described
 * the step instead of doing it. Measured on Qwen2.5-Coder-3B with one tool
 * offered: the call was right, down to the arguments, and it arrived as a
 * fenced block in `content` with `tool_calls` empty.
 *
 * Held to the same rule as the vendored repair: the *whole* answer has to be
 * calls, so a reply that explains a JSON example in prose is left as prose.
 * Every name must be one this run was given, and each object must carry an
 * `arguments` (or `parameters`) object. Anything less and nothing is promoted.
 */
export function promoteStandaloneJsonToolCalls(
  message: unknown,
  allowed: ReadonlySet<string>,
): Record<string, unknown> | undefined {
  const record = message as { role?: unknown; stopReason?: unknown; content?: unknown } | null;
  if (!record || record.role !== "assistant" || !REPAIRABLE_STOP_REASONS.has(record.stopReason)) {
    return undefined;
  }
  const blocks: Array<Record<string, unknown>> = Array.isArray(record.content)
    ? (record.content as Array<Record<string, unknown>>)
    : typeof record.content === "string"
      ? [{ type: "text", text: record.content }]
      : [];
  // Thinking is kept as it was; any other non-text block means this is not a
  // plain answer, and it is left alone.
  if (blocks.some((block) => block.type !== "text" && block.type !== "thinking")) {
    return undefined;
  }
  const text = blocks
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text as string)
    .join("");
  const calls = parseStandaloneJsonToolCalls(text, allowed);
  if (!calls) {
    return undefined;
  }
  return {
    ...(record as Record<string, unknown>),
    content: [
      ...blocks.filter((block) => block.type === "thinking"),
      ...calls.map((call) =>
        createPromotedPlainTextToolCallBlock(
          { arguments: call.arguments, name: call.name, raw: "", start: 0, end: 0 },
          call.name,
        ),
      ),
    ],
    stopReason: "toolUse",
  };
}

type JsonToolCall = { name: string; arguments: Record<string, unknown> };

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * The calls a message opens with, or null when it opens with anything else.
 *
 * Accepts one or more of ```` ```json {…} ``` ````, `<tool_call>{…}</tool_call>`
 * and a bare `{…}`, separated only by whitespace.
 *
 * Two things measured on Qwen2.5-Coder-3B in ARJUN's own harness are allowed
 * around them, and nothing else:
 *
 * - **Before:** the `<tools>…</tools>` list from its prompt, echoed back. That
 *   is the model repeating what it was given, not saying anything.
 * - **After:** anything. A model that writes a call and then carries on is
 *   writing about results it has not seen — one run followed a correct
 *   `workspace.write_text` call with an invented "Output:" section. The call
 *   is what it asked for; the rest is discarded with the turn it belongs to,
 *   and the real output arrives from the tool.
 *
 * A message that opens with prose is still left as prose.
 */
export function parseStandaloneJsonToolCalls(
  text: string,
  allowed: ReadonlySet<string>,
): JsonToolCall[] | null {
  const calls: JsonToolCall[] = [];
  let rest = text
    .trim()
    .replace(/^```(?:xml)?\s*<tools>[\s\S]*?<\/tools>\s*```\s*/, "")
    .replace(/^<tools>[\s\S]*?<\/tools>\s*/, "");
  while (rest.length > 0) {
    let body: string;
    const fenced = /^```[A-Za-z0-9_-]*[ \t]*\r?\n?([\s\S]*?)\r?\n?[ \t]*```/.exec(rest);
    const tagged = /^<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/.exec(rest);
    if (fenced) {
      body = fenced[1]!;
      rest = rest.slice(fenced[0].length).trim();
    } else if (tagged) {
      body = tagged[1]!;
      rest = rest.slice(tagged[0].length).trim();
    } else if (rest.startsWith("{")) {
      const end = endOfJsonObject(rest);
      body = end === null ? rest : rest.slice(0, end);
      rest = end === null ? "" : rest.slice(end).trim();
    } else {
      break;
    }
    const call = asJsonToolCall(parseLenientJson(body.trim()), allowed);
    if (!call) {
      break;
    }
    calls.push(call);
  }
  return calls.length > 0 ? calls : null;
}

/** Where the object that opens `text` closes, counting braces outside strings. */
function endOfJsonObject(text: string): number | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
    } else if (char === '"') {
      inString = true;
    } else if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) return index + 1;
    }
  }
  return null;
}

function asJsonToolCall(value: unknown, allowed: ReadonlySet<string>): JsonToolCall | null {
  if (!isObject(value) || typeof value.name !== "string") {
    return null;
  }
  const rawArguments = value.arguments ?? value.parameters;
  const args = isObject(rawArguments)
    ? rawArguments
    : typeof rawArguments === "string"
      ? parseLenientJson(rawArguments)
      : null;
  if (!isObject(args)) {
    return null;
  }
  const name = resolveToolName(value.name, allowed);
  return name ? { name, arguments: args } : null;
}

/**
 * The run's own name for what the model wrote. Exact first; otherwise the one
 * allowed name that differs only in `.` versus `_`, which is how a model
 * copying `sandbox.run_code` from a catalogue that went through a
 * provider-safe rename tends to write it.
 */
function resolveToolName(raw: string, allowed: ReadonlySet<string>): string | null {
  if (allowed.has(raw)) {
    return raw;
  }
  const flat = (name: string) => name.replace(/[._]/g, "_").toLowerCase();
  const matches = [...allowed].filter((name) => flat(name) === flat(raw));
  return matches.length === 1 ? matches[0]! : null;
}

/**
 * `JSON.parse`, forgiving one thing small models do: a raw line break or tab
 * inside a string value, which is what a multi-line program pasted into
 * `"source"` looks like. Escaped here, only inside strings; any other defect
 * still fails.
 */
function parseLenientJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    let out = "";
    let inString = false;
    let escaped = false;
    for (const char of text) {
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === '"') {
          inString = false;
        } else if (char === "\n") {
          out += "\\n";
          continue;
        } else if (char === "\r") {
          continue;
        } else if (char === "\t") {
          out += "\\t";
          continue;
        }
      } else if (char === '"') {
        inString = true;
      }
      out += char;
    }
    try {
      return JSON.parse(out);
    } catch {
      return null;
    }
  }
}

/**
 * Wire-contract tests for the message-stream translator.
 *
 * These tests pin the contract between the OpenClaw agent-loop event shape
 * and the Arjun chat surface's `AgentEvent` shape. They are the canary for
 * the bug that previously made the chat sit on "thinking…" for the entire
 * run duration: the live `message_*` events were forwarded with their
 * OpenClaw shape (`{ type, message, assistantMessageEvent }`), the chat
 * surface filtered on `event.messageId` (a field that did not exist), and
 * every event was dropped.
 *
 * The fix is in `run.ts::translateForWire`. These tests assert:
 *   1. `message_start` carries the front-end's `messageId`.
 *   2. `message_update` from a `text_delta` produces a `delta` string.
 *   3. `thinking_delta` is not exposed as visible text.
 *   4. `toolcall_delta` is not exposed as visible text.
 *   5. `message_end` carries the right `messageId` and a mapped `finishReason`.
 */

import { describe, expect, it } from "vitest";
import { translateForWire } from "./run.js";
import type { AgentEvent, AgentToolCall } from "@openclaw/agent-core";
import type { AssistantMessage, StopReason, Usage, TextContent } from "@openclaw/llm-core";

const RUN_ID = "run-test";
const MESSAGE_ID = "msg-test-1";

/** A minimal OpenClaw `AssistantMessage` with a controllable `stopReason` and `usage`. */
function assistantMessage(stopReason: StopReason, usage?: Usage): AssistantMessage {
  const content: TextContent[] = [{ type: "text", text: "" }];
  return {
    role: "assistant",
    content,
    api: "openai-completions",
    provider: "sovereign-local",
    model: "test-model",
    usage: usage ?? {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    timestamp: 0,
  };
}

/** An assistant message whose first text block carries the given text. */
function assistantMessageWithText(
  text: string,
  stopReason: StopReason = "stop",
  usage?: Usage,
): AssistantMessage {
  const base = assistantMessage(stopReason, usage);
  const content: TextContent[] = [{ type: "text", text }];
  return { ...base, content };
}

describe("translateForWire: message_start", () => {
  it("emits one message_start with the front-end's messageId", () => {
    const event: AgentEvent = {
      type: "message_start",
      message: assistantMessage("stop"),
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toEqual([{ type: "message_start", messageId: MESSAGE_ID, role: "assistant" }]);
  });
});

describe("translateForWire: message_update with text_delta", () => {
  it("emits a message_update carrying the delta text", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Hello" },
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toEqual([{ type: "message_update", messageId: MESSAGE_ID, delta: "Hello" }]);
  });

  it("concatenates word-by-word streams as the model emits them", () => {
    const deltas = ["Hi", ", ", "how", " can", " I help"];
    const results = deltas.flatMap((delta) =>
      translateForWire(
        {
          type: "message_update",
          message: assistantMessage("stop"),
          assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta },
        },
        RUN_ID,
        MESSAGE_ID,
      ),
    );
    expect(results).toEqual(
      deltas.map((delta) => ({ type: "message_update", messageId: MESSAGE_ID, delta })),
    );
  });

  it("does not emit a message_update when the text_delta is empty", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "" },
    };
    expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
  });
});

describe("translateForWire: message_update does not leak thinking", () => {
  it("drops thinking_delta entirely", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: {
        type: "thinking_delta",
        contentIndex: 0,
        delta: "private chain-of-thought that must not be shown",
        partial: assistantMessage("stop"),
      },
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toEqual([]);
  });

  it("drops thinking_start and thinking_end", () => {
    const starts: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "thinking_start", contentIndex: 0, partial: assistantMessage("stop") },
    };
    const ends: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "thinking_end", contentIndex: 0, content: "secret", partial: assistantMessage("stop") },
    };
    expect(translateForWire(starts, RUN_ID, MESSAGE_ID)).toEqual([]);
    expect(translateForWire(ends, RUN_ID, MESSAGE_ID)).toEqual([]);
  });
});

describe("translateForWire: text_start / text_end carry the visible text", () => {
  it("text_start emits a single message_update with the whole text block", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: {
        type: "text_start",
        contentIndex: 0,
        partial: assistantMessageWithText("Hello, world."),
      },
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toEqual([{ type: "message_update", messageId: MESSAGE_ID, delta: "Hello, world." }]);
  });

  it("text_end emits a single message_update with the whole text block", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: {
        type: "text_end",
        contentIndex: 0,
        content: "Done.",
        partial: assistantMessageWithText("Done."),
      },
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toEqual([{ type: "message_update", messageId: MESSAGE_ID, delta: "Done." }]);
  });
});

describe("translateForWire: message_update does not leak tool-call wire repair", () => {
  it("drops toolcall_start, toolcall_delta, and toolcall_end", () => {
    const toolCall: AgentToolCall = {
      type: "toolCall",
      id: "tc-1",
      name: "knowledge.search_authorized",
      arguments: { query: "x" },
    };
    const cases: AgentEvent[] = [
      {
        type: "message_update",
        message: assistantMessage("stop"),
        assistantMessageEvent: { type: "toolcall_start", contentIndex: 0, partial: assistantMessage("stop") },
      },
      {
        type: "message_update",
        message: assistantMessage("stop"),
        assistantMessageEvent: { type: "toolcall_delta", contentIndex: 0, delta: '{"name":', partial: assistantMessage("stop") },
      },
      {
        type: "message_update",
        message: assistantMessage("stop"),
        assistantMessageEvent: {
          type: "toolcall_end",
          contentIndex: 0,
          toolCall,
          partial: assistantMessage("stop"),
        },
      },
    ];
    for (const event of cases) {
      expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
    }
  });
});

describe("translateForWire: message_end", () => {
  it("carries the messageId and maps stopReason === 'stop' to 'stop'", () => {
    const event: AgentEvent = {
      type: "message_end",
      message: assistantMessage("stop", {
        input: 17,
        output: 12,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 29,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      }),
    };
    const out = translateForWire(event, RUN_ID, MESSAGE_ID);
    expect(out).toHaveLength(1);
    const wire = out[0]!;
    expect(wire.type).toBe("message_end");
    if (wire.type !== "message_end") return;
    expect(wire.messageId).toBe(MESSAGE_ID);
    expect(wire.finishReason).toBe("stop");
    expect(wire.tokensIn).toBe(17);
    expect(wire.tokensOut).toBe(12);
  });

  it("maps length, toolUse, error, and aborted to the chat surface's union", () => {
    const cases: Array<{ input: StopReason; expected: "length" | "tool_calls" | "error" }> = [
      { input: "length", expected: "length" },
      { input: "toolUse", expected: "tool_calls" },
      { input: "error", expected: "error" },
      { input: "aborted", expected: "error" },
    ];
    for (const { input, expected } of cases) {
      const event: AgentEvent = { type: "message_end", message: assistantMessage(input) };
      const out = translateForWire(event, RUN_ID, MESSAGE_ID);
      expect(out).toHaveLength(1);
      const wire = out[0]!;
      expect(wire.type).toBe("message_end");
      if (wire.type !== "message_end") return;
      expect(wire.messageId).toBe(MESSAGE_ID);
      expect(wire.finishReason).toBe(expected);
    }
  });
});

describe("translateForWire: non-message events pass through empty", () => {
  it("returns [] for tool_execution_start (forwarded as-is by the caller)", () => {
    const event: AgentEvent = {
      type: "tool_execution_start",
      toolCallId: "tc-1",
      toolName: "knowledge.search_authorized",
      args: { query: "x" },
    };
    expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
  });

  it("returns [] for agent_start, turn_start, turn_end", () => {
    for (const event of [
      { type: "agent_start" } as AgentEvent,
      { type: "turn_start" } as AgentEvent,
      { type: "turn_end", message: assistantMessage("stop"), toolResults: [] } as AgentEvent,
    ]) {
      expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
    }
  });
});

describe("translateForWire: messageId isolation", () => {
  it("two streams with different messageIds do not cross-contaminate", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "x" },
    };
    expect(translateForWire(event, RUN_ID, "msg-A")).toEqual([
      { type: "message_update", messageId: "msg-A", delta: "x" },
    ]);
    expect(translateForWire(event, RUN_ID, "msg-B")).toEqual([
      { type: "message_update", messageId: "msg-B", delta: "x" },
    ]);
  });
});

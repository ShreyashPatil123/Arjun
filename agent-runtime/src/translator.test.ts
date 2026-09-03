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
 *   3. `thinking_delta` is not exposed as visible text, and the
 *      content-free `model_thinking` signal that replaced silence carries
 *      no fragment of the reasoning.
 *   4. `toolcall_delta` is not exposed as visible text.
 *   5. `message_end` carries the right `messageId` and a mapped `finishReason`.
 */

import { describe, expect, it } from "vitest";
import { MessageTranslator, translateForWire } from "./run.js";
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

const SECRET = "private chain-of-thought that must not be shown";

function thinkingDelta(delta: string): AgentEvent {
  return {
    type: "message_update",
    message: assistantMessage("stop"),
    assistantMessageEvent: {
      type: "thinking_delta",
      contentIndex: 0,
      delta,
      partial: assistantMessage("stop"),
    },
  } as AgentEvent;
}

describe("translateForWire: reasoning streams on its own channel", () => {
  /**
   * The guarantee that survived the change, and the only one that was ever
   * load-bearing.
   *
   * Reasoning is now shown to the person watching — that was asked for, and
   * these tests were rewritten rather than deleted so the change is on the
   * record. What must not happen is reasoning reaching `message_update`: that
   * stream becomes `Message.content`, is persisted, is sent as `finalContent`,
   * and is what the verifier resolves citations against. A thought that leaked
   * into it would be signed as part of the deliverable.
   */
  it("never produces a message_update from a thinking_delta", () => {
    const out = translateForWire(thinkingDelta(SECRET), RUN_ID, MESSAGE_ID);
    expect(out.every((e) => e.type !== "message_update")).toBe(true);
  });

  it("carries the reasoning on model_thinking, and only there", () => {
    const out = translateForWire(thinkingDelta(SECRET), RUN_ID, MESSAGE_ID);
    expect(out).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "start",
        characters: SECRET.length,
        elapsedMs: 0,
        delta: SECRET,
      },
    ]);
    // Serialised rather than field-checked, so a future field carrying the
    // text somewhere it does not belong fails here too.
    const wire = JSON.stringify(out.filter((e) => e.type !== "model_thinking"));
    expect(wire).not.toContain(SECRET);
  });

  it("buffers between flushes rather than sending a frame per token", () => {
    let clock = 1_000;
    const translator = new MessageTranslator(MESSAGE_ID, () => clock);
    expect(translator.translate(thinkingDelta("abcde"))).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "start",
        characters: 5,
        elapsedMs: 0,
        delta: "abcde",
      },
    ]);

    // Inside the text window: buffered, not sent. A reasoning model emits one
    // delta per token, and a frame each would be thousands a second.
    clock = 1_020;
    expect(translator.translate(thinkingDelta("fghij"))).toEqual([]);

    // Past it: one frame carrying everything buffered since the last.
    clock = 1_120;
    expect(translator.translate(thinkingDelta("klm"))).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "active",
        characters: 13,
        elapsedMs: 120,
        delta: "fghijklm",
      },
    ]);
  });

  it("still ticks the counter when a provider sends no reasoning text", () => {
    // Some providers signal that reasoning is under way without streaming any
    // of it. The elapsed figure has to keep moving, or a long silent pass
    // looks like a stall again — the failure the counter was added for.
    let clock = 0;
    const translator = new MessageTranslator(MESSAGE_ID, () => clock);
    translator.translate(thinkingDelta(""));
    clock = 500;
    expect(translator.translate(thinkingDelta(""))).toEqual([]);
    clock = 1_100;
    expect(translator.translate(thinkingDelta(""))).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "active",
        characters: 0,
        elapsedMs: 1_100,
      },
    ]);
  });

  it("flushes the unsent tail on thinking_end rather than dropping it", () => {
    // The tail is the sentence the model was in the middle of when it stopped
    // reasoning. Without this the panel stops mid-word on every run.
    let clock = 1_000;
    const translator = new MessageTranslator(MESSAGE_ID, () => clock);
    translator.translate(thinkingDelta("abcde"));
    clock = 1_010;
    expect(translator.translate(thinkingDelta("tail"))).toEqual([]);

    clock = 4_000;
    const ends: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: {
        type: "thinking_end",
        contentIndex: 0,
        content: SECRET,
        partial: assistantMessage("stop"),
      },
    } as AgentEvent;
    expect(translator.translate(ends)).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "end",
        characters: 9,
        elapsedMs: 3_000,
        delta: "tail",
      },
    ]);
  });

  it("closes the thinking block when the model answers without a thinking_end", () => {
    let clock = 0;
    const translator = new MessageTranslator(MESSAGE_ID, () => clock);
    translator.translate(thinkingDelta(SECRET));
    clock = 900;
    const text: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Hello" },
    } as AgentEvent;
    expect(translator.translate(text)).toEqual([
      {
        type: "model_thinking",
        messageId: MESSAGE_ID,
        state: "end",
        characters: SECRET.length,
        elapsedMs: 900,
        // The opening frame already took the reasoning that had arrived by
        // then, so this one closes the block and carries no text. `delta` is
        // absent rather than an empty string, which is what lets a consumer
        // tell "no reasoning in this frame" from "reasoning that was blank".
      },
      { type: "message_update", messageId: MESSAGE_ID, delta: "Hello" },
    ]);
  });

  it("closes an open thinking block on message_end so the surface stops spinning", () => {
    let clock = 0;
    const translator = new MessageTranslator(MESSAGE_ID, () => clock);
    translator.translate(thinkingDelta("thought"));
    clock = 2_000;
    const out = translator.translate({
      type: "message_end",
      message: assistantMessage("stop"),
    } as AgentEvent);
    expect(out[0]).toEqual({
      type: "model_thinking",
      messageId: MESSAGE_ID,
      state: "end",
      characters: 7,
      elapsedMs: 2_000,
    });
    expect(out[1]?.type).toBe("message_end");
  });
});

describe("translateForWire: text_start / text_end carry the visible text", () => {
  /**
   * Reversed deliberately, and this is the assertion the streaming bug lived
   * behind.
   *
   * It used to require `text_start` to emit its whole payload. That payload is
   * `partial` — a live reference to the message the transport is still writing
   * into — so whenever one network chunk carried the block open plus its first
   * deltas, this fired with the text already accumulated, emitted it as one
   * lump, and marked the block as sent. Every later delta was then discarded,
   * and an answer spanning more than one chunk lost everything after the
   * first. See "the text_start / text_delta race" below, which reproduces that
   * interleaving against the real queue.
   *
   * A block opening carries no text worth forwarding. The text arrives as
   * deltas, and `text_end` reconciles whatever the deltas did not carry.
   */
  it("text_start emits nothing, because a block opening is not text", () => {
    const event: AgentEvent = {
      type: "message_update",
      message: assistantMessage("stop"),
      assistantMessageEvent: {
        type: "text_start",
        contentIndex: 0,
        partial: assistantMessageWithText("Hello, world."),
      },
    };
    expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
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

  it("maps length, error, and aborted to the chat surface's union", () => {
    const cases: Array<{ input: StopReason; expected: "length" | "error" }> = [
      { input: "length", expected: "length" },
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

  it("does not terminate the cell when an assistant turn stops to call a tool", () => {
    // `toolUse` is a hand-off, not an outcome: the loop runs the tools and
    // comes back with another assistant turn into the same cell. Emitting a
    // terminal event here truncated every tool-using run at its first call.
    const event: AgentEvent = { type: "message_end", message: assistantMessage("toolUse") };
    expect(translateForWire(event, RUN_ID, MESSAGE_ID)).toEqual([]);
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

describe("MessageTranslator: deduplication", () => {
  it("text_start + text_delta + text_end produces the text exactly once", async () => {
    const { MessageTranslator } = await import("./run.js");
    const t = new MessageTranslator("msg-1");
    const events: AgentEvent[] = [
      { type: "message_start", message: assistantMessageWithText("") },
      {
        type: "message_update",
        message: assistantMessageWithText("Hello, world."),
        assistantMessageEvent: {
          type: "text_start",
          contentIndex: 0,
          partial: assistantMessageWithText("Hello, world."),
        },
      },
      {
        type: "message_update",
        message: assistantMessageWithText("Hello, world."),
        assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Hello, world." },
      },
      {
        type: "message_update",
        message: assistantMessageWithText("Hello, world."),
        assistantMessageEvent: {
          type: "text_end",
          contentIndex: 0,
          // `text_end` carries the finished block as `content` as well as the
          // message so far as `partial` — see the provider that emits it,
          // `openai-completions.ts:163`. Omitting `content` made this a shape
          // no provider ever sends.
          content: "Hello, world.",
          partial: assistantMessageWithText("Hello, world."),
        },
      },
    ];
    const allWire = events.flatMap(e => t.translate(e));
    const textUpdates = allWire.filter(w => w.type === "message_update");
    expect(textUpdates).toHaveLength(1);
    expect(textUpdates[0]).toEqual({
      type: "message_update",
      messageId: "msg-1",
      delta: "Hello, world.",
    });
  });

  it("text_delta only (no text_start/text_end) streams each delta once", async () => {
    const { MessageTranslator } = await import("./run.js");
    const t = new MessageTranslator("msg-1");
    const events: AgentEvent[] = [
      { type: "message_start", message: assistantMessageWithText("") },
      {
        type: "message_update",
        message: assistantMessageWithText("He"),
        assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "He" },
      },
      {
        type: "message_update",
        message: assistantMessageWithText("llo"),
        assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "llo" },
      },
    ];
    const allWire = events.flatMap(e => t.translate(e));
    const textUpdates = allWire.filter(w => w.type === "message_update");
    expect(textUpdates).toHaveLength(2);
    expect(textUpdates[0]?.type === "message_update" && textUpdates[0].delta).toBe("He");
    expect(textUpdates[1]?.type === "message_update" && textUpdates[1].delta).toBe("llo");
  });

  it("message_start only emitted once even if multiple arrive", async () => {
    const { MessageTranslator } = await import("./run.js");
    const t = new MessageTranslator("msg-1");
    const ev: AgentEvent = {
      type: "message_start",
      message: assistantMessageWithText(""),
    };
    expect(t.translate(ev)).toHaveLength(1);
    expect(t.translate(ev)).toHaveLength(0);
    expect(t.translate(ev)).toHaveLength(0);
  });

  it("message_end only emitted once even if multiple arrive", async () => {
    const { MessageTranslator } = await import("./run.js");
    const t = new MessageTranslator("msg-1");
    t.translate({ type: "message_start", message: assistantMessageWithText("") });
    const ev: AgentEvent = {
      type: "message_end",
      message: assistantMessageWithText("done", "stop"),
    };
    expect(t.translate(ev)).toHaveLength(1);
    expect(t.translate(ev)).toHaveLength(0);
  });
});

/**
 * The full shape of one tool-using turn, as the loop actually emits it.
 *
 * This is the sequence that used to break the chat surface. `agent-core`
 * emits `message_start` and `message_end` for *every* message -- the user's
 * prompt and each tool result included -- and the translator treated the
 * first of each as the assistant's. The user's prompt opened the cell and
 * the user's own `message_end` closed it, so the terminal event went out
 * before the model had produced a token and every assistant turn after it
 * was dropped as a duplicate.
 *
 * The contract these tests pin:
 *   - exactly one `message_start`, from the first *assistant* turn;
 *   - exactly one `message_end`, from the *final* assistant outcome;
 *   - every assistant turn's text, in order, in between;
 *   - nothing at all from user or tool-result messages.
 */
describe("MessageTranslator: user -> assistant toolUse -> tool result -> final assistant", () => {
  /** The user's own prompt message, as the loop emits it. */
  function userMessage(text: string) {
    return { role: "user" as const, content: [{ type: "text" as const, text }], timestamp: 0 };
  }

  /** A tool-result message, as the loop emits it after executing a call. */
  function toolResultMessage(text: string) {
    return {
      role: "toolResult" as const,
      toolCallId: "tc-1",
      toolName: "knowledge.search_authorized",
      content: [{ type: "text" as const, text }],
      isError: false,
      timestamp: 0,
    };
  }

  function textDelta(message: AssistantMessage, delta: string, contentIndex = 0): AgentEvent {
    return {
      type: "message_update",
      message,
      assistantMessageEvent: { type: "text_delta", contentIndex, delta },
    } as AgentEvent;
  }

  /** The whole run, in emission order. */
  function lifecycle(): AgentEvent[] {
    const firstTurn = assistantMessageWithText("Let me look that up.", "toolUse");
    const finalTurn = assistantMessageWithText("The valve is rated to 40 bar.", "stop", {
      input: 120,
      output: 34,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 154,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
    return [
      { type: "agent_start" } as AgentEvent,
      { type: "turn_start" } as AgentEvent,
      // 1. The user's prompt. Must not open the cell, must not close it.
      { type: "message_start", message: userMessage("What is the valve rating?") } as AgentEvent,
      { type: "message_end", message: userMessage("What is the valve rating?") } as AgentEvent,
      // 2. The first assistant turn, which stops to call a tool.
      { type: "message_start", message: firstTurn } as AgentEvent,
      textDelta(firstTurn, "Let me look "),
      textDelta(firstTurn, "that up."),
      { type: "message_end", message: firstTurn } as AgentEvent,
      // 3. The tool runs and returns. Must not touch the cell either.
      {
        type: "tool_execution_start",
        toolCallId: "tc-1",
        toolName: "knowledge.search_authorized",
        args: { query: "valve rating" },
      } as AgentEvent,
      { type: "message_start", message: toolResultMessage("P-101 valve: 40 bar") } as AgentEvent,
      { type: "message_end", message: toolResultMessage("P-101 valve: 40 bar") } as AgentEvent,
      { type: "turn_end", message: firstTurn, toolResults: [] } as AgentEvent,
      // 4. The second assistant turn: the answer, into the same cell.
      { type: "turn_start" } as AgentEvent,
      { type: "message_start", message: finalTurn } as AgentEvent,
      textDelta(finalTurn, "The valve is "),
      textDelta(finalTurn, "rated to 40 bar."),
      { type: "message_end", message: finalTurn } as AgentEvent,
      { type: "turn_end", message: finalTurn, toolResults: [] } as AgentEvent,
      { type: "agent_end", messages: [finalTurn] } as AgentEvent,
    ];
  }

  function run(): Array<{ type: string }> {
    const t = new MessageTranslator("msg-lifecycle");
    return lifecycle().flatMap((event) => t.translate(event));
  }

  it("opens the cell exactly once, and from the assistant turn", () => {
    const starts = run().filter((w) => w.type === "message_start");
    expect(starts).toEqual([
      { type: "message_start", messageId: "msg-lifecycle", role: "assistant" },
    ]);
  });

  it("emits exactly one terminal event, from the final assistant outcome", () => {
    const ends = run().filter((w) => w.type === "message_end");
    expect(ends).toEqual([
      {
        type: "message_end",
        messageId: "msg-lifecycle",
        finishReason: "stop",
        tokensIn: 120,
        tokensOut: 34,
      },
    ]);
  });

  it("preserves both assistant turns' text, in order", () => {
    const deltas = run()
      .filter(
        (w): w is { type: "message_update"; messageId: string; delta: string } =>
          w.type === "message_update",
      )
      .map((w) => w.delta);
    expect(deltas).toEqual(["Let me look ", "that up.", "The valve is ", "rated to 40 bar."]);
  });

  it("orders the stream: start, all text, then the single end", () => {
    const kinds = run().map((w) => w.type);
    expect(kinds).toEqual([
      "message_start",
      "message_update",
      "message_update",
      "message_update",
      "message_update",
      "message_end",
    ]);
  });

  it("emits nothing for the user prompt or the tool result on their own", () => {
    const t = new MessageTranslator("msg-solo");
    expect(t.translate({ type: "message_start", message: userMessage("hi") } as AgentEvent)).toEqual(
      [],
    );
    expect(t.translate({ type: "message_end", message: userMessage("hi") } as AgentEvent)).toEqual(
      [],
    );
    expect(
      t.translate({ type: "message_start", message: toolResultMessage("result") } as AgentEvent),
    ).toEqual([]);
    expect(
      t.translate({ type: "message_end", message: toolResultMessage("result") } as AgentEvent),
    ).toEqual([]);
  });
});

describe("MessageTranslator: the terminal event is emitted exactly once, on every path", () => {
  it("closes a cell left open by a toolUse turn when the loop ends", () => {
    // A run stopped by its step budget, its deadline, or an operator: the last
    // assistant turn ended on `toolUse` and no further assistant message will
    // arrive. Without the `agent_end` backstop the chat cell streams forever.
    const stalled = assistantMessageWithText("Calling one more tool.", "toolUse");
    const t = new MessageTranslator("msg-stalled");
    t.translate({ type: "message_start", message: stalled } as AgentEvent);
    expect(t.translate({ type: "message_end", message: stalled } as AgentEvent)).toEqual([]);
    const out = t.translate({ type: "agent_end", messages: [stalled] } as AgentEvent);
    expect(out).toEqual([
      {
        type: "message_end",
        messageId: "msg-stalled",
        finishReason: "tool_calls",
        tokensIn: 0,
        tokensOut: 0,
      },
    ]);
  });

  it("does not emit a second terminal event on agent_end after a normal finish", () => {
    const done = assistantMessageWithText("Done.", "stop");
    const t = new MessageTranslator("msg-once");
    t.translate({ type: "message_start", message: done } as AgentEvent);
    expect(t.translate({ type: "message_end", message: done } as AgentEvent)).toHaveLength(1);
    expect(t.translate({ type: "agent_end", messages: [done] } as AgentEvent)).toEqual([]);
    expect(t.finalize([done])).toEqual([]);
  });

  it("finalize() closes a cell whose loop threw before agent_end", () => {
    const partial = assistantMessageWithText("I was saying", "error");
    const t = new MessageTranslator("msg-thrown");
    t.translate({ type: "message_start", message: partial } as AgentEvent);
    const out = t.finalize([partial]);
    expect(out).toHaveLength(1);
    expect(out[0]?.type === "message_end" && out[0].finishReason).toBe("error");
    // Idempotent: the caller in `startRun` runs it in a `finally` that also
    // fires on the paths where `agent_end` already closed the cell.
    expect(t.finalize([partial])).toEqual([]);
  });

  it("finalize() closes nothing when no assistant turn ever opened a cell", () => {
    // Nothing on the surface to terminate. A terminal event here would tell
    // the chat an answer finished that was never begun.
    const t = new MessageTranslator("msg-never");
    expect(t.finalize([])).toEqual([]);
    expect(t.translate({ type: "agent_end", messages: [] } as AgentEvent)).toEqual([]);
  });
});

/**
 * The streaming race, reproduced against the real queue.
 *
 * ## What is being reproduced
 *
 * `EventStream.push` (llm-core/utils/event-stream.ts) enqueues an event
 * whenever no consumer is currently parked in `waiting`. The SSE producer in
 * `openai-completions-stream.ts` parses a whole network chunk synchronously:
 * for one chunk carrying several frames it pushes `text_start` and then every
 * `text_delta` back to back, with no `await` between them, so the consumer
 * does not run until all of them are queued.
 *
 * The `text_start` frame carries `partial: output` — a **live reference** to
 * the message the producer is still mutating. By the time the consumer drains
 * it, `output.content[0].text` already holds everything that arrived in that
 * chunk.
 *
 * These tests build that interleaving by hand rather than mocking the
 * translator's inputs, because the ordering *is* the bug: a test that fed
 * `text_start` with an empty payload would pass against the broken code.
 */
describe("MessageTranslator: the text_start / text_delta race", () => {
  /** A block of text as the transport accumulates it, with a live partial. */
  function producer() {
    const output = { content: [{ type: "text", text: "" }] as Array<{ type: string; text: string }> };
    const queued: AgentEvent[] = [];
    return {
      /** What the transport does when it opens the block. */
      start() {
        queued.push({
          type: "message_update",
          message: assistantMessage("stop"),
          assistantMessageEvent: { type: "text_start", contentIndex: 0, partial: output },
        } as unknown as AgentEvent);
      },
      /** Mutate first, then queue — the order `appendTextDeltaInternal` uses. */
      delta(text: string) {
        const block = output.content[0];
        if (block) block.text += text;
        queued.push({
          type: "message_update",
          message: assistantMessage("stop"),
          assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text },
        } as unknown as AgentEvent);
      },
      end() {
        queued.push({
          type: "message_update",
          message: assistantMessage("stop"),
          assistantMessageEvent: { type: "text_end", contentIndex: 0, partial: output },
        } as unknown as AgentEvent);
      },
      /** Drain everything the producer queued, in order, through one translator. */
      drain(translator: MessageTranslator) {
        return queued.splice(0).flatMap((event) => translator.translate(event));
      },
      output,
    };
  }

  function deltasOf(events: Array<{ type: string; delta?: string }>): string[] {
    return events.filter((e) => e.type === "message_update").map((e) => e.delta ?? "");
  }

  it("streams every delta when the consumer keeps up with the producer", () => {
    const translator = new MessageTranslator(MESSAGE_ID);
    const p = producer();

    // One event per drain: the consumer is never behind, so `text_start`
    // carries an empty block and is correctly ignored.
    p.start();
    const started = p.drain(translator);
    p.delta("Hello");
    const first = p.drain(translator);
    p.delta(" world");
    const second = p.drain(translator);

    expect(deltasOf(started)).toEqual([]);
    expect(deltasOf(first)).toEqual(["Hello"]);
    expect(deltasOf(second)).toEqual([" world"]);
  });

  it("still streams every delta when one chunk carries the open and several deltas", () => {
    const translator = new MessageTranslator(MESSAGE_ID);
    const p = producer();

    // The real case: the producer runs to completion for this chunk before the
    // consumer sees any of it, so `text_start`'s live partial already reads
    // "Hello world".
    p.start();
    p.delta("Hello");
    p.delta(" world");
    expect(p.output.content[0]?.text).toBe("Hello world");

    const out = deltasOf(p.drain(translator));
    expect(out.join("")).toBe("Hello world");
    expect(
      out.length,
      "one lump instead of two deltas means the surface cannot render progressively",
    ).toBeGreaterThan(1);
  });

  it("loses no text when the answer spans more than one network chunk", () => {
    const translator = new MessageTranslator(MESSAGE_ID);
    const p = producer();

    // Chunk one.
    p.start();
    p.delta("Hello");
    p.delta(" world");
    const chunkOne = deltasOf(p.drain(translator));

    // Chunk two: deltas only, no second `text_start`.
    p.delta(" and");
    p.delta(" goodbye");
    const chunkTwo = deltasOf(p.drain(translator));

    // The close.
    p.end();
    const closing = deltasOf(p.drain(translator));

    expect(
      [...chunkOne, ...chunkTwo, ...closing].join(""),
      "text produced after the first chunk must reach the surface",
    ).toBe("Hello world and goodbye");
  });
});

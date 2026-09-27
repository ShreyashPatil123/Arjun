import { describe, expect, it } from "vitest";
import { parseStandaloneJsonToolCalls, promoteStandaloneJsonToolCalls } from "./repair.js";

const allowed = new Set(["sandbox.run_code", "workspace.write_text", "knowledge.search_authorized"]);

function answer(text: string, stopReason = "stop") {
  return { role: "assistant", content: [{ type: "text", text }], stopReason };
}

/**
 * What Qwen2.5-Coder-3B returned, verbatim in shape, with one tool offered:
 * the right call, as a fenced block, and nothing in `tool_calls`.
 */
const FENCED = [
  "```json",
  "{",
  '  "name": "sandbox.run_code",',
  '  "arguments": {"language": "python", "source": "A = 4.7\\nprint(A < 5.0)"}',
  "}",
  "```",
].join("\n");

describe("a tool call written as a JSON object", () => {
  it("is promoted from a fenced block", () => {
    const promoted = promoteStandaloneJsonToolCalls(answer(FENCED), allowed);
    expect(promoted?.stopReason).toBe("toolUse");
    const [call] = promoted?.content as Array<Record<string, unknown>>;
    expect(call).toMatchObject({
      type: "toolCall",
      name: "sandbox.run_code",
      arguments: { language: "python", source: "A = 4.7\nprint(A < 5.0)" },
    });
  });

  it("is promoted from tool_call tags and from a bare object", () => {
    const tagged = '<tool_call>\n{"name": "workspace.write_text", "arguments": {"path": "a.py", "text": "x"}}\n</tool_call>';
    expect(parseStandaloneJsonToolCalls(tagged, allowed)?.[0]?.name).toBe("workspace.write_text");
    const bare = '{"name": "workspace.write_text", "parameters": {"path": "a.py", "text": "x"}}';
    expect(parseStandaloneJsonToolCalls(bare, allowed)?.[0]?.arguments).toEqual({ path: "a.py", text: "x" });
  });

  it("keeps several calls in order", () => {
    const two = `${FENCED}\n\n<tool_call>{"name": "workspace.write_text", "arguments": {"path": "a.py", "text": "x"}}</tool_call>`;
    expect(parseStandaloneJsonToolCalls(two, allowed)?.map((call) => call.name)).toEqual([
      "sandbox.run_code",
      "workspace.write_text",
    ]);
  });

  /** The 1.5B pasted a multi-line program straight into "source". */
  it("forgives a raw line break inside a string value", () => {
    const raw = '```json\n{"name": "sandbox.run_code", "arguments": {"language": "python", "source": "A = 4.7\nprint(A)"}}\n```';
    expect(parseStandaloneJsonToolCalls(raw, allowed)?.[0]?.arguments.source).toBe("A = 4.7\nprint(A)");
  });

  /** Trial 1 on the 3B: a correct call, then an invented "Output:". */
  it("keeps the call and drops what the model wrote after it", () => {
    const withInventedOutput = `${FENCED}\n\n**Output:**\n\n\`\`\`\nFlagged: ['A']\n\`\`\``;
    const calls = parseStandaloneJsonToolCalls(withInventedOutput, allowed);
    expect(calls?.map((call) => call.name)).toEqual(["sandbox.run_code"]);
  });

  /** Trials 2 and 3: the prompt's tool list echoed back, then the calls. */
  it("skips an echoed tool list in front of the calls", () => {
    const echoed = `\`\`\`xml\n<tools>\n{"type": "function", "function": {"name": "sandbox.run_code"}}\n</tools>\n\`\`\`\n\n${FENCED}`;
    expect(parseStandaloneJsonToolCalls(echoed, allowed)?.[0]?.name).toBe("sandbox.run_code");
  });

  it("maps an underscored name onto the run's dotted one", () => {
    const underscored = '{"name": "sandbox_run_code", "arguments": {"language": "python", "source": "1"}}';
    expect(parseStandaloneJsonToolCalls(underscored, allowed)?.[0]?.name).toBe("sandbox.run_code");
  });
});

describe("what is left as the model wrote it", () => {
  it("an answer that explains a call in prose", () => {
    expect(promoteStandaloneJsonToolCalls(answer(`Run this:\n${FENCED}`), allowed)).toBeUndefined();
  });

  it("code that is not a call", () => {
    expect(promoteStandaloneJsonToolCalls(answer("```python\nprint(1)\n```"), allowed)).toBeUndefined();
  });

  it("a tool this run was not given", () => {
    const other = '{"name": "network.fetch", "arguments": {"target": "anything"}}';
    expect(parseStandaloneJsonToolCalls(other, allowed)).toBeNull();
  });

  it("an object with no arguments", () => {
    expect(parseStandaloneJsonToolCalls('{"name": "sandbox.run_code"}', allowed)).toBeNull();
  });

  it("a turn that already called a tool, or did not end normally", () => {
    const called = { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", name: "x" }] };
    expect(promoteStandaloneJsonToolCalls(called, allowed)).toBeUndefined();
    expect(promoteStandaloneJsonToolCalls(answer(FENCED, "error"), allowed)).toBeUndefined();
  });
});

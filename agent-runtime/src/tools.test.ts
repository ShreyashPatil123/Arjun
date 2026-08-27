import { describe, expect, it, vi } from "vitest";
import type { BeforeToolCallContext } from "@openclaw/agent-core";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { GrantLedger, authorizeToolCall, buildTools, type Verdict } from "./tools.js";

/** A peer whose `request` is scripted, so the gateway's answer is the variable. */
function scriptedPeer(script: (method: string, params: unknown) => Promise<unknown>): RpcPeer {
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);
  vi.spyOn(peer, "request").mockImplementation((method, params) => script(method, params));
  return peer;
}

function callContext(toolCallId = "tc-1", name = "search_documents", args: unknown = { query: "seal spec" }) {
  return {
    toolCall: { type: "toolCall", id: toolCallId, name, arguments: args },
    args,
    assistantMessage: {},
    context: {},
  } as unknown as BeforeToolCallContext;
}

describe("authorizeToolCall", () => {
  it("records the grant and lets the call through when the gateway allows it", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({ outcome: "allow", tool: "search_documents", grant: "g-1" } satisfies Verdict));

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result).toBeUndefined();
    expect(ledger.size).toBe(1);
  });

  it("blocks with the gateway's own words when refused", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({ outcome: "refuse", reason: "SearchKnowledge not held" } satisfies Verdict));

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result).toEqual({ block: true, reason: "SearchKnowledge not held" });
    expect(ledger.size).toBe(0);
  });

  it("blocks rather than assuming consent when a person is required", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () =>
      ({ outcome: "needsApproval", tool: "write_scoped_file", summary: "Write 5 bytes to note.txt" } satisfies Verdict),
    );

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext("tc-2", "write_scoped_file"));

    expect(result?.block).toBe(true);
    expect(result?.reason).toContain("Write 5 bytes to note.txt");
    expect(ledger.size).toBe(0);
  });

  it("fails closed when the gateway cannot be reached", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => {
      throw new Error("core closed the channel");
    });

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result?.block).toBe(true);
    expect(result?.reason).toContain("authorisation is unavailable");
    expect(ledger.size).toBe(0);
  });

  it("sends the run, call, tool and arguments the gateway needs to decide", async () => {
    const ledger = new GrantLedger();
    const seen: unknown[] = [];
    const peer = scriptedPeer(async (method, params) => {
      seen.push({ method, params });
      return { outcome: "allow", tool: "search_documents", grant: "g" } satisfies Verdict;
    });

    await authorizeToolCall(peer, ledger, "run-42", callContext("tc-9"));

    expect(seen[0]).toEqual({
      method: "tool.authorize",
      params: { runId: "run-42", toolCallId: "tc-9", tool: "search_documents", args: { query: "seal spec" } },
    });
  });
});

describe("GrantLedger", () => {
  it("hands a grant out exactly once", () => {
    const ledger = new GrantLedger();
    ledger.put("tc-1", "g-1");

    expect(ledger.take("tc-1")).toBe("g-1");
    expect(ledger.take("tc-1")).toBeUndefined();
  });

  it("clears everything, so a grant cannot outlive its run", () => {
    const ledger = new GrantLedger();
    ledger.put("tc-1", "g-1");
    ledger.put("tc-2", "g-2");
    ledger.clear();
    expect(ledger.size).toBe(0);
  });
});

describe("host tools", () => {
  it("exposes only tools the Rust catalogue also knows", () => {
    const peer = scriptedPeer(async () => ({}));
    const names = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").map((tool) => tool.name);
    // Rust's `ToolName` enum is the authority; a name here that is absent
    // there is refused by the gateway regardless of what this declares.
    expect(names.slice().sort()).toEqual([
      "create_docx",
      "create_xlsx",
      "execute_code",
      "load_more_evidence",
      "memory_promote_approved",
      "memory_recall_authorized",
      "read_scoped_file",
      "run_calculation",
      "search_documents",
      "validate_artifact",
      "write_scoped_file",
    ]);
  });

  it("refuses to execute without a grant, so the gateway cannot be skipped", async () => {
    const peer = scriptedPeer(async () => ({ text: "should never be reached" }));
    const tool = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").find((t) => t.name === "search_documents")!;

    await expect(tool.execute("tc-unauthorised", { query: "x" })).rejects.toMatchObject({
      code: "refused",
    });
  });

  it("spends the grant on execution and cannot reuse it", async () => {
    const ledger = new GrantLedger();
    const calls: unknown[] = [];
    const peer = scriptedPeer(async (method, params) => {
      calls.push({ method, params });
      return { text: "3 passages", details: { hits: 3 } };
    });
    const tool = buildTools(peer, ledger, "run-1", "qwen2.5-coder-7b").find((t) => t.name === "search_documents")!;
    ledger.put("tc-1", "g-1");

    const result = await tool.execute("tc-1", { query: "seal spec" });

    expect(result.content).toEqual([{ type: "text", text: "3 passages" }]);
    expect(result.details).toEqual({ hits: 3 });
    expect(calls[0]).toEqual({
      method: "tool.execute",
      params: {
        runId: "run-1",
        toolCallId: "tc-1",
        tool: "search_documents",
        args: { query: "seal spec" },
        grant: "g-1",
        model: "qwen2.5-coder-7b",
      },
    });

    // A second execution of the same call has no grant left to spend.
    await expect(tool.execute("tc-1", { query: "seal spec" })).rejects.toMatchObject({ code: "refused" });
  });

  it("describes the tool well enough for a model to know when to reach for it", () => {
    const peer = scriptedPeer(async () => ({}));
    const tool = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").find((t) => t.name === "search_documents")!;
    expect(tool.description).toMatch(/permitted to read/i);
    expect(tool.description).toMatch(/do not answer such questions from memory/i);
  });
});

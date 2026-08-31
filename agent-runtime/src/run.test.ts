/**
 * The Phase 1 loop, end to end, against a real HTTP model server.
 *
 * This is the test that proves the parts fit: a prompt goes to an
 * OpenAI-compatible endpoint, the reply asks for a tool, the tool call is put
 * to the gateway, the gateway's grant is spent executing it, the result goes
 * back to the model, and the model answers. Everything except the model's
 * judgement and the Rust core is real — the server speaks genuine SSE and the
 * agent loop is OpenClaw's, unmodified.
 *
 * The two fakes are deliberate and opposite in kind. The model server is fake
 * because a real one would make this test need a GPU and a 5 GB download. The
 * Rust core is fake because its own behaviour is tested in Rust; what is under
 * test here is that this side asks it the right questions in the right order.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { startRun, type RunRequest } from "./run.js";
import { TOOL_DEFINITIONS } from "./catalogue.js";

/** One SSE chunk in the shape an OpenAI-compatible server emits. */
function chunk(delta: unknown, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`;
}

/**
 * A local inference server that replies with a scripted turn each time.
 *
 * Requests are recorded so the test can assert what the model was actually
 * shown — which is the only way to know the tool result reached it.
 */
function modelServer(turns: string[][]): Promise<{
  baseUrl: string;
  requests: unknown[];
  close: () => Promise<void>;
}> {
  const requests: unknown[] = [];
  let turn = 0;

  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => {
      body += c;
    });
    req.on("end", () => {
      requests.push(JSON.parse(body || "{}"));
      const script = turns[Math.min(turn, turns.length - 1)] ?? [];
      turn += 1;
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      for (const line of script) res.write(line);
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });

  return new Promise((resolve) => {
    // Port 0 so parallel test files cannot collide, and loopback because the
    // runtime refuses anything else.
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        requests,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

/**
 * The eligibility answer Rust gives a run with an ordinary plan.
 *
 * Built from this runtime's own catalogue so a tool added later is offered to
 * these tests automatically — a fixed list here would silently stop exercising
 * whatever was added after it was written.
 */
function eligibleTools() {
  return {
    tools: TOOL_DEFINITIONS.map((definition) => ({
      name: definition.name,
      summary: definition.label,
      readOnly: definition.readOnly,
      approvalClass: definition.readOnly ? "automatic" : "personBeforeEffect",
      network: "none",
      maxResponseBytes: 16 * 1024,
      timeoutSeconds: 30,
    })),
    mode: "Work",
  };
}

/** A peer standing in for the Rust core, scripted per method. */
function coreStub(handlers: Record<string, (params: unknown) => unknown>) {
  const calls: Array<{ method: string; params: unknown }> = [];
  const events: unknown[] = [];
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);

  peer.request = ((method: string, params: unknown) => {
    calls.push({ method, params });
    // Served by default so each test can be about its own property rather than
    // about the one-off eligibility fetch every run makes. A test that cares
    // what the catalogue said overrides it like any other handler.
    const handler = handlers[method] ?? (method === "tool.catalogue" ? eligibleTools : undefined);
    if (!handler) return Promise.reject(new Error(`core stub has no ${method}`));
    try {
      return Promise.resolve(handler(params));
    } catch (error) {
      return Promise.reject(error);
    }
  }) as RpcPeer["request"];

  peer.notify = ((method: string, params: unknown) => {
    if (method === "run.event") events.push(params);
  }) as RpcPeer["notify"];

  return {
    peer,
    calls,
    events,
    /**
     * The methods this run called about tools, without the eligibility fetch.
     *
     * `calls` still holds every request, so nothing is hidden. This is what a
     * test asserting a call *sequence* wants: the fetch happens once at start-up
     * and says nothing about whether authorisation preceded execution.
     */
    get toolMethods() {
      return calls.map((call) => call.method).filter((method) => method !== "tool.catalogue");
    },
  };
}

let server: Awaited<ReturnType<typeof modelServer>> | undefined;

afterEach(async () => {
  await server?.close();
  server = undefined;
});

function request(baseUrl: string, prompt = "What is the seal specification?"): RunRequest {
  return {
    runId: "run-1",
    messageId: "msg-1",
    prompt,
    systemPrompt: "Search before answering.",
    model: { id: "test-model", provider: "sovereign-local", baseUrl, maxTokens: 256 },
  };
}

describe("a run that uses a tool", () => {
  beforeEach(async () => {
    server = await modelServer([
      // Turn one: ask for the tool.
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"seal specification"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      // Turn two: answer from what the tool returned.
      [chunk({ role: "assistant", content: "" }), chunk({ content: "The seal is 9.0 mm." }), chunk({}, "stop")],
    ]);
  });

  it("authorises before executing, and executes only with the grant it was given", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found. Maintenance SOP p.4: seal 9.0 mm." }),
    });

    const outcome = await startRun(core.peer, request(server!.baseUrl), () => {});

    const methods = core.toolMethods;
    expect(methods).toEqual(["tool.authorize", "tool.execute"]);

    // The grant issued by the authorise step is the one spent executing.
    expect(core.calls[1]!.params).toMatchObject({
      runId: "run-1",
      toolCallId: "call_1",
      tool: "knowledge.search_authorized",
      grant: "g-1",
    });
    expect(outcome.text).toBe("The seal is 9.0 mm.");
  });

  it("gives the model the tool result, so the answer is grounded in it", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found. Maintenance SOP p.4: seal 9.0 mm." }),
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    // Two turns means the tool result went back and the model spoke again.
    expect(server!.requests).toHaveLength(2);
    const secondTurn = JSON.stringify(server!.requests[1]);
    expect(secondTurn).toContain("Maintenance SOP p.4");
  });

  it("reports lifecycle to the operator without echoing tool arguments", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found." }),
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    const types = core.events.map((event) => (event as { event: { type: string } }).event.type);
    expect(types).toContain("agent_start");
    expect(types).toContain("tool_execution_start");
    expect(types).toContain("agent_end");

    // The arguments are in the audit record under access control; sending them
    // again over the event channel would put document text on a second path.
    const toolEvents = core.events.filter((event) =>
      (event as { event: { type: string } }).event.type.startsWith("tool_execution"),
    );
    expect(toolEvents.length).toBeGreaterThan(0);
    for (const event of toolEvents) {
      expect((event as { event: { args?: unknown } }).event.args).toBeUndefined();
    }
  });
});

describe("a run whose tool call is refused", () => {
  beforeEach(async () => {
    server = await modelServer([
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"salary list"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      [
        chunk({ role: "assistant", content: "" }),
        chunk({ content: "I am not permitted to search that." }),
        chunk({}, "stop"),
      ],
    ]);
  });

  it("never executes, and hands the refusal back as something the model can read", async () => {
    const core = coreStub({
      "tool.authorize": () => ({
        outcome: "refuse",
        reason: "You do not hold SearchKnowledge for that collection.",
      }),
      "tool.execute": () => {
        throw new Error("execute must not be reached after a refusal");
      },
    });

    const outcome = await startRun(core.peer, request(server!.baseUrl), () => {});

    expect(core.toolMethods).toEqual(["tool.authorize"]);
    // The refusal reaches the model as a tool result, so it can say so rather
    // than stall — the run completes.
    expect(JSON.stringify(server!.requests[1])).toContain("SearchKnowledge");
    expect(outcome.text).toContain("not permitted");
  });
});

describe("a run the gateway cannot be asked about", () => {
  beforeEach(async () => {
    server = await modelServer([
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"x"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      [chunk({ role: "assistant", content: "" }), chunk({ content: "I could not check." }), chunk({}, "stop")],
    ]);
  });

  it("fails closed rather than running the tool anyway", async () => {
    const core = coreStub({
      "tool.authorize": () => {
        throw new Error("core closed the channel");
      },
      "tool.execute": () => {
        throw new Error("execute must not be reached when authorisation failed");
      },
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    expect(core.toolMethods).toEqual(["tool.authorize"]);
    expect(JSON.stringify(server!.requests[1])).toContain("authorisation is unavailable");
  });
});

describe("endpoint policy", () => {
  it("refuses a model endpoint that is not loopback, before any socket opens", async () => {
    const core = coreStub({});
    await expect(
      startRun(
        core.peer,
        { ...request("https://api.openai.com/v1"), runId: "run-x" }, // arjun-egress-ok: a fixture proving this endpoint is refused, never reached
        () => {},
      ),
    ).rejects.toThrow(/not loopback/);
    expect(core.calls).toHaveLength(0);
  });

  it("refuses a private-network endpoint too, not just the public internet", async () => {
    const core = coreStub({});
    await expect(
      startRun(core.peer, request("http://192.168.1.50:8000/v1"), () => {}), // arjun-egress-ok: a fixture proving a private-network endpoint is refused
    ).rejects.toThrow(/not loopback/);
  });

  it("accepts the loopback forms a local server actually binds to", async () => {
    // localhost, 127.x and ::1 are all legitimate llama-server/vLLM bindings.
    server = await modelServer([[chunk({ role: "assistant", content: "" }), chunk({ content: "ok" }), chunk({}, "stop")]]);
    const port = new URL(server.baseUrl).port;
    const core = coreStub({});
    const outcome = await startRun(core.peer, request(`http://localhost:${port}/v1`), () => {});
    expect(outcome.text).toBe("ok");
  });
});

describe("aborting", () => {
  it("registers a handle the core can use to stop the run", async () => {
    server = await modelServer([[chunk({ role: "assistant", content: "" }), chunk({ content: "done" }), chunk({}, "stop")]]);
    const core = coreStub({});
    let registered: { abort: (reason?: unknown) => void } | undefined;

    await startRun(core.peer, request(server.baseUrl), (run) => {
      registered = run;
    });

    expect(registered).toBeDefined();
    // Aborting a finished run is a normal race and must not throw.
    expect(() => registered!.abort("operator stopped it")).not.toThrow();
  });
});

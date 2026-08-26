import { describe, expect, it, vi } from "vitest";
import { RpcError, RpcPeer, type PeerTransport } from "./peer.js";
import { FrameDecoder, encodeFrame, type Frame } from "./protocol.js";

/** A transport whose far side is a test, not a process. */
function harness() {
  let dataSink: (chunk: string) => void = () => {};
  let closeSink: () => void = () => {};
  const sent: Frame[] = [];
  const decoder = new FrameDecoder();

  const transport: PeerTransport = {
    write: (line) => {
      for (const frame of decoder.push(line)) sent.push(frame);
    },
    onData: (sink) => {
      dataSink = sink;
    },
    onClose: (sink) => {
      closeSink = sink;
    },
  };

  return {
    transport,
    sent,
    /** Deliver a frame as though the far side sent it. */
    receive: (frame: Frame) => dataSink(encodeFrame(frame)),
    receiveRaw: (chunk: string) => dataSink(chunk),
    endStream: () => closeSink(),
  };
}

describe("RpcPeer requests", () => {
  it("resolves a request when its matching result arrives", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);

    const pending = peer.request("tool.execute", { tool: "search_documents" });
    expect(h.sent[0]).toMatchObject({ method: "tool.execute" });

    const id = (h.sent[0] as { id: string }).id;
    h.receive({ id, result: { text: "found" } });

    await expect(pending).resolves.toEqual({ text: "found" });
  });

  it("correlates by id, not by arrival order", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);

    const first = peer.request("a");
    const second = peer.request("b");
    const ids = h.sent.map((f) => (f as { id: string }).id);
    const [idA, idB] = [ids[0]!, ids[1]!];

    // Replies come back inverted, which is normal when one tool is slower.
    h.receive({ id: idB, result: "B" });
    h.receive({ id: idA, result: "A" });

    await expect(first).resolves.toBe("A");
    await expect(second).resolves.toBe("B");
  });

  it("rejects with the far side's code so callers can branch on it", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    const pending = peer.request("tool.execute");
    const id = (h.sent[0] as { id: string }).id;

    h.receive({ id, error: { code: "refused", message: "Not permitted" } });

    await expect(pending).rejects.toMatchObject({ code: "refused", message: "Not permitted" });
    await expect(pending).rejects.toBeInstanceOf(RpcError);
  });

  it("drops a reply that matches no outstanding request", () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    expect(() => h.receive({ id: "999", result: "orphan" })).not.toThrow();
  });
});

describe("RpcPeer as a server", () => {
  it("replies with the handler's return value", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    peer.handle("health", () => ({ ready: true }));

    h.receive({ id: "7", method: "health" });
    await vi.waitFor(() => expect(h.sent).toHaveLength(1));

    expect(h.sent[0]).toEqual({ id: "7", result: { ready: true } });
  });

  it("turns a handler throw into an error frame rather than dying", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    peer.handle("boom", () => {
      throw new Error("exploded");
    });

    h.receive({ id: "8", method: "boom" });
    await vi.waitFor(() => expect(h.sent).toHaveLength(1));

    expect(h.sent[0]).toEqual({ id: "8", error: { code: "internal", message: "exploded" } });
  });

  it("preserves an RpcError's code through the reply", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    peer.handle("gated", () => {
      throw new RpcError("refused", "gateway said no");
    });

    h.receive({ id: "9", method: "gated" });
    await vi.waitFor(() => expect(h.sent).toHaveLength(1));

    expect(h.sent[0]).toEqual({ id: "9", error: { code: "refused", message: "gateway said no" } });
  });

  it("answers an unregistered method with unknown_method", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);

    h.receive({ id: "10", method: "nope" });
    await vi.waitFor(() => expect(h.sent).toHaveLength(1));

    expect(h.sent[0]).toMatchObject({ id: "10", error: { code: "unknown_method" } });
  });

  it("never replies to a notification", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    const seen = vi.fn();
    peer.onNotification("run.event", seen);

    h.receive({ method: "run.event", params: { n: 1 } });

    await vi.waitFor(() => expect(seen).toHaveBeenCalledWith({ n: 1 }));
    expect(h.sent).toHaveLength(0);
  });
});

describe("RpcPeer lifecycle", () => {
  it("rejects everything outstanding when the channel closes", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    const pending = peer.request("slow");

    h.endStream();

    await expect(pending).rejects.toMatchObject({ code: "peer_closed" });
    expect(peer.closed).toBe(true);
  });

  it("refuses new requests once closed instead of hanging forever", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    peer.close();
    await expect(peer.request("anything")).rejects.toMatchObject({ code: "peer_closed" });
  });

  it("treats a desynchronised channel as fatal and stops", async () => {
    const h = harness();
    const peer = new RpcPeer(h.transport);
    const fatal = vi.fn();
    peer.onFatal(fatal);
    const pending = peer.request("inflight");

    h.receiveRaw("this is not a frame\n");

    expect(fatal).toHaveBeenCalledOnce();
    expect(peer.closed).toBe(true);
    await expect(pending).rejects.toBeDefined();
  });

  it("swallows notification failures so a dropped event cannot fail a run", () => {
    const failing: PeerTransport = {
      write: () => {
        throw new Error("pipe closed");
      },
      onData: () => {},
      onClose: () => {},
    };
    const peer = new RpcPeer(failing);
    expect(() => peer.notify("run.event", { n: 1 })).not.toThrow();
  });
});

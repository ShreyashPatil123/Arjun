import { describe, expect, it } from "vitest";
import { FrameDecoder, encodeFrame, isError, isNotification, isRequest, isResult } from "./protocol.js";

describe("FrameDecoder", () => {
  it("reassembles a frame split across chunk boundaries", () => {
    const decoder = new FrameDecoder();
    const line = encodeFrame({ id: "1", method: "run.start", params: { runId: "r" } });
    const split = Math.floor(line.length / 2);

    expect(decoder.push(line.slice(0, split))).toEqual([]);
    const frames = decoder.push(line.slice(split));

    expect(frames).toHaveLength(1);
    expect(frames[0]).toEqual({ id: "1", method: "run.start", params: { runId: "r" } });
  });

  it("returns every frame when a chunk carries several", () => {
    const decoder = new FrameDecoder();
    const chunk =
      encodeFrame({ method: "run.event", params: { n: 1 } }) +
      encodeFrame({ method: "run.event", params: { n: 2 } }) +
      encodeFrame({ method: "run.event", params: { n: 3 } });

    expect(decoder.push(chunk)).toHaveLength(3);
  });

  it("ignores blank lines rather than treating them as frames", () => {
    const decoder = new FrameDecoder();
    expect(decoder.push("\n\n" + encodeFrame({ method: "health" }))).toHaveLength(1);
  });

  it("throws on a line that is not JSON, because the channel is now untrustworthy", () => {
    const decoder = new FrameDecoder();
    expect(() => decoder.push("not json\n")).toThrow(/Malformed frame/);
  });

  it("throws on a JSON value that is not a frame shape", () => {
    const decoder = new FrameDecoder();
    expect(() => decoder.push('{"hello":"world"}\n')).toThrow(/matches no known shape/);
    expect(() => decoder.push("[1,2,3]\n")).toThrow(/must be a JSON object/);
  });

  it("refuses to buffer without bound so one bad peer cannot exhaust memory", () => {
    const decoder = new FrameDecoder(64);
    expect(() => decoder.push("x".repeat(65))).toThrow(/Channel desynchronised/);
    // Buffer is dropped, so the decoder does not throw forever afterwards.
    expect(decoder.pending).toBe(0);
  });

  it("reports a truncated tail so EOF mid-frame is detectable", () => {
    const decoder = new FrameDecoder();
    decoder.push('{"id":"1","result":');
    expect(decoder.pending).toBeGreaterThan(0);
  });
});

describe("frame discrimination", () => {
  it("tells the four shapes apart", () => {
    expect(isRequest({ id: "1", method: "m" })).toBe(true);
    expect(isResult({ id: "1", result: null })).toBe(true);
    expect(isError({ id: "1", error: { code: "c", message: "m" } })).toBe(true);
    expect(isNotification({ method: "m" })).toBe(true);

    // A request is not a notification even though both carry `method`.
    expect(isNotification({ id: "1", method: "m" })).toBe(false);
    expect(isRequest({ method: "m" })).toBe(false);
  });
});

describe("encodeFrame", () => {
  it("terminates with exactly one newline so framing is unambiguous", () => {
    const line = encodeFrame({ method: "health" });
    expect(line.endsWith("\n")).toBe(true);
    expect(line.slice(0, -1)).not.toContain("\n");
  });

  it("escapes newlines inside payloads rather than emitting them raw", () => {
    const line = encodeFrame({ id: "1", result: { text: "line one\nline two" } });
    expect(line.split("\n")).toHaveLength(2);
    const decoded = new FrameDecoder().push(line);
    expect(decoded[0]).toEqual({ id: "1", result: { text: "line one\nline two" } });
  });
});

import { describe, expect, it } from "vitest";
import {
  applyThinkingPolicy,
  DEFAULT_BASE_URL,
  hasSeparableReasoning,
  LOCAL_PROVIDERS,
  payloadPolicy,
  PROVIDER_LABEL,
} from "./providers.js";

describe("recognising models that emit a reasoning block", () => {
  it("spots the families that do", () => {
    for (const id of [
      "Qwen3-8B-Instruct",
      "qwen3-30b-a3b",
      "Qwen/Qwen3-Coder-30B",
      "QwQ-32B-Preview",
      "nemotron-3-nano",
      "Nemotron-3_Super",
    ]) {
      expect(hasSeparableReasoning(id), id).toBe(true);
    }
  });

  it("leaves everything else alone", () => {
    // Qwen2.5 does not emit a reasoning block, so patching it would send a
    // template argument its chat template does not know.
    for (const id of [
      "Qwen2.5-Coder-7B-Instruct",
      "Meta-Llama-3-8B-Instruct",
      "mistral-7b-instruct",
      "gemma-3-4b-it",
      "phi-4",
    ]) {
      expect(hasSeparableReasoning(id), id).toBe(false);
    }
  });
});

describe("the thinking policy", () => {
  it("turns thinking off explicitly rather than relying on the default", () => {
    // The whole point: a Qwen3 served without this thinks by default and the
    // block lands in the visible answer.
    const patched = applyThinkingPolicy({ model: "Qwen3-8B" }, "Qwen3-8B", false);
    expect(patched).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
  });

  it("turns thinking on when the routed model is a reasoning one", () => {
    const patched = applyThinkingPolicy({ model: "Qwen3-8B" }, "Qwen3-8B", true);
    expect(patched).toMatchObject({
      chat_template_kwargs: { enable_thinking: true },
    });
  });

  it("keeps Nemotron from returning an empty turn when thinking is off", () => {
    const patched = applyThinkingPolicy({}, "nemotron-3-nano", false) as Record<string, never>;
    expect(patched.chat_template_kwargs).toMatchObject({
      enable_thinking: false,
      force_nonempty_content: true,
    });
  });

  it("does not force content when Nemotron is allowed to think", () => {
    const patched = applyThinkingPolicy({}, "nemotron-3-nano", true) as Record<string, never>;
    expect(patched.chat_template_kwargs).not.toHaveProperty("force_nonempty_content");
  });

  it("returns a model that needs nothing completely untouched", () => {
    const payload = { model: "Qwen2.5-Coder-7B", messages: [] };
    expect(applyThinkingPolicy(payload, "Qwen2.5-Coder-7B", false)).toBe(payload);
  });

  it("preserves template arguments the caller already set", () => {
    const patched = applyThinkingPolicy(
      { chat_template_kwargs: { custom: "keep me" } },
      "Qwen3-8B",
      false,
    ) as Record<string, Record<string, unknown>>;
    expect(patched.chat_template_kwargs).toMatchObject({
      custom: "keep me",
      enable_thinking: false,
    });
  });

  it("does not mutate the payload it was given", () => {
    const payload = { model: "Qwen3-8B" };
    applyThinkingPolicy(payload, "Qwen3-8B", false);
    expect(payload).toEqual({ model: "Qwen3-8B" });
  });

  it("ignores a payload that is not an object rather than throwing", () => {
    for (const payload of [null, undefined, "text", 42, []]) {
      expect(() => applyThinkingPolicy(payload, "Qwen3-8B", false)).not.toThrow();
      expect(applyThinkingPolicy(payload, "Qwen3-8B", false)).toBe(payload);
    }
  });
});

describe("payloadPolicy", () => {
  it("keys on the model the transport is actually about to call", () => {
    const hook = payloadPolicy(false);
    expect(hook({}, { id: "Qwen3-8B" })).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
    expect(hook({ a: 1 }, { id: "Meta-Llama-3-8B" })).toEqual({ a: 1 });
  });
});

describe("provider identity", () => {
  it("covers both runtimes and nothing else", () => {
    // Two providers is the claim: C++ and Python. A third appearing here
    // without a matching Rust runtime would mean the two sides disagree.
    expect(Object.values(LOCAL_PROVIDERS).sort()).toEqual(["llama-cpp", "vllm"]);
  });

  it("has a label and a default endpoint for each", () => {
    for (const provider of Object.values(LOCAL_PROVIDERS)) {
      expect(PROVIDER_LABEL[provider]).toBeTruthy();
      expect(DEFAULT_BASE_URL[provider]).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
    }
  });

  it("defaults to loopback for both, never to a public host", () => {
    for (const url of Object.values(DEFAULT_BASE_URL)) {
      expect(new URL(url).hostname).toBe("127.0.0.1");
    }
  });
});

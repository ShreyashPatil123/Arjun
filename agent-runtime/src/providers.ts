/**
 * The quirks of the two local inference servers ARJUN runs against.
 *
 * `llama-server` (C++/GGUF) and vLLM or SGLang (Python) both speak the
 * OpenAI-compatible chat API, which is why one agent loop drives both. They are
 * not, however, identical — and the differences that matter are not in the
 * protocol but in what the *model* does when served by each.
 *
 * ## The reasoning-block problem
 *
 * Qwen3 and Nemotron emit a reasoning block before their answer. Whether that
 * block arrives as a separate field or gets glued into the visible content
 * depends on how the server was told to apply the chat template — and the two
 * servers take that instruction in different places. Left alone, a run against
 * vLLM produces answers that begin with the model thinking out loud, and an
 * approval note that opens with `<think>` is not a deliverable.
 *
 * The knowledge here is adapted from OpenClaw's `extensions/vllm`
 * (`thinking-policy.ts`, `stream.ts`, MIT). ARJUN applies it through the
 * transport's `onPayload` hook rather than through OpenClaw's plugin host,
 * which it does not adopt.
 *
 * ## Why this is a payload patch and not configuration
 *
 * An operator should not have to know that the Qwen they registered needs a
 * different flag from the Llama next to it. The model id is enough to tell, so
 * ARJUN tells, and the registry entry stays a description of the model rather
 * than a description of its server's foibles.
 */

/** The provider ids Rust sends, one per runtime. */
export const LOCAL_PROVIDERS = {
  /** `llama-server` — GGUF weights, started by ARJUN. */
  llamaCpp: "llama-cpp",
  /** vLLM or SGLang — Python weights, run by an operator. */
  vllm: "vllm",
} as const;

export type LocalProvider = (typeof LOCAL_PROVIDERS)[keyof typeof LOCAL_PROVIDERS];

/**
 * Endpoints these servers bind by default.
 *
 * Not used to *reach* anything — Rust always supplies the real endpoint — but
 * to say something useful when it supplies one that is not answering, and as
 * the value a setup screen offers first. Taken from the upstream adapters'
 * `defaults.ts`, which is where the conventional ports are recorded.
 */
export const DEFAULT_BASE_URL: Record<LocalProvider, string> = {
  [LOCAL_PROVIDERS.llamaCpp]: "http://127.0.0.1:8080/v1",
  [LOCAL_PROVIDERS.vllm]: "http://127.0.0.1:8000/v1",
};

/** Human-readable, for the trace. */
export const PROVIDER_LABEL: Record<LocalProvider, string> = {
  [LOCAL_PROVIDERS.llamaCpp]: "llama.cpp",
  [LOCAL_PROVIDERS.vllm]: "vLLM",
};

/** Model families whose reasoning has to be asked for, or suppressed, explicitly. */
const QWEN_THINKING = /\bqwen-?3\b|\bqwq\b/i;
const NEMOTRON_THINKING = /\bnemotron-3(?:[-_](?:nano|super|ultra))?\b/i;

/** Whether a model id names a family that emits a reasoning block. */
export function hasSeparableReasoning(modelId: string): boolean {
  return QWEN_THINKING.test(modelId) || NEMOTRON_THINKING.test(modelId);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Adds the chat-template arguments a model needs, without disturbing the rest.
 *
 * Returns the payload unchanged when the model needs nothing, so this is safe to
 * install unconditionally — which matters, because the alternative is a
 * per-model switch somebody has to remember to set.
 *
 * `enable_thinking` is passed inside `chat_template_kwargs` rather than at the
 * top level. Both forms exist in the wild; the nested one is what current vLLM
 * and llama-server builds read, and the top-level form is silently ignored by
 * servers that do not know it — which is the failure mode this is meant to
 * prevent, so the quieter option is the wrong one.
 */
export function applyThinkingPolicy(
  payload: unknown,
  modelId: string,
  reasoningWanted: boolean,
): unknown {
  if (!isRecord(payload) || !hasSeparableReasoning(modelId)) {
    return payload;
  }

  const existing = isRecord(payload.chat_template_kwargs) ? payload.chat_template_kwargs : {};
  const kwargs: Record<string, unknown> = {
    ...existing,
    // An explicit false is the point: a Qwen3 served without it defaults to
    // thinking, and the block lands in the visible answer.
    enable_thinking: reasoningWanted,
  };

  if (NEMOTRON_THINKING.test(modelId) && !reasoningWanted) {
    // Nemotron with thinking off can return an empty content block, which the
    // loop reads as an assistant turn that said nothing and the operator reads
    // as a hang.
    kwargs.force_nonempty_content = true;
  }

  return { ...payload, chat_template_kwargs: kwargs };
}

/**
 * The `onPayload` hook for a run.
 *
 * Curried over the model id because the transport hands the hook a `Model`, and
 * the id ARJUN routed to is the one the policy should key on.
 */
export function payloadPolicy(
  reasoningWanted: boolean,
): (payload: unknown, model: { id: string }) => unknown {
  return (payload, model) => applyThinkingPolicy(payload, model.id, reasoningWanted);
}

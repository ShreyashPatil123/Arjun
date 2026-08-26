/**
 * Transport-aware stream factory selection.
 *
 * Routes models that need OpenClaw-managed proxy/TLS/local-service semantics onto built-in transport implementations.
 */
import type { Api, Model, StreamFn } from "@openclaw/llm-core";
import { getAiTransportHost } from "../host.js";
import { createOpenAICompletionsTransportStreamFn } from "./openai-completions-transport.js";

// ARJUN sovereign build: openai-completions is the only transport-aware API.
// Upstream also lists openai-responses, openai-chatgpt-responses,
// azure-openai-responses, anthropic-messages and google-generative-ai; each of
// those dispatches to a vendor endpoint, so they are removed from the switch
// below and their transport modules are not vendored. A model whose api is one
// of those now fails closed at dispatch rather than opening a connection.
const SUPPORTED_TRANSPORT_APIS = new Set<Api>(["openai-completions"]);

const SIMPLE_TRANSPORT_API_ALIAS: Record<string, Api> = {
  "openai-completions": "openclaw-openai-completions-transport",
};

type ProviderTransportStreamContext = {
  cfg?: unknown;
  agentDir?: string;
  workspaceDir?: string;
  env?: NodeJS.ProcessEnv;
};

function createSupportedTransportStreamFn(
  model: Model,
  _ctx?: ProviderTransportStreamContext,
): StreamFn | undefined {
  switch (model.api) {
    case "openai-completions":
      return createOpenAICompletionsTransportStreamFn();
    default:
      return undefined;
  }
}

function hasOpenClawTransportRequirement(model: Model): boolean {
  return getAiTransportHost().requiresManagedTransport(model);
}

/** Returns whether OpenClaw has a managed transport implementation for this API. */
function isTransportAwareApiSupported(api: Api): boolean {
  return SUPPORTED_TRANSPORT_APIS.has(api);
}

/** Maps public model APIs to the internal transport API id used by simple runtime dispatch. */
export function resolveTransportAwareSimpleApi(api: Api): Api | undefined {
  return SIMPLE_TRANSPORT_API_ALIAS[api];
}

/** Creates a managed transport stream only when request overrides require it. */
export function createTransportAwareStreamFnForModel(
  model: Model,
  ctx?: ProviderTransportStreamContext,
): StreamFn | undefined {
  if (!hasOpenClawTransportRequirement(model)) {
    return undefined;
  }
  if (!isTransportAwareApiSupported(model.api)) {
    throw new Error(
      `Model-provider request.proxy/request.tls/localService is not yet supported for api "${model.api}"`,
    );
  }
  const streamFn = createSupportedTransportStreamFn(model, ctx);
  if (!streamFn) {
    throw new Error(`Managed transport stream is unavailable for api "${model.api}"`);
  }
  return streamFn;
}

/** Creates a managed OpenClaw transport stream for explicit fallback/runtime callers. */
export function createOpenClawTransportStreamFnForModel(
  model: Model,
  ctx?: ProviderTransportStreamContext,
): StreamFn | undefined {
  // Explicit fallback callers use this when they need OpenClaw's HTTP
  // transport semantics regardless of the default embedded-runner strategy.
  // Native OpenAI HTTP still depends on this path for strict tool shaping,
  // attribution, cache-boundary stripping, and runtime credential injection.
  if (!isTransportAwareApiSupported(model.api)) {
    return undefined;
  }
  return createSupportedTransportStreamFn(model, ctx);
}

export function createBoundaryAwareStreamFnForModel(
  model: Model,
  ctx?: ProviderTransportStreamContext,
): StreamFn | undefined {
  // Default embedded-runner fallback. Keep OpenAI-family APIs here while native
  // HTTP streams preserve the same OpenClaw request contract.
  if (!isTransportAwareApiSupported(model.api)) {
    return undefined;
  }
  return createSupportedTransportStreamFn(model, ctx);
}

export function prepareTransportAwareSimpleModel<TApi extends Api>(
  model: Model<TApi>,
  ctx?: ProviderTransportStreamContext,
): Model {
  const streamFn = createTransportAwareStreamFnForModel(model as Model, ctx);
  const alias = resolveTransportAwareSimpleApi(model.api);
  if (!streamFn || !alias) {
    return model;
  }
  return getAiTransportHost().inheritManagedTransport(model, {
    ...model,
    api: alias,
  });
}

export function buildTransportAwareSimpleStreamFn(
  model: Model,
  ctx?: ProviderTransportStreamContext,
): StreamFn | undefined {
  return createTransportAwareStreamFnForModel(model, ctx);
}

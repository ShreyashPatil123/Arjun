// ARJUN sovereign build: this file replaces upstream's transport matrix test.
//
// Upstream asserts that openai-responses, openai-chatgpt-responses,
// azure-openai-responses, anthropic-messages and google-generative-ai each
// route to their managed transport. In this build those transports are not
// vendored, so the upstream expectations are not merely failing -- they assert
// the opposite of the property we need.
//
// What is asserted here is the property the sovereignty claim rests on: the
// dispatcher can produce a stream for exactly one API, the local
// OpenAI-compatible one, and every cloud API fails closed instead of silently
// returning undefined and letting a caller fall through to a direct SDK path.
import type { Api, Model } from "@openclaw/llm-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { configureAiTransportHost, getAiTransportHost } from "../host.js";
import {
  buildTransportAwareSimpleStreamFn,
  createBoundaryAwareStreamFnForModel,
  createOpenClawTransportStreamFnForModel,
  createTransportAwareStreamFnForModel,
  prepareTransportAwareSimpleModel,
  resolveTransportAwareSimpleApi,
} from "./provider-transport-stream.js";

const managedTransportModels = new WeakSet<object>();
const initialHost = getAiTransportHost();

/** Marks a model as carrying a proxy/TLS/localService override. */
function attachManagedTransport<TModel extends object>(model: TModel): TModel {
  const attached = { ...model };
  managedTransportModels.add(attached);
  return attached;
}

beforeAll(() => {
  configureAiTransportHost({
    requiresManagedTransport: (model) => managedTransportModels.has(model),
    inheritManagedTransport: (source, target) => {
      if (managedTransportModels.has(source)) {
        managedTransportModels.add(target);
      }
      return target;
    },
  });
});

afterAll(() => {
  configureAiTransportHost(initialHost);
});

function buildModel<TApi extends Api>(api: TApi, provider: string, baseUrl: string): Model<TApi> {
  const id = `${provider}-model`;
  return {
    id,
    name: id,
    api,
    provider,
    baseUrl,
  } as Model<TApi>;
}

/** Every API upstream supports that reaches a vendor endpoint. */
const CLOUD_APIS: Api[] = [
  "openai-responses",
  "openai-chatgpt-responses",
  "azure-openai-responses",
  "anthropic-messages",
  "google-generative-ai",
] as Api[];

describe("sovereign transport dispatch", () => {
  it("routes a local OpenAI-compatible model through the managed transport", () => {
    // llama-server, vLLM and SGLang all present as openai-completions.
    const model = attachManagedTransport(
      buildModel("openai-completions", "llama-cpp", "http://127.0.0.1:8080/v1"),
    );
    expect(createTransportAwareStreamFnForModel(model)).toBeTypeOf("function");
    expect(buildTransportAwareSimpleStreamFn(model)).toBeTypeOf("function");
  });

  it("aliases only the local completions API for simple dispatch", () => {
    expect(resolveTransportAwareSimpleApi("openai-completions" as Api)).toBe(
      "openclaw-openai-completions-transport",
    );
    for (const api of CLOUD_APIS) {
      expect(resolveTransportAwareSimpleApi(api)).toBeUndefined();
    }
  });

  it("fails closed when a cloud API carries a transport override", () => {
    // Failing closed matters more than failing quietly: a caller that receives
    // undefined here is free to fall back to a direct provider SDK, which is
    // precisely the egress path this build removes.
    for (const api of CLOUD_APIS) {
      const model = attachManagedTransport(buildModel(api, "vendor", "https://api.example.com/v1"));
      expect(() => createTransportAwareStreamFnForModel(model)).toThrow(
        /not yet supported for api/,
      );
    }
  });

  it("offers no managed or boundary stream for cloud APIs", () => {
    for (const api of CLOUD_APIS) {
      const model = buildModel(api, "vendor", "https://api.example.com/v1");
      expect(createOpenClawTransportStreamFnForModel(model)).toBeUndefined();
      expect(createBoundaryAwareStreamFnForModel(model)).toBeUndefined();
    }
  });

  it("leaves a model without transport overrides untouched", () => {
    const model = buildModel("openai-completions", "vllm", "http://127.0.0.1:8000/v1");
    expect(createTransportAwareStreamFnForModel(model)).toBeUndefined();
    expect(prepareTransportAwareSimpleModel(model)).toBe(model);
  });

  it("re-aliases a local model onto the managed transport api", () => {
    const model = attachManagedTransport(
      buildModel("openai-completions", "sglang", "http://127.0.0.1:30000/v1"),
    );
    const prepared = prepareTransportAwareSimpleModel(model);
    expect(prepared.api).toBe("openclaw-openai-completions-transport");
  });
});

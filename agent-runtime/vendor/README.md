# Vendored OpenClaw

`vendor/openclaw/` holds a pruned copy of [openclaw/openclaw](https://github.com/openclaw/openclaw)
(MIT), pinned to commit **`ed56f3c001a6d18427bc399493edabc7166233bd`**.

It is vendored rather than depended on because the packages we need are marked
`private` upstream and are not published to npm, and because an air-gapped build
has to be able to reproduce its whole dependency tree from a local store.

## What this copy is for

ARJUN runs open-weight models served two different ways — llama.cpp/`llama-server`
(C++, GGUF) and vLLM/SGLang (Python). Both speak the OpenAI-compatible chat API, so
one transport covers both. OpenClaw's `packages/ai` is that transport, and
`packages/agent-core` is the loop that drives it.

## What was removed, and why

The upstream repository is a personal assistant with roughly sixty cloud providers,
a dozen chat channels, web search, a browser, and remote execution backends. None of
that belongs in an air-gapped industrial workbench, so the copy is pruned rather than
configured.

| Removed | Reason |
|---|---|
| `@anthropic-ai/sdk`, `@google/genai`, `@mistralai/mistralai` | Vendor SDKs. Dropped from `packages/ai/package.json`; the five source files that imported them are deleted. |
| `providers/anthropic.ts`, `google*.ts`, `mistral*.ts` | The provider entrypoints that used those SDKs. |
| `providers/openai-responses*`, `openai-chatgpt-responses*`, `azure-openai-responses*` | OpenAI's cloud Responses API, Codex backend, and Azure. Fetch-based rather than SDK-based, but still vendor endpoints. |
| `transports/openai-responses-*`, `transports/anthropic-*` | Their transport layers, including a WebSocket client to OpenAI. |
| `prepareCodexSimpleTransportModel` | Routed models to `chatgpt.com/backend-api`. |
| All `*.live.test.ts` | Tests that call real provider APIs over the network. |
| `extensions/` | Moved to `../reference/openclaw-extensions`. See "Why the extensions are not vendored". |
| `packages/plugin-sdk` | A re-export shim over the root application's `src/plugin-sdk/`, which is not vendored. |

Two registries were narrowed rather than emptied:

- `providers/register-builtins.ts` registers **only** `openai-completions`.
- `transports/provider-transport-stream.ts` dispatches **only** `openai-completions`,
  and throws for any other API rather than returning `undefined` — a caller that
  received `undefined` would be free to fall back to a direct provider path.

Deliberately **kept** despite vendor-sounding names: `clean-for-gemini.ts`,
`cloudflare.ts`, `github-copilot-headers.ts`, `anthropic-model-contract.ts` and
similar. These are pure schema/URL/header helpers with no SDK import and no network
call, and the shared transport imports them. Deleting them would mean patching files
we otherwise take unchanged, which raises the cost of every future re-sync for no
security gain.

## Why the extensions are not vendored

`extensions/vllm`, `extensions/sglang` and `extensions/llama-cpp` look like the
obvious thing to reuse, but they import `openclaw/plugin-sdk/provider-model-shared`
— a path inside the **root OpenClaw application**, not inside any package. They are
plugins for the Gateway host, not standalone libraries, and cannot be lifted without
dragging in the application we are specifically not adopting.

The part with actual value is small: each is a few dozen lines of configuration
(base URL, API key environment variable, thinking profile, stream wrapper) around a
host helper. ARJUN supplies its own equivalent against `packages/ai`'s registry.
The upstream sources are kept at `../reference/openclaw-extensions` as a reference
for those values.

`extensions/llama-cpp` additionally contains a managed-install path
(`llama-server-install.ts`, `llama-server-assets.ts`) that downloads server builds
and model weights at runtime. That is incompatible with the air gap regardless.
Models and binaries reach ARJUN through its signed offline package import.

## Reductions

Two packages were reduced to the single module `packages/ai` imports, rather than
vendored whole:

- `markdown-core` — kept `reasoning-tags.ts` and `reasoning-tag-parser.ts` (44 source
  files → 2). Dropped `markdown-it`, `markdown-it-cjk-friendly` and `yaml` with the
  modules that needed them. The three `mdast`/`micromark` dependencies are genuinely
  required by the parser.
- `media-core` — kept `base64.ts`. Dropped the `file-type` dependency.

## How source is resolved

The vendored manifests point `main`/`exports` at `dist/`, which upstream builds with
tsdown. **We do not vendor a build step.** The source is what gets reviewed for the
sovereignty claim, and a prebuilt `dist` would be an unreviewed artifact in the audit
path.

So `vendor/openclaw/tsconfig.json` carries a `paths` map from package specifier to
source file, and `vitest.config.ts` resolves `@openclaw/*` through that same map.
One table, used by both, because subpath and file layout genuinely disagree upstream:
`@openclaw/ai/event-stream` is declared as `dist/event-stream.mjs` but lives at
`src/utils/event-stream.ts`.

`workspace:*` was rewritten to `*` throughout, because this repository installs with
npm and `workspace:` is a pnpm protocol.

## Re-syncing to a newer upstream commit

A re-sync will silently reintroduce everything removed above. The guard is
`scripts/audit-vendor.mjs`, which fails the build if a forbidden SDK, an excluded
provider module, or an unexpected registered API reappears, and if any relative
import stops resolving.

```bash
node scripts/audit-vendor.mjs
npx vitest run
```

Both must be green. When re-syncing, re-apply the table above, then run the audit —
it is the record of what this copy is supposed to be, and it is more reliable than
reading the diff.

## Modified upstream tests

Three test files were rewritten rather than deleted, because the behaviour they
covered still matters but the fixtures depended on removed providers. Each carries a
header explaining the change:

- `transports/provider-transport-stream.test.ts` — now asserts that only the local
  API dispatches and that cloud APIs fail closed.
- `transports/simple-completion-transport.test.ts` — Codex blocks removed.
- `host.test.ts` — the registration-replay test no longer routes through Codex.
- `package-dependencies.test.ts` — derives the expected dependency set from the
  sources instead of hardcoding a list that the prune invalidated.

One test was deleted outright: `transports/provider-compaction-replay.test.ts`. Its
fixtures were built from the Responses helpers. The module it covered,
`provider-compaction-replay.ts`, is retained and is currently **uncovered**.

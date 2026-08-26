# Third-Party Notices

ARJUN incorporates third-party software. This file records those components and
their licences, as their licences require.

## OpenClaw

A pruned copy of [openclaw/openclaw](https://github.com/openclaw/openclaw) is
vendored at `agent-runtime/vendor/openclaw/`, pinned to commit
`ed56f3c001a6d18427bc399493edabc7166233bd`.

Portions of that copy have been modified: cloud provider adapters, their SDK
dependencies, cloud transport layers, chat channels and the plugin host shim are
removed, and two registries are narrowed to a single OpenAI-compatible transport.
`agent-runtime/vendor/README.md` records every change. Modified files carry an
"ARJUN sovereign build" comment at the point of change.

The upstream licence is reproduced verbatim at
`agent-runtime/vendor/openclaw/LICENSE`:

> MIT License
>
> Copyright (c) 2026 OpenClaw Foundation
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

OpenClaw records its own incorporated third-party code in its
`THIRD_PARTY_NOTICES.md`; that file is not reproduced here because the adapted
code it covers (Pi/pi-mono) is not part of the vendored subset.

## Agent runtime npm dependencies

The Node agent runtime resolves 50 third-party packages at runtime. The
authoritative, versioned list is `agent-runtime/package-lock.json`; regenerate a
readable inventory with:

```bash
npm --prefix agent-runtime ls --all --omit=dev
```

Direct runtime dependencies:

| Package | Version | Licence | Used for |
|---|---|---|---|
| `openai` | 7.5.0 | Apache-2.0 | OpenAI-compatible client, pointed at local llama-server / vLLM / SGLang endpoints |
| `typebox` | 1.3.15 | MIT | Tool and message schema validation |
| `zod` | 4.4.3 | MIT | Model catalogue schema validation |
| `ipaddr.js` | 2.5.0 | MIT | Network policy IP classification |
| `partial-json` | 0.1.7 | MIT | Incremental parsing of streamed tool-call arguments |
| `libphonenumber-js` | 1.13.11 | MIT | Pulled by `normalization-core`; not used by ARJUN |
| `mdast-util-from-markdown` | 2.0.3 | MIT | Reasoning-tag partitioning |
| `mdast-util-gfm-table` | 2.0.0 | MIT | Reasoning-tag partitioning |
| `micromark-extension-gfm-table` | 2.1.1 | MIT | Reasoning-tag partitioning |

The remaining packages are transitive dependencies of the three markdown parsers,
predominantly the `micromark` and `unist` families, all MIT.

No cloud model provider SDK is present. `agent-runtime/scripts/audit-vendor.mjs`
enforces that.

## Python document sidecar

`sidecars/document_sidecar/` uses Docling and pypdf. Their licences apply as
distributed; record versions in the deployment bundle's SBOM.

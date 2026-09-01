# ARJUN

A desktop workbench for running large language models entirely on your own
machine. ARJUN downloads, sizes, serves, and talks to local models — and is
built so that you can *prove* it never phones home.

Built with [Tauri 2](https://tauri.app): a React 19 + TypeScript frontend, a
Rust core, a vendored TypeScript agent runtime, and Python sidecars for
document and memory work.

> Status: `0.1.0`, in active development.

---

## Why it looks the way it does

Most local-LLM tools ask you to take "it runs offline" on faith. ARJUN treats
that claim as something the build has to demonstrate, so a few design choices
follow from it:

- **One egress chokepoint.** Exactly one Rust module
  (`src-tauri/src/sovereignty/broker.rs`) is allowed to construct an outbound
  HTTP client. `npm run check:egress` fails the build if a second one appears,
  and every external hostname anywhere in the tree has to be on a reviewed
  allowlist. Exemptions require a `arjun-egress-ok: <reason>` comment, so each
  one documents itself in review.
- **The agent loop is a sidecar, not a server.** `agent-runtime/` vendors
  OpenClaw's agent loop and its OpenAI-compatible transport, with the cloud
  providers stripped out. It speaks JSON-RPC over stdio to the Rust core and
  never opens a listening socket.
- **Gates inspect the artifact, not just the source.** `npm run check:bundle`
  reads the bundled runtime (`agent-runtime/dist/arjun-agent-runtime.mjs`) and
  reports which protocol adapters actually survived bundling — a stronger claim
  than any source-tree grep.
- **An SBOM ships with the evidence.** `npm run sbom` regenerates
  `evidence/sbom.md` and `evidence/sbom.cdx.json` (CycloneDX).

---

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Node.js | >= 22.19 | enforced by `agent-runtime/package.json` |
| Rust | stable, 2021 edition | plus the [Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS |
| Python | 3.10+ | only needed for the sidecars and their tests |

GPU acceleration is optional. ARJUN can build against **CUDA** or **Vulkan**,
or fall back to CPU.

---

## Getting started

```bash
npm install
npm run runtime:install   # installs the agent-runtime workspace, offline
```

Then start the app with the backend that matches your hardware:

```bash
npm run dev:auto          # let scripts/select-backend.mjs choose
npm run tauri:dev:gpu     # force CUDA
npm run tauri:dev:vulkan  # force Vulkan
```

`npm run dev` starts only the Vite frontend, which is useful for UI work but
will not have a Rust backend behind it.

### Building

```bash
npm run build:auto        # pick a backend and build
npm run tauri:build:gpu   # CUDA
npm run tauri:build:vulkan
```

On a fresh configuration, ARJUN uses
`lmstudio-community/gemma-4-12B-it-QAT-GGUF` (`Q4_0`) as the default
orchestrator and loads it automatically at startup. The automatic load requires
a CUDA- or Vulkan-enabled build and at least one layer must be resident on the
GPU; a CPU fallback is rejected rather than reported as GPU execution. Install
the model from Discover before restarting ARJUN. An administrator can choose
any ready installed model variant from **Models → Set as orchestrator**; ARJUN
persists its provider, model ID, and quantization and uses that exact variant on
future startups. Startup loading can be disabled with
`ai_settings.auto_load_on_startup`.

---

## Verifying a build

`npm run verify` runs the whole chain — egress gate, offline-build check,
vendor audit, typecheck, runtime tests, runtime build, bundle gates, SBOM, and
the Rust and Python test suites. It is the single command to run before
shipping or reviewing.

The individual gates, if you want them one at a time:

| Command | What it checks |
|---|---|
| `npm run check:egress` | only the broker can make outbound calls; hostnames are allowlisted |
| `npm run check:offline` | the build completes with no network access |
| `npm run runtime:audit` | the vendored OpenClaw copy still has its cloud providers removed |
| `npm run runtime:typecheck` | types across the agent runtime |
| `npm run runtime:test` | agent-runtime unit tests (Vitest) |
| `npm run test:ui` | frontend logic tests — run recovery from the durable record |
| `npm run check:bundle` | inspects the built runtime artifact for surviving providers |
| `npm run sbom` | regenerates the CycloneDX SBOM under `evidence/` |
| `npm run test:rust` | Rust unit tests |
| `npm run test:integration` | agent-runtime and two-runtime integration tests |
| `npm run test:baseline` | acceptance baseline |
| `npm run test:sidecar` | Python document-sidecar tests |
| `npm run accept` | acceptance run against `acceptance-baseline.json` |

---

## Project layout

```
src/                    React 19 + TypeScript frontend
  pages/                Workbench, Browse, Health, Approvals, AuditNetwork, …
  services/             typed bridges to Tauri commands
  sdk/  hooks/  contexts/

src-tauri/              Rust core (crate: sarathi)
  agent_runtime/        supervises the TS runtime; workspace sandbox,
                        capability grants, approval gating, artifact capture
    events/             durable, ordered task history in SQLite — snapshots
                        for the UI, idempotency keys for side effects, and
                        recovery of runs a restart interrupted
  serving/              model serving lifecycle and probing
  model_manager/        download, sizing, and installation
  model_intelligence/   hardware-aware recommendation
  model_package/        base-model package manifests and repair
  sovereignty/          the single egress broker
  policy/  capability/  audit/  identity/
  memory_engine/  knowledge/  documents/

agent-runtime/          vendored OpenClaw agent loop (TypeScript)
  src/                  protocol, run loop, providers, tools, compaction
  vendor/openclaw/      pruned upstream packages
  scripts/              vendor audit and bundler

sidecars/               Python sidecars (documents, memory engine, packs)
scripts/                build, verification, and evidence gates
evidence/               generated SBOM and audit output
```

---

## Architecture in one pass

1. The **React frontend** calls typed service wrappers in `src/services/`.
2. Those invoke **Tauri commands** in `src-tauri/src/commands/`.
3. The Rust core handles model download, hardware sizing, and serving, and
   supervises the **agent runtime** as a child process over stdio.
4. The agent runtime runs the loop and calls back into the host for tools,
   which are gated by capability grants and, where required, user approval.
5. Any outbound request — and there should be very few — goes through the
   sovereignty broker, the one audited chokepoint.
6. Everything a run does is written to an ordered, append-only history as it
   happens (`agent_runtime/events/`), separately from the task record a
   finished run leaves behind. That is what lets a window reattach to a run
   after a remount, and what lets the next start find the runs the previous
   process was carrying when it went away.

---

## Contributing

Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`,
`perf:`, `ci:`). Run `npm run verify` before opening a pull request; the
verification gates are the point of the project, and a change that trips one
needs a reason in review rather than a new exemption.

Third-party attributions live in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
ARJUN does not yet carry a license file.

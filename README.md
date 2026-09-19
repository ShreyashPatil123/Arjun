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

## Recent Updates & Engineering Log (September 2026)

ARJUN has undergone intensive architectural evolution over the recent days across its agent runtime, shared memory bus, multi-agent governance, hardware planning, and user interface. Every change is logged below:

### 1. Shared Agent Memory Bus & Dynamic Context Compiler
- **1,000,000-Token Chat Memory Bus (`chat_memory_bus.rs`):** Implemented an ultra-long context memory bus supporting up to 1M tokens with proactive, budget-aware context window truncation and deterministic token accounting.
- **Dynamic Context Compiler (`context_compiler.rs`):** Assembles system instructions, active conversational history, grounded knowledge passages, and persistent context pins into a compiled, hardware-bounded context window tailored to the active model.
- **Context Manifests & State Commits (`context_manifest.rs`, `state_commit.rs`):** Structured manifests capture the exact state of compiled context and pin bindings per turn, providing auditable state commits for reproducibility and session resumption.
- **Token Pinning Architecture (`pins.rs`):** Allows crucial system instructions, active tool definitions, and user-selected knowledge snippets to be pinned in memory so they are never evicted during context compaction.
- **Stateful Model Handoff & Model Transitions (`model_handoff.rs`, `model_transition.rs`):** Seamless runtime handoff between the primary orchestrator and specialist models (e.g., coding, reasoning, vision), preserving conversational turn state and memory boundaries without session restarts.

### 2. Multi-Agent Administration & Governance Dashboard
- **Agent Definitions & Store (`src-tauri/src/agents/`):** Persistent, declarative agent definitions with configurable system prompts, tool bindings, model preference orders, and role-based access control.
- **Agent Administration IPC (`commands/agent_admin.rs`, `commands/agents.rs`):** Robust IPC command suite for agent lifecycle management (creation, quarantine, activation, model bindings, skill verification).
- **Agents Management UI (`src/pages/Agents.tsx`):** Dedicated administration view in the desktop interface for configuring autonomous agents, monitoring active workloads, and managing model/skill assignments.
- **Frontend Agent Registry Service (`src/services/agentRegistry.service.ts`):** Typed TypeScript bridge connecting UI components to the Rust agent governance backend.

### 3. Interactive Knowledge & Memory Graph Canvas
- **Real-Time Memory Feed & Storage (`runtime_feed.rs`, `runtime_memory.rs`, `runtime_store.rs`):** Reactive event streaming architecture that publishes memory graph mutations (entity discoveries, relationship links, topic clusters) directly from the agent loop.
- **High-Performance Graph Canvas (`src/components/graph/MemoryGraphCanvas.tsx`):** Accelerated canvas renderer supporting hundreds of concurrent nodes and edges with smooth zoom, pan, force simulation, and hierarchical clustering.
- **Label Geometry & Collision Avoidance (`labelGeometry.ts`):** Multi-pass collision detection and bounding box placement preventing text overlap across dense entity clusters.
- **Visual Evidence & Stress Test Harness (`evidence/memory-graph/`):** Comprehensive automated visual regression and stress test harness (`harness.html`, `harness.tsx`) with recorded evidence for up to 500-node live mutation benchmarks, reduced-motion adaptations, and parent/child aggregation.

### 4. Subagent Worker Loop & Live Model Execution
- **Subagent Child Loop & Scheduler (`child_loop.rs`, `scheduling.rs`):** Concurrency-safe background worker execution loop enabling agents to spawn autonomous subagents for parallel research and task execution.
- **Bi-Directional Graph I/O (`graph_io.rs`):** Subagents stream findings, extracted relations, and memory nodes directly back into the shared workspace memory graph.
- **Live Integration Tests (`tests/model_binding_handoff_live.rs`, `tests/subagent_model_loop_live.rs`):** End-to-end integration test coverage ensuring deterministic execution of the subagent loop and model transitions under real inference loads.

### 5. Spark Orchestrator, Token Budgeting & VRAM Planning
- **Spark-X2.5-4B Orchestrator Support (`orchestrator-spark-registry.json`):** Integrated `Spark-X2.5-4B (Q8_0)` as an official orchestrator candidate with 32,768 served context tokens and FlashAttention acceleration.
- **Hardware-Aware VRAM Planner (`vram_planner.rs`):** Upgraded memory estimation engine that parses GGUF header geometry, context KV-cache requirements, and available VRAM to determine exact GPU layer offloading.
- **Fine-Grained Token Budgeting (`token_budget.rs`):** Dynamically allocates token quotas across prompt ingestion, tool grammar constraints, and completion limits.
- **Transparent Response Continuation (`continuation.rs`):** Multi-part generation handler that automatically detects output token boundary exhaustion and transparently continues generation without losing reasoning state.

### 6. Document Ingestion, Notebook Research & Evidence Citations
- **Deep Document & Attachment Extraction (`sidecars/document_sidecar/attachment_extract.py`):** Multi-format parser capable of extracting and indexing text, tables, and images from PDFs, Word documents, Excel spreadsheets, and PowerPoint presentations.
- **Evidence Citations & Notebook Scope (`EvidenceCitations.tsx`, `NotebookScopeControls.tsx`):** UI components that render verifiable citations linking model statements directly back to source document chunks.
- **Seamless Notebook-to-Chat Handoff (`notebookHandoff.ts`):** Smooth transfer of grounded research notes and context boundaries directly into active agent chat sessions.

### 7. Reliability, Platform Hardening & Verification Gates
- **Windows Console Popup Suppression (`src-tauri/src/serving/mod.rs`):** Applied `CREATE_NO_WINDOW` process flags when probing `llama-server.exe` on Windows to eliminate flickering background command prompt windows.
- **Synchronous Crash Logger & Standard Logging (`src-tauri/src/logging/mod.rs`):** Replaced custom telemetry early boot logging with standard log facilities and added a synchronous crash reporter to capture unhandled panics.
- **Ratcheted Lint Gate (`scripts/lint-budget.json`):** Lowered the unused-code ceiling from 43 to 40 warnings via the staged lint budget ratchet.
- **CycloneDX SBOM Regeneration (`evidence/sbom.cdx.json`, `evidence/sbom.md`):** Updated full software bill of materials cataloging 1,153 verified components.
- **Autonomous E2E CDP Test Suite (`ARJUN_SPARK_TEST_REPORT.md`):** Full end-to-end test run against the live desktop application via Chrome DevTools Protocol on Edge WebView2, validating reasoning, structured JSON decode (154 tok/s), and river crossing problem-solving.

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
  pages/                Workbench, Agents, Browse, Health, Approvals, AuditNetwork, …
  components/           ChatSurface, AgentMemoryPanel, MemoryGraphCanvas, RunView, …
  services/             typed bridges to Tauri commands (agentRegistry, memoryGraph, …)
  sdk/  hooks/  contexts/

src-tauri/              Rust core (crate: sarathi)
  agent_runtime/        supervises the TS runtime; workspace sandbox,
                        chat memory bus (1M tokens), dynamic context compiler,
                        context manifests, token pins, state commits, model handoff
    events/             durable, ordered task history in SQLite — snapshots
                        for the UI, idempotency keys for side effects, and
                        recovery of runs a restart interrupted
  agents/               agent definitions, persistent store, and governance
  subagents/            subagent worker loops, scheduling, and graph I/O
  knowledge/            graph runtime store, memory feeds, notebook retrieval
  ai_engine/            token budgeting, response continuation, VRAM planner
  serving/              model serving lifecycle, llama-server probing, and admission
  model_manager/        download, sizing, and installation
  model_intelligence/   hardware-aware recommendation
  model_package/        base-model package manifests and repair
  sovereignty/          the single egress broker
  policy/  capability/  audit/  identity/
  memory_engine/  documents/

agent-runtime/          vendored OpenClaw agent loop (TypeScript)
  src/                  protocol, run loop, context refresh, state commit, providers, tools
  vendor/openclaw/      pruned upstream packages
  scripts/              vendor audit and bundler

sidecars/               Python sidecars (documents, memory engine, packs, graph)
scripts/                build, verification, and evidence gates
evidence/               generated SBOM, test reports, and memory graph visual artifacts
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
7. The **shared agent memory bus** (`chat_memory_bus.rs`) and **dynamic context compiler**
   assemble active dialogue turns, token pins, and grounded knowledge passages into
   hardware-bounded manifests, supporting up to 1M tokens with seamless model transitions.
8. **Autonomous subagents** execute concurrently in supervised child loops (`subagents/`),
   streaming extracted knowledge, relations, and memory nodes directly into the reactive
   **memory graph runtime store**, visualizable live on the interactive canvas.

---

## Contributing

Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`,
`perf:`, `ci:`). Run `npm run verify` before opening a pull request; the
verification gates are the point of the project, and a change that trips one
needs a reason in review rather than a new exemption.

Third-party attributions live in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
ARJUN does not yet carry a license file.

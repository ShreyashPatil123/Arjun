# Agent system — execution ledger

The running record for [`2026-09-20-agent-system-build-plan.md`](2026-09-20-agent-system-build-plan.md),
prompts P00–P16. One ledger, updated at the end of each phase.

| Fact | Value |
|---|---|
| Plan baseline SHA | `4164fc26bbdc86eeed61fcf6513f0b97f2cc7e0e` (the plan's own header) |
| HEAD when P00 started | `b9ff7fbcbab3c610dffd541b674ec04706710561`, working tree clean |
| Branch | `main` |
| Host | Windows 11 Home Single Language 10.0.26200 |
| CPU / RAM | AMD Ryzen 7 250 (8C/16T) · 25 025 695 744 B (≈23.3 GiB) |
| GPU | NVIDIA GeForce RTX 5060 **Laptop** GPU, 8151 MiB, driver 595.95, CUDA 13.2, cc 12.0 |
| node / npm | v24.12.0 / 11.6.2 |
| cargo / rustc | 1.93.1 (083ac5135) / 1.93.1 (01f6ddf75) |
| Python | 3.11.9 |
| llama-server | 0.4.1-dev, build **10970**, commit `bfdc32183`, Clang 20.1.8 |
| App data identity | `com.arjun.workbench` (`src-tauri/tauri.conf.json`) |

> **This machine is not established to be the target machine.** The plan targets
> "RTX 5060, 8 GB VRAM". This host reports an RTX 5060 **Laptop** GPU — a
> different SKU with a different power envelope and memory bandwidth. Every
> figure below that was measured here says so; none of them is presented as a
> target-machine measurement. Only the hardware owner can close that gap.

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| P00 | Baseline, inventory, executable acceptance fixtures | **Complete** — see below |
| P01 | Agent definitions, jobs, tool/result contracts | **Complete** (portable code and deterministic tests; no native gate) — see below |
| P02 | Shared memory correctness and authority | **Complete for the graph, receipts, versions, outbox, sharing and the sixth legacy store**; cutover of the other five legacy writers and graph-backed agent recall remain open — see below |
| P03–P16 | — | Not started |

---

# P00 — Reproducible baseline and acceptance harness

## Implemented

| File | What it is |
|---|---|
| `scripts/qualification-inventory.mjs` | The machine-readable qualification inventory. Every value is `{value, source}` with `source ∈ measured \| declared \| unknown`; an unknown carries `unknown_because`. Parses GGUF headers for architecture, trained context and chat-template hash. `--hash` computes model sha-256s. |
| `scripts/fixture-manifest.mjs` | Builds and verifies `fixtures/agent-system/v1/manifest.json`. `--check` fails if a fixture changed without its hash being updated. |
| `scripts/agent-baseline.mjs` | The harness. Five-state report: executed / passed / failed / blocked / skipped. |
| `src-tauri/tests/agent_baseline.rs` | Four cases driven through the **production task driver** — `AgentRuntime::spawn` starting the real Node bundle over the real stdio protocol against real stores. |
| `fixtures/agent-system/v1/` | The fixture pack: 19 files, plus 3 reused in place. |
| `package.json` | Six entry points: `inventory`, `inventory:hash`, `fixtures:manifest`, `fixtures:check`, `baseline:agents`, `test:baseline:agents`. |
| `docs/plans/2026-09-20-agent-system-build-plan.md` | The plan itself, checked in so every `docs/plans/…` reference in P00–P16 resolves. |

Nothing existing was replaced. No registry row, model file or agent profile was
written. No model was downloaded.

## Contract decisions taken in this phase

1. **`{value, source}` everywhere in the inventory.** A bare `null` cannot be
   told apart from a field that was never asked about. The three sources are
   `measured` (a command ran here), `declared` (a file on this machine asserts
   it, unchecked) and `unknown` (with `unknown_because` naming what would
   establish it).
2. **Hashes are generated, never typed.** `fixture-manifest.mjs --check` is the
   mechanism; the same reasoning the repo already applies to the SBOM through
   `check:generated`.
3. **The grading key lives in the checkout, not the pack's source tree.** A run's
   workspace is `<app data>/runs/<run_id>` (`agent_runtime/workspace.rs:53`), so
   `fixtures/agent-system/v1/expected/` is unreachable from any run. The harness
   asserts this before grading rather than trusting the layout.
4. **Plain substrings, not regular expressions, in the fixture check files.**
   They are read by Node, Rust and Python; one escaping convention per language
   is three chances to get it wrong.
5. **A blocked case is never a pass, and a group that selected nothing is
   reported as failed.** Both are enforced in `agent-baseline.mjs`, and the
   hidden tests exit non-zero on zero selected cases.

## Revalidation of the nine §3 findings at HEAD `b9ff7fb`

Each was re-read in current source. Line numbers below are **current**, and
differ from the plan's where the file has moved.

| # | Finding | Classification | Current location | Note |
|---|---|---|---|---|
| 1 | Graph cursor accuracy | **Present** | `agent_runtime/context_compiler.rs:304` calls `self.graph.snapshot(…)`; `:471` records `graph_revision: scope.graph_revision`. `knowledge/graph/runtime_store.rs:946` is `snapshot_at`. | Unchanged. `snapshot_at` has exactly one production caller, `commands/memory_graph.rs:73`, and context compilation is not it. The manifest labels rows with a revision the read did not come from. |
| 2 | Real receipt provenance | **Present** | `subagents/worker.rs:391` publishes `event_seq: 0`; `knowledge/graph/runtime_memory.rs:507` admits only when `event_seq > 0`. | Unchanged (the plan cited `:388`/`:504`). The system is at least consistent: every worker finding is kept as a **proposal**, not admitted. The gap is that nothing carries the real event identity. |
| 3 | Semantic retrieval wiring | **Present, and narrowed** | `knowledge/embedding.rs:64` declares `LocalEmbedder`; re-exported at `knowledge/mod.rs:35`; **no production caller**. `context_compiler.rs:552` reports `mode: "lexical"`. | The code is honest — it refuses to call itself semantic. **New fact:** two embedding models *are* installed and registered on this machine (`Qwen3-Embedding-0.6B-Q8_0`, `nomic-embed-text-v2-moe.Q8_0`). So this is a code gap, not a weights gap. |
| 4 | Memory authority migration | **Present, and documented as such** | `knowledge/graph/migration.rs`, module header lines 44–48. | Its own words: *"What this is **not** is a cutover. Every legacy store is still written to directly by its own code."* Five legacy sources are covered and `uncovered` is computed on every run. The remaining work is routing writers, not writing the copier. |
| 5 | Executable new roles | **Present** | `subagents/worker.rs:143` lists five performable profiles; `subagents/profile.rs:162` `SchemaKind` has five variants (extraction/retrieval/calculation/review/code). | Unchanged. `agents/` holds five role files; orchestrator, document-author, presentation-author and spreadsheet-author have no file and no schema. |
| 6 | Visual identity | **Present** | `agents/mod.rs:82` — `AGENT_PALETTE: [&str; 8]`. | Unchanged. Nine roles, eight colours. `agents/store.rs:734` assigns `AGENT_PALETTE[held.len() % 8]`, so the ninth agent silently reuses the first's colour. |
| 7 | Model/projector discovery | **Present, with a concrete instance** | `registry/discovery.rs:151` and `registry/scan.rs:124` both require a filename starting `mmproj-`. | **New fact:** the installed Unlimited-OCR directory holds *two* matching files, `mmproj-Unlimited-OCR-F16.gguf` and `mmproj-ref-q8_0.gguf`. `discovery.rs` sorts by filename length and takes the shortest, which is `mmproj-ref-q8_0.gguf` — the wrong one. The OCR registry rows dodge this by pinning `projector` explicitly; auto-discovery of a new package does not. |
| 8 | Memory sharing policy wiring | **Present** | `agents/mod.rs:223` stores `shared_with_task`; `agents/mod.rs:502` (`to_profile`) carries `memory_scope` and **not** the flag; `subagents/profile.rs` `AgentProfile` has no field for it. `src/pages/Agents.tsx:671` renders the control. | Unchanged. The setting is visible, editable and reaches nothing. `agents/store.rs:750` hardcodes `shared_with_task: false` on import. |
| 9 | Registry versions reaching children | **Present** | `subagents/packet.rs:121` has `agent_id` and no `definition_version`. `subagents/manager.rs:209` snapshots profiles at construction; `subagents/worker.rs:169` holds instructions at construction. | Unchanged. A registry edit cannot reach the next child, because neither the manager's map nor the worker's string is re-read. |
| §13 | `store.rs` loses the role body | **Present** | `agents/store.rs:705` and `:741` both assign `agent.instructions = profile.description`. | Confirmed. `AgentProfile` carries a separate `instructions` field which both import paths ignore, so an imported agent is instructed by its one-line description. |

**Nine of nine present. None already fixed. None requires further evidence to
classify** — each was confirmed by reading current source, not by inference.

Two findings were *narrowed* by measurement rather than reclassified: #3 (the
weights exist, the wiring does not) and #7 (a real wrong-projector case exists
in the installed OCR directory).

## Qualification inventory

`evidence/agent-system/P00/qualification-inventory.json` — 15 registered models,
15 present on this machine, **15 with a sha-256** (3 declared by the registry,
12 computed here).

Architecture and trained context come from each file's own GGUF header, so they
are `measured`:

| Model | Arch (header) | Trained ctx | Quantization | sha-256 |
|---|---|---|---|---|
| Spark-X2.5-4B-Q8_0 | `spark2_5` | 1 048 576 | Q8_0 | measured |
| Qwen3.5-9B-Q4_K_S | `qwen35` | 262 144 | Q4_K_S | measured |
| gemma-4-E4B-it-qat-UD-Q4_K_XL | `gemma4` | 131 072 | registry says **`unknown`** | measured |
| gemma-4-12b-it-UD-Q4_K_XL | `gemma4` | 262 144 | registry says **`unknown`** | measured |
| NVIDIA-Nemotron3-Nano-4B-Q4_K_M | `nemotron_h` | 1 048 576 | **Q4_K_M** | measured |
| unlimited-ocr-q6-k / q4-k-m | `deepseek2-ocr` | 32 768 | Q6_K / Q4_K_M | declared by registry |
| Qwen3-Embedding-0.6B-Q8_0 | `qwen3` | 32 768 | Q8_0 | measured |
| nomic-embed-text-v2-moe.Q8_0 | `nomic-bert-moe` | 512 | Q8_0 | measured |
| Qwen3.6-35B-A3B-UD-Q4_K_M | `qwen35moe` | 262 144 | Q4_K_M | measured |
| gemma-3-12b-it, Qwen2.5-VL-3B, 2× mtp-gemma-4-12b, rebel-large | — | — | — | measured |

Questions the inventory settles that the plan left open:

- **The Nemotron variant question (§4) is answered.** It is
  `NVIDIA-Nemotron3-Nano-4B`, architecture `nemotron_h` — not the
  Llama-3.1-Nemotron variant. It is installed at **Q4_K_M**, not the Q8 the
  plan's "8-bit around 4B" policy calls for. Changing that is a decision, not a
  fix, and nothing here changed it.
- **Unlimited-OCR's architecture is `deepseek2-ocr`**, both tiers, sharing one
  chat template (`sha256 177c96be…`). The registry pins the **patched**
  projector, `mmproj-Unlimited-OCR-F16-patched.gguf`.
- **Two Gemma rows record `quantization: "unknown"`** for files whose names end
  `UD-Q4_K_XL`. The inventory records the registry's word and the filename's
  separately rather than reconciling them silently.
- **No registry row carries `minLlamaBuild`.** `registry/mod.rs:205` documents
  that Spark's `spark2_5` needs b10828 and the field is absent everywhere, so
  the launch gate cannot fire. The installed build is 10970, which satisfies it
  — by luck, not by check.
- **Container daemon: not reachable.** Docker CLI is installed at
  `C:\Program Files\Docker\Docker\resources\bin\docker.exe`; the daemon is not
  running, so its image list is `unknown`, not empty.
- **Renderers: none.** No `soffice`, no `libreoffice`. Python has PIL 12.3.0,
  PyMuPDF 1.28.0, python-docx 1.2.0, python-pptx 1.0.2, reportlab 5.0.0 —
  **openpyxl is absent**, which is what an independent workbook reopen needs.
- **A second, legacy app-data identity exists.** `com.sarathi.app` holds a
  partial mirror of the model tree and a stale `registry.json`. Several Rust
  tests still hard-code paths into it (`ai_engine/runtime.rs:1387`, `:1513`,
  `:1554`, `:1599`, `:1646`). Recorded, not touched.

## Measured on this machine (not on the target machine)

`evidence/agent-system/P00/served-context-observations.json`.

Real `llama-server` b10970 serving Spark-X2.5-4B Q8_0, driven by the production
task driver:

| Requested ctx | Observed served ctx | VRAM used at rest | Free | Load | Run outcome |
|---|---|---|---|---|---|
| 4096 | 4096 | 4813 MiB / 8151 | 3087 MiB | ~6 s | **failed** — 8852-token request exceeds 4096 |
| 16384 | 16384 | 5266 MiB / 8151 | 2634 MiB | ~4 s | **passed** — 12 characters generated |

Three findings came out of it:

- **P00-OBS-1 — the tool schemas cost 8584 tokens before anything else.** From
  the driver's own `context_ledger` event: `sections.toolSchema = 8584`. A
  one-word question costs 8852 input tokens. **A 4096-token served context
  cannot complete a single turn of this runtime, whatever the model.** Whether
  role-scoped tool loading (plan §4) reduces this is not measured.
- **P00-OBS-2 — the ledger reports `window: 0` and `headroom: 0`.** The runtime
  is not told the served window, so its budget accounting bounds nothing. Not in
  the plan's §3 list; it belongs to P03.
- **P00-OBS-3 — Spark Q8 at 16 k leaves 2634 MiB free** with the desktop
  resident. Consistent with the plan's one-heavy-slot policy. No second model,
  projector or OCR service was measured alongside it.

## Fixture pack

`fixtures/agent-system/v1/`, version 1.0.0, 19 files + 3 reused in place, every
one hashed in `manifest.json`. Nonconfidential throughout: every tag, reading,
name and certificate number is invented (PS-K).

| Input | What it is |
|---|---|
| 3 scanned pages | Rendered by `sources/scan/make-scans.py` — rotation, noise, bleed-through, blur. Deterministic (seed 26117): re-running gives byte-identical files. Page 3 carries a signature degraded past legibility **on purpose**. |
| SOP Revision C | Reused at `src-tauri/tests/fixtures/maintenance-sop.md`. Minimum allowable 9.0 mm. |
| SOP Revision D | `sources/sop/maintenance-sop-rev-d.md`. Reduces the minimum to 8.0 mm — and §3.1 suspends that reduction wherever pitting is recorded. |
| 6 calculation cases | Unit-aware: mm vs inch, percentage-of-what, 90 days vs three months, a corrosion rate over a non-integer interval, and one with no answer at all. |
| 3 briefs | Word approval note, four-slide deck, recalculating workbook. |
| Coding task | Two functions, graded by 11 hidden tests. |
| 8 failure cases | Each with a *correct* failure: blocked, partial, refused, or completed-with-the-conflict-surfaced. |

**The conflict is the point.** Pitting is recorded adjacent to point C, no
pitting assessment exists, so Revision D §3.1 is engaged and **9.0 mm governs**.
An agent that cites the newest revision answers 8.0 mm and concludes the vessel
is fine. Answering 9.0 mm *for the wrong reason* also fails: the number and the
reasoning are both graded.

**The grading key is held out and the holding-out is enforced.**
`expected/` sits in the checkout; a run's workspace is `<app data>/runs/<run_id>`;
`agent-baseline.mjs` asserts no `expected/` path resolves inside a workspace root
before grading anything.

**The coding fixture grades in both directions, and both were run.** The
reference implementation passes 11 of 11; the deliberately wrong one fails 8 of
11. A suite nobody has seen pass is not a grading key, and a suite that passes a
broken program grades nothing.

## Baseline result

`evidence/agent-system/P00/baseline.json`. Exact command:

```bash
ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 ARJUN_BASELINE_MODEL_ID=Spark-X2.5-4B-Q8_0 node scripts/agent-baseline.mjs
```

| | |
|---|---|
| cases | 39 |
| **executed** | **9** |
| passed | 9 |
| failed | 0 |
| blocked | 30 |
| skipped | 0 |

**Nine executed out of thirty-nine is the honest reading of this build**, not a
score. Thirty cases are blocked because the agents they grade do not exist yet;
each names the phase that delivers it.

The nine that ran:

| Case | Kind | Result |
|---|---|---|
| `heldout-01-key-outside-every-workspace` | precondition | passed |
| `fixture-01-manifest-current` | precondition | passed |
| `fixture-02-inputs-match-their-hashes` | precondition | passed — 17 inputs |
| `transport-01-health` | **deterministic-transport** | passed |
| `transport-02-unknown-method-refused` | **deterministic-transport** | passed |
| `driver-01-no-model-server-fails-honestly` | real-driver | passed |
| `model-01-real-generation` | **real-model** | passed |
| `coding-01-reference-passes-the-hidden-tests` | **deterministic-fixture** | passed — 11 selected |
| `coding-02-wrong-implementation-is-reported-as-failed` | **deterministic-fixture** | passed — 8 of 11 failed |

Two of the nine are marked `deterministic-transport` and two
`deterministic-fixture`, as §11.1 requires. The other five reach real
infrastructure.

`driver-01` is the one worth reading. `run.start` was pointed at a loopback
address nothing serves. The driver returned `outcome: {kind: "failed", detail:
"Connection error."}`, `stopReason: "error"`, empty text, with a real 7-event
trace. It did not invent an answer.

### A defect found in this harness, and fixed

`model-01-real-generation` first reported **passed** for a run whose own outcome
was `failed: request (8852 tokens) exceeds the available context size (4096)`.
The case checked only that `run.start` returned `Ok` — that the driver answered,
not that the run worked. That is precisely the fabricated pass the plan forbids:
a green case on top of a failed run, with a real trace underneath it.

Fixed at `src-tauri/tests/agent_baseline.rs` — the case now reads
`outcome.kind`, requires non-empty text, and reports `failed` with the reason
otherwise. Re-run at 4096 it correctly reports **failed**; re-run at 16384 it
reports **passed** with 12 characters generated. Both runs are recorded.

## Repository gates run

| Gate | Result |
|---|---|
| `node scripts/check-ipc.mjs` | pass — 172 commands |
| `node scripts/check-reachable.mjs` | pass — 196 modules |
| `node scripts/check-egress.mjs` | pass — one chokepoint, no unapproved hosts |
| `node scripts/check-no-lora.mjs` | pass — 588 product files |
| `cargo check --all-targets` | pass — no errors; pre-existing warnings only |

Raw output: `evidence/agent-system/P00/log_repo-gates.txt`.

## Unverified or blocked

| What | Why | Runnable command |
|---|---|---|
| Sandbox / code execution | The container daemon is not running | Start Docker Desktop, then `npm run baseline:agents` |
| Independent workbook reopen | `openpyxl` is not installed | `python -m pip install openpyxl` |
| Rendered page / slide review | No `soffice` on PATH | Install LibreOffice and put `soffice` on PATH |
| Served context for the other 14 models | Only Spark was served | `llama-server -m <weights> --port 8080 -c <n>`, then `npm run baseline:agents` with `ARJUN_BASELINE_MODEL_URL` set |
| Peak VRAM during generation | Only at-rest VRAM was sampled | Sample `nvidia-smi` during a generation |
| **Every figure on the target machine** | This machine is not established to be it | Confirm the target hardware, then re-run `npm run inventory:hash` and `npm run baseline:agents` there |
| OCR page accuracy | No OCR page was run | Blocked on P06 |
| 30 fixture cases | Their agents do not exist | Blocked on P02, P04, P06–P09, P11–P13 |

## PS 26117 evidence matrix after P00

IDs are this plan's tracking labels (§12.1), not official steps.

| ID | P00's contribution | Evidence |
|---|---|---|
| PS-A | No network path added. Egress gate re-run and clean; the model server runs on loopback; the inventory reaches nothing off-machine. | `log_repo-gates.txt` |
| PS-B | 15 open-weight models registered and inventoried with architecture, quantization and hash. Routing itself is untouched — P01/P03/P05. | `qualification-inventory.json` |
| PS-C | Not addressed. The multi-step agent loop is P01–P05. | — |
| PS-D | Sandbox availability measured and **blocked**: CLI present, daemon down. Honest refusal is a defined failure case (`fail-04`). | inventory `containers`; `failures/failure-cases.json` |
| PS-E | Scanned-page fixtures for the printed and degraded classes; the P&ID excerpt reused for the drawing class. Handwriting and photographs are **not covered** — a gap this pack does not close. | `manifest.json` |
| PS-F | Objective reopen-based checks defined for docx/pptx/xlsx before any authoring agent exists. All blocked on P09/P12/P13. | `expected/artifact-checks.json` |
| PS-G | Two-revision SOP corpus with a real conflict, plus missing-source, revoked-source and denied-source failure cases. | `sources/sop/`, `failures/` |
| PS-H | **Measured on this machine**: served context, VRAM at two context sizes, load time, and the 8584-token schema floor. Explicitly not a target-machine measurement. | `served-context-observations.json` |
| PS-I | Not addressed. End-to-end routing is P05–P10. | — |
| PS-J | Failure cases defined for all three demonstrations. None exercisable yet. | `failures/failure-cases.json` |
| PS-K | **Satisfied for the fixture pack.** Every fixture is synthetic or already in the repo; nothing came from a plant, vendor or customer; the manifest records provenance and licence. | `manifest.json` |

## P00 close-out

- **Implemented** — inventory collector, fixture-manifest generator, baseline
  harness, Rust driver target, 19-file fixture pack, 6 package entry points,
  the plan checked in, this ledger.
- **Wired** — all six entry points run from `package.json`; the Rust target
  compiles under `cargo check --all-targets` and runs under
  `cargo test --test agent_baseline`; the harness invokes the real production
  driver and the real hidden tests.
- **Tested** — 9 cases executed, 9 passed, 0 failed. Real-model generation
  confirmed through the production driver against Spark-X2.5-4B Q8_0 at a 16384
  served context. One harness defect found and fixed during the phase.
- **Unverified or blocked** — 30 fixture cases (agents not built), the sandbox
  (daemon down), renderers (absent), 14 models' served context (not served), and
  every target-machine figure (machine not established).
- **Remaining** — nothing in P00.

**Exact next step:** P01 — make agent definitions and tools executable
contracts. Begin at `src-tauri/src/subagents/packet.rs:121`, which carries
`agent_id` and no `definition_version`; that single omission is finding 9 and is
the dependency for proving an administrator edit reaches the next child.

---

# P01 — Agent definitions, jobs, tool and result contracts

Worked across two sessions on 2026-09-20 and 2026-09-24, on `main` at HEAD
`b9ff7fb`. Nothing is committed: the working tree carries P00 and P01 together.
The first session's P01 work was found uncommitted and not recorded here; it
was re-read, kept, tested and is recorded below with the second session's.

**Contract map:** [`2026-09-20-agent-system-contract-map.md`](2026-09-20-agent-system-contract-map.md)
— every tool's real routing path, the job and result contracts, and the known
deviations with their owning phase. Machine-readable: `agent-runtime/src/tool-contract.json`.

## Traced before changing

Registration: `lib.rs` loads `agents/*.md` → `SubagentManager` +
`AgentRegistry::import_bundled`. Dispatch: runtime `tool.execute` →
`agent_runtime::execute` → fallback → `runner_for` → `LocalToolRunner::delegate_to_subagent`
→ `SubagentManager::spawn` → worker. Result: `ChildResult` → `settle` →
rendered text for the parent. Tools: TS `catalogue.ts` → `tool.catalogue` →
`buildTools` → `tool.authorize` → `ToolGateway::decide` → `tool.execute`.
`orchestrator::executor` and `orchestrator::grammar` have **no production
caller**; the agent path is the only dispatcher.

## Implemented

| Area | Change | Files |
|---|---|---|
| Definitions at dispatch (finding 9) | `DefinitionSource::resolve` reads the registry once per dispatch; the copy is pinned on the packet; keys are `ag-` ids or bundled role keys, never display names; the worker is found by capability (`capability_for(output_schema)`); a disabled agent is refused and not replaced by its bundled profile. | `subagents/definitions.rs` (new), `manager.rs`, `lib.rs`, `runner.rs` |
| §13 import defect | Imports store the profile body, not its description; rows written wrongly by the old import are repaired once, with a version bump. | `agents/store.rs` |
| Job packet | Adds definition version/origin, capability, instructions + hash (hash only is serialised), skills, attempt id, model policy (with a computed `within_eligible`), sharing policy; `job_id()`; a field-by-field contract table on the type. | `subagents/packet.rs`, `definitions.rs`, `manager.rs`, `agent_runtime/mod.rs` |
| Result contract | `partial`, `blocked`; artifact versions, receipts, validation checks, `missing`; new lists enter the hash only when non-empty; `with_receipt` refuses a receipt with no event. | `subagents/result.rs` |
| New role schemas | `document`, `deck`, `workbook` registered; `capability_for` gives them no worker; dispatch and the test-run command refuse them as "registered and cannot yet be run"; the Agents screen labels them "(no worker yet)". | `subagents/profile.rs`, `definitions.rs`, `src/services/agentRegistry.service.ts`, `src/pages/Agents.tsx` |
| Tool contract | `orchestrator::contract`: one `ToolContract` per tool gathering the existing authorities and adding route, prerequisite (+ where checked), output kind and cancellation, all as exhaustive matches; published as JSON and checked from both languages. | `orchestrator/contract.rs` (new), `agent-runtime/src/tool-contract.json` (generated) |
| Gateway | Arguments must be an object; **undeclared arguments refused**; `List` kind; 12 tools' specs corrected to match the handlers and the model's schema (§3 of the contract map). | `orchestrator/gateway.rs`, `tools.rs`, `grammar.rs` |
| Catalogue | Offered prerequisites asked of the contract; the fallback refuses, by name, a tool the contract routes to the agent path. | `agent_runtime/mod.rs` |
| TS mirror | `artifact.list` / `artifact.read` defined and named (never offered before); delegation inputs offered; search `page` removed (nothing read it); `notebook.delete` side-effecting. | `agent-runtime/src/{catalogue,tool-names}.ts` |
| Writer delegation | `DelegationMode::{ReadOnly, Writer}`: read-only refuses writing roles and withholds write tools; writer never widens and takes the exclusive lane whenever it holds a write tool; no model-facing tool can request it. | `subagents/manager.rs` |
| Name dependence removed | The runner's empty-input rule (was the literal `"knowledge-retriever"`); the Agents screen's `workerAvailable`/`ready` (was profile name, else display name); `agent_test_run` (dispatched a clone by display name — never resolvable); `agent_dependents` (matched only the bundled name — registry children were invisible). | `orchestrator/runner.rs`, `commands/agents.rs`, `commands/agent_admin.rs` |
| Plausible-number fallback | An artifact reference with no revision was given revision 1. Now refused; `art-7@4` or `revision` accepted. | `orchestrator/runner.rs` |

No IPC command was added (172, unchanged). `AgentView`'s JSON shape is
unchanged; only its values are now correct for clones and for schemas with no worker.

## Contract decisions

1. **The contract gathers; it does not duplicate.** `spec_for`, `class_of`,
   `is_side_effecting`, `retry_policy_of` and the `ToolName` spellings stay the
   authority for their own facts. `ToolContract` reads them and adds the four
   facts that had no home.
2. **Generated, never typed — across languages.** `tool-contract.json` is
   rendered by Rust and checked byte-for-byte by a Rust test; the runtime's
   conformance test checks the TS catalogue against it (names, required and
   optional arguments, kinds, read-only, side-effecting, output kind, aliases).
   The hand-typed lists on the TS side had drifted exactly where it mattered.
3. **The gateway's schema is closed**, like the model's. Required and optional
   are the whole input schema. The grammar learned optional members rather than
   the specs pretending they were required.
4. **A capability is decided by output schema**, and is the one key tied to
   code. Everything else (id, display name, role key) may change or multiply.
5. **Registered is not executable.** A schema with no worker has a contract, is
   savable, and is refused at dispatch, at test-run and in readiness.
6. **Recorded is not enforced, and says so.** The packet carries the model
   policy and the skills; routing does not yet hold the eligible set and the
   child loop does not load skills. Both are named as such (P03, P05).
7. **Writer semantics exist before any writer tool.** `Writer` is a manager
   contract with a single constructor that no model-facing tool calls.

## Tests

All deterministic. Every test below drives the production `SubagentManager`
against a real `AgentRegistry` in a temporary directory, or the production
gateway, or the production `authorize` → `execute` path — no mocked handler.

| Property the plan names | Test |
|---|---|
| admin edits affect the next child | `subagents::definitions_tests::an_administrators_edit_reaches_the_next_child` |
| active jobs keep their definition | `…::a_running_child_keeps_the_definition_it_was_sent_under` (edit saved while the worker is held mid-run) |
| clone / rename | `…::a_clone_is_performed_by_its_capability_and_answers_to_its_own_id`, `…::renaming_an_agent_changes_nothing_about_how_it_is_dispatched`, `commands::agents::tests::the_worker_is_the_capability_of_the_output_and_not_a_name` |
| unsupported capabilities stay blocked | `…::a_schema_with_no_worker_is_registered_and_refused`, `…::a_disabled_agent_is_refused_and_not_replaced_by_its_bundled_profile`, `…::the_agents_screen_offers_exactly_the_schemas_a_worker_produces` |
| malformed tool arguments refused | `orchestrator::gateway::tests::{an_undeclared_argument_is_refused_and_the_real_ones_are_named, a_tool_with_no_arguments_refuses_any, a_payload_that_is_not_an_object_is_refused, a_list_argument_given_as_anything_else_is_refused}`, `agent_runtime::tests::tool_contract_path::an_undeclared_argument_is_refused_on_the_production_path`, `orchestrator::runner::tests::an_artifact_reference_without_a_revision_is_refused_not_defaulted` |
| child never exceeds parent | `…::an_edit_cannot_widen_a_child_beyond_its_parent`, `…::no_shipped_role_is_ever_granted_what_its_parent_does_not_hold` (every shipped role × five parent grants), `…::a_read_only_dispatch_withholds_a_write_tool_the_parent_does_hold`, `…::a_role_declared_as_writing_is_refused_by_a_read_only_dispatch`, `…::two_writers_never_share_the_reader_lane` |
| older saved records | `…::a_packet_recorded_before_definitions_were_pinned_still_reads` (strips every added field), `…::a_result_recorded_before_the_contract_grew_reads_and_keeps_its_hash`, `…::the_new_schemas_survive_a_round_trip_through_the_registry_file`, `…::a_row_the_earlier_import_wrote_wrongly_is_repaired_exactly_once` |
| canonical aliases | `orchestrator::gateway::tests::every_accepted_spelling_is_decided_as_the_tool_itself`, `orchestrator::contract::tests::every_listed_alias_resolves_to_its_own_tool_and_to_no_other`, runtime `catalogue.conformance.test.ts` "resolves every alias this runtime knows…" |
| the real path | `agent_runtime::tests::tool_contract_path::{a_delegation_the_schema_allows_reaches_the_delegation_handler, a_page_read_without_its_optional_end_page_is_not_refused_by_the_gateway}`, `orchestrator::runner::tests::the_runner_serves_exactly_what_the_contract_routes_to_it` |
| the job contract | `…::a_packet_carries_the_whole_job_contract`, `…::a_model_outside_the_eligible_set_is_recorded_as_outside_it`, `…::a_receipt_with_no_event_behind_it_is_refused` |

**Seen failing.** Three regression tests were run against the fix reverted —
`two_writers_never_share_the_reader_lane`, `an_undeclared_argument_is_refused_…`,
`a_tool_with_no_arguments_refuses_any` — and all three failed; restored, all
pass. A test nobody has seen fail grades nothing.

## Checks run

Raw output: `evidence/agent-system/P01/log_checks.txt`.

| Check | Result |
|---|---|
| `cargo test --lib` (full, before the grammar fix) | 2675 passed, **1 failed** — `grammar::a_tool_that_takes_arguments_can_be_sent_them`, which the optional-argument change broke. Fixed by teaching the grammar optional members. |
| `cargo test --lib -- orchestrator:: subagents:: commands::agents commands::agent_admin agent_runtime::tool_policy agent_runtime::tests` | 421 passed, 0 failed (every P01 test named above confirmed in the run) |
| `cargo test --lib -- agent_runtime::tests::tool_contract_path` | 3 passed |
| `cargo test --lib --no-fail-fast` (full, after every fix) | **2679 passed, 0 failed**, 2 ignored |
| `npm run check:targets` | pass |
| `npm run runtime:typecheck`, `npm run runtime:test` | pass; 131 files, 2291 tests |
| `npx tsc --noEmit -p tsconfig.json`, `npm run test:ui` | pass; 43 files, 618 tests |
| `npm run runtime:build`, `npm run check:bundle:self` | pass; bundle sha-256 `7eafd6f0…` |
| `check-ipc`, `check-reachable`, `check-egress`, `check-no-lora` | pass — 172 commands, 196 frontend modules, one chokepoint, 591 files |
| `node scripts/agent-baseline.mjs --out evidence/agent-system/P01/baseline.json` | 39 cases: 8 executed, 8 passed, 0 failed, 31 blocked. P00 executed 9: `model-01-real-generation` is **blocked** here because no model server was running — not a regression, and not a pass. |

## Measured

**P01-OBS-1 — the tool-schema floor rose by 455 estimated tokens**, from 8584
(P00) to **9039**, per the driver's own `context_ledger` in `driver-01`. That is
the cost of actually offering `artifact.list`, `artifact.read` and the
delegation inputs, net of removing search's `page`. P00-OBS-1 stands and is
slightly worse: a 4096-token served context cannot complete one turn. Role-scoped
tool loading (P03) is the remedy; nothing here measured it.

## Unverified or blocked

| What | Why | Runnable command |
|---|---|---|
| Real-model generation through the new catalogue | No model server was running this session | `llama-server -m <Spark-X2.5-4B-Q8_0.gguf> --port 8080 -c 16384`, then `ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 ARJUN_BASELINE_MODEL_ID=Spark-X2.5-4B-Q8_0 node scripts/agent-baseline.mjs` |
| A model choosing `artifact.list` / delegation unprompted | Needs a real model on the full catalogue | as above, with a prompt that refers to an earlier artifact |
| The installed desktop app | Not rebuilt or redeployed; the installed copy predates P01 (see the deploy memory: the app is a manual copy) | `npm run tauri build`, then replace the installed binary |

## P01 close-out

- **Implemented** — definitions resolved and pinned at dispatch; the full job
  packet; the extended result contract with partial/blocked; three writer role
  schemas registered and not executable; one tool contract, published and
  checked from both languages; a closed gateway schema; writer delegation
  semantics; name-keyed dispatch replaced by capability in five places.
- **Wired** — through the production path: runtime schema → `tool.catalogue` →
  `tool.authorize`/`ToolGateway` → `tool.execute` → route → handler; the manager
  used by `lib.rs` reads the registry the Agents screen writes.
- **Tested** — the table above; mutation-checked; full lib, runtime and UI
  suites green.
- **Unverified or blocked** — real-model behaviour on the enlarged catalogue;
  the installed app.
- **Remaining** — nothing in P01's scope. Deviations found and deliberately left
  to their owning phases are listed in §7 of the contract map.

**Exact next step:** P02 — begin at `subagents/worker.rs:391`, which publishes
`event_seq: 0`, and at `knowledge/graph/runtime_memory.rs:507`, which admits only
`event_seq > 0`; the receipt must name the child's own successful tool event.

---

# P02 — Shared memory, provenance and graph authority

Worked on 2026-09-24, on `main` at HEAD `b9ff7fb`, uncommitted, on top of P01.
Raw output: `evidence/agent-system/P02/log_checks.txt`.

## Found before changing anything

| # | Finding | Where |
|---|---|---|
| 1 | P00 finding 2, and wider: the four mechanical worker routines wrote **no tool event at all**; the model path pinned every claim to the *first* tool the child called, on the *parent's* run, at `event_seq: 0`. | `subagents/worker.rs` |
| 2 | `admit()` accepted any positive `event_seq` with a tool and run id beside it — a number anybody can write. | `knowledge/graph/runtime_memory.rs` |
| 3 | P00 finding 1: the context compiler read the **latest** rows with a plain `snapshot` and labelled them with a cursor frozen earlier. | `agent_runtime/context_compiler.rs` |
| 4 | Items were upserted in place with no version history; supersede, tombstone and source invalidation changed `status`/`body` **without moving the revision**, so an optimistic writer could not see the change. | `runtime_store.rs` |
| 5 | `correct()` was two transactions; a crash between them left a correction and the fact it corrected both current. | `runtime_store.rs` |
| 6 | The outbox table existed and **nothing ever drained it**. | `runtime_store.rs` |
| 7 | `migrate_all` has **no production caller**, and the store `memory_api` answers from (`<app data>/memory/*.json`) was **not one of its five sources** — the sixth store its own docs predicted. | `migration.rs`, `agent_runtime/memory.rs` |
| 8 | P00 finding 8: `shared_with_task` was carried by P01 and enforced nowhere; the registry import hard-coded `false`. | `agents/store.rs` |
| 9 | `child_loop` sent `"definitionVersion": 0` to the runtime — a plausible number for one nobody had. | `subagents/child_loop.rs` |

## Implemented

| Area | Change |
|---|---|
| Receipts resolved in storage | `knowledge/graph/receipts.rs` (new): `ReceiptLedger::verify` reads the event back and requires it to exist, be `tool_succeeded`, name the same tool (either spelling) and record the same output hash. `MemoryGraph` asks before admitting; with no ledger in reach **nothing** is admitted by receipt. `Provenance::ToolReceipt` carries `output_sha256`. |
| One receipt per finding | Each mechanical action records its own `tool_succeeded` event on the **child's** run (`record_tool_receipt`, idempotent per action). On the model path `remember_outcome` now returns the event it wrote, and `ToolCallRecord` carries `eventSeq`, `outputSha256`, `toolCallId` and, for retrievals, the chunk ids that call returned — so each passage is tied to the one search that returned it. `Claim.receipt` is per claim; `Work.tool` is gone. |
| Source / measured / inferred / supplied | `Basis` computed by the store from provenance and the tool's contract (`SourceText` for evidence tools, `Measured` for others, `Inferred`, `Supplied`, `Carried`); never chosen by the writer. |
| Versions and historical reads | `agent_memory_versions` (immutable, backfilled once), one `write_revision` path every change goes through (row + version + dependency index + feed entry, one transaction), `snapshot_as_of(cursor)` and `versions_of`. A cursor ahead of the head is refused. |
| Compiler cursor (finding 1) | Reads `snapshot_as_of(frozen cursor)` or the atomic `snapshot_at`, and records the cursor actually read. |
| Correction, staleness, revocation | `correct`/`correct_at` in one transaction with an optional expected revision; `Stale` status; staleness propagated transitively through pinned dependencies and `derivedFrom`/`cites` edges on correction, tombstone, source invalidation, migration re-sync and rollback; `revoke_reader` (per-reader, propagated to derived items, new revisions so the feed drops them). |
| Dependency revalidation | `MemoryItem.depends_on` pins; at commit each pin is re-read — moved, withdrawn or missing inputs publish the result **stale** with the reason; the calculation checker pins the shared items its expressions came from. |
| Derived restrictions | At commit a derived item takes the intersection of cleared roles, the input's project and owner (two different ones are refused), a non-`Internal` classification, and every revoked reader. |
| Per-record authority | `Authority::{Graph, Legacy{store}}`; migrated copies are `Legacy`, and a graph write or correction to one is refused until its store is cut over. |
| Outbox | `deliver_pending` + `record_delivery_failure` (attempts, last error); `TaskEventLog` is the `events` consumer, writing a new `memory_published` event whose id is derived from the outbox key — a redelivery is refused as a duplicate. Delivered at start-up and after every commit (`lib.rs`). Each worker publication commits its outbox row in the same transaction as the item. |
| Sharing | `MemoryScope::Scratch{task, agent}`; a worker publishes to the task only when its pinned definition shares (`packet.shared_with_task`), otherwise to its own scratch, which it reads and its siblings do not. Import now records `sharedWithTask: true`. |
| Migration | `LegacySource::RuntimeScopedMemory` (sixth store) with `durable_items_on_disk`; `upsert_migrated` (Inserted / Unchanged / Updated / Restored); orphan retirement; `rollback_source`; `verify_runtime_scoped_memory`; a run ledger (`agent_memory_migration_runs`); the sixth store is mirrored at every start-up. |
| Service | Tauri commands `memory_graph_neighbours` (≤ 200, inside the authorised set), `memory_graph_history`, `memory_graph_correct` (operator provenance from the session, at the revision read) and `memory_graph_revoke_reader` (administrators). IPC 172 → 176, all four `frontend-pending`. |

## Migration and rollback notes

- **Schema changes are additive only.** Three new tables, three new columns; no
  row is dropped or rewritten. A database from before P02 opens, gains the tables,
  and has each existing item's *current* body backfilled as its one known version
  — earlier revisions were never stored and are not invented.
- **Restartable.** Every migrated record is addressed by
  `migrated_item_id(store, legacy_id)`; a pass interrupted part-way is simply run
  again. The run ledger shows a pass with no `finished_at`.
- **Reversible without deletion.** `rollback_source(source)` tombstones each record
  migrated from that source as a new revision and marks what was derived from them
  stale. The legacy store was never written by the migration and remains the
  authority; running the migration again restores each record (`Restored`).
- **Verified before any cutover.** `verify_runtime_scoped_memory` reports records
  missing, differing, orphaned or rolled back; `clean()` is the precondition for
  retiring that legacy writer. **No legacy writer has been retired**: every
  migrated record is `Authority::Legacy`, and the graph refuses to write it.
- **A legacy change is followed, not overwritten**: a changed value is a new
  revision (`Updated`), a vanished record is tombstoned (`retired`) — only when
  every legacy file was read, since an unreadable file is not evidence of absence.
- **Existing registry rows keep `sharedWithTask: false`** where the old import
  wrote it. That value now takes effect: those agents publish to their own scratch
  and their results say so. An administrator turns sharing on per agent on the
  Agents screen; nothing flips it automatically, because a stored `false` may be a
  person's choice.

## Tests

| Property named by the plan | Test |
|---|---|
| A publishes a real observation; B reads its version | `subagents::worker_tests::a_publishes_real_observations_each_on_its_own_receipt_and_b_reads_that_version` (production delegation path; two findings, two distinct verified receipts on the child run; B reads the version at the revision it landed) and `…::a_retrieval_is_published_as_source_text_on_its_own_search` |
| conflicting updates do not overwrite silently | `knowledge::graph::p02_tests::a_write_against_a_revision_that_moved_is_a_named_conflict` |
| correction invalidates an artifact | `…::a_correction_makes_everything_derived_from_the_corrected_fact_stale` (transitive, versions kept, feed told), `…::a_result_whose_input_moved_during_the_work_is_published_stale` |
| denied users see no hidden data or metadata | `…::a_revoked_reader_sees_nothing_of_the_item_on_any_path` (snapshot, historical read, history, neighbours, count, feed), `…::a_derived_record_inherits_the_restrictions_of_its_inputs` |
| failed outbox delivery recovers once after restart | `…::a_failed_delivery_is_recovered_exactly_once_after_a_restart` (file-backed; reopened; redelivery absorbed), `subagents::worker_tests::a_publication_reaches_the_parent_run_through_the_outbox` |
| repeat migration does not duplicate rows | `…::the_migration_is_repeatable_verifiable_and_reversible`, `…::a_changed_or_removed_legacy_record_is_found_and_followed_by_the_next_pass` |
| invalid receipts remain proposals | `subagents::worker_tests::a_receipt_the_event_log_does_not_back_stays_a_proposal` (missing event, failed call, wrong tool), `…::only_a_receipt_resolved_in_the_event_log_admits` (tampered hash, no log), `knowledge::graph::receipts::tests::*` |
| cursor honesty (P00 finding 1) | `agent_runtime::context_compiler::tests::a_context_frozen_at_a_revision_reads_that_revision_and_not_the_latest`, `…::a_read_as_of_a_cursor_is_that_cursor_and_not_the_latest_rows` |
| scratch versus task | `subagents::worker_tests::a_definition_that_does_not_share_publishes_privately` |
| older records | `knowledge::graph::runtime_memory::tests::an_item_written_before_the_p02_fields_still_reads` |

Two existing tests encoded the defects and were changed to prove the fix
instead: the compiler test that froze at 412 on a graph whose head was 1, and
`production_acceptance` step 2, which asserted that a receipt naming events 12
and 19 — never written — was "corroborated". Its journey now records real
receipts and resolves them.

**Seen failing.** With the receipt refusal turned into a pass and staleness
propagation disabled, `a_receipt_the_event_log_does_not_back_stays_a_proposal`,
`only_a_receipt_resolved_in_the_event_log_admits` and
`a_correction_makes_everything_derived_from_the_corrected_fact_stale` all failed;
restored, all pass.

## Checks run

| Check | Result |
|---|---|
| `cargo test --lib --no-fail-fast` (final) | **2735 passed, 0 failed**, 3 ignored |
| `cargo test --test production_acceptance --test production_hardening` | 12 passed · 8 passed |
| `npm run check:targets` | pass |
| `node scripts/check-ipc.mjs` | pass — 176 commands (20 frontend-pending) |

Not re-run in P02 because nothing they cover changed: the runtime TypeScript
suite, the UI suite, `check-reachable`, `check-egress`, `check-no-lora`, the
baseline harness. P02 changed no TypeScript and no network path.

## Unverified, open or deliberately left

| What | State | Owner |
|---|---|---|
| Cutover of the other five legacy sources | `migrate_all` still has no production caller; their writers are live and authoritative. Only the sixth store is mirrored at start-up. | P14 |
| Agent recall through the graph | `memory.recall_authorized` still answers from the legacy store (now mirrored in the graph). Routing it through the graph is the cutover of that store. | P03/P14 |
| Coverage of partial results | Only `tool_succeeded` events establish anything, and each passage rests on the call that returned it; there is no finer per-claim coverage model inside one successful call. | P06/P07 |
| Retrieval caches on source revocation | The graph and the compiler read current authorisation every time; the run's passage table (`RunPassages`) is not invalidated when a source is withdrawn mid-run. | P07 |
| UI for correct / history / neighbours / revoke | Commands exist and are gated; no screen calls them. The reconnect path (`reset` → fresh atomic snapshot) was read, not changed, and has no new test. | P14 |
| Commit-to-visible latency | Not measured. | P14 |
| The installed app | Not rebuilt or redeployed. | — |

## P02 close-out

- **Implemented** — verified receipts per finding; basis; immutable versions and
  historical reads; one-transaction corrections; staleness, dependency
  revalidation, derived restrictions and per-reader revocation; per-record
  authority; a working outbox; scratch versus task sharing; a restartable,
  verifiable, reversible migration including the sixth store; four bounded service
  commands.
- **Wired** — `lib.rs` gives the graph the event log as its ledger, drains the
  outbox at start and after each commit, and mirrors the runtime's memory; the
  workers and the child loop publish through `TaskMemory` with real receipts; the
  compiler reads at its cursor.
- **Tested** — the table above; mutation-checked; full lib and the two changed
  integration suites green.
- **Remaining** — the five-source cutover and graph-backed agent recall (see
  above).

**Exact next step:** P03 — begin at `agent_runtime/mod.rs` `context.refresh`
(the `FrozenScope` built near line 814), which now reads at its frozen cursor;
the served window (P00-OBS-2, `window: 0`) and the 9 039-token tool-schema floor
(P01-OBS-1) are what the context budget has to be built against.

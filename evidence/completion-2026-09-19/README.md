# Completion and validation record — 19 September 2026

What was actually run on this machine, what passed, what failed, and — at the
end — what is still not done. Every figure here is copied from a command's
output; the raw logs are beside this file.

## Provenance

| Fact | Value |
|---|---|
| Commit at session start | `41ea9dd796a8c48da7225583dd9e9480c3cfc626`, 112 dirty paths |
| Commit at session end | `c04c231cd8cb180f255674d6f1c4646ffbe93650`, 11 dirty paths |
| Timestamp (UTC) | 2026-09-19T06:00:40Z → 2026-09-19T09:52:32Z |
| Host | Windows 11 Home Single Language 10.0.26200 |
| CPU / RAM | AMD Ryzen 7 250 (8C/16T) · 23 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8151 MiB, driver 595.95 (+ AMD Radeon 780M iGPU) |
| node / npm | v24.12.0 / 11.6.2 |
| cargo / rustc | 1.93.1 (083ac5135) / 1.93.1 (01f6ddf75) |
| Python | 3.11.9 |
| llama-server | 0.4.1-dev, **build 10970**, commit `bfdc32183` |

> **HEAD moved mid-session, and not by this work.** Two commits were made from
> outside this session while it was paused: `16af905` (142 files, +37 177/−476)
> and `c04c231` (a README update). They swept up four early fixes recorded below
> together with the ~100 files already uncommitted at session start. No
> `git commit` was run from this session.

### Model artifacts (two genuinely distinct local files)

| Model | Bytes | SHA-256 | Arch | Trained ctx |
|---|---|---|---|---|
| Spark-X2.5-4B Q8_0 | 4 375 021 152 | `5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2` | `spark2_5` | 1 048 576 |
| Qwen3-4B-Instruct-2507 Q6_K | 3 306 261 600 | `cd7b21b38b3e71400587c184b6a9b04d3beb4d13fdae6464d4075dee4f1bc5ad` | `qwen3` | 262 144 |
| Babelscape/rebel-large | 1 625 590 959 (weights) | `407feb8d55cae8ee077aa032a4aab5577a5503f910d090593626ebd6fccb6cff` | BART seq2seq | 1024 enc |

---

## 1. Graph-sidecar packaging — FIXED and proven in the installer

The gate's own words before the fix:

```
x  graph-sidecar is declared Bundled but no bundle.resources entry covers
   "sidecars/graph_sidecar/main.py". The installer will not contain it, and the
   app will fall back to the checkout — which exists only on a developer machine.
```

One line added to `bundle.resources` in `src-tauri/tauri.conf.json`.

**Three independent confirmations, in increasing strength:**

1. `npm run check:deployment` — all six bundled dependencies `ok`, exit 0.
2. A full `tauri build` ran. `npm run rehearse:offline-install` now reports
   `ok  Knowledge graph sidecar ships: _up_/sidecars/graph_sidecar/main.py`, and
   the staged `main.py` is byte-identical to the source
   (`31bb7f93bb4ad061a8bf5eafc00b5b92d2f41100d4188a8cf788b889297910c7`).
3. The **MSI itself** was searched: `graph_sidecar`, `main.py`, `rebel.py` and
   `router.py` all occur in `ARJUN_0.2.0_x64_en-US.msi` (17 188 472 bytes).
   NSIS installer also produced (11 293 833 bytes).

### Proof that no checkout fallback is required

`graph-sidecar-offline-probe.txt` is the verbatim output of running the sidecar
from a copy **staged outside the repository**, working directory also outside
it, repository absent from `PYTHONPATH` (`"repo_on_path": false`):

- `graph.ping` — **110 ms**, no model load.
- `graph.extract_triplets` — **20.86 s** including the 1.6 GB first load,
  returning one real triplet: `Acme Pumps Ltd —[headquarters location]→ Pune`.
- Exit code 0.

That triplet is the one `tests/test_decode.py` pins as `RAW_REPEATED`, which
corroborates that the committed fixture came from the real model.

### Dependency audit

| Requirement | Source | On this machine |
|---|---|---|
| `main.py`, `router.py`, `rebel.py` | repo, now in `bundle.resources` | yes, and in the MSI |
| `torch==2.14.0` (CPU) | vendored wheels in an air-gapped install | 2.14.0+cpu |
| `transformers==5.15.1` | same | 5.15.1 |
| `Babelscape/rebel-large` | model library, outside the repo | 1.6 GB, hash above |

**Still failing:** `rehearse:offline-install` overall, on its *other* half —
node, python and llama-server are not staged in a deployment pack. The script
says so plainly (`no --pack given`). Building one needs an operator to stage
those three runtimes deliberately; `scripts/build-offline-pack.mjs` fetches
nothing by design. **Blocked on a human decision, not on code.**

---

## 2. Spark pinned binary and manifests — verified against the bytes

| Claim in `orchestrator-spark-registry.json` | Verified |
|---|---|
| `sha256: 5c2c3c19…` | **matches** the file byte-for-byte |
| `weightsBytes: 4375021152` | **matches** `stat` |
| `minLlamaBuild: 10828` | local binary is **build 10970** — satisfied |
| architecture `spark2_5` | **matches** GGUF `general.architecture` |
| `contextLength: 1048576` | **matches** GGUF `spark2_5.context_length` |
| `supportsStructuredOutput: true` | chat template present, contains `tools` / `tool_call` |
| tokenizer | `gpt2` model, pre-tokenizer `spark2_5`, bos 0 / eos 1 |

Sliding-window attention is declared (`sliding_window: 512`, 3:1 pattern over 36
blocks). **Effective context** is decided by `vram_planner`, whose 30 tests pass
including the one pinning 65 536 for an 8 GB card against a 1 048 576 trained
window — the figure the registry cites for this exact GPU. That is the planner's
own test, **not** a measurement from a running llama-server in this session.

---

## 3. Five real defects found and fixed

### 3.1 Conversations silently disappearing
`ConversationStore::list` dropped unparseable files with `.ok().flatten()` — no
error, no log, no count. **Two real files on this machine** (`61f02186-…`,
`f2f46622-…`) are valid JSON followed by the tail of a longer earlier version:
write-without-truncate damage from a build predating the current `save`. The
writer was already fixed; the reader made the loss invisible.

Now logged, and returned by a new
`list_with_diagnostics(owner) -> (Vec<Conversation>, Vec<PathBuf>)`. Two
regression tests reproduce the exact corruption shape.

### 3.2 A live administrator password in git history
`src-tauri/tests/seed_local_accounts.rs` held `const PASSWORD = "Shreyash@123"`
and echoed it to stderr. Replaced with `seed_password()`, which reads
`ARJUN_SEED_PASSWORD` and **panics when unset** rather than defaulting.

> **This does not remove it from history.** Rotating that credential and
> deciding whether to rewrite history are the owner's calls.

### 3.3 A green gate that asserted nothing
`sidecars/graph_sidecar/tests/test_decode.py` held 8 good pytest-style tests.
`unittest discover` collected **zero** and exited 0, and no npm script pointed
at it. Converted to `unittest.TestCase` (stdlib, so an air-gapped install can
run its own suite) and wired in: `test:sidecar` now runs **133 tests**.

### 3.4 A gate that could ratchet itself into permanent failure
`check-lint-budget.mjs` writes its count back into tracked
`scripts/lint-budget.json` whenever it falls. The count is not a property of the
code — measured on this tree, same source, consecutive runs:

| Run | Cargo state | Count |
|---|---|---|
| 1 | fully cached | **0** |
| 2 | after `touch src-tauri/src/lib.rs` | **40** |
| 3 | cached again | **40** |

A `0` would have been written as the new ceiling, after which no change to the
code could ever satisfy the gate. This is also why an earlier draft recorded
"44 warnings, ceiling 43" as a standing failure attributable to uncommitted
work — **that attribution was wrong**; the numbers were readings of the cache.

Fixed: a run that re-checked nothing **and** saw nothing now fails loudly, and
the ceiling is lowered only on a run that actually re-checked the crate.

### 3.5 Every software version reported empty, and Python reported missing
Found by running the real app. `run_command_with_timeout` spawned children with
**inherited** stdio, so `wait_with_output()` returned empty buffers every time
and the child's text leaked to ARJUN's console instead.

Direct evidence from the app's own log, before the fix:

```
✓ Found Rust: version="", path="…rustc.exe"
✓ Found Node.js: version="", path="…node.exe"
ℹ️ Python not found on PATH
✓ Software Detection Summary: Python=false, …
```

Every tool's version was empty because the parsed buffer was always empty. For
Python it was fatal: the Windows Store alias answers first, exits non-zero, and
the condition that would have rescued it (`!stdout.is_empty()`) could never be
true. **The app believed Python was absent on a machine with three copies — and
Python is what both sidecars need.**

Fixed in two places: stdio is now piped, and `check_executable` requires exit
success (with output captured, accepting "wrote something to stderr" would newly
report the Store alias's *refusal* as an installed interpreter).

Verified by rebuilding and re-running the real app:

| | before | after |
|---|---|---|
| `Software Detection Summary` | `Python=false` | **`Python=true`** |
| versions | all `""` | `Python 3.11.9`, `rustc 1.93.1 (01f6ddf75 2026-02-11)`, `git version 2.52.0.windows.1`, `v24.12.0`, `11.6.2` |
| child output leaked to console | 5 lines | **0** |

Two regression tests pin stdout and stderr capture.

---

## 4. The single memory authority — migration built

`Provenance::Migrated` had been in the data model since runtime memory was
written and **nothing in the tree ever constructed it**. `runtime_store.rs`'s own
docs named the condition: *"Two other authorities — the conversation store and
the agent registry — are JSON files and are not in that database at all."*

**Built:** `src-tauri/src/knowledge/graph/migration.rs`, covering **all five**
legacy stores.

| Source | Status |
|---|---|
| `agents/registry.json` | migrated |
| `conversations/pinnedContext` | migrated |
| `knowledge/assertions` | migrated, with source hash + extraction revision |
| `memory_engine/scoped` | migrated |
| `artifacts/references` | migrated, one item per **version** |

Design points:

- **Stable ids.** `migrated_item_id(store, legacy_id)` is sha-256 derived — the
  *opposite* of `runtime_memory::item_id`'s deliberate randomness, and not a
  contradiction of it: two agents asserting one fact are two observations; one
  legacy record read twice is one record arriving twice.
- **Interrupt and retry needs no cursor.** The derived id is also the
  `idempotency_key`, so recognition happens inside the store's transaction
  rather than in a check that would race a concurrent pass. No resume state to
  lose, so nothing to roll back.
- **ACLs do not widen in transit.** An agent's `classification_ceiling` becomes
  the item's classification and ACL; a pin stays user-scoped with `owner` set.
- **Unreadable records are named, never dropped** — the two damaged
  conversations above would otherwise have been migrated past in silence.

**Rehearsal:** `migration_tests.rs`, **9 tests**, each in its own `TempDir` over
the real `AgentRegistry` and real `ConversationStore`. Covers id determinism,
separator collision-resistance, a second run writing nothing, an interrupted run
finished by re-running it, provenance round-tripping to the id,
classification/ACL survival, owner confinement, a damaged file being named, and
that `migrate_all` attempts every declared source.

> **This is not a cutover.** Every legacy store is still written to directly by
> its own code. This copies from them under one authority; routing the writers
> through it is a separate change. Until then the honest position is "two
> stores, one of them derived" — stated rather than left to infer.

---

## 5. Production acceptance journey — built, 8 steps

`src-tauri/tests/production_acceptance.rs`, **12 tests**, over the real
`MemoryGraph`, `AgentRegistry` and `TaskEventLog`.

| Step | Asserted |
|---|---|
| 1 | A and B created with distinct palette colours, sharing one task scope |
| 2 | Fact (tool receipt → admitted), operator correction (admitted), model guess (**stays a proposal**), versioned artifact |
| 3 | B reads the fact and the artifact at exact revision + hash; unauthorised readers see nothing; retrieval trace recorded |
| 4 | B rebound Spark→Qwen, window cut to 4 096, **3 compactions + 1 trim** all survive into the replayed trace |
| 5 | Interrupt after a **real completed tool**, recover the worker, restart, resume the **original assistant message** |
| 6 | Pending approval; plan hash; **one external write happens exactly once** — the second attempt under the same key is replayed, not re-run |
| 7 | Conflicting update refused with both revisions named; duplicate key returns the *original* item; out-of-order cursor invents nothing; source invalidation |
| 8 | The **real** verifier: a citation pointing at nothing is refused, a groundless organisation-record answer is refused, a correct citation passes; terminal result persisted and survives rebuild |

**Three of my first assertions were wrong, and the code was right.** They are
documented in the tests rather than quietly deleted:

- Every `Classification` clears the same two roles under the 2-role model, so
  classification confers no differential clearance. Project and owner are what
  separate readers. Asserting otherwise would have pinned a guarantee the
  product does not make.
- `invalidate_source` sets `Rejected`, which is **deliberately still readable** —
  *"kept, because deleting it invites the same wrong thing to be proposed
  again"*. That is invalidation, not access revocation. **Per-reader source
  access revocation does not exist**, and the test says so in a table rather
  than asserting a guarantee that is absent.
- The verifier checks **citation integrity and grounding**, not semantic
  entailment. A short uncited claim beside retrieved passages is not something
  it can refuse without refusing ordinary prose.

### Hardening fixtures — `production_hardening.rs`, 8 tests

Large history (601 items, past `MAX_BATCH` 500) pages with no loss and no
repeat; `usize::MAX` is clamped and `0` is clamped upward; lexical and semantic
retrieval reach the same chunk and stay **labelled differently**; a vector
search with no vectors returns nothing rather than falling back to keyword; a
prompt injection in stored content cannot promote itself, reclassify itself,
widen its ACL or cross a project; two projects and two owners are isolated **in
both directions**.

---

## 6. Gate results

All run on this machine at `c04c231` with the working tree as it stands.

| Gate | Result |
|---|---|
| `check:egress` | PASS |
| `check:no-lora` | PASS |
| `check:deployment` | PASS — **was failing before this work** |
| `check:ipc` | PASS — 172 commands |
| `check:reachable` | PASS |
| `check:whitespace` | PASS |
| `check:bundle:self` / `check:bundle` | PASS |
| `check:offline` | PASS |
| `check:targets` | PASS |
| `check:lint-budget` | PASS — 40/40, stable across cache states after §3.4 |
| `check:generated` | PASS — SBOM matches HEAD, no drift |
| `runtime:audit` / `runtime:typecheck` / `runtime:build` | PASS |
| `runtime:test` | PASS — **2188** tests / 131 files |
| `test:ui` | PASS — **618** tests / 43 files |
| `test:rust` | PASS — **2599** passed, 0 failed, 2 ignored |
| `test:baseline` | PASS |
| `test:integration` | PASS — 11 suites, **63** passed, 1 correctly ignored |
| `test:sidecar` | PASS — **133** tests |

| Gate | Result | Why |
|---|---|---|
| `rehearse:offline-install` | **FAIL** | Bundled half fully passes (all six resources ship, graph sidecar included). Fails on node/python/llama-server because no deployment pack exists; building one requires an operator to stage those runtimes. |
| `test:rust:models` | not run | Live-model suite. |

### One latent risk

`agent-runtime`'s test script is `vitest run --passWithNoTests`. It found 2188
tests, so the pass is real — but a config break collecting zero would still
report success. Same shape as §3.3.

---

## 7. Native UI journey — partial, and blocked on a password

The **real** application was built and run, not a dev server:

- `tauri build` produced `sarathi.exe`, an MSI and an NSIS installer.
- The app was launched with `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port`,
  and CDP attached to the genuine Tauri WebView2 page at `http://tauri.localhost/`.
- Pre-auth state captured: title `ARJUN`, heading `Sign in`, six accounts
  listed, one `input[type=password]`, 30 924-byte screenshot.
- **Zero network requests** were observed during startup. Recorded precisely:
  that is an observation of the pre-auth startup window, **not** a general
  zero-egress proof.
- §3.5 was found here and fixed here — the only defect in this record that could
  only have been found by running the application.

**The authenticated journey was not done.** The app is gated by a password
field, and entering passwords is not something I will do. Everything from
"admin creates agents A and B" onward needs the operator signed in. The
acceptance journey in §5 asserts those same behaviours against the real stores,
which is genuine evidence — but it is not the same claim as "an independent
observer watched it happen on screen", and it is not offered as one.

---

## 8. Verdicts

| Class | Verdict |
|---|---|
| Fixture / unit | **PASS** — 2599 Rust lib, 2188 runtime, 618 UI, 133 sidecar |
| Migration rehearsal | **PASS** — all 5 sources, 9 tests, isolated copies. **No cutover.** |
| Acceptance journey | **PASS** — 12 tests over real stores, all 8 steps |
| Hardening fixtures | **PASS** — 8 tests |
| Real-model | **PARTIAL** — REBEL ran end-to-end for real; Spark and Qwen verified as artifacts (hash, arch, template, context) but **not served** in this session |
| Native UI | **PARTIAL** — real app built, launched, driven by CDP, defect found and fixed; authenticated journey **blocked on sign-in** |
| Installed / offline | **PARTIAL** — sidecar proven in the MSI and proven to run outside the checkout; external-runtime pack **not built** |

### Still not done

1. **Cutover.** The legacy writers still write to the legacy stores.
2. **Authenticated native UI journey** — needs an operator to sign in.
3. **Offline deployment pack** — needs an operator to stage node, python and
   llama-server deliberately.
4. **Per-reader source access revocation** — does not exist (§5); `Rejected`
   items stay readable by design.
5. **Serving either GGUF** — neither Spark nor Qwen was loaded into
   llama-server in this session, so no tokens-per-second, recall or reliability
   figure appears anywhere in this file.

No speed, recall, reliability, graph non-overlap or zero-egress number here is
estimated. Every figure is copied from a command's output.

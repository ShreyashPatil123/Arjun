# Phase 0 — Checked evidence map and baseline results

**Date:** 17 September 2026
**Branch:** `main`
**HEAD:** `41ea9dd796a8c48da7225583dd9e9480c3cfc626`
**Upstream:** `shreyash/main`, parity `0 ahead / 0 behind`
**Machine:** Windows 11, RTX 5060 Laptop (8 GB), CUDA 13.1, node v24.12.0, cargo 1.93.1

Companion documents: [contracts](2026-09-17-shared-agent-memory-contracts.md),
[ledger](2026-09-17-shared-agent-memory-ledger.md).

This is a snapshot of what exists, not a plan. It is deliberately separate from
[`docs/implementation-baseline.md`](../implementation-baseline.md), which is an
older snapshot at `49520ef` (27 August 2026) covering different work; that file
is left intact because later documents measure against it.

---

## 0. Provenance of this audit, and one document that does not exist here

The task names "the two 2026-09-16 shared-agent-memory planning documents". Only
**one** is present on this machine:

| Document | Status |
|---|---|
| `C:\Users\lenovo\Downloads\2026-09-16-shared-agent-memory-prompts.md` | Present, 34,134 bytes, read in full |
| `…/ARJUN/docs/plans/2026-09-16-shared-agent-memory-audit.md` | **Absent.** Referenced by the prompts document at a path under `C:\Users\parag\Downloads\New folder\ARJUN` — a different user profile. A full-profile search for `*2026-09-16*` and `*shared-agent-memory*` finds no copy here. |

Consequence, and it is the reason this document is shaped the way it is: the
audit's *findings* reach this session only as claims quoted inside the prompts
document. Every one of them is treated below as a **hypothesis** and carries its
own verification against the working tree. Nothing is recorded as fact on the
strength of the prompts document alone.

The prompts document also states its baseline as `f02ed8d0…`. This tree is one
commit further on, at `41ea9dd7…` ("token budget, response continuation, spark
orchestrator registry, VRAM planning"), so Phase 3 findings in particular are
re-checked against the newer commit rather than inherited.

Historical `origin/Context-memory` branch material (`docs/agent-runtime-architecture.md`,
`docs/context-memory-status-2026-09-05.md`, `agent-runtime/vendor/openclaw/…`,
`vendor/letta`, `vendor/mem0`, `vendor/zep`) exists on that branch and **is not**
on `main`. It has not been used as evidence for anything below.

---

## 1. Working tree, preserved as found

```
 M src/components/layout/AppMenu.tsx
?? @AutomationLog.txt
?? ARJUN_SPARK_TEST_REPORT.md
```

- **`src/components/layout/AppMenu.tsx`** — 29 insertions / 2 deletions of local,
  uncommitted work making the "New conversation" menu item actually mint a
  conversation (`useConversation`, a `creatingRef` double-click lock, an
  `isStreaming` guard). Unrelated to this feature and **left untouched**.
  Flagged because Phase 10 edits this same file to add `Administration → Agents`;
  that phase must build on this change rather than over it.
- **`ARJUN_SPARK_TEST_REPORT.md`** — untracked output of a prior automated CDP
  session claiming 26 tests / 53.8% pass. **Not treated as evidence.** It was not
  produced or reproduced here, it is not in version control, and its numbers are
  not reconciled against any gate in this repository. Its individual observations
  are used only as *hypotheses*, and two of them are independently confirmed
  below (§4.7 routing, §4.4 budget pressure) by reading code, not by trusting it.
- **`@AutomationLog.txt`** — untracked scratch output. Ignored.

Worktrees: `main` here, plus `.claude/worktrees/sweet-sammet-1d7186` at `fcb8853`.
No stashes. Nothing was reset, merged or overwritten.

---

## 2. Baseline check results

Run at this SHA with the working tree as found. **One failure, pre-existing.**

| Check | Result | Evidence |
|---|---|---|
| `npm run check:ipc` | **pass** | 154 commands (10 admin, 3 external, 129 frontend, 12 frontend-pending) |
| `npm run check:egress` | **pass** | one chokepoint, no unapproved hosts |
| `npm run check:deployment` | **FAIL (pre-existing)** | 1 problem — see §2.1 |
| `npm run runtime:typecheck` | **pass** | `tsc --noEmit` clean |
| `npm run runtime:audit` | **pass** | 287 source files, 10 manifests, provider exclusion holds |
| `npm run runtime:test` | **pass** | 131 files, 2,188 tests |
| `npm run test:ui` | **pass** | 39 files, 518 tests |
| `npm run test:rust` (lib) | **pass** | 2,293 passed, 0 failed, 2 ignored |
| `npm run test:integration` | **pass** | 9 suites, 43 passed, 0 failed, 1 ignored — **but see §2.2** |

Ignored tests, named so a later run cannot mistake them for coverage:

- `ai_engine::runtime::tests::gguf_metadata_matches_the_model`
- `artifacts::audit_emit_tests::emit_one_of_each` (needs `ARJUN_AUDIT_OUT`)
- `seed_local_accounts_…_except_modeladmin` (writes to the live credential store)

### 2.1 The one pre-existing failure

```
x  graph-sidecar is declared Bundled but no bundle.resources entry covers
   "sidecars/graph_sidecar/main.py".
```

Confirmed directly: `src-tauri/tauri.conf.json` `bundle.resources` lists
`agent-runtime/dist`, `skills/`, `agents/`, `sidecars/document_sidecar/`,
`sidecars/memory_engine_sidecar/`, `sidecars/voice_sidecar/` — and **not**
`sidecars/graph_sidecar/`, which exists on disk with `main.py`, `rebel.py`,
`router.py`, `requirements.txt`. An installed build falls back to a checkout that
only exists on a developer machine. This matches the failure the prompts document
predicted. It is Phase 11 work and is not to be papered over earlier.

### 2.2 A contract broken while establishing the baseline — reported, not buried

The shared contract says *"Never run the legacy memory diagnostic against the
user's AppData."* `npm run test:integration` is one of the checks Phase 0 is told
to run, and it includes `src-tauri/tests/test_memory_pipeline.rs`, which does
exactly that:

```rust
// src-tauri/tests/test_memory_pipeline.rs:13
fn make_app_data_dir() -> PathBuf {
    let appdata = std::env::var("APPDATA")
        .unwrap_or_else(|_| r"C:\Users\lenovo\AppData\Roaming".to_string());
    PathBuf::from(appdata).join("com.sarathi.app")
}
```

**What happened:** the run at 09:49:53 IST modified
`C:\Users\lenovo\AppData\Roaming\com.sarathi.app\sarathi.db`. Inspecting it
afterwards: one row appended to `memory_nodes`
(`mem_bbc53851-…`, content `"User's name is Shreyash Patil"`, `created_at`
`2026-09-17T04:19:52Z`) and `user_profile.name` re-stamped with the same instant.

**Scope, stated precisely rather than minimised:** the directory is
`com.sarathi.app`. The application's current Tauri identifier is
`com.arjun.workbench` (`src-tauri/tauri.conf.json:5`), and the live data lives in
`…\Roaming\com.arjun.workbench`. So this is a *stale* profile that the shipping
application no longer opens, and the written value is the same fixture text this
test appended on 15 September and earlier. No live data was touched. It is still
a write to the user's AppData that the contract forbids, and it will recur on
every `test:integration` run until the test is fixed.

**Remediation is the first item in the ledger** (L0.1), ahead of all feature
work: `test_memory_pipeline.rs` must take a `tempfile::TempDir`, and
`test:integration` must not be run again on this machine until it does.

---

## 3. Production flow, traced

`agent_start_run` → `drive_run` (`src-tauri/src/commands/agent.rs:1468`, ~2,480 lines)
is the single production entry for a chat turn. `agent_resume_run:5637` re-enters
the same function with `existing_run_id: Some(..)`.

Order of operations inside `drive_run`, each step read rather than assumed:

1. `require_permission(UseModel)`, then `AuditHealth::refusal()` — a run that
   cannot be recorded is refused before anything is touched.
2. `RunCancellations::register` on correlation-id + existing-run-id, guarded by
   `CancelGuard`, so Stop works before weights load.
3. `StageReporter` (`Accepted`).
4. Attachments → `commands::ocr::read_attachment` per file → `PreparedDocument::of`
   → structure-aware chunking.
5. `run_id = existing_run_id.unwrap_or(uuid)` (`:2017`).
6. `resolve_turn_identity` (`:2085`, defined `:467`) settles conversation + cell.
7. Routing → model choice → `served_window` (the window the *server* was started
   with, not `entry.context_length`).
8. `CheckpointSeed` written into `RunCheckpoints` (`:2437`).
9. Notebook research, when `request.research` is `Some`: `notebook_retrieval::resolve`
   (owner-checked) → `retrieve` → `retrieval::record` → `notebook_retrieval::manifest`
   → `record_research_turn`. Manifest is written **before** the model runs.
10. `compose_system_prompt` folds workspace / plan / notebooks / documents /
    skills / research notes.
11. History: `turn_context::fit_with_memory_bus` (`:2946`) against
    `budget_for(served_window, prompt + system + 4096 reply + tool_floor)`.
12. Node loop over JSON-RPC; tool calls return through the Rust gateway;
    `recording::remember_outcome` writes `ToolSucceeded`/`ToolFailed` and takes a
    checkpoint (`recording.rs:264`).
13. Continuation chain reads `outcome.notes` when the output cap is hit (`:3283`).
14. Finalisation reads `outcome.notes` → `working_notes` and `outcome.ledger` →
    `context_ledger` onto the task record (`:3410`).

Events: `record_and_publish` writes durably **then** emits `AGENT_DURABLE_EVENT`
with a sequence number (`:856`). Commit-before-publish already holds.

---

## 4. Hypotheses from the prompts document, each checked

Legend: **CONFIRMED** (found in a production path), **CONFIRMED+** (true and worse
than stated), **PARTLY** (true with an important qualification).

### 4.1 "Production checkpoints contain `RunMemory::default()`" — **CONFIRMED**

`src-tauri/src/agent_runtime/recording.rs:264-268`, in `remember_outcome`, which
runs after **every** tool result in production:

```rust
deps.checkpoint_or_note(
    &call.run_id,
    events::RunState::ToolResultRecorded,
    crate::agent_runtime::memory::RunMemory::default(),   // empty
);
```

The only non-test writer of real notes is the finalisation block, and it runs **at
run end**. Working notes exist in Node (`agent-runtime/src/working-notes.ts`), are
returned in `RunResult.notes`, and are read by Rust in exactly two places —
`agent.rs:3288` (continuation) and `agent.rs:3413` (finalisation). There is no
`run.note`-driven durable commit. A crash mid-run therefore leaves a checkpoint
whose notes are empty, and `notes_to_resume_from` (`:5880`) filters empty notes
out of both candidates, so such a resumption starts from nothing.

### 4.2 Attempt identity — **CONFIRMED+** (three minters, not two)

On a resumption, `agent_resume_run`:

1. `assess_resumability` returns the checkpoint's `attempt_id` — and it is
   **immediately discarded**: `let _ = attempt_id;` (`agent.rs:5683`).
2. `Attempt::new` mints a fresh UUID (`resume.rs:184`), which is what the durable
   `RunResumed` event and the audit line carry.
3. `drive_run` then mints a **third** UUID for the `CheckpointSeed`
   (`agent.rs:2437`), and that is the `attempt_id` stamped on every checkpoint
   the resumed attempt writes.

So the `RunResumed` event's `attemptId` can never equal the `attempt_id` on the
checkpoints of the attempt it announces. Leases are keyed separately again, by
`worker_id()` (`agent.rs:104`). The `checkpoints` table does persist `attempt_id`
(`events/store.rs:200`), so the storage is ready; only the minting is wrong.

### 4.3 "Resume enters the new-conversation branch" — **CONFIRMED+**

`agent_resume_run` builds its `StartRunRequest` with
`conversation_id: None, message_id: None` (`agent.rs:5742-5745`), with a comment
arguing a resumption is not a conversation turn. `resolve_turn_identity` then
takes its third branch and **creates a brand-new conversation** titled from the
first line of the original prompt, appending a user turn and an assistant cell.

Worse than the prompts document states: the reserved cell id is
`format!("a-{run_id}")` and `run_id` **is** preserved across resume, so the new
conversation receives a cell carrying the *same* message id as the original
conversation's cell. Two conversations, one message id, and the original
assistant cell is orphaned with no answer.

`research: None` on the same request means the frozen notebook scope and evidence
manifest are also not rehydrated (the code comments acknowledge this and call the
original manifest authoritative).

### 4.4 Chat-memory-bus regressions — **CONFIRMED+**

The production path is `fit_with_memory_bus` (`turn_context.rs:386`), reached
only from `agent.rs:2946`. The older `fit` (`turn_context.rs:260`) has **no
production caller at all**.

| Behaviour | `fit` (tested, dead) | `fit_with_memory_bus` (production) |
|---|---|---|
| `tool_summary` appended | yes, via `with_tool_summary` | **no** — `chat_memory_bus` uses `msg.content.trim()` |
| `[E#]` neutralised | yes, before costing | yes, but **after** the bus has budgeted |
| Pin by message id | case-**in**sensitive (`eq_ignore_ascii_case`) | exact `p == &msg.id`, **case-sensitive** |
| Pin by document hash in content | yes (`upper.contains`) | **not supported at all** |
| Empty-string pin guard | yes | **absent** |
| Oversized pin | rescued over budget, with an explicit ordering marker | silently `continue`d (`chat_memory_bus.rs:230`) |
| Hole marker between rescued pins and tail | yes | **no** |
| Tests | ~18 | **zero** |

Two consequences worth naming separately:

- **Costing accuracy.** `fit_with_memory_bus` recomputes `tokens` from the
  post-neutralisation text, but the bus made its selection on pre-neutralisation
  estimates. `[E12]` (5 chars) becomes `[cited earlier]` (15 chars), so the
  transform *expands*. A projection can therefore exceed the budget it was given.
- **Silent pin loss.** A pinned turn larger than the remaining budget is dropped
  with no typed signal; it lands only in the undifferentiated
  `retained_not_projected` count.

Pins themselves are untyped: `agent_pin_context` takes `pinned: Vec<String>`
(`agent.rs:4451`), capped at `MAX_PINNED_CONTEXT = 64`
(`conversations.rs:320`), stored as `Conversation.pinned_context: Vec<String>`.

`CHAT_RETENTION_LIMIT = 1_000_000` (`chat_memory_bus.rs:34`) is declared and
**never read** — nothing enforces, measures or reports it.

### 4.5 "No `ChildWorker` is registered; routing is a placeholder" — **CONFIRMED**

- `lib.rs:685` constructs `SubagentManager::new(profiles, events)` and never calls
  `with_worker`. The only `impl ChildWorker` in the repository is `Fake` in
  `subagents/tests.rs:99`. `lib.rs:701-714` logs the gap at start-up and
  `tool_catalogue` withholds `agent.delegate_readonly`.
- `orchestrator/runner.rs:915` hard-codes
  `Decision { model_id: "parent-inherited", … }` — a literal, not a model id —
  and calls `spawn(…, Vec::new(), decision)` with empty inputs.

So delegation is *declared* (5 profiles in `agents/`), *inherits correctly*
(`subagents/inherit.rs` intersects parent policy and is well covered), and is
**not executable**.

### 4.6 Embedding / hybrid retrieval — **CONFIRMED**

`LocalEmbedder` (`knowledge/embedding.rs:64`) and `hybrid::search`
(`knowledge/hybrid.rs`) have **zero production callers**: the only references
outside their own modules are the `pub use` in `knowledge/mod.rs:35` and a
doc-comment. No command and no `agent_runtime` path constructs an embedder. All
retrieval reaching a model today is lexical. `hybrid.rs` already models the
honest-degradation case (`vector_covered`, `degraded`), which Phase 6 should
reuse rather than reinvent.

Note in passing: two embedding models *are* registered on this machine
(`Qwen3-Embedding-0.6B-Q8_0`, `nomic-embed-text-v2-moe.Q8_0`), so the Phase 6
prerequisite is a wiring question, not a missing-model question.

### 4.7 Spark-X2.5-4B Q8_0 — **PARTLY**, and the gap is migration, not code

Verified here, by measurement:

- Weights present at `C:\Users\lenovo\models\Spark-X2.5-4B-Q8\Spark-X2.5-4B-Q8_0.gguf`,
  **4,375,021,152 bytes** — matches the reference exactly.
- `sha256sum` → `5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2`
  — **matches the pinned value exactly.** The file at
  `…\com.arjun.workbench\models\local\Spark_Spark-X2.5-4B\base\` is a hard link
  (link count 2) to the same bytes.
- `src-tauri/config/orchestrator-spark-registry.json` declares
  `id: "orchestrator.spark-x2-5-4b"` with that sha256, `license: "apache-2.0"`,
  `serving.mode: "managed"`, `routing.preferred: true`,
  `permittedClassifications: []`.
- `registry/mod.rs:748` elects an entry whose id is `"orchestrator"` or starts
  with `"orchestrator."`. The election code is real and unit-tested (`:1400`).

But the **installed** registry at `…\com.arjun.workbench\models\registry.json`
contains, instead:

```json
{ "id": "Spark-X2.5-4B-Q8_0", "license": "unstated", "sha256": null,
  "serving": null, "routing": { "preferred": false, "rankWithinBand": 0 },
  "permittedClassifications": [] }
```

The config file was never merged. The orchestrator tag is absent, so Spark is not
elected; `sha256` is `null`, so no integrity check binds; `serving` is `null`, so
managed-launch settings do not apply. That fully explains the routing symptom the
untracked report describes (prompts mentioning Rust/Python routed to
`mtp-gemma-4-12b-it-BF16`) — and it means Phase 3's remaining work is
**registry migration and an administrator path**, not model integration.

No `revision` field exists anywhere in `ModelEntry`. Phase 3's "pin an immutable
model revision" is therefore a schema addition, not a value change.

---

## 5. Inventory of every memory authority

### 5.1 The seven stores

| # | Authority | Backing | Identity / scope key | Reached from production by |
|---|---|---|---|---|
| 1 | `memory_engine` (legacy) | `sarathi.db` tables `memory_nodes`, `user_profile`, `projects`, `working_memory` | **per-machine, no `user_id`** | `commands::inference::send_chat_message` **only**. All 10 IPC commands deleted (`memory_engine/api.rs` documents why: every one authenticated then queried unscoped). |
| 2 | `agent_runtime::memory::MemoryStore` | `<appdata>/memory/{run-,workspace-,user-}*.json`; `Run` scope in RAM only | `MemoryScope::{Run,Workspace,User}` + `Acl` + `Classification` | `memory_api::{recall_authorized, promote_approved, remember_for_run}`; tools `memory.recall_authorized`, `memory.promote_approved` |
| 3 | Conversation history | `<appdata>/conversations/{id}.json` | `owner_user_id` filter on every read | `ConversationStore`; projected by `fit_with_memory_bus` |
| 4 | Task events + checkpoints + leases + idempotency | `sarathi.db` (`task_events`, `checkpoints`, lease and outcome tables) | `run_id`, `actor`, monotonic `seq` | `record_and_publish`, `recording::*`, `resume::*` |
| 5 | Notebook graph + research | `sarathi.db`, tables prefixed `notebook_*` / `graph_*` | `notebook_id` + `owner_user_id` **bound into the SQL** | `commands::notebook*`, `notebook_retrieval::{resolve,retrieve,manifest}` |
| 6 | Documents / extractions | `<appdata>/documents/{originals,derived}` + `extractions/{sha}.json` + `sarathi.db` | content-addressed by sha256 | `DocumentStore`, `agent_runtime::documents` |
| 7 | Conversation artifacts | `sarathi.db` + `<appdata>/artifacts/blobs` | conversation-scoped, revisioned | `ConversationArtifacts`, `agent_runtime::artifacts` |

Plus `audit_log` and `credentials`, also in `sarathi.db`, and
`knowledge/state/*.json` collections.

### 5.2 The finding that shapes Phase 5 — one file, many connections

Authorities 1, 4, 5, 6, 7 (and audit, and credentials) all call
`Connection::open(app_data_dir.join("sarathi.db"))` **independently**. There is no
shared connection and no shared migration runner. `knowledge/graph/mod.rs:9-19`
documents the consequence in the code itself:

> `sarathi.db` already contains a `documents` table created *twice*, with two
> incompatible schemas, by `crate::documents::store` and by
> `knowledge::multimodal` — whichever opens first wins and the other's inserts
> fail. It is latent today only because one of the two is never constructed
> outside tests. `projects` and `memory_nodes` are likewise taken, by
> `memory_engine::persistence`.

`events/store.rs:115-120` sets a 5-second `BUSY_TIMEOUT` for the same reason.

Two conclusions, both load-bearing for the contracts document:

- A single SQLite *file* does **not** give cross-domain atomicity here, because a
  transaction cannot span two `Connection` handles. Phase 5's outbox protocol is
  therefore **required**, not a precaution.
- Authorities 2 and 3 are JSON files outside the database entirely, so any
  "commit a memory record and a conversation change together" story must be an
  outbox, not a transaction.

### 5.3 What the graph is today, and what it is not

`knowledge/graph` holds a **document co-occurrence and research graph**: nodes are
`node_id(notebook_id, normalised_term)` and `document_node_id(notebook_id, sha256)`;
edges are co-occurrence and extracted relations; `Assertion` carries
`provenance: {Model, User}`, `status: {Proposed, Accepted, Rejected}`,
`direction_certain`, `extractor` + `extractor_version`, `source_revision`,
`stale`, and per-passage `AssertionEvidence { chunk_id, document_sha256, page, quote }`.

`AssertionStatus::usable_as_evidence` already encodes "a rejected claim is never
retrieved; a proposed one is retrieved and labelled unreviewed". That is exactly
the proposed-vs-validated distinction the contract asks for, and it should be
extended rather than re-invented.

There are **no** agent nodes, task nodes, run nodes or memory-assertion nodes.
Everything is keyed by `notebook_id`.

`NotebookStore::graph_revision` (`research.rs:611`) returns
`MAX(updated_at)` — **a timestamp string**. Usable as a staleness hint, which is
what it was built for. **Not** usable as a changefeed cursor: no total order, no
gap detection, vulnerable to equal timestamps and clock movement.

---

## 6. Frontend baseline

- **Graph**: `src/components/graph/GraphCanvas.tsx` (704 lines) draws on a 2D
  canvas driven by `simulation.ts` (380 lines). The background already reads
  `--bg-primary`, default `#000000` (`GraphCanvas.tsx:393`) — the black surface
  requirement is already met.
- **The renderer's stated design rule is that type is never carried by colour
  alone** (`GraphCanvas.tsx:27-36`): types are shapes, colour carries selection
  and focus only. Phase 9 introduces per-agent colour for **ownership**, which is
  a different axis from type. The rule survives only if shape, name and status
  cues are all retained. This is recorded as a contract, not an exception.
- **`GraphSimulation` has no incremental update path.** The constructor sorts all
  node ids, seeds a PRNG from them, and lays every node out on a ring
  (`simulation.ts:153-172`). Adding one node means rebuilding and losing every
  position. Phase 9's "retain positions by stable node id" is new work.
- **`clampToBox` forces every node inside the viewport** (`simulation.ts:371`) —
  precisely the behaviour Phase 9 forbids.
- **Collision uses `node.radius` only** (`applyCollision`, `:332`); label bounds
  are not in the geometry.
- **Graph delivery is pull-only.** `notebook_graph` returns a snapshot; there is
  no subscription, no cursor, no replay.
- **Durable events are broadcast unscoped.** `record_and_publish` calls
  `app.emit(AGENT_DURABLE_EVENT, event.envelope())`, and `envelope()`
  (`events/model.rs:431`) includes the full `payload`. Every window receives every
  payload regardless of the viewer.
- **Administration menu** (`AppMenu.tsx:54-63`): Models, Approvals, Audit &
  Network, Health, Model Health, Settings. No Agents entry; no `/agents` route.

---

## 7. Agent definitions today

`subagents::AgentProfile` (`profile.rs:212`) is **name-keyed** — the file name is
the identity. Fields: `name`, `description`, `version`, `model_role`,
`eligible_models`, `allowed_tools`, `disallowed_tools`, `limits`, `isolation`,
`memory_scope`, `network_permitted`, `write_policy`, `classification_ceiling`,
`required_schema`, `sha256`.

Absent: stable id, colour, enabled/archive state, skill bindings, default/fallback
model split, output schema beyond `SchemaKind`, concurrency budget.

Profiles load from the **installer resource directory** (`lib.rs:660`, and
`agents/` is in `bundle.resources`). Mutable agent configuration cannot be written
there, which is why Phase 4 needs an AppData-backed registry.

`Permission` (`identity/mod.rs:48`) has 11 variants. There is **no** agent-registry
permission; `ModifyPolicy` is the closest existing analogue.

---

## 8. What is already strong, and should be reused rather than rebuilt

Naming these matters as much as naming the gaps — the contract says to extend one
authority, not add another.

1. **Commit-before-publish with sequence numbers** (`record_and_publish`) — the
   durable channel Phase 9's changefeed should extend.
2. **Owner-bound SQL** in `knowledge/graph` and `knowledge/index` — clearance is
   bound into the statement, not filtered after the fetch. This is the pattern
   Phase 5's "authorize before traversal" must follow.
3. **`EvidenceManifest`** (`research.rs:229`) already carries run / notebook /
   conversation / message ids, scope, retrieval mode, entries, `limitations`,
   `graph_revision` and `sources_unused`. Phase 2's `ContextManifest` should
   extend this shape, not compete with it.
4. **`ApprovalBinding` + `source_fingerprint`** (`memory.rs:384-461`) — an
   approval that does not verify against *this* request is refused. The admission
   rule Phase 5 needs for model-proposed assertions already exists in miniature.
5. **`subagents::inherit`** — intersection-only policy narrowing with a policy
   hash carried in the packet. Phase 8 gets its authorization model free.
6. **`AssertionStatus` / `AssertionProvenance`** — proposed / accepted / rejected
   with rejection retained.
7. **Idempotency, leases and unknown-effect reconciliation**
   (`events/idempotency.rs`, `events/lease.rs`, `agent_unknown_effects`,
   `agent_reconcile_effect`) — Phase 8's duplicate-dispatch story already has
   primitives.
8. **`AuditHealth` refusal gate** — "nothing runs that cannot be recorded" is
   already enforced at the top of `drive_run`.

---

## 9. Verification status summary

| Claim | Status |
|---|---|
| Baseline SHA, branch, parity, dirty files | measured |
| `check:ipc`, `check:egress`, `runtime:typecheck`, `runtime:audit` | run, pass |
| `runtime:test`, `test:ui`, `test:rust` (lib), `test:integration` | run, pass |
| `check:deployment` graph-sidecar failure | run, fails, cause read in `tauri.conf.json` |
| Spark weights size + SHA256 | measured on this machine |
| Installed registry lacks the orchestrator entry | read from AppData |
| Every §4 hypothesis | read in the named production file and line |
| §5 store inventory | read from each `open()` |
| `ARJUN_SPARK_TEST_REPORT.md` figures | **not verified, not used as evidence** |
| Any Context-memory branch claim | **not used as evidence** |
| Real-model behaviour, native UI, installed/offline | **not attempted in Phase 0** |

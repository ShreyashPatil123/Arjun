# Intent routing — Laya semantic reader in front of the model router

## Read this first

- **Implemented and tested:** the Laya intent sidecar, the Rust client and
  engine, language detection, the calibrated gate, the keyword fail-safe, router
  integration, logging, a 180-case validation set and the calibration harness.
- **Not yet measured:** Laya's accuracy, latency and memory *on ARJUN's
  validation set*. The build environment could not download the weights
  (`huggingface.co` is blocked by that environment's network policy). The
  harness that measures them exists (`capability::intent_eval::measure`) and
  fails loudly without weights.
- **What ARJUN does until then:** without a calibration file fitted for the
  loaded question, Laya runs in **shadow mode**. Its verdict is logged next to
  the keyword verdict, but routing is exactly what it was before this change.
  No threshold that was never measured decides which model answers.
- **Where the numbers come from:** every figure here is either measured in this
  repository (and says where) or quoted from Laya's own repository (and says
  so). None is an estimate presented as a measurement.

## The flow

```text
User prompt
  → language::detect          en | hi | hi-en | und        (Rust, microseconds)
  → Laya `choice` over 6 intents, via intent sidecar       (stdio JSON-RPC, CPU, deadline)
       │ absent · loading · busy · timed out · crashed · wrong answer shape · uncalibrated
       └──────────► keyword IntentClassifier (unchanged; the fail-safe)
  → gate: top probability ≥ τ AND lead over runner-up ≥ μ  (fitted, not chosen)
       fail → ambiguous → general reasoning route
  → IntentAnalysis ─► ModelRouter::route_*_analyzed
                       role → candidates (enabled, floor, clearance, modality, licence)
                            → orchestrator → largest that fits VRAM → below-floor rescue
                            → sticky conversation model
  → the model answers
```

Laya returns an intent, never a model. The router, registry, VRAM planner,
orchestrator preference, stickiness and every refusal are unchanged; the only
thing they now receive is an `IntentAnalysis` instead of a keyword score. The
original `route`, `route_with_orchestrator` and `route_sticky` signatures still
exist. They build a keyword `IntentAnalysis` and delegate, so every caller
without an engine behaves exactly as before, down to the wording of the trace.

## Files

| File | What it is |
| --- | --- |
| `sidecars/intent_sidecar/intent_question.json` | The question: six labels and their descriptions. The single source of truth, shared byte-for-byte by Python and Rust. |
| `sidecars/intent_sidecar/engine.py` | Loads the installed checkpoints from a local directory only, checks the head budget, answers one `choice` per prompt. |
| `sidecars/intent_sidecar/main.py` | JSON-RPC over stdio; the Hub switched off before anything imports it; stdout reserved for the protocol. |
| `sidecars/intent_sidecar/tests/test_engine.py` | `unittest` without weights: question, refusals, protocol, budget arithmetic. |
| `src-tauri/src/capability/language.rs` | English / Hindi / Hindi-English detection, and the checkpoint ARJUN forces for Hindi. |
| `src-tauri/src/capability/intent_analysis.rs` | `IntentAnalysis`, `LayaVerdict`, `LayaGate`, the combine rule, trace and log wording. |
| `src-tauri/src/capability/laya_sidecar.rs` | The sidecar process client (deadline, backoff, containment) and `IntentEngine`. |
| `src-tauri/src/capability/intent_eval.rs` | Validation set loader, scoring, gate fitting, the ignored `measure` test. |
| `fixtures/intent-routing/v1/validation.jsonl` | 180 labelled prompts, split 90/90 into `fit` and `test`. |
| `src-tauri/src/registry/router.rs` | `route_analyzed` / `route_sticky_analyzed`; `RoutingDecision.intent_analysis`. |
| `src-tauri/src/commands/agent.rs`, `commands/registry.rs`, `lib.rs` | The engine in the run path, the preview and `prepare_model_for`, started at launch. |
| `src-tauri/src/deployment/mod.rs`, `tauri.conf.json` | `intent-sidecar` as a bundled Feature dependency. |

## Classification schema

One Laya `choice` question, asked against the state `{"request": <prompt>}`:

> Which kind of work is the user asking for in `request`?

| Label | Description given to Laya | Router role |
| --- | --- | --- |
| `coding` | write, debug or explain software code, scripts or SQL (not ASME/API codes) | Coding |
| `mathematics` | solve a calculation, equation, proof or unit conversion with numbers | Reasoning |
| `reasoning` | weigh a decision: compare options, trade-offs, risks or root causes | Reasoning |
| `tool-calling` | emit machine-readable JSON, schema output or function-call arguments | Reasoning |
| `research` | summarise or extract findings from documents, reports, papers or manuals | Reasoning |
| `general` | greeting, thanks, small talk, general knowledge, or a follow-up with no task | Reasoning |

Why it is written this way:

- **Labels are ARJUN's capability keys.** A label Rust cannot map to a
  `PromptIntent` is a protocol fault, and a test pins the mapping.
- **No boolean-word labels.** Laya's README warns that labels like
  `true`/`false` or `yes`/`no` get followed instead of the descriptions.
- **Short descriptions.** On the English checkpoint the six options and the
  instruction share a 192-token head budget, and past it
  `laya.common.build_sequence` silently cuts every option to about 29 tokens.
  The sidecar measures the budget with the checkpoint's own tokenizer and
  refuses to start over it. A Rust test is the cheap early warning, and it
  caught the first draft at about 160 tokens of options.
- **The plant collision is named.** `coding` says "not ASME/API codes". The
  point is to separate engineering codes and plant vocabulary from software
  without relying on a keyword table.
- **Hierarchy, not breadth.** More intents later (for example, coding →
  write / debug / review) go in a second question asked only after the first
  answers `coding`. They don't go into this list. On Banking77, with 77 flat
  labels, Laya's README reports 0.425 against Jev's 0.870 and attributes the
  gap to this same head budget (3–4 tokens per label).

## Language

Laya's `Router` picks the multilingual checkpoint for non-Latin script, and
does so correctly for Devanagari. It does **not** for romanized Hindi. Laya
0.3.20's `lang.py` has stopword lists for romanized Bangla, Azerbaijani and the
Romance languages, but none for Hindi. This was checked by running the real
package: `Router().route("is report ka summary do")` returns
`english — "English Latin text"`. Laya's own benchmark gives the English
checkpoint 0.100 on Hindi at 20 options (random is 0.050).

So ARJUN detects Hindi first (`language.rs`) and names the `multilingual`
checkpoint for `hi` and `hi-en`. The romanized-Hindi marker list excludes every
word that is also ordinary English (`is`, `to`, `me`, `the`, `main`, `do`,
`par`, …). A single marker counts only when it is at least a fifth of the words.
On the validation set's hand-written labels the detector matches **180/180**.

## Gating and calibration

Laya's `choice` answer carries `probabilities`, `answer_confidence` (the top
probability after temperature scaling) and `confidence` (normalised entropy).
Laya's `common.py` says the two confidences must not share a threshold. ARJUN
gates on the probabilities:

- **top probability ≥ τ** catches prompts Laya cannot read;
- **top − runner-up ≥ μ** catches prompts that read as two things at once.

Neither τ nor μ has a default. Laya's README says both checkpoints ship
over-confident and `laya-multilingual` ships with no fitted temperatures.
BENCHMARKS.md reports the opposite direction, under-confidence, on a routing
task. The direction depends on the task, so the gate is fitted on this task:

1. `intent_eval::measure` runs every validation prompt through the production
   path: language detection, the real sidecar, the real checkpoints. It records
   the full distribution and the round-trip time.
2. Every τ ∈ [0.20, 0.95] × μ ∈ [0.00, 0.60] (steps of 0.01), under both
   policies (`laya`, and `hybrid` where a confident keyword verdict for the
   *same* intent may rescue a near miss), is scored on the **fit** half.
3. It keeps gates with no more false-specialist routes than the keyword
   classifier makes on the same prompts, then picks the fewest routing errors.
   Ties go to higher intent accuracy, then `laya` over `hybrid`, then the
   stricter gate.
4. It reports the chosen gate on the held-out **test** half, and writes
   `arjun-intent-calibration.json` with the question fingerprint, checkpoints,
   Laya version, dataset hash and the measured comparison.

A calibration applies only to the question bytes it was fitted on (SHA-256) and
the checkpoints it saw. Change a description, or answer from an unmeasured
checkpoint, and Laya goes back to shadow mode, with a logged reason.

Keywords can confirm Laya but never overrule it. A confident keyword verdict
that disagrees with a gated Laya verdict loses, because those disagreements are
the cases keyword counting gets wrong.

## Integration choice

| Option | Verdict | Evidence (Laya 0.3.20 source) |
| --- | --- | --- |
| Python stdio sidecar, CPU | **Chosen** | Same transport and CPU-torch rationale as `graph_sidecar` (REBEL, 400M). No socket, no port, no HTTP client for `check-egress.mjs` to exempt. |
| `laya-serve` HTTP | Rejected | `LAYA_HOST` defaults to `0.0.0.0` (`laya/serve.py`); ARJUN would need a loopback HTTP client and an egress exemption; one worker anyway. |
| ONNX Runtime (`laya.onnx_agent.ONNXAgent`) | Rejected for now | Still imports torch (`from .agent import …`); returns no `answer_confidence`; picks `CUDAExecutionProvider` automatically when present; the export script (`scripts/export_onnx.py`) is not in the pip package. |
| GPU for Laya | Opt-in only (`ARJUN_LAYA_DEVICE=cuda`) | 421M + 322M parameters is about 1.7 + 1.3 GB of weights at fp32 before activations. That's arithmetic; Laya publishes no VRAM figure. On an 8 GB card that is the margin between the chat model fitting in VRAM and running partly on the CPU. |

## Measured in this repository

Environment: a 4-vCPU Linux container with no GPU and Python 3.11.15.

**Keyword classifier on the validation set** (`cargo test --lib intent_eval -- --nocapture`):

| | n | intent (6-way) | role | false specialist | missed specialist | abstain |
| --- | --- | --- | --- | --- | --- | --- |
| all | 180 | 73.3% | 93.3% | 6 | 6 | 60.0% |
| fit | 90 | 76.7% | 92.2% | 3 | 4 | 57.8% |
| test | 90 | 70.0% | 94.4% | 3 | 2 | 62.2% |

| category | n | intent | role | false specialist |
| --- | --- | --- | --- | --- |
| clear | 66 | 66.7% | 97.0% | 0 |
| refinery | 30 | 70.0% | 83.3% | 4 |
| ambiguous | 14 | 100.0% | 100.0% | 0 |
| short | 16 | 100.0% | 100.0% | 0 |
| hindi | 20 | 40.0% | 85.0% | 0 |
| hinglish | 20 | 75.0% | 90.0% | 2 |
| multi | 14 | 100.0% | 100.0% | 0 |

The six prompts the keyword classifier sends to the coding model:

- `refinery-007` "Debug why the heat exchanger outlet temperature keeps dropping"
- `refinery-008` "Compile a list of all pumps due for maintenance next month"
- `refinery-010` "Refactor the maintenance schedule so the two turbines are not serviced in the same week"
- `refinery-011` "Build a unit test plan for the new flare system instrumentation"
- `hinglish-018` "maintenance schedule ko refactor karo taaki dono turbine ek hi hafte mein service na ho"
- `hinglish-019` "heat exchanger ka outlet temperature kyun gir raha hai, debug karo"

The keyword classifier's perfect scores on ambiguous, short and multi-intent
prompts come from abstaining, which is the correct behaviour there. On Hindi it
finds no signal at all and abstains too. That is safe (no false specialist)
but it misses every Hindi coding request.

**The 0.55 keyword threshold, swept** (all 180):

| threshold | 0.30–0.40 | 0.45 | 0.50 | **0.55** | 0.60 | 0.65 | 0.70 | 0.80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| routing errors | 16 | 13 | 14 | **12** | 19 | 21 | 22 | 33 |
| false specialist | 12 | 9 | 9 | 6 | 5 | 1 | 0 | 0 |

The existing threshold is the error minimum on this set, so it is kept.

**Cost of the keyword path:** 0.006 ms p50 and 0.011 ms p95 per prompt over
the 180 cases in a release build (0.067 / 0.109 ms in a debug build). It runs
in process, with no model and nothing resident beyond its static signal tables.

**Cost of the sidecar, without a model:**
- stdio JSON-RPC round trip to the real `main.py`: 0.049 ms p50, 0.070 ms p95
  over 1,000 calls;
- 29.9 ms from spawn to the first answer;
- 14.7 MB RSS before a checkpoint loads.

The IPC adds nothing measurable next to a forward pass.

## Laya's own published figures (not measured on ARJUN's data)

| Figure | Value | Source in Laya 0.3.20 |
| --- | --- | --- |
| One `choice`, checkpoint resident, T4 GPU | 32.8 ms (multilingual) / 39.5 ms (English) | README, "Why Route" |
| One question, preloaded, CPU | 193–464 ms | README, "Production Preload & Memory" |
| One question, Ryzen 9 6900HX, inter-op pinned | 910 ms at 1 thread; 329 ms at 8 | BENCHMARKS.md |
| Checkpoint reload | 7.4 s median CPU, 10.3 s T4 | README |
| MASSIVE intent (20 options), English | 0.783 English ckpt / 0.657 multilingual | router.py, README |
| MASSIVE intent, Hindi | 0.100 on the English checkpoint | lang.py |

## Comparison: keyword vs Laya vs hybrid

| | keyword | Laya, ungated | Laya, gated | hybrid |
| --- | --- | --- | --- | --- |
| intent accuracy (test half) | 70.0% | not measured | not measured | not measured |
| role accuracy (test half) | 94.4% | not measured | not measured | not measured |
| false specialist (test half) | 3 | not measured | not measured | not measured |
| latency | 0.006 ms p50 (release) | not measured; Laya reports 193–910 ms CPU | ← + gate (ns) | ← + keyword |
| resident memory | none | not measured; ≈3 GB of fp32 weights (arithmetic) | same | same |

Filling in the three right-hand columns is one command once the weights are on
the machine:

```bash
ARJUN_LAYA_DIR=<models>/local/convaiinnovations_laya ARJUN_PYTHON=python3 \
  cargo test --lib intent_eval::measure -- --ignored --nocapture       # CPU row
ARJUN_LAYA_DEVICE=cuda ... (same command)                               # GPU row
cp target/intent-eval/arjun-intent-calibration.json <models>/local/convaiinnovations_laya/
```

## Operating it

- **Weights:** the `convaiinnovations/laya` bundle, with the English checkpoint
  at its root and `multilingual/` beside it, at
  `<app data>/models/local/convaiinnovations_laya`, or wherever
  `ARJUN_LAYA_DIR` points. It's provisioned like any other model; the sidecar
  never downloads.
- **Python:** `pip install -r sidecars/intent_sidecar/requirements.txt` from
  vendored wheels (CPU torch; the same torch and transformers pins as the graph
  sidecar).
- **Costs once weights are installed:** every turn waits for Laya up to the
  deadline, **including in shadow mode**, where the answer is only logged. The
  sidecar keeps its checkpoints resident in CPU memory (about 3 GB of fp32
  weights for both; arithmetic, not measured). A site that never sees Hindi can
  remove `multilingual/` to load only the English checkpoint. Hindi prompts
  then fall back to keywords, with that reason logged.
- **Switches:** `ARJUN_LAYA=off` · `ARJUN_LAYA_DEVICE=cpu|cuda` ·
  `ARJUN_LAYA_THREADS` · `ARJUN_LAYA_DEADLINE_MS` (default 1500, which clears
  Laya's slowest published CPU p95 of 1,023 ms) · `ARJUN_LAYA_CALIBRATION`.
- **Logs**, one line per turn plus the routing result:

```text
[INTENT] source=laya lang=en intent=coding confidence=0.820 runner_up=research 0.090 ambiguous=false checkpoint=english latency_ms=312.0
[INTENT] source=keyword lang=hi-en intent=general confidence=0.000 runner_up=- ambiguous=true checkpoint=- latency_ms=0.1 fallback="Laya did not answer within 1500 ms"
[ROUTING] intent=coding source=laya confidence=0.820 ambiguous=false fallback=- -> model=qwen-coder-14b role=coding used_fallback=false
```

  The routing trace (`RoutingDecision.reasons`) says which engine read the turn,
  and why the keyword classifier was used when it was. The full `IntentAnalysis`
  is recorded on the decision.

## PS 26117

Checked against [`sih/ps-26117-official.md`](sih/ps-26117-official.md).

| PS text (verbatim) | How this change relates |
| --- | --- |
| "automatically pick the right one for a given task based on what that task needs, a coding request handled differently from a document summary request" | The intent that decides the role can now be read semantically. The pick itself is still the registry's. `the_two_demo_task_types_reach_different_models` is unchanged and passes. |
| "New open weight models should be addable later without redesigning the system" | Laya never names a model; adding one is still a registry entry. |
| "air gapped … Nothing leaves the premises" | Weights load from a local directory; `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set by both parent and child; no socket; the egress gate passes. |
| "show, through logs or a visible network monitor, that no external calls are made at any point" | Static checks only. No live network observation was run for this change. |

Hindi and Hinglish support, the confidence gate and the sidecar design are
ARJUN's additions. The problem statement doesn't mention languages.

## Limitations

1. **Laya is unmeasured on ARJUN's data** (see the top of this page). Until
   `measure` runs and its calibration is installed, routing is the keyword
   classifier and Laya only logs.
2. **180 prompts is small.** A gate fitted on 90 of them is a starting point;
   re-fit on real, anonymised turns once there are some. The test half is 90
   prompts, so one prompt is 1.1 points.
3. **The labels are one annotator's.** `accept` lists the readings a second
   annotator could defend. The set has not been independently re-labelled.
4. **The plain-chat capability layer still uses keywords.** `CapabilityLayer`
   (prompt profile and sampling, not model choice) keeps the keyword classifier:
   its hysteresis thresholds are calibrated to that scale and would need
   refitting against Laya's.
5. **Romanized-Hindi detection is a word list.** It covers the common function
   words and verbs. Unusual spellings fall to English and reach the English
   checkpoint, as they would have anyway.
6. **One prompt at a time.** Laya's batching helps throughput, not the latency
   of one turn, and Laya notes that batch size alone may not speed up CPU
   inference. No batching is used.
7. **The head-budget check against the real tokenizer has not run here.** Only
   the character estimate and the arithmetic against a stand-in tokenizer have.
   The sidecar enforces the real check at load.
8. **Provisioning the weights isn't wired into the downloader.** An operator
   places the bundle; the offline pack does not yet carry it.

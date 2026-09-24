# P00 evidence — 21 September 2026

What was actually run on this machine, and what it produced. Every figure here
is copied from a command's output; the raw logs are beside this file.

Read the ledger first: [`docs/plans/2026-09-20-agent-system-execution-ledger.md`](../../../docs/plans/2026-09-20-agent-system-execution-ledger.md).

## Provenance

| Fact | Value |
|---|---|
| HEAD at phase start | `b9ff7fbcbab3c610dffd541b674ec04706710561`, working tree clean |
| Host | Windows 11 Home Single Language 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5060 **Laptop** GPU, 8151 MiB, driver 595.95, CUDA 13.2 |
| llama-server | build 10970, commit `bfdc32183` |
| App data identity | `com.arjun.workbench` |

**Not a target-machine measurement.** The plan targets an "RTX 5060"; this is a
Laptop SKU. Whether they are the same deployment target is not established.

## Files

| File | What it is |
|---|---|
| `qualification-inventory.json` | 15 models with architecture and trained context read from their own GGUF headers, 15 sha-256s, runtime build, container state, renderers, OCR settings, embedding models. Every value carries `measured`, `declared` or `unknown`. |
| `served-context-observations.json` | Two real `llama-server` runs of Spark-X2.5-4B Q8_0 with VRAM, load time and observed served context, plus three findings. |
| `baseline.json` | The harness report: 39 cases, 9 executed, 9 passed, 0 failed, 30 blocked. |
| `log_qualification-inventory.txt` | Output of the hashing inventory run. |
| `log_agent-baseline.txt` | Output of the final baseline run. |
| `log_repo-gates.txt` | Output of the four repository gates. |
| `log_llama-server-spark.txt` | The 4096-context server's own log. |
| `log_llama-server-spark-16k.txt` | The 16384-context server's own log. |

## Exact commands

Inventory, with model hashes:

```bash
node scripts/qualification-inventory.mjs --hash
```

Fixture manifest, and the staleness check:

```bash
node scripts/fixture-manifest.mjs && node scripts/fixture-manifest.mjs --check
```

Serve the orchestrator model:

```bash
llama-server -m "$APPDATA/com.arjun.workbench/models/local/Spark_Spark-X2.5-4B/base/Spark-X2.5-4B-Q8_0.gguf" --port 8080 -c 16384 -ngl 99 --host 127.0.0.1 --no-warmup
```

The baseline, against that server:

```bash
ARJUN_BASELINE_MODEL_URL=http://127.0.0.1:8080/v1 ARJUN_BASELINE_MODEL_ID=Spark-X2.5-4B-Q8_0 node scripts/agent-baseline.mjs
```

The driver target alone:

```bash
cargo test --manifest-path src-tauri/Cargo.toml --test agent_baseline -- --nocapture --test-threads=1
```

## What 9 of 39 means

Thirty cases are **blocked**, not failed: the agents that would answer them do
not exist in this build yet. Each blocked case names the phase that delivers it.
Nine cases ran, and all nine passed.

Four of the nine are labelled deterministic — two transport contract cases and
two fixture cases — as §11.1 requires. Five reach real infrastructure: the real
production driver twice, a real served model once, and two preconditions over
real files.

## The two results worth reading

**`driver-01-no-model-server-fails-honestly`.** `run.start` was pointed at a
loopback address nothing serves. The production driver returned
`outcome: {kind: "failed", detail: "Connection error."}`, `stopReason: "error"`,
empty text, with a real seven-event trace. It did not invent an answer.

**P00-OBS-1, from the driver's own `context_ledger` event.** The tool schemas
occupy **8584 tokens** before any prompt, system message, history or evidence.
A one-word question costs 8852 input tokens, so a 4096-token served context
cannot complete a single turn of this runtime — whatever model is behind it.
That is what "report actual served context rather than the advertised maximum"
looks like in a number.

## A defect found here, and fixed here

`model-01-real-generation` first reported **passed** for a run whose own outcome
was `failed: request (8852 tokens) exceeds the available context size (4096)`.
It checked only that `run.start` returned `Ok` — that the driver answered, not
that the run worked.

That is the fabricated pass this repository has a standing rule against, and it
was in the harness written to prevent it. The case now reads `outcome.kind` and
requires non-empty text. At 4096 it correctly reports failed; at 16384 it
reports passed with 12 characters generated. Both runs are in
`served-context-observations.json`.

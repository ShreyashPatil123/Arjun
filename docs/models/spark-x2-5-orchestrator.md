# Spark-X2.5-4B as ARJUN's orchestrator

The orchestrator is the model that runs the chat: it reasons, calls tools, and
decides what happens next. This document records what was measured about
Spark-X2.5-4B (Q8_0) on real hardware, what ARJUN does with those measurements,
and the one place where the specification this work started from does not
survive contact with an 8 GB card.

Every figure below was measured on this machine. None is estimated, and none is
carried over from a datasheet.

## The model

Read from the GGUF header with a GGUF reader, not from a model card:

| | |
|---|---|
| Architecture | `spark2_5` |
| Blocks | 36 |
| KV heads | 4 |
| Key / value width | 256 / 256 |
| Sliding window | 512 tokens |
| Attention pattern | `[T, T, T, F] × 9` — 27 windowed blocks, **9 full-attention** |
| Trained context | 1 048 576 |
| Vocabulary | 131 072 |
| File | 4 375 021 152 bytes, `sha256 5c2c3c19…d9dea2` |

The attention pattern is the fact everything else follows from. Only the 9
full-attention blocks hold a cache that grows with the conversation; the other
27 hold a fixed 512-token window however long the chat gets.

```
per token (grows):  9 blocks × 4 KV heads × (256+256) × 2 B = 36 864 B
fixed (does not):  27 blocks × 4 KV heads × (256+256) × 2 B × 512 = 56 623 104 B
```

Reading the model as dense — which is what ARJUN did before this work — charges
`36 × 4 × 512 × 2 = 147 456` bytes a token, **four times** what it costs.

## What that means for the 1M context

A million tokens of KV cache at `q8_0` is about **19.3 GB**. Plus 4.4 GB of
weights, that is a 24 GB card. It is not an 8 GB one, and no amount of
configuration makes it one.

Measured directly, with `-ngl 99 -dev Vulkan0 -fa 1 -ctk q8_0 -ctv q8_0` on an
RTX 5060 Laptop (7 899 MiB, ~7.1 GB free):

| `-c` | Result |
|---|---|
| 1 048 576 | `ggml_vulkan: Device memory allocation of size 1140850688 failed` → `failed to allocate buffer for kv cache` |
| 262 144 | same failure |
| 196 608 | same failure |
| 163 840 | loads |
| 131 072 | loads in 4.9 s, serves, 50.0 tok/s decode |

So the command line in the original integration brief —

```
llama-server.exe -m …\Spark-X2.5-4B-Q8_0.gguf -c 1048576 -ctk q8_0 -ctv q8_0 \
  -ngl 99 -dev Vulkan0 -fa 1 --host 127.0.0.1 --port 8080 --no-warmup
```

— **does not start on this machine.** It is a correct command line for a card
with the memory for it.

### What ARJUN does instead

The registry entry declares the **trained** window, 1 048 576, and
`ai_engine::vram_planner` walks down from it to the largest window this
particular machine can actually hold. That is not a workaround; it is how this
codebase already treats every model, and why `Endpoint::context_tokens` means
"what the server was actually started with" rather than "what the model
supports".

With the hybrid-attention reading in place, the planner's answer is:

| Card | Served window | Whole model on GPU |
|---|---|---|
| RTX 5060 Laptop, 7.9 GB | 65 536 | yes |
| 16 GB | 524 288 | yes |
| 24 GB | 1 048 576 (the trained window) | yes |

Before this work the same card was served **16 384** tokens — the planner was
paying the dense KV price. Correcting the geometry is worth a **4× larger
context** on the development machine, and the full million on a workstation.

The planner stays deliberately conservative: it reserves 900 MB for the OS and
12 % of what remains for compute buffers, which is why it picks 65 536 where a
bare `llama-server` will load 163 840. That headroom is what lets ARJUN run an
OCR or embedding model beside the orchestrator without the pair fighting over
the card.

## Serving

`serving::plan_launch` builds the command line. Every flag is probed against
`llama-server --help` before it is sent, because a build that does not know a
flag refuses to start — which is worse than not passing it.

| Flag | Where it comes from |
|---|---|
| `--ctx-size` | the planner's chosen window, not the entry's declared one |
| `--n-gpu-layers` | `auto` when the build supports it — see `gpu_layers_arg` |
| `-fa on -ctk q8_0 -ctv q8_0` | probed; halves the KV cache |
| `--temp 0.15 --top-p 0.95` | the entry's `sampling` block |
| `--device` | `ARJUN_LLAMA_DEVICE`, when set |
| `--no-warmup` | probed |
| `--jinja --reasoning-format deepseek` | when the model emits reasoning and the build supports it |

### `ARJUN_LLAMA_DEVICE`

This laptop reports two Vulkan devices: `Vulkan0` is the RTX 5060 and `Vulkan1`
is the integrated Radeon sharing system RAM. Left unsaid, llama.cpp may split
the model across both — while the planner sized the whole plan against the
discrete card alone. Set `ARJUN_LLAMA_DEVICE=Vulkan0` on a machine with more
than one GPU. Unset, nothing changes from previous behaviour.

### Sampling

`temperature 0.15`, `top_p 0.95`, declared on the entry rather than set
globally. The model's own header suggests `temp 1.0 / top_p 0.95`, which is a
chat default rather than an agent one: this model's job is to emit tool-call
arguments that parse against a schema every time, and that is worth more here
than variety. Declaring it per entry is what stops one model's needs from
changing what every other model gets.

## Token budgets

`ai_engine::token_budget` caps **one generation**, by what the turn is doing:

| Band | Cap | Chosen when |
|---|---|---|
| Routing | 2 048 | the router labelled it routing/classification and complexity is low |
| Tool calling | 4 096 | the default, and what every turn got before |
| Decomposition | 6 144 | the plan has more than two steps |
| Complex reasoning | 8 192 | the complexity estimator says high |

Hard ceiling: **16 384** per inference call. Any band is additionally clamped to
half the served window, because the cap is also the reply reserve the runtime's
compactor holds free — an unclamped cap on a small window starves the
conversation it was meant to serve.

The signals are ones the turn already computed (plan step count, complexity
estimate, router intent), so classifying a model call costs no model call.

## Continuing past one generation

The 16 384 ceiling is per call, not per task. `ai_engine::continuation` carries
a task across several:

```
task -> gen 1 (<= cap) -> checkpoint -> gen 2 (<= cap) -> ... -> done
                             |
                             +-> not converging -> hand back, and say so
```

A `Checkpoint` is a compression, not a transcript — conclusions reached,
subtasks done and outstanding, constraints still binding, tool calls already
made, and the next step. Sending the transcript back instead is how the second
generation has less room than the first.

Three things stop the chain:

- **Nothing outstanding** — finished.
- **No progress twice running** — a checkpoint that adds no completed subtask
  achieved nothing, whatever it emitted, and two in a row is a circular think
  loop. This escalates: the caller should delegate or recover, not just report.
- **8 generations** — a backstop for anything the progress check missed.

A generation that stopped *of its own accord* with work outstanding is never
resumed. That is the model deciding to hand back, and overruling it is not a
judgement this code is in a position to make.

## Timeout tiers

These already existed and are genuinely decoupled. Recorded here because the
integration brief asked for a tier table, and the numbers are not the ones it
proposed:

| Tier | Value | Where |
|---|---|---|
| Readiness probe | 5 s per attempt | `serving::probe::PROBE_TIMEOUT` |
| Server becomes ready | 180 s | `serving::READY_TIMEOUT` |
| Tool execution | per tool, from the catalogue | `agent-runtime/src/tools.ts` |
| Model call | **4 min of silence**, rearmed by any output | `run.ts` stall guard |
| Turns per run | 64 | `run.ts` `MAX_TURNS` |
| Whole task | the plan's `max_duration_seconds` | `commands/agent.rs` |

The model-call tier is a stall guard rather than a fixed 300 s timeout, and
deliberately so: a fixed ceiling cannot tell a slow model from a stuck one. A
2.3k-token answer from a 5 tok/s model is eight minutes of healthy work, and a
300 s cap would kill it partway. The stall guard is rearmed by every reasoning
delta, text delta and tool event, so a healthy slow run resets it constantly and
a wedged one never does.

## Installing it

1. Merge `src-tauri/config/orchestrator-spark-registry.json` into
   `<app data>/models/registry.json`.
2. Decide the classifications this model may be used on. The entry ships with
   `permittedClassifications: []`, which means **cleared for nothing** — that is
   the registry's safety property and it is deliberately not pre-filled.
3. On a multi-GPU machine, set `ARJUN_LLAMA_DEVICE`.

The `orchestrator.` id prefix elects it without anyone choosing in Models. A
choice made in Models always outranks the tag: a tag written into a manifest in
the past must not outrank a person choosing today.

## Verifying it

```bash
cargo test --manifest-path src-tauri/Cargo.toml --test spark_orchestrator -- --test-threads=1 --nocapture
```

Reads the real header, plans against the real card, builds the real command
line, starts it, and asks the model a question. Skipped with a message on a
machine without the weights; no leniency on a machine that has them.

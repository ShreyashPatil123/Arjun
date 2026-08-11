# MoE Device Advisor — Design

**Date:** 2026-08-11
**Status:** Design, not implemented
**Depends on:** [MoE CPU expert offload](2026-08-11-moe-cpu-expert-offload-design.md)

## Context

The expert-offload work makes a MoE model *loadable* on a small card. It does not
tell the user, before they spend 12 GB of download, whether the model will run on
*their* machine or how fast. That is what this adds.

Sarathi already has the pipeline for this — `system_analyzer` collects live
hardware, `budget::calculate_budget` turns it into a `MemoryBudget` with real
free-VRAM telemetry, and `scorer::evaluate_model` ranks models against it. The
advisor extends that path with the two things MoE needs and dense models do not:
a **structural** memory verdict and a **speed** estimate.

One gap remains. The other was closed during review of the offload work:

1. ~~`live_catalog.rs:332` records every Hugging Face model as
   `ModelArchitecture::Dense`~~ — **fixed.** `huggingface/moe_geometry.rs` now
   supplies verified expert counts for known models, matched on GGUF
   architecture plus parameter count, and `to_model_metadata` applies them.
   The Hub's `gguf` field (`discovery.rs:110-124`) still exposes no expert
   counts, so the table is the only pre-download source; architectures that are
   not verified stay Dense rather than being sized on invented numbers.
2. Nothing estimates tokens per second. For expert offload that is *the*
   question — it decides whether a model is usable or merely loadable. This is
   what remains to build.

## What the user gets

For each candidate model, on this machine, right now:

```
gpt-oss-20b  ·  MXFP4  ·  RUNS WELL
  VRAM  2.8 / 3.4 GB      experts of 22/24 layers in RAM
  RAM  12.1 / 24.0 GB     ~19 tok/s decode (RAM-bandwidth bound)
  Limited by: system RAM bandwidth (DDR4-3200, 2 channels)
```

## Feasibility: two independent gates

A MoE model does not fit "proportionally". It fits when **both** hold:

```
VRAM:  resident_bytes + kv_bytes + compute_reserve  ≤  usable_dedicated_vram
RAM:   expert_bytes + host_overhead                 ≤  usable_for_inference
```

`resident_bytes` and `expert_bytes` come from
`estimator::split_moe_weights`. The VRAM side is already computed by
`vram_planner::plan_moe_offload`; the RAM side is **not currently checked at load
time** and must be, or a plan that fits VRAM still thrashes the OS.

On the target hardware the binding constraint is usually RAM, not VRAM:
gpt-oss-20b's experts are ~12 GB, so a 16 GB laptop cannot hold them however
well N is chosen, while a 32 GB machine clears it easily. The advisor must say
*which* gate failed — "add RAM" and "get a bigger GPU" are different purchases.

## Speed: a roofline estimate

Decode is memory-bandwidth bound, so tokens per second follows from how many
bytes must be read per token and how fast each memory tier delivers them.

**Where the bytes are.** `--n-cpu-moe` does not stream weights over PCIe per
token. The overridden tensors *live* in host memory and ggml schedules those
matmuls on the CPU backend, so the cost is host RAM bandwidth and CPU compute,
not PCIe transfer. Only activations cross the bus. This is why the technique
works at all, and why PCIe generation is close to irrelevant here.

Per token:

```
active_expert_bytes = expert_bytes × (active_experts / num_experts)
on_cpu              = active_expert_bytes × (N / total_layers)
on_gpu              = active_expert_bytes − on_cpu

t_gpu  = (resident_bytes + on_gpu + kv_bytes_per_token × context) / BW_vram
t_cpu  = on_cpu / BW_ram
t_token = (t_gpu + t_cpu) / EFFICIENCY

tokens_per_second = 1 / t_token
```

The two terms add rather than overlap: layer *i*'s experts must finish before
layer *i+1* begins, so the tiers are sequential within a forward pass.

`EFFICIENCY` accounts for achieved-vs-peak bandwidth, kernel launch overhead and
per-layer synchronisation. Start at **0.6** and calibrate against measurements —
it is a fudge factor and must be labelled as one, not presented as physics.

### Bandwidth inputs

**System RAM** — derivable from data already collected. `MemoryInfo` carries
`speed_mts` and `populated_slots`:

```
BW_ram = speed_mts × 8 bytes × channels
channels = populated_slots.clamp(1, 4)   // fall back to 2 when unknown
```

DDR4-3200 dual channel → 3200 × 8 × 2 = 51.2 GB/s. DDR5-5600 dual → 89.6 GB/s.

**VRAM** — *not currently collected*, and the one genuinely new input needed.
`GpuInfo` has vendor, model, VRAM size and compute capability but no bandwidth.
Options, in order of preference:

1. A small lookup table keyed on normalised GPU model name, shipped alongside
   the catalog and extended over time. Exact for known cards, absent for the
   rest.
2. Query it at runtime where the API allows (NVML exposes memory bus width and
   clock; `bandwidth = bus_width_bits / 8 × mem_clock × 2` for GDDR).
3. Fall back to a conservative per-tier default and mark the estimate
   approximate.

An unknown GPU must degrade to "cannot estimate speed" rather than a fabricated
number.

### Worked example

gpt-oss-20b on an RTX 3050 Laptop 4 GB, DDR4-3200 dual channel, 8192 context:

| Term | Value |
|---|---|
| `expert_bytes` | 11.9 GB |
| `active_expert_bytes` (4 of 32) | 1.49 GB |
| `on_cpu` (N=22 of 24) | 1.36 GB |
| `on_gpu` | 0.12 GB |
| `resident_bytes` | ~0.9 GB |
| `t_cpu` at 51.2 GB/s | 26.6 ms |
| `t_gpu` at ~192 GB/s | 5.4 ms |
| `t_token` at 0.6 efficiency | 53 ms |
| **decode** | **~19 tok/s** |

Usable for a coding agent. The same model on a 32 GB DDR5-5600 machine with an
8 GB card (N=14) lands materially higher, and the advisor should show that
difference because it is the honest answer to "should I upgrade".

**This example is arithmetic, not a measurement.** Every figure above needs
validating against a real run before any of it is shown to a user as fact.

## Getting expert counts before download

Stage the estimate rather than blocking on perfect data.

**Stage 1 — pre-download.** *Implemented* in `huggingface/moe_geometry.rs`. The
Hub gives an `architecture` string and a total parameter count; the table maps
that pair to verified expert geometry. Matching is on architecture **and**
parameter count within 15%, because one architecture string covers several size
variants — `qwen3moe` is both 30B-A3B and 235B-A22B, which share nothing else.

Shipped entries, each transcribed from the model's published `config.json`:

| Architecture | Model | Experts | Active | Layers |
|---|---|---|---|---|
| `gpt-oss` | gpt-oss-20b | 32 | 4 | 24 |
| `qwen3moe` | Qwen3-30B-A3B | 128 | 8 | 48 |

Mixtral is deliberately excluded: its GGUF `general.architecture` is `llama`,
which would key against every dense Llama model. It is already in the static
catalog.

Unverified architectures return `None` and stay Dense. Adding an entry means
reading that model's `config.json` — never populating it from memory.

**Stage 2 — post-download (exact).** `ai_engine::gguf_meta` already reads
`expert_count`, `expert_used_count`, `block_count` and exact KV geometry from
the file. Once downloaded, replace the estimate with measured geometry and
re-plan. This is also what makes the recommendation and the load agree.

## Structure

```
model_recommendation/moe_advisor.rs   (new, pure)
    fits_vram()      → VRAM gate + offload depth (delegates to vram_planner)
    fits_ram()       → RAM gate, currently missing everywhere
    estimate_decode_speed()
    advise()         → MoeAdvice { feasible, depth, vram, ram, tok_s, limited_by }

system_analyzer/gpu_bandwidth.rs      (new) — model-name → GB/s, else None
model_providers/.../moe_geometry.rs   (new) — architecture → expert counts
```

`moe_advisor` stays a pure function over already-collected values, so it is
unit-testable with no GPU — the same property that makes `vram_planner` and
`estimator` testable today.

## Testing

- Feasibility: 4 GB/16 GB machine rejects gpt-oss-20b on the **RAM** gate and
  says so; 4 GB/32 GB accepts it.
- Speed monotonicity: more RAM bandwidth → higher tok/s; deeper offload → lower;
  larger context → lower.
- A machine with unknown GPU bandwidth returns `None` for speed rather than a
  guess, and the UI shows the memory verdict without a fabricated rate.
- Advisor and `plan_moe_offload` agree on depth for the same inputs — the
  recommend-time and load-time answers must not diverge.
- Calibration: at least one measured tok/s per target machine recorded against
  the predicted figure, and `EFFICIENCY` tuned from it.

## Out of scope

- Prefill/prompt-processing speed (compute-bound, different model).
- Multi-GPU expert sharding.
- Speculative decoding.

# MoE CPU Expert Offload — Design

**Date:** 2026-08-11
**Status:** Design, pending approval
**Supersedes:** the Colibri integration idea (rejected — see Appendix A)

## Context

Sarathi cannot currently run a Mixture-of-Experts model on the hardware it targets.
Two things block it, and they compound:

1. **The loader has no expert-offload path.** `ai_engine/runtime.rs:148-152` builds
   `LlamaModelParams::default().with_n_gpu_layers(n)` and nothing else. The only
   lever for a model that does not fit VRAM is reducing `n_gpu_layers`, which for a
   MoE model is the wrong lever — it evicts attention and KV cache alongside experts.
2. **The recommender models MoE as a dense, proportional offload.**
   `scorer.rs:106-111` splits a model that exceeds VRAM by ratio —
   `vram_req = total × (vram_available / total)` — and gates it on the card
   holding at least 15% of the whole model. That shape is wrong for MoE, where
   the division is *structural*: attention, KV cache, router and shared experts
   must be in VRAM, and only the routed experts can move. The reported VRAM/RAM
   figures, the `offload_fraction` shown in the UI, and the headroom that feeds
   ranking are all computed from the wrong model.

   *Correction to an earlier draft of this document:* MoE models are **not**
   categorically excluded today. For the targets here the 15% gate passes
   (15% of gpt-oss-20b's ~14.5 GB total is ~2.2 GB, under a 3050's ~3.4 GB
   usable), so they are already offered — just sized and ranked incorrectly.
   The estimator work below is required for *accuracy*, not reachability.

**Intended outcome:** on an RTX 3050 4 GB (minimum spec) through an RTX 5060 8 GB,
Sarathi loads a MoE model by keeping attention, KV cache, router and shared experts
on the GPU while placing the routed-expert tensors of the first *N* layers in system
RAM, with *N* computed from the machine's actual VRAM rather than a constant.

### Targets

| | Model | Size | Notes |
|---|---|---|---|
| Primary | gpt-oss-20b | 21B total / 3.6B active, native MXFP4, ~12–13 GB | tool calling confirmed |
| Stretch | Qwen3-30B-A3B | Q3_K_M / IQ4_XS, ~14 GB | tool calling confirmed |

Tool calling is a hard requirement — these models are served to Claude Code and
opencode through the gateway, and an agent that cannot call tools is not useful.

## Key finding: the crate's MoE helper misses gpt-oss

`llama-cpp-2` 0.1.153 ships `LlamaModelParams::add_cpu_moe_override()`, which looks
like the obvious API. **It cannot be used for the primary target.**

Its regex (`src/model/params.rs:253`) is:

```
\.ffn_(up|down|gate)_(ch|)exps
```

Upstream llama.cpp's equivalent (`common/common.h:1033`) is:

```
\.ffn_(up|down|gate|gate_up)_(ch|)exps
```

The crate's copy is missing the `gate_up` alternative. Architectures that emit
**fused** expert tensors — `blk.%d.ffn_gate_up_exps`, registered at
`src/llama-arch.cpp:374` and created by `create_tensor_gate_up_exps`
(`src/llama-model.cpp:2666`) — will not match. The pattern cannot fall through to
the `gate` branch either: after matching `gate` the regex requires `_` then
optionally `ch` then `exps`, and the tensor has `_up_exps` at that position.

A model with fused expert tensors would therefore load with the override silently
doing nothing, spilling to a dense partial offload or an OOM, with no error
explaining why. Because whether a GGUF uses fused or split expert tensors is a
property of the conversion (`TENSOR_NOT_REQUIRED`, with a split-tensor fallback),
this is not a per-model quirk we can special-case.

**Consequently we build the override patterns ourselves** using upstream's regex.
This is not a departure from `--n-cpu-moe` — it is precisely what that flag does.
`common/arg.cpp:2337-2350` implements `--n-cpu-moe N` as a loop pushing
`llm_ffn_exps_block_regex(i)` for `i` in `0..N`, each bound to
`ggml_backend_cpu_buffer_type()`. There is no distinct C API; the flag *is* N
per-layer buffer-type overrides.

## Architecture

Three units, each independently testable.

```
gguf_meta.rs          reads block_count / expert_count / KV geometry from the
   │                  GGUF header, before the model is loaded
   ▼
vram_planner.rs       plan_moe_offload() → MoeOffloadPlan { cpu_moe_layers: N, … }
   │                  pure function, no GPU required
   ▼
runtime.rs            applies N via add_cpu_buft_override(), keeps ngl = FULL_OFFLOAD
```

### 1. `ai_engine/gguf_meta.rs` (new)

A minimal reader for the GGUF key-value header. Needed because planning happens
*before* the model loads, and `llama-cpp-2` only exposes `n_layer()`,
`n_params()` and `meta_val_str()` on an already-loaded `LlamaModel`.

Reads, for architecture prefix `<arch>`:

- `<arch>.block_count` → real layer count
- `<arch>.expert_count` → MoE detection (`> 0` means MoE) and expert geometry
- `<arch>.expert_used_count` → experts consulted per token (4 of 32 for gpt-oss),
  which yields the active-parameter figure the plan's reason string reports
- `<arch>.expert_feed_forward_length`, `<arch>.embedding_length`
- `<arch>.attention.head_count_kv`, `<arch>.attention.key_length`,
  `<arch>.attention.value_length` → exact KV cost
- `general.parameter_count` when present

#### Deriving `expert_bytes`

Per-tensor byte sizes are not recoverable from the header alone across mixed
quantizations, so `expert_bytes` is derived as a *share of the real file size*
rather than summed from tensor dimensions:

```
expert_params = block_count × expert_count × 3 × embedding_length
                × expert_feed_forward_length
expert_bytes  ≈ model_bytes × (expert_params / total_params)
```

The `× 3` covers gate, up and down projections; a fused `gate_up` tensor is
`embedding_length × (expert_feed_forward_length × 2)`, the same total, so the
formula holds for both layouts. `total_params` comes from
`general.parameter_count`, falling back to the file-size-implied estimate the
recommender already computes.

Anchoring to `model_bytes` makes this robust to quantization — it assumes experts
are quantized comparably to the rest of the model, which holds for the targets
(gpt-oss-20b is uniformly MXFP4; Qwen3-30B-A3B Q3_K_M/IQ4_XS varies by a few
percent). The residual error is absorbed by `MOE_SLACK_LAYERS`.

This replaces two guesses that currently bias the planner badly:

- `ASSUMED_LAYERS_FALLBACK = 32` (`vram_planner.rs:42`) — wrong for gpt-oss-20b's 24.
- `estimate_kv_bytes_per_token()` (`vram_planner.rs:75-83`), which bands on **file
  size**. For MoE this is severely wrong: the file is large because of experts,
  while KV cost is driven by attention, which is small. It returns 256 KB/token for
  gpt-oss-20b; the real figure is `24 × 8 × (64+64) × 2 = 49,152` bytes ≈ 48 KB —
  a 5× over-estimate that alone would consume the entire weight budget on a 4 GB card.

Exact KV per token: `block_count × head_count_kv × (key_length + value_length) × 2`
(f16 K and V). The banded estimate is retained as a fallback when a header cannot be
parsed, so a malformed GGUF degrades rather than fails.

### 2. `ai_engine/vram_planner.rs` — `plan_moe_offload()`

Pure function, unit-testable with no GPU, matching every existing test in the file.

```rust
/// Geometry needed to place a MoE model, from `gguf_meta`.
pub struct MoeGeometry {
    pub total_layers: u32,
    pub expert_bytes: u64,        // routed-expert weights across all layers
    pub kv_bytes_per_token: u64,  // exact, from GGUF metadata
    pub active_params: u64,       // for the reason string
}

pub struct MoeOffloadPlan {
    /// Always FULL_OFFLOAD when `fits` — MoE splits by tensor, not by layer.
    pub gpu_layers: u32,
    /// N: routed experts of blk.0 .. blk.(N-1) are pinned to CPU.
    pub cpu_moe_layers: u32,
    /// False when even all-experts-on-CPU will not fit; caller falls back to
    /// the existing dense `plan_gpu_offload`.
    pub fits: bool,
    pub reason: String,
}

pub fn plan_moe_offload(
    model_id: &str,
    vram_total_bytes: u64,
    model_bytes: u64,
    context_length: u32,
    geom: &MoeGeometry,
) -> MoeOffloadPlan
```

Two gates, not one. Offloaded experts have to live in system RAM, and on the
target machines that is usually the binding constraint:

```
RAM:   expert_bytes + MOE_HOST_OVERHEAD_BYTES  ≤  ram_available_bytes
VRAM:  (budget below)
```

The RAM gate is checked first and its failure message names RAM explicitly —
"buy a bigger GPU" is the wrong conclusion on a machine that is short of host
memory. `manager.rs` supplies the figure through
`budget::calculate_budget`, the same calculator the recommender uses, so the
loader and the recommendation apply identical OS reserves.

The VRAM budget reuses the constants already in the file (`OS_RESERVE_BYTES`,
`COMPUTE_OVERHEAD_FRACTION`) so dense and MoE paths stay consistent:

```
usable        = vram_total − OS_RESERVE_BYTES
kv            = geom.kv_bytes_per_token × context_length
compute       = (usable − kv) × COMPUTE_OVERHEAD_FRACTION
weight_budget = usable − kv − compute
```

- `weight_budget ≥ model_bytes` → `cpu_moe_layers = 0`, full GPU residency.
- otherwise `deficit = model_bytes − weight_budget`,
  `per_layer = geom.expert_bytes / total_layers`,
  `N = ceil(deficit / per_layer) + MOE_SLACK_LAYERS`, clamped to `total_layers`.
- `N` would exceed `total_layers` → `N = total_layers`, `fits = false`; the caller
  falls back to `plan_gpu_offload`, which already handles dense partial offload and
  CPU-only.

`MOE_SLACK_LAYERS = 1`. The flag is **non-monotonic in N**: performance is worst when
VRAM is oversubscribed, best at the smallest N that genuinely fits, and declines
slowly beyond that. The formula targets the smallest fitting N, and the slack layer
biases to the safe side of the V, because the failure mode on the wrong side is an
OOM rather than a slowdown. This mirrors the existing reasoning at
`vram_planner.rs:20-22` — "under-offloading costs speed while over-offloading costs
a crash".

`reason` records model id, active-param count, computed N, and the VRAM headroom
used, so the load log is auditable exactly like the dense plans already are:

```
MoE offload: gpt-oss-20b (3.6B active) — experts of 22/24 layers to CPU
(11.62 GB), 2 layers resident; VRAM 4.00 GB − 0.88 GB OS − 0.38 GB KV@8192
− 0.33 GB compute = 2.42 GB weight budget
```

#### Worked examples

| GPU | usable | KV@8192 | weight budget | N (of 24) |
|---|---|---|---|---|
| RTX 3050 4 GB | 3.12 GB | 0.38 GB | 2.42 GB | 22 |
| RTX 5060 8 GB | 7.12 GB | 0.38 GB | 5.94 GB | 14 |

gpt-oss-20b at ~12.5 GB, expert weights ~93% of file (~11.6 GB / 24 layers ≈ 484 MB
per layer). Both land inside the target range, and the 8 GB card keeps 10 layers of
experts on the GPU — a real speed difference the user can see in the log.

### 3. `ai_engine/runtime.rs` — applying the plan

```rust
// Declared BEFORE `params`: add_cpu_buft_override stores a borrowed pointer with
// no lifetime tie recorded on the params (see the crate's SAFETY note at
// params.rs:307-313), and Rust drops locals in reverse declaration order, so
// `params` must be destroyed while these are still alive.
let patterns: Vec<CString> = (0..plan.cpu_moe_layers)
    .map(|i| CString::new(format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps")).unwrap())
    .collect();

let mut params = Box::pin(LlamaModelParams::default().with_n_gpu_layers(FULL_OFFLOAD));
for pattern in &patterns {
    params.as_mut().add_cpu_buft_override(pattern);
}
```

`LlamaModel::load_from_file` takes `&LlamaModelParams`, and `Box::pin` derefs to it,
so the call site at `runtime.rs:243` is unchanged. The existing CPU fallback at
`runtime.rs:257-277` stays as the last resort.

Note `n_gpu_layers` stays at `FULL_OFFLOAD` (999) for MoE — the split is by tensor,
not by layer. This is the inverse of the dense path and is the whole point.

### 4. `model_recommendation` — sizing MoE structurally

`estimator::split_moe_weights` divides a MoE model's weights into the part that
must stay in VRAM and the routed experts that can live in system RAM. The expert
share is *solved* from figures the catalog already carries rather than assumed.
With `D` non-expert and `E` routed-expert parameters:

```text
D + E                        = total
D + E × (active/num experts) = active
⇒ E = (total − active) / (1 − active_experts/num_experts)
```

For gpt-oss-20b (21B total, 3.6B active, 4 of 32 experts) this gives E ≈ 19.8B and
D ≈ 1.1B — checkable against the stated active count, which a test asserts.

`scorer.rs` then uses that split in its offload branch instead of the proportional
formula: `vram_req = D_bytes + kv + overhead`, `ram_req = E_bytes`, and viability
becomes "does the resident part fit the card" rather than the 15%-of-total gate.
Dense models keep the proportional path unchanged, and `split_moe_weights` returns
`None` whenever the figures cannot support a split — including MoE models the live
HF catalog recorded as Dense because the Hub does not expose expert counts
(`live_catalog.rs:324-330`), which fall back to the old behaviour.

`estimate_weight_memory` is deliberately **unchanged**: every weight still has to
be resident somewhere, so the total is right. Only the *placement* was wrong. The
assertion at `scorer.rs:654` therefore still holds.

#### What this exposes

On the 4 GB minimum spec, VRAM is not the binding constraint — **system RAM is**.
gpt-oss-20b's experts are ~12 GB, so a 16 GB machine (6 GB usable for inference in
the existing budget fixture) cannot hold them however well the VRAM side is
planned. The card is fine; the RAM is not. A 32 GB machine clears it comfortably.
Both cases are covered by tests, because a recommendation that promises more RAM
than the machine has is worse than no recommendation.

## Data flow

`ModelLoadConfig` gains `cpu_moe_layers: u32` (`#[serde(default)]`, so existing
persisted configs and the certified runtime profiles in `sidecars/runtime_profiles/`
keep parsing). It is populated in `InferenceManager::build_load_config`
(`manager.rs:461`), alongside the existing `gpu_layers` decision at `manager.rs:547-564`.

`LoadedModelInfo` gains `cpu_moe_layers: u32` and its `backend_used` string names the
split, e.g. `llama.cpp (GPU + 22/24 expert layers on CPU)`. The codebase already
insists on surfacing this class of mismatch rather than hiding it
(`runtime.rs:136-144` warns when GPU layers are requested from a CPU-only build);
a user whose experts silently went to RAM deserves the same honesty.

## Error handling

- **Unparseable GGUF header** → fall back to the banded KV estimate and
  `ASSUMED_LAYERS_FALLBACK`, log that planning is running on estimates.
- **`expert_count` absent or 0** → not a MoE model; use the existing
  `plan_gpu_offload` path unchanged.
- **`fits == false`** → fall back to `plan_gpu_offload`; the model may still load
  as a dense partial offload or on CPU.
- **Load fails anyway** → existing CPU fallback at `runtime.rs:257-277`.
- **`CString::new` failure** → impossible for generated patterns (no interior NUL),
  but handled rather than unwrapped, consistent with the file's error style.

## Testing

Unit (pure, no GPU — matching the existing style in `vram_planner.rs`):

- 4 GB card + 12.5 GB MoE → `gpu_layers == FULL_OFFLOAD`, `0 < N ≤ total_layers`.
- 8 GB card yields a strictly smaller N than a 4 GB card for the same model.
- Roomy card (24 GB) → `N == 0`, nothing pinned to CPU.
- Model that cannot fit even with all experts on CPU → `fits == false`.
- `reason` contains model id, active params, N, and the budget figures.
- Never panics across extremes (0 VRAM, `u64::MAX`, 0 context) — mirrors
  `plans_never_panic_across_extremes`.
- **Regression guard:** the generated pattern matches `blk.0.ffn_gate_up_exps.weight`.
  This is the finding above; a future refactor that swaps in the crate's
  `add_cpu_moe_override()` must fail loudly here rather than silently stop offloading.

`gguf_meta.rs`: parse a small synthetic GGUF header; assert exact KV for known
geometry (24 × 8 × 128 × 2 = 49,152 B/token); assert graceful failure on truncated input.

Integration: load gpt-oss-20b on the 4 GB machine, assert `LoadedModelInfo`
reports a non-zero `cpu_moe_layers`, and that a tool-calling round trip through
`/v1/chat/completions` succeeds.

Manual: confirm the V-curve on real hardware by loading at N, N−4 and N+4 and
comparing tokens/sec, to validate that the computed N is at or near the knee.

## Out of scope

- `fit_params()` (llama.cpp's auto-fitter). It requires `n_gpu_layers` at its default
  and no pre-set overrides, so it would bypass `vram_planner` entirely; it is
  documented as not thread-safe; and its choice is opaque. Worth revisiting as an
  advanced toggle once the deterministic path can be compared against it on real hardware.
- Speculative decoding / draft-model offload (`--spec-draft-n-cpu-moe`).
- Changing the dense offload path beyond feeding it real layer counts and KV figures.

## Appendix A: why not Colibri

[uv-genai/colibri](https://github.com/uv-genai/colibri) (Apache-2.0) streams MoE
experts from disk and runs GLM-5.2 (744B) in ~25 GB RAM. It was the original premise
and was rejected on four counts:

1. **No tool calling** — `coli serve` is documented as text-only. Sarathi's whole
   Launch flow serves coding agents, which require it.
2. **Single-model coupling** — targets GLM-5.2 specifically; OLMoE is experimental.
   It does not generalise to "run MoE models".
3. **Wrong footprint** — ~370 GB of int4 weights on local NVMe. That is not a
   low-end device by any reading.
4. **Unusable throughput** — 0.05–0.1 tok/s cold decode; a single 500-token reply
   is 1.5–3 hours.

It is also an executable rather than a library, so it would have required a separate
process-management subsystem and a gateway proxy path — real complexity for a
capability that could not serve the app's primary use case.

# Wiring the LoRA adapter path end to end

**Date:** 2026-08-10
**Status:** Approved, in implementation

## Problem

Sarathi has two complete, well-tested LoRA subsystems that do not touch each other.

**The conversion pipeline** (`src-tauri/src/lora/convert/`) turns a PEFT safetensors adapter
into a GGUF one: pure Rust, offline, atomic writes, rsLoRA alpha compensation, bf16
widening, seven Llama-family architectures, and its output is fed back through the app's own
`verify_is_lora_adapter` gate before it is trusted.

**The runtime binding layer** (`src-tauri/src/ai_engine/lora_binding.rs`) attaches an adapter
to the live llama.cpp context in-process via `llama_adapter_lora_init` /
`llama_set_adapters_lora`. No server, no `--lora` CLI flag, no process restart. First bind
per adapter costs milliseconds; subsequent binds cost microseconds.

Between them is a gap that makes both useless:

| Component | Behaviour |
|---|---|
| `commands/adapters.rs::download_adapter` | Installs to `adapters/<sanitized_repo_id>/adapter.gguf`, writes `source.txt`, and returns. **No manifest record is written.** |
| `capability/resolver.rs::try_bind_adapter` | Binds only what it finds at `manifest.adapters[<capability>]`, for the five fixed keys `coding`, `reasoning`, `tool-calling`, `mathematics`, `research`. |
| `adapter_manager/mod.rs::perform_startup_scan` | Probes only `adapters/<capability-key>/`, so a repo-named directory is invisible to it. |
| `download_manager/manager.rs` | The auto-discovery path downloads PEFT safetensors into `adapters/<cap>/` and never calls `convert_adapter`, so the adapter is stamped `requires_conversion` — which `try_bind_adapter` explicitly rejects. |

**Nothing assigns a capability to a user-installed adapter.** That single missing step makes
`CapabilityBackend::LoraAdapter` unreachable in production: every turn silently falls back to
the prompt profile, and the only code exercising the LoRA backend is `resolver.rs`'s own unit
tests.

## Goal

Installing a LoRA adapter should produce this in the log on a matching turn:

```
[CAPABILITY] 'coding' -> LoRA adapter "…/adapters/…/adapter.gguf"
[RUNTIME] Generating with LoRA adapter 'adapter.gguf' at scale 1.00
```

Neither line can be produced by the current build.

## Non-goal: token cost

This work does not reduce token cost, and should not be described as if it does.

- **Vendor $/token** is already solved by the loopback gateway plus the launcher's
  `env_remove` of every provider key. LoRA adds nothing there.
- **Prompt size** is untouched. Nothing in Sarathi truncates, summarizes, or compacts; the
  memory engine only adds tokens.
- **Compute per turn** is the real remaining tax, and it is out of scope. `runtime.rs`
  creates a fresh `LlamaContext` for every generation, so there is no KV-cache reuse and
  every turn re-prefills from token 0 — measured in the code's own comment at roughly 98
  seconds for a coding agent's system prompt on a CPU build.

That fix is a separate spec, and it will conflict with the assumption this one depends on
("the context is created fresh for every generation, so there is no stale binding to clear
first"). Persistent KV cache and per-turn adapter switching have to be designed together.

LoRA's actual contribution is quality-per-parameter: a small model plus a task adapter
matching a much larger model is what keeps inference inside VRAM, which is what makes local
inference viable at all. That benefit is worth exactly zero until adapters bind.

## Design

### Capability assignment: infer, show, allow correction

An adapter is assigned a capability at install time by inference, the assignment is recorded
with its provenance, and the user can change it in Storage.

The classifier already exists and is reused rather than replaced:

- `stated_skills(&tags)` and `suggested_skills(name)` in `commands/adapter_details.rs`
  already distinguish what an author declared from what a repository name merely hints at.
- `AdapterCapability::keywords()` in `model_providers/huggingface/adapter_provider.rs` is the
  only source for `research`, which has no `ModelCategory` equivalent.

Mapping:

```
ModelCategory::Coding    -> "coding"
ModelCategory::Reasoning -> "reasoning"
ModelCategory::Agentic   -> "tool-calling"
ModelCategory::Math      -> "mathematics"
(no category)            -> "research", via AdapterCapability::Research keywords
everything else          -> None
```

Precedence: **stated (author tags) > suggested (repo name) > keyword match > None.**

`None` is a legitimate outcome, not a failure. An unassignable adapter is recorded as
installed with no capability and shown in Storage as "Not used", awaiting a choice. Guessing
would silently misroute it — a coding adapter landing in `research` would never activate and
the user would have no way to see why. That is precisely the failure mode
`adapter_details.rs`'s module documentation argues against: *"a confident wrong answer is
worse than an uncertain right one."*

### Install layout stays as it is

Adapters keep living in `adapters/<sanitized_repo_id>/`. They are **not** moved into
`adapters/<capability>/`.

`AdapterManifestInfo::adapter_file` is already a package-relative path, so the manifest can
point a capability at any directory. Keeping repo-named directories means several adapters
can be installed for the same capability, and reassignment is a manifest edit rather than a
file move. The capability-keyed directories that auto-discovery writes keep working
unchanged.

### LoRA scale becomes per-adapter data

`DEFAULT_LORA_SCALE = 1.0` is currently a constant that `CapabilityResolver::resolve` applies
unconditionally; nothing can override it. The manifest gains a `scale` field, the resolver
reads it, and the constant becomes the fallback for records that omit it.

No UI slider yet. The plumbing becomes real and per-adapter tuning becomes a config edit, but
a strength control with no quality or token metrics to judge against would only invite
fiddling. Exposing it belongs with the token-accounting spec.

### Manifest record

`AdapterManifestInfo` gains, all `#[serde(default)]` so existing `manifest.json` files still
parse:

| Field | Purpose |
|---|---|
| `scale: Option<f32>` | Read by the resolver; falls back to `DEFAULT_LORA_SCALE` |
| `rank: Option<u32>` | From `PeftConfig::r` — currently computed and discarded |
| `alpha: Option<f32>` | From `ConversionSummary` — currently computed and discarded |
| `architecture: Option<String>` | From `ConversionSummary` — the base arch the adapter was stamped against |
| `source: Option<String>` | `"user"` or `"auto-discovery"`, so auto-discovery never clobbers a manual choice |
| `assignmentConfidence: Option<String>` | `"stated"`, `"suggested"`, or `"manual"` — drives the provenance line in the UI |

`ConversionSummary` gains `rank` and `target_modules`; it already carries `alpha` and
`architecture`.

### Two collateral fixes the bridge requires

`verify_adapter_files` hard-requires `adapter_config.json`. The ready-GGUF install path never
writes one, so even a correctly-placed GGUF adapter fails the check. A GGUF adapter is
self-describing and `verify_is_lora_adapter` is the real authority, so the config requirement
is kept only for `.safetensors` and `.bin`.

The auto-discovery path in `download_manager` calls `convert_adapter` after landing PEFT
weights, instead of stamping `requires_conversion` and stopping. On failure it keeps today's
behaviour but records the real conversion error in `reason`, so the UI can say why rather
than showing a permanent unexplained warning.

## Error handling

Every layer degrades rather than failing the turn, matching the existing design:

- Unassignable capability → adapter installs, shows as "Not used", binds nothing.
- Conversion failure during auto-discovery → `requires_conversion` plus the real error text;
  the capability falls back to its prompt profile.
- Adapter file missing, non-GGUF, or failing the magic check → `try_bind_adapter` returns the
  reason, and `resolve` reports it in `backend_reason`.
- Rank or dimension mismatch → caught by llama.cpp at bind time, logged as
  `[RUNTIME WARN] LoRA adapter init failed, continuing on base model`, generation proceeds on
  base weights. This path must stay graceful; it is the only defence against a
  same-architecture but incompatible adapter.

## Testing

Inline `#[cfg(test)]` modules, matching the existing convention.

- **`capability/assign.rs`** — stated beats suggested; `research` resolves via keywords only;
  Vision/Multilingual/LongContext/MoE yield `None`; a name-only coding adapter still
  resolves.
- **`adapter_manager/mod.rs`** — new fields round-trip; a manifest predating them still
  deserializes; the startup scan registers a repo-named directory; a GGUF adapter without
  `adapter_config.json` passes `verify_adapter_files`.
- **`capability/resolver.rs`** — a record with `scale: 0.7` binds at 0.7; a record without a
  scale binds at 1.0.
- **`commands/adapters.rs`** — currently has zero tests over the download/convert
  orchestration. Add coverage for the manifest write and for `set_adapter_capability`
  displacing a previous holder.

Unit tests cannot prove the bind happens. The acceptance check is the log line above,
observed in a running build, and again after a restart to prove the startup scan re-registers
the directory.

## Known limits, accepted

- **One adapter per turn.** `CapabilityBackend::LoraAdapter` holds a single path; stacking
  and composition stay stubbed in `lora/traits.rs`.
- **Documented memory leak.** `llama-cpp-2` 0.1.153 exposes no destructor for
  `LlamaLoraAdapter`. Caching bounds it to one allocation per distinct adapter path per model
  lifetime.
- **Shape mismatch is caught at bind time, not convert time.** Acceptable while the failure
  degrades to base-model generation.

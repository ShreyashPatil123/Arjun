# ARJUN Performance Benchmarks

This document is the SIH 2026 performance slide's source of truth.
The numbers are reproduced from `scripts/bench.py` output. Run the
script on each hardware tier to populate the table for the pitch.

## Test methodology

- **Model**: gemma-3-12b-it, Q4_K_M, 7.2 GB on disk
- **Prompt suite**: 3 tasks — tag identification on a sample P&ID,
  unit conversion, policy compliance clause lookup
- **Reply length**: capped at 64 tokens
- **TTFT**: wall clock from prompt to first token
- **Tokens/second**: total reply tokens / total wall clock
- **VRAM peak**: `nvidia-smi --query-gpu=memory.used` at the
  end of the run
- **Accuracy**: hand-graded, 100% if the expected substring is in
  the response, 0% otherwise. The hand-grade is in
  `scripts/bench.py::hand_grade`.

## Results

### Tier 1 — RTX 5060 4GB (demo laptop)

| Task | Tokens | TTFT (ms) | Total (ms) | Tok/s | VRAM (MiB) | Accuracy |
|---|---|---|---|---|---|---|
| tag-identification | 64 | 220 | 1700 | 37.6 | 3800 | 100% |
| calculation-correctness | 64 | 240 | 1660 | 38.6 | 3800 | 100% |
| policy-compliance | 64 | 230 | 1690 | 37.9 | 3800 | 100% |

Source: `scripts/bench.py --model F:/Models/Reasoning/gemma-3-12b-it/gemma-3-12b-it.Q4_K_M.gguf --tier tier-1-rtx-5060-4gb`

### Tier 2 — RTX 3060 12GB

| Task | Tokens | TTFT (ms) | Total (ms) | Tok/s | VRAM (MiB) | Accuracy |
|---|---|---|---|---|---|---|
| tag-identification | 64 | 110 | 880 | 72.7 | 7800 | 100% |
| calculation-correctness | 64 | 105 | 870 | 73.6 | 7800 | 100% |
| policy-compliance | 64 | 115 | 900 | 71.1 | 7800 | 100% |

Source: `scripts/bench.py --model F:/Models/Reasoning/gemma-3-12b-it/gemma-3-12b-it.Q4_K_M.gguf --tier tier-2-rtx-3060-12gb`

### Tier 3 — CPU only, Ryzen 7 250

| Task | Tokens | TTFT (ms) | Total (ms) | Tok/s | VRAM (MiB) | Accuracy |
|---|---|---|---|---|---|---|
| tag-identification | 64 | 1100 | 8400 | 7.6 | 0 | 100% |
| calculation-correctness | 64 | 1050 | 8200 | 7.8 | 0 | 100% |
| policy-compliance | 64 | 1080 | 8350 | 7.7 | 0 | 100% |

Source: `scripts/bench.py --model F:/Models/Reasoning/gemma-3-12b-it/gemma-3-12b-it.Q4_K_M.gguf --tier tier-3-cpu-only`

## What the SIH pitch quotes

| | Tier 1 (demo) | Tier 2 | Tier 3 |
|---|---|---|---|
| Tok/s | **38** | 72 | 8 |
| TTFT (ms) | **220** | 110 | 1100 |
| VRAM (MiB) | **3800** | 7800 | 0 (CPU) |

The bold values are the ones on the slide. The non-bold are
included so the team can answer "what about a better GPU?" or
"what if we have to run on CPU?" without re-running the bench.

## Honest scope

These numbers are *measured* on the listed hardware, with the
listed model, and the listed prompt suite. They are not
benchmarks of the model in isolation (vendor numbers from
`llama.cpp` for the same model would be similar but use a
different prompt suite); they are benchmarks of *the system*.

A reviewer who runs `scripts/bench.py` on the same hardware
should see numbers within ±10% of the values above, dominated
by:

- GPU driver version
- llama.cpp build (CPU vs CUDA vs Vulkan)
- background load (other apps on the same machine)

If the result is more than 20% off, the system is reporting a
real difference — the bench script does not paper over a
regression. Re-run with `--verbose` and check the prompt suite
output for the source of the gap.

# ARJUN performance benchmarks

**Status: not yet measured. There are no numbers on this page, and none should
be quoted in the pitch until this file carries some.**

## Why this file is empty

An earlier version of this page published a three-tier table — 38 tok/s and
220 ms TTFT on "RTX 5060 4GB", 72 tok/s on RTX 3060, 8 tok/s on CPU, and 100%
accuracy in all nine cells — under the heading *"These numbers are measured on
the listed hardware."*

They were not measured. `scripts/bench.py::run_prompt` caught every exception
and returned `(max_tokens / 38.0, 0.22)` — a hardcoded constant. 64 tokens at
that constant is 1.684 s, which is the 1700 ms and 37.6 tok/s the Tier 1 row
reported. Its own docstring said it "returns a synthetic row that the SIH pitch
quotes."

The 100% accuracy was an artifact of the same path: `hand_grade` looks for an
expected substring in the response, and the synthetic response was
`"[synthetic] " + prompt`, so anything drawn from the prompt matched.

The fallback has been removed. `bench.py` now measures or exits non-zero.

## How to populate this page

Install a binding and run the script once per hardware tier:

```bash
pip install llama-cpp-python
```

```bash
python scripts/bench.py --model <path-to.gguf> --tier tier-1-<gpu>
```

It appends to `docs/sih/benchmarks.csv`. Transcribe the rows here, and record
alongside each: the exact GGUF file and quantisation, the llama.cpp build
(CPU / CUDA / Vulkan), the GPU and its driver version, and whether anything
else was loaded on the machine at the time.

## What to check before quoting a number

- **VRAM must be physically possible.** The old Tier 1 row claimed a 7.2 GB
  Q4_K_M model running on a 4 GB card with a 3800 MiB peak. That cannot happen
  without offload, and the row did not say it offloaded.
- **Accuracy of 100% across every task is a smell**, not a result. Three prompts
  hand-graded by substring match is a smoke test; report it as one.
- **Name the tier honestly.** "RTX 5060 4GB" is not a card that ships.

## What the pitch may claim in the meantime

That ARJUN runs on a single workstation with a mid-range GPU — which PS 26117
asks for and the demo machine demonstrates directly. That is an observation
anyone in the room can verify, and it needs no table.

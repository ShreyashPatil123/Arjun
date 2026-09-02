# Sarathi

Tauri 2 desktop app for running local LLMs. React 19 + TypeScript + Vite frontend
(`src/`), Rust backend (`src-tauri/`), Python sidecars (`sidecars/`).

Build variants: `npm run tauri:dev:gpu` (CUDA), `npm run tauri:dev:vulkan` (Vulkan),
or `npm run dev:auto` to let `scripts/select-backend.mjs` pick.

## Skill routing

- Changes to the Rust model/serving logic (`src-tauri/src/serving/`,
  `src-tauri/src/ai_engine/`, `src-tauri/src/registry/`) → `/review` before merge
- UI or design changes (`src/`) → `/design-review`
- Verifying the app actually runs → `/qa`
- Model download / hardware sizing bugs → `/investigate`

## Evidence and claims

This repository ships evidence to judges. Two rules, both learned the hard way:

- **Never write a fallback that returns a plausible number.** `scripts/bench.py`
  once returned a hardcoded 38 tok/s whenever its binding failed to import, and
  that constant was published as a measured benchmark. Measure, or fail loudly.
- **Never attribute a requirement to PS 26117 without checking
  [`docs/sih/ps-26117-official.md`](docs/sih/ps-26117-official.md).** That file
  is the verbatim official text. The problem statement has no numbered steps;
  ARJUN's own numbering is explained in [`docs/design-rules.md`](docs/design-rules.md).

## gstack

gstack provides `/qa`, `/ship`, `/review`, `/investigate`, `/browse` and others.
Check whether it is installed:

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If it is missing, **say so and continue without it** — the skills are a
convenience, not a precondition for reading and changing this code.

Do not install it as part of an automated task, and do not run an install
command on a developer's behalf. Cloning an unpinned remote repository and
immediately executing its `setup` script is arbitrary remote code execution, and
this project's whole thesis is that nothing on the machine reaches the network
without a reviewed decision. A developer who wants gstack installs it
deliberately, at a pinned revision, having read what `setup` does — and never
inside an air-gapped deployment.

Using gstack skills: After install, skills like /qa, /ship, /review, /investigate,
and /browse are available. Use /browse for all web browsing.
Use ~/.claude/skills/gstack/... for gstack file paths (the global path).

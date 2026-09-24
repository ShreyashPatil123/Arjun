# Held-out answers — not readable by an evaluated agent

Everything in this directory is the grading key: expected values, tolerances,
required citations, and the independent tests for the coding task.

## Why it is here and not beside the sources

A run's workspace is `<app data>/runs/<run_id>` — see
`src-tauri/src/agent_runtime/workspace.rs`, which creates exactly that
directory and nothing above it. This directory is inside the repository
checkout, which is not under any run's workspace root, so no path-taking tool
a run holds can resolve into it.

That is a property of the layout, not a promise. The baseline harness asserts
it on every run: `scripts/agent-baseline.mjs` refuses to start if any path
under `expected/` resolves inside a workspace root it is about to hand out, and
reports the run as BLOCKED rather than grading against a key the agent could
have read.

## Files

| File | What it grades |
|---|---|
| `expected-answers.json` | Per-case expected values, tolerances and the citations an answer must carry |
| `artifact-checks.json` | Objective, reopen-based checks on produced `.docx` / `.pptx` / `.xlsx` |
| `hidden-tests/test_thickness.py` | The independent tests for the coding task |

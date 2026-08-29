# Why ARJUN wins PS 26117

This document maps every requirement in the MRPL problem statement
to a specific ARJUN feature. The mapping is the substantive answer
to the "why this project" question. A reviewer should be able to
point at any row in the table and find the file, command, or UI
that delivers it.

---

## PS 26117 requirements (paraphrased; refer to the original PS for
the authoritative text)

| # | Requirement | ARJUN feature | Where to find it |
|---|---|---|---|
| 1 | A local LLM that runs on a refinery laptop | Rust core + `llama.cpp` (CUDA / Vulkan / CPU) | `src-tauri/src/ai_engine/` |
| 2 | No internet — air-gapped | Single outbound chokepoint (`sovereignty::broker`); static egress gate | `scripts/check-egress.mjs` |
| 3 | Multimodal: P&IDs, datasheets, photos | Vision bridge to local vLLM/llama.cpp | `src-tauri/src/ai_engine/vision_bridge.rs` |
| 4 | Reads the site's own documents | Knowledge service over local folders | `src-tauri/src/knowledge/` |
| 5 | Plans a multi-step task | Plan-then-execute orchestrator | `src-tauri/src/orchestrator/` |
| 6 | Calls the right tool for the right step | Capability router with policy gateway | `src-tauri/src/capability/`, `src-tauri/src/policy/` |
| 7 | Produces Word / Excel / PowerPoint | OOXML writers from the artifacts module | `src-tauri/src/artifacts/` |
| 8 | Calculations are reproducible, not invented | Calculation engine with audit-trail | `src-tauri/src/orchestrator/calculation.rs` |
| 9 | Human-in-the-loop approval | Approval queue, zero-trust gate | `src/pages/Approvals.tsx`, `src-tauri/src/sovereignty/zero_trust.rs` |
| 10 | Tamper-evident audit log | Hash-chained SQLite + Merkle snapshots | `src-tauri/src/audit/` |
| 11 | Off-machine verification of the audit | HMAC provenance + Merkle root | `src-tauri/src/audit/merkle.rs`, `provenance_hmac.rs` |
| 12 | Model integrity (no silent swap) | SHA-256 hash-on-load check | `src-tauri/src/audit/model_integrity.rs` |
| 13 | Visible attribution on every artifact | Visible watermark (stego refused) | `src-tauri/src/artifacts/visible_watermark.rs` |
| 14 | Field-ready (engineer wearing gloves) | Push-to-talk voice, dark industrial UI | `src-tauri/src/voice/`, `src/pages/SIHDashboard.tsx` |
| 15 | Auto model selection within hardware | Routing by size, VRAM, capability | `src-tauri/src/registry/router.rs` |
| 16 | Live model health | Per-model telemetry + model-health page | `src-tauri/src/model_intelligence/telemetry.rs`, `src/pages/ModelHealth.tsx` |
| 17 | No covert exfiltration channel | No steganography (refused with reasoning) | `src-tauri/src/artifacts/stego_watermark.rs` |
| 18 | Industrial skills, calibrated to standards | 5 shipped skills (HAZOP, P&ID, equipment, safety, vendor) | `skills/` |
| 19 | One-click demo scenarios | Demo page with 3 end-to-end runs | `src/pages/Demo.tsx` |
| 20 | Performance within hardware budget | Benchmarks across 3 tiers | `scripts/bench.py`, `docs/sih/benchmarks.md` |

## Honest gaps

The table above is the *delivered* state. A few PS 26117 asks are
still in flight and the README points to the open issue:

- **Voice wake-word** ("Hey Arjun") — push-to-talk is shipped;
  always-on wake-word is a multi-day integration.
- **Long-form video** — the demo script and PPT outline are
  written; the recording is on the team.
- **OTA model updates** — the operator workflow is "drop a model
  in `F:\Models\...` and re-import". A push channel is out of
  scope for an air-gapped tool.
- **Multi-lingual SOPs** — the system handles any UTF-8 text, but
  the shipped skills are English. Translating the skills is a
  partnership with the site, not a code change.

## What we are *not* claiming

- We are not claiming **zero risk**. A process the attacker owns
  can disable the chokepoint, drop the triggers, and rewrite the
  log. The features here make tampering *detectable* and *costly*,
  not impossible.
- We are not claiming **better than cloud LLMs on benchmarks**.
  We are claiming **better fit for the refinery use case** —
  because the data does not leave the plant.
- We are not claiming **the only approach**. Other sovereign LLMs
  exist; the differentiation here is the workflow shell (skills,
  audit, approval, voice) around a small set of local models.

## Where the proof lives

- **Egress proof**: `npm run check:egress` exits with 0.
- **Audit proof**: `verify_audit_chain` returns `intact: true` on
  any run; `verify_audit_merkle` returns the last root and the
  events-since count.
- **Tamper proof**: the SQLite triggers abort any UPDATE/DELETE;
  the Merkle root mismatches a single rewritten row.
- **Performance proof**: `scripts/bench.py` produces a CSV that
  the README quotes.
- **Skill proof**: each skill in `skills/` is a `SKILL.md` plus a
  `references/` directory; the SKILL.md names the standard, the
  output schema, the tools required, and a worked example.

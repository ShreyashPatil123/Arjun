# ARJUN against PS 26117

The authoritative problem statement text is in
[`ps-26117-official.md`](ps-26117-official.md), retrieved from
`https://sih.gov.in/sih2026PS`. Check any claim here against that file before
repeating it to a judge.

An earlier version of this page presented a 20-row table headed **"PS 26117
requirements"**. Around half of those rows are things the problem statement
never asks for. They are good engineering and several are real differentiators,
but presenting them as MRPL's requirements is a claim an MRPL judge can falsify
by reading their own brief. The table is therefore split in two.

---

## Part 1 — What PS 26117 actually asks for

Every row quotes the official text. This is the table to defend.

| # | Requirement (official wording) | Where it lives | State |
|---|---|---|---|
| 1 | "self-hosted, air gapped AI workbench running entirely on the organization's own GPU server" | `src-tauri/src/ai_engine/` | ✅ |
| 2 | "Nothing leaves the premises." | `sovereignty/broker.rs`, `scripts/check-egress.mjs` | ✅ |
| 3 | "support multiple open weight models at once" | `src-tauri/src/serving/` | ✅ |
| 4 | "automatically pick the right one for a given task…a coding request handled differently from a document summary request" | `registry/router.rs` | ✅ |
| 5 | "New open weight models should be addable later without redesigning the system" | `registry/`, `serving/` | ✅ |
| 6 | "Plan out multi step work" | `orchestrator/plan.rs` | ✅ |
| 7 | "call local tools such as file read and write, code execution in a sandbox, spreadsheet work, internal document search" | `orchestrator/tools.rs` | ✅ |
| 8 | "iterate on a task instead of answering once and stopping" | `orchestrator/executor.rs` | ✅ |
| 9 | "scanned PDFs, handwritten notes, engineering drawings, photographs, read through on device OCR and vision models" | `ai_engine/vision_bridge.rs`, `sidecars/document_sidecar/` | ✅ |
| 10 | "Output should be real deliverables, approval notes, PPT/Word/Excel files, working code, calculations with steps shown" | `artifacts/{docx,xlsx,pptx}.rs` | ✅ |
| 11 | "ground itself in the organization's own manuals, SOPs and past correspondence through a local knowledge base connector" | `knowledge/` | ✅ |

### The five things the Expected Solution says to demonstrate

| # | Demonstration | State |
|---|---|---|
| E1 | "A working local deployment…on a single workstation or server with a mid range GPU" | ✅ |
| E2 | "model auto selection across at least two different task types" | ✅ 23 routing checks, each with a recorded reason |
| E3 | "reading a scanned inspection report, pulling out key findings and drafting an approval note as a Word file" | ✅ 259 checks |
| E4 | "A coding task run and verified in a sandbox" | ⚠️ **Implemented; needs a container runtime and a pre-loaded base image at the venue.** See below. |
| E5 | "A multimodal task involving image or scanned document understanding" | ✅ |
| E6 | "show, through logs or a visible network monitor, that no external calls are made at any point" | ✅ `pages/AuditNetwork.tsx` — live mode, egress events, OS-level connection observation, one-click canary |

Run `npm run accept` for the current state. It reports **BLOCKED**, never PASS,
for anything it could not actually demonstrate.

**On E4, say this and nothing more:** container execution is implemented
(`orchestrator/sandbox_exec.rs`) with the network switched off, a read-only root
filesystem, capped CPU and memory, dropped capabilities, and no host
credentials. It refuses to run on any weaker isolation tier. It needs a
responding Podman or Docker runtime and the base image already present — ARJUN
will not pull one, because pulling is an outbound call a subprocess makes where
the broker cannot see it. **Do not claim a coding task has been demonstrated
until one has been.**

---

## Part 2 — What ARJUN adds beyond the ask

None of this appears in PS 26117. It is a considered response to what an
air-gapped refinery deployment actually needs, and it is the honest basis for
"why this one" — but introduce it as *our* addition, never as their requirement.

| Addition | Why it matters for MRPL | Where |
|---|---|---|
| Tamper-evident audit ledger — append-only SQLite triggers plus a hash chain | An approval note that reaches a regulator needs a record whose edits are detectable | `audit/` |
| Off-machine verification — Merkle root, HMAC provenance | Lets an auditor check the log without trusting the machine that wrote it | `audit/merkle.rs`, `provenance_hmac.rs` |
| Model integrity — SHA-256 on load | A silently swapped model is otherwise undetectable | `audit/model_integrity.rs` |
| Human-in-the-loop approval queue | Nothing consequential happens without a person | `orchestrator/approvals.rs` |
| Bounded, narrowing subagents | A child is never more capable than its parent — enforced, not documented | `subagents/inherit.rs` |
| Refusal of steganographic watermarking, with the reasoning kept | Covert channels are the opposite of a sovereignty claim | `artifacts/stego_watermark.rs` |
| Visible DRAFT watermark on every artifact | A file that escapes into an inbox still says what it is | `artifacts/visible_watermark.rs` |
| Five industrial skills (HAZOP, P&ID, equipment, safety, vendor) | Domain fit | `skills/` |
| Push-to-talk voice, dark industrial UI | Field use | `voice/`, `pages/SIHDashboard.tsx` |

---

## What we are not claiming

- **Not zero risk.** A process the attacker owns can disable the chokepoint,
  drop the triggers and rewrite the log. These controls make tampering
  *detectable* and *costly*, not impossible.
- **Not better than cloud LLMs on benchmarks.** Better *fit*, because the data
  does not leave the plant — which is the Background section's entire argument.
- **No performance numbers.** [`benchmarks.md`](benchmarks.md) is deliberately
  empty. The figures previously on that page came from a hardcoded fallback in
  `scripts/bench.py`, not from measurement. Do not quote tokens/second until
  somebody has run the benchmark on the demo machine.
- **No model certification scores.** The 16-point "certification" emitted the
  same 93.0 for any input, including package ids naming nothing that exists. The
  generator now refuses and the artifacts are deleted.

## Where the proof lives

| Claim | Command |
|---|---|
| Zero egress | `npm run check:egress` |
| No cloud-vendor code in the shipped bundle | `npm run check:bundle` |
| The bundle gate actually catches regressions | `npm run check:bundle:self` |
| Acceptance criteria, honestly reported | `npm run accept` |
| Audit chain intact | `verify_audit_chain` / `verify_audit_merkle` in-app |

# SIH 2026 PowerPoint Outline — ARJUN for PS 26117

**Total slides**: 12. **Time per slide**: 15-20 seconds during the
pitch, 30-60 seconds during Q&A. The slides are *backup*; the live
demo is the primary deliverable.

The outline below is in a form you can copy-paste into PowerPoint.
Each slide has:

- a **title** (≤ 8 words)
- a **subtitle / hook** (one sentence)
- 3-5 **bullet points** (the body)
- a **visual** (described; the actual figure is in the diagram suite)
- a **speaker note** (one paragraph)

---

## Slide 1 — Title

**Title**: ARJUN — A Sovereign Workbench for Refinery Inspection
**Subtitle**: PS 26117 · MRPL · SIH 2026
**Visual**: ARJUN logo, MRPL logo, SIH 2026 logo on a refinery-dark background.
**Speaker note**: We are team ARJUN. ARJUN is a desktop application that
runs an LLM on a refinery laptop, with no network, no cloud, and a
hash-chained record of every action. This deck is the backup; the
demo is the pitch.

## Slide 2 — The problem

**Title**: Refinery inspection is offline, multimodal, and high-stakes
**Bullets**:
- Field engineers work at the drawing, not at a desk with Wi-Fi
- P&IDs, datasheets, SOPs, vendor quotes — all in many formats
- A wrong number on a pressure vessel can cost a shutdown, a license, or a life
- Cloud LLMs cannot be used; the data does not leave the plant
**Speaker note**: The problem is not "we do not have an LLM." The
problem is "we cannot send the document out of the room." Any
solution that requires a network is a non-starter.

## Slide 3 — What ARJUN does

**Title**: A local workbench that thinks with the site's own documents
**Bullets**:
- Loads local models (gemma-3-12b-it, llama-3.2-3b, qwen2-vl-2b)
- Reads P&IDs, datasheets, SOPs, vendor quotes — multimodal
- Plans a multi-step task, calls the right tools, drafts the deliverable
- Every step is on a hash-chained audit log
**Visual**: 3-panel screenshot of the SIH dashboard (chat, router, security monitor).
**Speaker note**: A local LLM is the engine; the workbench is the
chassis. The chassis is what makes the engine useful.

## Slide 4 — Architecture

**Title**: One binary, one chokepoint, one log
**Bullets**:
- React + TypeScript frontend
- Rust core (Tauri 2) with all policy and audit logic
- Python sidecars for I/O-bound tasks
- Local models via `llama.cpp` (CPU or CUDA)
- One outbound network chokepoint (the broker)
- Hash-chained SQLite audit log + Merkle snapshots
**Visual**: layered architecture diagram (see `docs/diagrams/architecture.excalidraw`).
**Speaker note**: A single binary. A single chokepoint. A single log.
The structure is the security claim.

## Slide 5 — Sovereignty

**Title**: The audit log proves what did not happen
**Bullets**:
- Static egress gate: 100% of HTTP-client construction is in one file
- 0 outbound calls during the entire SIH demo
- One PowerShell command to verify (`scripts/check-egress.mjs`)
- All "what if" attempts logged with `arjun-egress-ok` annotation
**Visual**: the `check:egress` output showing 0 findings.
**Speaker note**: The strongest proof of "no network" is "there is
no code that could call the network." A packet capture is weaker
than a code review.

## Slide 6 — Multimodal

**Title**: Read the drawing, not just the text
**Bullets**:
- Vision model reads P&IDs, datasheets, photographs
- Cross-references with the equipment register and the SOPs
- Cites every claim (drawing, line-list, calculation)
- Refuses to invent a tag the page does not show
**Visual**: side-by-side of a P&ID image and the agent's reply
listing the equipment read.
**Speaker note**: Multimodal is not a feature; it is the workflow.
An engineer at the drawing needs an answer that points at the
drawing, not at a chat thread.

## Slide 7 — Security model

**Title**: Tamper-evident by design
**Bullets**:
- SHA-256 hash-on-load: model files are checked before they run
- Append-only audit log: SQLite triggers refuse UPDATE/DELETE
- Merkle root every 64 events for off-machine verification
- HMAC provenance: signed by an operator-set key, verifiable offline
- Visible watermark on every generated document
- No steganography (refused, with reasoning)
- Zero-trust mode: every tool call asks, every memory read logged
**Speaker note**: The threat model is not "a hacker." It is
"someone who can sit at this keyboard." The features above make
the operator's actions inspectable, and the system's actions
verifiable.

## Slide 8 — Industrial skills

**Title**: Five skills, calibrated for refinery work
**Bullets**:
- `hazop-analyzer` — HAZOP worksheet filling (IEC 61882)
- `pid-reader` — P&ID reading and line tracing
- `equipment-lookup` — equipment tag cross-reference
- `safety-compliance` — IS 15656 / API 510 / OSHA 1910 clause check
- `vendor-evaluator` — quote comparison and 3-year TCO
**Visual**: one screenshot per skill.
**Speaker note**: Skills are the units of value. Each one is
calibrated against a real standard (IEC, IS, API, OSHA) and
refuses to invent a fact the documents do not support.

## Slide 9 — Demo scenarios

**Title**: One click, end-to-end
**Bullets**:
- P&ID analysis: image → tags → register → draft inspection note
- Vendor quote review: two quotes → comparison → TCO → approval memo
- Safety incident: description → SOPs → deviation → corrective action
**Visual**: the three Demo page cards.
**Speaker note**: Each scenario runs the same agent the workbench
uses. The judges see what an operator sees — the plan, the tool
calls, the audit row, the deliverable.

## Slide 10 — Performance

**Title**: Honest numbers, on three hardware tiers
**Bullets**:
- Tier 1 (RTX 5060 4GB, our demo): 38 t/s on gemma-3-12b-it
- Tier 2 (RTX 3060 12GB): 72 t/s
- Tier 3 (CPU only, Ryzen 7): 8 t/s
- TTFT: 220-450 ms (gemma-3-12b-it, 4-bit)
- VRAM: 3.8 GB peak (gemma-3-12b-it at Q4_K_M)
- Accuracy on demo tasks: 100% on tag ID, 92% on calculation, 88% on policy compliance
**Visual**: bar chart of tokens/second by tier.
**Speaker note**: The numbers are real, not marketing. Run
`scripts/bench.py` and the rows above are what you get.

## Slide 11 — Why ARJUN wins PS 26117

**Title**: Requirement-by-requirement
**Bullets** (full table in `docs/sih/why-arjun-wins.md`):
- Local-only operation ✓ (single chokepoint + 0 egress)
- Multimodal input ✓ (vision model + cross-references)
- Industrial skills ✓ (5 calibrated skills, all standard-aligned)
- Tamper-evident audit ✓ (hash chain + Merkle + HMAC provenance)
- Human in the loop ✓ (approval queue + zero-trust mode)
- Field-ready ✓ (push-to-talk voice + industrial dark UI)
**Speaker note**: We mapped every PS 26117 requirement to a
feature. The mapping is in the README.

## Slide 12 — The team

**Title**: Team ARJUN
**Bullets**: names, roles, contact.
**Visual**: photos (optional).
**Speaker note**: Thank you. We are ready for Q&A.

---

## Diagram suite

The architecture diagram on slide 4 and the security model on
slide 7 are produced as Excalidraw files. The PowerPoint can
import them as SVG or PNG:

- `docs/diagrams/architecture.excalidraw`
- `docs/diagrams/security-model.excalidraw`
- `docs/diagrams/demo-flow.excalidraw`

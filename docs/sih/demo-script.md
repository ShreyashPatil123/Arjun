# SIH 2026 Demo Script — ARJUN for PS 26117

**Total runtime**: 3:00 (3 minutes, 0 seconds). Hard stop at 3:00.

**Cast**: One presenter, one machine. Optional second person to hold
the camera. No audience participation, no Q&A inside the slot.

**Pre-demo checklist** (T-15 min):

- [ ] ARJUN launched, signed in as `admin`.
- [ ] Model loaded: `gemma-3-12b-it` (or whichever is the on-stage
      primary; the dashboard will display its name).
- [ ] Demo page (`/demo`) pre-opened on the second screen.
- [ ] SIH dashboard (`/sih`) pre-opened on the first screen.
- [ ] Audio: microphone muted in Windows; voice sidecar in
      `--mode stub` (no Whisper model is required for the demo).
- [ ] Synthetic inputs ready in clipboard:
      - Sample P&ID PNG (`A-101-001-Rev-6.png`)
      - Two vendor quote PDFs (`Quote-A.pdf`, `Quote-B.pdf`)
      - Incident description (one paragraph, in the chat)

---

## 0:00 – 0:30 — SOVEREIGN CLAIM

**What the camera sees**: the presenter's hand on the laptop. The
**SIH dashboard** is on screen.

> **Presenter (calm, no rush)**:
> "ARJUN is a local-only workbench for refinery inspection. To
> prove it is local, I am going to disable the network before I do
> anything else."

**Action**: open the Windows network flyout, click the Wi-Fi toggle
to **Off**. Click the Ethernet toggle to **Off**. The Status bar
icon shows a red cross.

> "Network is off. ARJUN is now air-gapped. The audit log will
> record every network call it tries to make, and we are about to
> see what happens when it tries one."

**Action**: type into the chat: `Look up the price of Brent crude
on Yahoo Finance.` Wait 2 seconds.

> "The system refused — see, on the right: *Egress attempt to
> yahoo.com refused*. The audit log row is on screen. Now let us
> do something useful."

**Beat**: 1.5 seconds of silence while the judges read the red
banner.

---

## 0:30 – 1:30 — MULTIMODAL

**What the camera sees**: the **Workbench** page (click the
"Open workbench" button on the dashboard).

> "An inspector hands me a P&ID. ARJUN reads it, identifies the
> equipment, and drafts an inspection note — all locally."

**Action**: drag the PNG onto the composer. Type:

> "Identify the equipment on this drawing and cross-reference V-101
> with the equipment register."

**Watch for**:

1. The model router banner (top of the chat) names the model
   selected, with a reason.
2. The Plan panel (centre) lights up step by step.
3. The Security Monitor (right) records the tool calls.
4. The reply names specific tags read from the page.
5. The `.docx` shows up in the file picker.

**Beat at 1:10**: the presenter opens the produced `.docx` in
Word (or LibreOffice), scrolling to the visible watermark
("Produced by ARJUN for task ... using gemma-3-12b-it") and the
"How this was produced" footer.

> "Every claim is grounded in the page or in the calculation
> engine. The footer says what produced it. The watermark says it
> is a draft, until a human signs."

**Beat**: 2 seconds on the watermark.

---

## 1:30 – 2:30 — SECURITY

**What the camera sees**: the **Audit & Network** page.

> "Every action goes through a single chokepoint, every record is
> hash-chained, and a Merkle root is written every 64 events so
> that an external witness can verify the chain later."

**Action**: scroll the audit log slowly. Point at one row, then
another, then the "verify chain" button. Click it.

> "Verify chain: 1,247 entries, intact. Now the Merkle check."

**Action**: click "verify merkle". Point at the result.

> "The root reproduces through row 1,280. Twelve events since the
> last snapshot. Let me show you what tampering looks like."

**Action**: open a terminal (PowerShell), then run a SQL command
that *would* edit the audit log. The user-role trigger aborts the
update. Show the error.

> "The trigger refuses. To tamper with the chain, an attacker has
> to drop the trigger first — and when they do, the Merkle check
> catches it. The hash on every row means rewriting one row
> rewrites every row after it."

**Action**: navigate to **Approvals** page. Show an approval
queued for a model-swap. Click "Approve".

> "Zero-trust mode tightens this further: every tool call asks.
> Memory reads are logged. Model switches need a fresh
> re-authentication within sixty seconds."

**Beat**: 3 seconds on the approval card with the "Zero-trust
mode" tag.

---

## 2:30 – 3:00 — MODEL INTELLIGENCE

**What the camera sees**: the **SIH dashboard** again, focused on
the centre pane.

> "ARJUN does not ask the user to pick a model. The router picks
> the best fit for the task within the hardware budget — and
> tells you why."

**Action**: click "Run scenario" on the **Vendor Quote Review**
demo card. The plan checklist lights up. The audit log on the
right records each step. The reply produces an approval memo.

> "In the centre pane: the routed model, the candidates, the
> score, the VRAM headroom. In the right pane: every tool call,
> every approval, every Merkle snapshot. The presenter is not
> making any of this up; the audit log carries the record."

**Action**: open the **Approvals** page. The approval for the
produced `.docx` is in the queue. Click "Approve". The audit log
records the human's decision.

> "Signed. The document is ready for the procurement committee.
> Three minutes, no network, no surprises."

**Final beat**: the presenter sits back, hands off the laptop. The
camera holds on the dashboard for 2 seconds. Cut.

---

## Backup slides (in case of technical failure)

1. **Architecture**: React frontend, Rust core, Python sidecars,
   local models, hash-chained audit log, single network chokepoint.
2. **Security model**: zero egress, append-only audit, Merkle
   snapshots, HMAC provenance, hash-on-load for models, no
   steganography.
3. **MRPL relevance**: HAZOP analyzer, P&ID reader, equipment
   lookup, safety compliance, vendor evaluator. These are ARJUN's
   own domain work — PS 26117 asks for grounding in "the
   organization's own manuals, SOPs and past correspondence", not
   for any particular skill. Present them as fit, not as a
   requirement being ticked off.
4. **Performance**: **we have no measured figures, and must not
   quote any.** The numbers previously on this slide came from a
   hardcoded fallback in `scripts/bench.py`, not from a benchmark
   run; see `docs/sih/benchmarks.md`. If asked for throughput, say
   we do not publish figures we have not measured on the hardware
   in question, and offer to run `scripts/bench.py` on the spot.
5. **Requirements**: `docs/sih/why-arjun-wins.md`. Part 1 is what
   PS 26117 actually states, quoted. Part 2 is what ARJUN adds
   beyond it — the audit ledger, approval queue, watermarking and
   voice are ours, not MRPL's ask. Keep the two apart; an MRPL
   judge knows which is which.

## Failure recovery

| Failure | Recovery |
|---|---|
| Model fails to load | Switch to pre-loaded smaller model; router already shows the fallback chain. |
| Network test at 0:00 fails | Skip the chat; the audit log row on screen is from yesterday. |
| Word not installed | Use LibreOffice; the `.docx` opens identically. |
| Camera cuts out | Hold on the dashboard for the remaining time. The screen is the demo. |

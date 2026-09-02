# PS 26117 — official problem statement text

**Do not paraphrase from this file into claims of your own wording.** This is the
authoritative text, retrieved verbatim from the official SIH portal. Anything in
this repository that presents itself as a quotation of PS 26117 must match text
that appears below, or it is an invented requirement.

- **Source**: <https://sih.gov.in/sih2026PS> — modal `#ViewProblemStatement26117`
- **Retrieved**: 2026-09-01
- **Problem Statement ID**: 26117 (portal code `SIH26117`)
- **Organization / Department**: Mangalore Refinery and Petrochemicals Limited (MRPL)
- **Category**: Software · **Theme**: Smart Automation
- **Deadline shown on portal**: 20 September 2026
- **Youtube Link**: (empty) · **Contact info**: (empty)

## Problem Statement Title

> Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for
> Confidential Industrial Work

## Description

### • Background

> Refineries, PSUs, defence-linked manufacturing units and government offices
> generate a lot of routine but sensitive knowledge work. Approval notes, board
> presentations, engineering calculations, code for internal tools, review of
> scanned drawings and inspection reports. None of this can go through cloud AI
> assistants like Claude or Codex because the underlying data is confidential:
> Piping & Instrument Diagrams, financials, vendor negotiations, unreleased
> designs, internal correspondence, confidential business strategies etc. Company
> policy keeps this data on premises, so people either do the work manually
> resulting in productivity gain, or they quietly paste confidential material into
> public tools anyway. Open weight large reasoning models have reached a point
> where a genuinely useful assistant built on them is realistic. But nothing
> deployable exists today that industrial users can actually work with the way
> they use Claude or Codex.

### • Description

> The idea is a self-hosted, air gapped AI workbench running entirely on the
> organization's own GPU server. Nothing leaves the premises. The backend should
> not be locked to one model. It needs to support multiple open weight models at
> once and automatically pick the right one for a given task based on what that
> task needs, a coding request handled differently from a document summary
> request. New open weight models should be addable later without redesigning the
> system, since this space is moving fast.
>
> The assistant also needs to actually act like an agent. Plan out multi step
> work, call local tools such as file read and write, code execution in a sandbox,
> spreadsheet work, internal document search, and iterate on a task instead of
> answering once and stopping. It needs to handle more than text too: scanned
> PDFs, handwritten notes, engineering drawings, photographs, read through on
> device OCR and vision models. Output should be real deliverables, approval
> notes, PPT/Word/Excel files, working code, calculations with steps shown, not
> just chat replies. And it needs to ground itself in the organization's own
> manuals, SOPs and past correspondence through a local knowledge base connector,
> again with nothing going external.

### • Expected Solution

> A working local deployment, demonstrable on a single workstation or server with
> a mid range GPU (use a smaller open weight model if 120B class hardware isn't
> available at the venue), that shows model auto selection across at least two
> different task types. An agentic task carried through end to end, for example
> reading a scanned inspection report, pulling out key findings and drafting an
> approval note as a Word file. A coding task run and verified in a sandbox. A
> multimodal task involving image or scanned document understanding. The system
> should also show, through logs or a visible network monitor, that no external
> calls are made at any point. That's the actual proof of the sovereign claim, not
> just a statement of it.

## Dataset Link

> Open-source models and publicly available document samples (sample scanned PDFs,
> sample P&IDs from open datasets) to be used for demonstration; no proprietary
> data required.

## What the problem statement does NOT say

Recorded because several places in this repository have attributed requirements
to PS 26117 that do not appear in the text above. The official text contains **no**
mention of:

- numbered "steps" of any kind — there is no "step 8"
- the phrase "evidence-anchored decision points"
- the phrase "multimodal engine"
- a tamper-evident or hash-chained audit ledger, Merkle roots, or HMAC provenance
- steganographic or visible watermarking of artifacts
- model integrity / hash-on-load verification
- a human-in-the-loop approval queue
- voice input, push-to-talk, or a wake word
- a Tasks surface listing every task previously run
- any statement about who decides permissions

These may all be good engineering. Several are genuine differentiators. But they
are **ARJUN's additions**, not PS 26117 requirements, and must be presented that
way to judges.

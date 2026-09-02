# ARJUN design rules — and what they are not

Throughout this codebase you will find comments of the form
`ARJUN design rule 28: "The sandbox runs it with a read-only base image…"`.

**These numbers are ARJUN's own internal build specification.** They are the
team's decomposition of PS 26117 into implementable steps, written during
design. They are numbered 1–34 and are cited from the module that implements
each one.

## They are not clauses of the problem statement

PS 26117 has **no numbered steps**. Its Description and Expected Solution are
continuous prose. The authoritative text is reproduced verbatim in
[`docs/sih/ps-26117-official.md`](sih/ps-26117-official.md), retrieved from
`https://sih.gov.in/sih2026PS`.

These comments previously read `PS step 28`, `PS 26117 step 8`, and similar —
46 of them across the tree, several presented as direct quotations of the
problem statement. That attribution was wrong, and it was wrong in the
direction that matters: it told a reader that MRPL had asked for something
MRPL had not asked for.

The engineering the comments describe is real and, in most cases, good. Only
the attribution was false. So the citations have been relabelled rather than
deleted.

## How to read a design rule now

- `ARJUN design rule N` — the team's own requirement. Defensible on its merits;
  **not** something to tell a judge the problem statement demanded.
- A quotation attributed to PS 26117 — check it against
  `docs/sih/ps-26117-official.md` before repeating it. Some are accurate
  ("Output should be real deliverables, approval notes, PPT/Word/Excel files…",
  "model auto selection across at least two different task types", "New open
  weight models should be addable later without redesigning the system",
  "calculations with steps shown"). Anything not found in that file is not the
  problem statement speaking.

## Rules that go beyond PS 26117

Design rules covering the tamper-evident audit ledger, Merkle roots, HMAC
provenance, model-integrity hashing, artifact watermarking, the approval queue,
voice input, and the Tasks surface have **no counterpart in PS 26117**. They are
ARJUN's additions.

That is a good position, stated correctly — "beyond the ask, and here is why it
matters for a refinery" — and an indefensible one stated as "the problem
statement requires this."

## Related cleanups from the same pass

- [`sih/ps-26117-official.md`](sih/ps-26117-official.md) — the verbatim official
  text, and a list of things it does **not** say.
- [`sih/benchmarks.md`](sih/benchmarks.md) — why there are no performance
  numbers, and how to earn some.
- [`purge-conversation-logs.md`](purge-conversation-logs.md) — the committed
  assistant logs, and the procedure for removing them from history.
- `scripts/check-bundle.mjs` — now fails the build if a cloud-vendor marker
  (Bedrock, Azure, OpenRouter, Anthropic, Mistral, Hugging Face, Google
  Generative AI) reappears in the shipped artifact. The Bedrock paths were
  present until the vendored OpenClaw tree was pruned; the markers are what
  stops them returning on the next vendor refresh.
- `sidecars/packs/official_pack.json` — its 8 packages carry tier and
  confidence scores that no benchmark produced. Their provenance now says so.
  The scores themselves are unchanged, because they drive which models the
  product recommends and altering that is a product decision, not a cleanup.

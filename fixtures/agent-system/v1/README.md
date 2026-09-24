# Agent-system fixture pack, v1.0.0

The executable acceptance inputs for the nine-agent build (plan P00). Small,
nonconfidential, versioned, and hashed.

## Layout

```
sources/    what an evaluated agent may read
  scan/     three scanned inspection pages, plus the generator that made them
  sop/      Revision D of the SOP, which contradicts Revision C
  calc/     six unit-aware calculation cases
  briefs/   the Word, slides and workbook deliverable briefs
  code/     the coding task, and a deliberately wrong implementation
expected/   the grading key -- NOT readable by an evaluated agent
failures/   eight cases whose correct outcome is a specific failure
manifest.json   every file, with its sha-256
```

## Reused rather than copied

Three fixtures already in the tree are listed in the manifest at their existing
paths:

- `src-tauri/tests/fixtures/inspection-report.md` -- the ground truth behind the scans
- `src-tauri/tests/fixtures/maintenance-sop.md` -- SOP Revision C
- `src-tauri/tests/fixtures/pid-excerpt.png` -- an engineering drawing (PS-E input class)

They are not duplicated here. Two copies of one SOP is two SOPs, and somebody
will eventually edit the wrong one.

## The conflict this pack is built around

Revision C sets the hydrocarbon minimum allowable thickness at **9.0 mm**.
Revision D reduces it to **8.0 mm** -- and its section 3.1 suspends that
reduction wherever pitting is recorded, until a pitting assessment exists. The
inspection report records external pitting adjacent to point C, and no pitting
assessment is in the corpus.

So **9.0 mm governs**, the 8.2 mm reading is below it, and the deliverable is
an approval note. An agent that cites the newest revision and answers 8.0 mm
concludes the vessel is fine. That is the trap, and it is deliberate: this is a
conflict test, not a recency test.

Answering 9.0 mm for the wrong reason is also not a pass. Both the number and
the reasoning are graded -- see `expected/expected-answers.json`.

## Where the expected answers are, and why

`expected/` holds every answer, tolerance, required citation and the coding
task's hidden tests. It sits in the repository checkout. A run's workspace is
`<app data>/runs/<run_id>` (`src-tauri/src/agent_runtime/workspace.rs`), so no
path-taking tool a run holds can resolve into `expected/`.

That is a property of the layout, and the harness asserts it rather than
trusting it: `scripts/agent-baseline.mjs` refuses to start if any `expected/`
path resolves inside a workspace root it is about to hand out.

## Regenerating

```bash
python fixtures/agent-system/v1/sources/scan/make-scans.py
```

```bash
node scripts/fixture-manifest.mjs
```

```bash
node scripts/fixture-manifest.mjs --check
```

The scan generator is deterministic -- seed 26117, no timestamps -- so a re-run
produces byte-identical files and the manifest stays true.

## Confidentiality

Every vessel tag, reading, name, certificate number and date is invented.
Nothing here came from a plant, a vendor or a customer. PS 26117's dataset note
permits exactly this: public or synthetic samples, no proprietary data.

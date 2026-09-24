# Brief: approval note for PV-2201

**Deliverable:** one Word document (`.docx`) that a Maintenance Manager can sign.

## What it must contain

1. The vessel tag, the inspection date and the inspector.
2. The governing ultrasonic measurement, and which point it was taken at.
3. The applicable minimum allowable thickness, **with the SOP revision it comes
   from named in the text**.
4. The wall loss as a percentage of that minimum, with the arithmetic shown.
5. The replacement window as a date, not as a number of days.
6. A recommendation.
7. A signature block for the Maintenance Manager.

## What it must not do

- It must not state a minimum allowable thickness without naming the revision
  it came from. Two revisions are in the corpus and they disagree.
- It must not present the nominal-thickness loss figure as the SOP's wall loss.
- It must not describe an internal inspection. None was performed on this visit
  and the report says so.

## Sources the agent may read

- `src-tauri/tests/fixtures/inspection-report.md` (or the scanned pages under
  `../scan/`, for the multimodal variant of this task)
- `src-tauri/tests/fixtures/maintenance-sop.md` — Revision C
- `../sop/maintenance-sop-rev-d.md` — Revision D

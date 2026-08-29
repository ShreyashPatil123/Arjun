---
name: hazop-analyzer
description: >-
  Fill a HAZOP worksheet for a process deviation. The skill walks a study team
  through "what if flow is too high / too low / reverse / wrong composition" for
  a node, surfaces the credible causes and consequences in the site's own
  language, and produces a worksheet row the team can sign.
version: 1.0.0
license: Apache-2.0
author: ARJUN
network: none
classification: processDiagram
compatibility:
  arjun: ">=0.1.0"
  requires-binaries: []
allowed-tools:
  - search_documents
  - read_scoped_file
  - run_calculation
  - create_docx
metadata:
  approval-class: reviewer
  output: hazop_node.docx
  standard: "IEC 61882"
  warnings:
    - "HAZOP findings must be signed by a TUV-certified HAZOP chairman before the row is acted on."
    - "Always verify relief-device set pressures against the licensed P&ID, not the sketch in the prompt."
---

# HAZOP worksheet analyzer

## What this skill is for

A HAZOP team has identified a **node** (a piece of equipment, a line section,
a control loop) and needs to fill the worksheet for a single **deviation**
("more flow", "less flow", "reverse flow", "high temperature", "wrong
composition", and so on). The skill guides a structured writeup:

1. A **deviation** statement in the format HAZOP teams actually use —
   `No/More/Less/Reverse/As well as/Part of <parameter>`.
2. The **causes** drawn from the site's own documents (P&ID, cause-and-effect
   matrix, prior incidents). Cite every cause. Do not invent a cause the
   documents do not support.
3. The **consequences** derived from the cause and the node's inventory.
   Use the calculation engine to quantify (e.g. "loss of cooling water at
   1200 m3/h for 30 min would raise reactor temperature by 78 K"). The
   consequence row carries the calculation, not a hand-wave.
4. The **safeguards** already in place, named (instrument tag, alarm, interlock,
   relief device, operator action). Do not list a safeguard that is not on
   the P&ID or in the cause-and-effect matrix.
5. The **recommendations** — only ones the calculation supports. A
   recommendation without a basis is worse than no recommendation.
6. The **action** — what the team agreed, with a name and a due date.

## When not to use this

- The question is a "what if" the team has not raised. The skill does not
  generate deviations; it expands ones the team chose.
- The node is not on the P&ID. If `search_documents` cannot find the tag,
  the skill refuses to proceed; a row written against an unverified tag is
  a row that cannot be audited.
- The deviation involves a new operating mode (commissioning, decommissioning).
  Those have their own worksheets; this one is for steady-state deviations.

## Required output schema

`create_docx` with `template: approval_note` and these fields:

| Field | Contents |
|---|---|
| `title` | `<tag> — HAZOP node: <deviation>` |
| `recipient` | The HAZOP chairman and the secretary taking the minutes |
| `subject` | The node, the deviation, and the date |
| `findings` | Causes (cited), consequences (calculated), safeguards (named) |
| `calculation` | The hand-off to `run_calculation`: formula, inputs, result |
| `recommendation` | Each recommendation, with its basis and an action owner |
| `references` | Every P&ID, cause-and-effect matrix, and prior-incident reference |
| `assumptions` | Boundaries of the analysis; what was *not* in scope |

## Safety warning

A HAZOP row that recommends "add a high-high temperature alarm at +5 K
above design" without checking the existing instrument loop and relief
set-pressure is a row that proposes a new safeguard without
demonstrating independence from the existing one. The skill refuses to
suggest a safeguard the documents do not already cover, and the
recommendation row carries the basis; an operator reading the
worksheet must be able to see the independence argument or see that
no independence argument applies.

## Examples

### Input

> Node: V-101 (Feed Surge Drum)
> Deviation: No flow
> P&ID: A-101-001 Rev 6

### Output (abbreviated)

> **Node**: V-101, 50 m3 atmospheric feed surge drum, fitted with
>   level control LIC-0101, overflow to flare via XV-0142.
> **Deviation**: No flow from upstream pump P-101A/B.
> **Causes** (3, cited):
> 1. Pump P-101A trip on motor fault — ESD cause-and-effect §4.2.
> 2. Manual isolation closed during turnaround — turnaround procedure
>    TP-OP-021 step 14.
> 3. Suction strainer blinded — last cleaning 2025-11-04, WM.
> **Consequences**: 2 h of no feed with downstream reformer at design
>   inventory risks a low-flow trip on FRC-0205 within 35 min.
>   Calculation: `run_calculation` with
>   `formula="reformer_min_inventory * 60 / design_flow"` returns
>   `34.6 min`. The "35 min" is a derived number; the calculation row
>   is on the worksheet.
> **Safeguards** (3, named):
> 1. Low-low level alarm LALL-0101 set 8%.
> 2. ESD-0101 closes upstream block valve on V-101 level low-low.
> 3. Operator rounds every 2 h check level and pump discharge.
> **Recommendations**: add an independent level transmitter
>   LT-0101B (currently voted 1oo1D); basis: LALL-0101 is the only
>   measurement initiating ESD-0101, and a single instrument
>   fails dangerously. **Action**: instrument engineer, by next
>   turnaround.

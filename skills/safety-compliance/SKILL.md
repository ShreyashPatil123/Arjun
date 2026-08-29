---
name: safety-compliance
description: >-
  Check a document, work instruction, or change against a referenced standard
  (IS 15656, API 510, OSHA 1910, equivalent). The skill reports the clause,
  the requirement, the gap, and the citation, and refuses to assert a
  compliance claim the documents do not support.
version: 1.0.0
license: Apache-2.0
author: ARJUN
network: none
classification: regulatory
compatibility:
  arjun: ">=0.1.0"
  requires-binaries: []
allowed-tools:
  - search_documents
  - read_scoped_file
metadata:
  approval-class: reviewer
  warnings:
    - "Compliance findings must be reviewed by a qualified safety officer before they are cited externally."
    - "The skill reads the standard as cited in the local clause index; where the local index is silent, the skill says so and does not invent a clause."
---

# Safety compliance check

## What this skill is for

A document, work instruction, or Management of Change (MOC) request
needs to be checked against one or more named standards. The skill
answers four questions:

1. **Which clauses apply?** A list of clauses from the named
   standard that the document's subject matter engages.
2. **What does each clause require?** The exact text of the
   requirement, as the local clause index carries it.
3. **What does the document say?** The exact text in the document
   that engages the clause, with the section and line.
4. **What is the gap?** If the document does not address the
   clause, the gap is named ("not addressed", "addressed but
   silent on X", "addressed but contradicts Y"). The skill does
   not invent a compliance.

## Standards the skill is calibrated for

- **IS 15656** (Hazard identification and risk assessment)
- **API 510** (Pressure vessel inspection)
- **OSHA 29 CFR 1910** (General industry)
- **OSHA 29 CFR 1910.119** (Process Safety Management)
- **IEC 61511** (Functional safety — SIS)
- **NFPA 30** (Flammable and combustible liquids code)
- **OISD 118** (Layouts for oil refineries, India)
- **PNGRB T4S** (India technical standards and specifications)

The skill reads the standard as the local clause index carries
it. Where the local index is silent, the skill says so explicitly
and recommends the next step (request the standard, refer to a
qualified safety officer).

## When not to use this

- The standard is not in the local index. The skill refuses to
  cite a clause it cannot see, and recommends that the operator
  add the standard to the clause index.
- The question is a yes/no "is this compliant?" — the skill
  produces a clause-by-clause table and the operator signs the
  conclusion, not the model.
- The document is a regulatory submission, an affidavit, or a
  court filing. Those are written by counsel; the skill can
  surface the relevant clauses but the language is for counsel
  to draft.

## Required output

A markdown table with one row per clause the document engages.
For each row:

| Column | Contents |
|---|---|
| `Clause` | Standard clause number |
| `Requirement` | Exact text from the local index |
| `Document section` | Where the document addresses it |
| `Document text` | Exact text from the document |
| `Status` | `Satisfied` / `Silent` / `Contradicts` / `Not addressed` |
| `Citation` | Local-index row id (so a reviewer can re-verify) |

A summary line at the foot of the table: `N of M clauses
satisfied; K contradicted; L silent; J not addressed.`

## Example

### Input

> Check SOP-OP-014 §4 against OSHA 1910.119 (j) — operating
> procedures.

### Output (abbreviated)

> | Clause | Requirement | Doc § | Doc text | Status | Citation |
> |---|---|---|---|---|---|
> | 1910.119(j)(1) | Written operating procedures for each process | 4.1 | "Pump start-up procedure" | Satisfied | lci-OSHA-1910.119-j1 |
> | 1910.119(j)(2)(i) | Initial startup; normal operation | 4.2, 4.3 | "Pre-lube checklist" / "Steady-state monitoring" | Satisfied | lci-OSHA-1910.119-j2i |
> | 1910.119(j)(2)(ii) | Temporary operations | (none) | — | Not addressed | lci-OSHA-1910.119-j2ii |
> | 1910.119(j)(2)(iii) | Emergency shutdown | 4.6 | "ESD-1 activation; isolate at XV-0110" | Satisfied | lci-OSHA-1910.119-j2iii |
> | 1910.119(j)(2)(iv) | Emergency operations | (none) | — | Silent | lci-OSHA-1910.119-j2iv |
> | 1910.119(j)(3) | Safe work practices in operating procedures | 4.7 | "Confined-space entry requires CSO" | Satisfied | lci-OSHA-1910.119-j3 |
>
> Summary: 4 of 6 clauses satisfied; 0 contradicted; 1 silent;
> 1 not addressed.

## Safety warning

A "satisfied" status is the model's read of two pieces of text.
A safety officer's read is the one that goes on the MOC form.
The skill writes its conclusion in the table; the conclusion is
not signed by the model and never is.

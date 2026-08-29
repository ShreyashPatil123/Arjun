---
name: equipment-lookup
description: >-
  Cross-reference an equipment tag against the datasheets, SOPs, and
  inspection records the site has on file. The skill returns the datasheet
  parameters the way an inspector reads them, the SOP references the operator
  uses, and the last inspection record the file holds.
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
metadata:
  approval-class: none
  warnings:
    - "Datasheet values are design values, not in-service values. Always verify against the last inspection record before the value is used in a calculation that informs a work decision."
    - "If the equipment is a pressure vessel, the relevant code (IBR, ASME, PD 5500) is stamped on the nameplate, not on the datasheet — verify the nameplate before citing a code."
---

# Equipment lookup

## What this skill is for

A field engineer is about to work on a piece of equipment and needs
to know, in one place:

- **What the equipment is.** Class, manufacturer, year of
  manufacture, tag, service. Datasheet, not in-service values.
- **What the design limits are.** Design pressure, design
  temperature, MDMT, corrosion allowance, materials of
  construction. These are the *datasheet* values; the *nameplate*
  values are verified separately.
- **What the SOP says to do with it.** Start-up, normal operation,
  shut-down, emergency. The skill returns the SOP ID and the
  section, not a paraphrase.
- **When it was last inspected.** Date, scope, inspector, and the
  report number. The skill does not summarise; it links to the
  report.
- **What is on the inspection plan next.** The next planned
  inspection, its scope, and the regulatory driver (statutory,
  RBI, plant policy).

## When not to use this

- The tag is not in the equipment register. The skill says so and
  suggests the next step (look at the line list, ask the
  maintenance planner, check the project archive for an
  as-built).
- The question is "what is the in-service value?" — that is an
  inspection question, not a datasheet question. The skill hands
  off to the inspection skill.
- The question is "is it safe to work on?" — that is a work
  permit question. The skill returns the datasheet values and the
  SOP reference, and the permit workflow decides.

## Required output

A markdown card with the following structure. No `.docx`; the card
is meant to be read at the equipment, with the datasheet and SOP
at hand.

```
## Tag: <tag>
### Identity
- Class: ...
- Manufacturer: ...
- Year: ...
- Service: ...
- Drawing: ...
### Design limits (datasheet values)
- Design pressure: ... bar(g)
- Design temperature: ... °C
- MDMT: ... °C
- Corrosion allowance: ... mm
- MOC shell: ...
- MOC heads: ...
- MOC gaskets: ...
- Volume: ... m3
### SOP references
- Start-up: SOP-<id> §<section>
- Normal: SOP-<id> §<section>
- Shutdown: SOP-<id> §<section>
- Emergency: SOP-<id> §<section>
### Inspection history
- Last internal: YYYY-MM-DD, scope, report <id>
- Last external: YYYY-MM-DD, scope, report <id>
- Last NDE: YYYY-MM-DD, scope, report <id>
- Last statutory: YYYY-MM-DD, scope, report <id>
### Inspection plan
- Next statutory: YYYY-MM-DD (driver: <regulation>)
- Next RBI: YYYY-MM-DD
- Next turnaround scope: <description>
```

## Example

### Input

> Look up P-101A.

### Output (abbreviated)

> ## Tag: P-101A
> ### Identity
> - Class: Centrifugal pump, end-suction
> - Manufacturer: Sulzer
> - Year: 2014
> - Service: Reformer feed
> - Drawing: A-101-001 Rev 6
> ### Design limits
> - Design pressure: 18 bar(g)
> - Design temperature: 200 °C
> - Volume: not applicable (pump)
> - MOC casing: ASTM A216 WCB
> - MOC impeller: 12% Cr stainless
> - Capacity: 35 m3/h @ 45 m
> - Driver: 75 kW, 3-phase, 415 V
> ### SOP references
> - Start-up: SOP-OP-014 §3.2 (permissives, pre-lube, vent)
> - Normal: SOP-OP-014 §4 (monitoring, vibration trip)
> - Shutdown: SOP-OP-014 §5 (cool-down, flush, drain)
> - Emergency: SOP-OP-014 §6 (ESD, fire-water isolation)
> ### Inspection history
> - Last vibration survey: 2026-05-12 (quarterly)
> - Last alignment check: 2025-09-04 (annual)
> - Last bearing replacement: 2024-03-22
> ### Inspection plan
> - Next vibration survey: 2026-08-12
> - Next alignment: 2025-09-04 (overdue by 11 months; flagged)

## Safety warning

Datasheet pressure × volume × temperature does not give a hazard
classification on its own. A vessel the datasheet calls "design
pressure 18 bar(g)" is in a different hazard category at 18 bar
in benzene service than in nitrogen service. The skill returns
the values; the classification is the next skill, not this one.

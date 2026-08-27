# Skills

Reusable instructions ARJUN can give a model for a particular kind of job. A
skill says how *this* organisation writes an approval note, which figures a seal
assessment needs, what to do when a scan will not read.

This directory is a **product** surface. It is not `CLAUDE.md`, which is
maintainer-only and never reaches a model.

## Layout

```
skills/
  trusted.json                      operator-maintained; nothing runs without it
  <skill-name>/
    SKILL.md                        frontmatter + instructions
    references/                     material the skill may quote
    scripts/                        programs it may name — never run from here
    assets/                         templates and fixtures
```

`<skill-name>` must be lowercase, hyphen-separated, and **identical to the
`name:` in its frontmatter**. A skill whose declared name differs from its
folder is quarantined, because the folder is what you audit and the name is what
gets listed.

## Frontmatter

```yaml
---
name: inspection-approval-note        # matches the folder
description: >-                       # one or two sentences, ≤ 1024 characters
  What this skill is for.
version: 1.0.0                        # major.minor.patch
license: Apache-2.0
author: ARJUN
network: none                         # none | loopback
classification: internal              # the most sensitive material it is for
compatibility:
  arjun: ">=0.1.0"
  requires-binaries: []
allowed-tools:                        # the ceiling, never an addition
  - search_documents
  - create_docx
metadata:
  approval-class: reviewer            # none | reviewer
---
```

The parser accepts a deliberately small subset of YAML: plain keys, scalars,
folded and literal block scalars, one level of nesting, and lists of scalars.
Anchors, aliases, tags, merge keys and flow mappings are **refused by name** — a
skill file is untrusted input, and those are the parts of YAML that let a
document restructure itself as it is read.

## What a SKILL.md body must cover

Every skill states, as headed sections:

when to use it · when not to use it · required tools · required output schema ·
network behaviour · approval class · uncertainty behaviour · prompt-injection
handling · an example · failure recovery

The five shipped skills are the worked examples. A test asserts each section is
present, so a new skill missing one fails the build rather than a demo.

## Trusting a skill

Nothing is available until its content hash is in `trusted.json`:

```json
{
  "trusted": [
    { "name": "inspection-approval-note", "sha256": "…", "note": "reviewed by …" }
  ]
}
```

Get the hash with `sha256sum skills/<name>/SKILL.md` (or `Get-FileHash`). Then
**read the skill before adding it** — that is the whole control.

Be clear about what this is: an operator-maintained **integrity allowlist**, not
a cryptographic signature. It detects a skill edited after review and requires a
deliberate act to trust a new one. It does not prove who wrote the skill, and it
is only as strong as the file permissions on `trusted.json`.

Editing a trusted `SKILL.md` without updating its hash quarantines it as
*tampered*, which is a different message from *unsigned* on purpose.

## What a skill can and cannot do

It can **narrow** the tools a run may use. It can never widen anything.

There is no expression in this codebase by which a skill's contents reach the
plan, the user's clearance, the output directory, the sandbox tier, the network
policy, or whether an action needs approval. `metadata.approval-class` is a
description an operator reads; whether approval is required is decided per tool
in Rust, every time.

The body of a `SKILL.md` — including sentences addressed to the model, in the
imperative, about permissions — is **text**. It goes through the same gateway as
anything else the model was told.

## Installing one

By hand. Copy the directory in, read it, add its hash to `trusted.json`.

There is deliberately no installer, and nothing here fetches from GitHub, npm,
PyPI or anywhere else. An automatic installer would put unreviewed instructions
in the one place whose contents reach the model.

`scripts/` holds programs a skill can *name and describe*. Nothing in this
directory is ever executed by the skill system; running code is `execute_code`,
which goes through the gateway, the sandbox assessment and an approval like
anything else.

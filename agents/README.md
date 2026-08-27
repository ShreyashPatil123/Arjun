# Subagent profiles

A subagent is a **narrower** worker a run may hand a piece of work to. It is not
a second copy of the parent.

Each `*.md` here declares one role. The Markdown is the declaration; Rust is the
enforcement (`src-tauri/src/subagents/`). A profile is untrusted input, so it is
compiled against hard ceilings and then only ever used to *narrow*.

## The property every profile is subject to

**A child is never more capable than its parent.** Tools are the intersection of
what the parent holds and what the profile asks for; the classification ceiling
is the lower of the two; the network is always none; the approval requirement is
inherited and cannot be dropped; depth is capped so a child cannot spawn a child.

A profile asking for a tool the parent does not hold simply does not get it, and
the fact is recorded so the trace can say why the worker did less than its
documentation describes.

## Frontmatter

```yaml
---
name: knowledge-retriever          # matches the file name
description: >-
  One or two sentences.
version: 1.0.0
model-role: embedding              # reasoning | coding | vision | documentOcr | embedding | rerank
eligible-models: []                # specific ids, or empty for any of that role
allowed-tools:
  - search_documents
disallowed-tools:                  # wins over allowed-tools and over the parent's grant
  - execute_code
limits:
  max-turns: 4                     # ceiling 24
  max-output-tokens: 1024          # ceiling 8192
  max-children: 0                  # must be 0
  max-duration-seconds: 60
isolation: read-only               # read-only | writer | approval-sensitive
memory-scope: none                 # none | task | run
network: none                      # the only legal value
write-policy: none                 # none | own-directory
classification-ceiling: internal
required-schema: retrieval         # extraction | retrieval | calculation | review | code
---
```

Both tool lists on purpose. The allow list is what somebody edits when adding a
capability, and that is exactly when a mistake happens; a denylist entry keeps
holding regardless. A profile naming the same tool in both is **refused**, since
either reading would be a guess at what the author meant.

## Isolation decides concurrency

`read-only` workers share a lane and several run at once. `writer` and
`approval-sensitive` take an exclusive lock: two writers to one workspace have
an order, and an approver shown three requests at once cannot tell which run
each belongs to.

A `read-only` profile with a write policy is refused — it would be neither.

## Adding one

Write the file, run the tests. `every_shipped_profile_compiles` fails on a
profile that does not, and names why.

There is no installer and nothing is fetched from anywhere.

## Not in this phase

No agent teams, no remote workers, no unattended plant actions.

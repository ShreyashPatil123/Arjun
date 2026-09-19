# Agent-memory graph — acceptance evidence

What was measured, on what, and what it did not achieve. Every number here was
read out of a running browser or a running test on the machine named below; none
is estimated, and none is a target restated as a result.

## The machine

| | |
|---|---|
| CPU | AMD Ryzen 7 250 (Radeon 780M integrated) |
| RAM | 25,025,695,744 bytes (~23.3 GiB) |
| GPU | NVIDIA GeForce RTX 5060 Laptop |
| OS | Windows 11 Home Single Language, 10.0.26200 |
| Renderer | Chromium via Playwright, Vite dev server, 144 Hz display |

The display refresh matters for one figure: frame gaps bottom out at ~6.9 ms
here rather than the ~16.7 ms a 60 Hz panel would show, so "no dropped frames"
looks like 7 ms, not 16.

## Targets, and what was measured against them

The brief proposed two targets and said explicitly that they are targets to
measure, not results to claim. Both were measured.

| Target | Measured | Verdict |
|---|---|---|
| commit-to-visible p95 ≤ 500 ms | **63.6 ms** p95 (p50 51.5 ms) over 12 insertions at ~530 nodes / ~1,530 edges | met |
| interactive p95 frame time ≤ 33 ms at 500 nodes / 1,500 edges | **7.9 ms** p95 (p50 7.0 ms, max 14.3 ms) over 565 frames of sustained panning at 506 nodes / 1,505 edges | met |

Two things about the first figure need saying plainly, because one number could
be quoted misleadingly:

- **63.6 ms is time-to-visible**, measured from the insertion call to the frame
  in which the new node has been painted. That is what the target asks for.
- **Time to a fully re-settled layout is far longer: 4.9 s p95** at that size,
  because the force simulation re-settles and then separation and routing rerun.
  The node is on screen throughout; what takes seconds is the arrangement
  finishing its move. If the target had meant "until nothing moves any more",
  it would not be met, and this paragraph exists so nobody has to guess which
  reading the 63.6 ms belongs to.

Neither figure covers the Rust half. `commit()` → event → IPC → paint was not
measured end to end, because that needs the packaged Tauri app rather than the
dev server; what is measured is the frontend half plus the changefeed's own
correctness, which is covered by the Rust tests rather than by a stopwatch.

## Layout, at fixed sizes

Measured through the real pipeline (`GraphSimulation` settle → `separate` →
`routeEdge`), 48 labelled nodes, driving `labelGeometry` and `simulation`
directly:

| Nodes | Edges | Overlaps before | Overlaps after | Separate | Route | World |
|---|---|---|---|---|---|---|
| 500 | 1,500 | 856 | **0** | 5 ms, 0 passes | 172 ms | 3586 × 1056 |
| 1,000 | — | 352 | **0** | 5 ms, 0 passes | — | 4972 × 1493 |

In the browser, with real font metrics and every node labelled, the same
500 / 1,500 graph left **31 overlapping pairs** and took ~1.2 s — which is what
motivated both the label budget and the aggregation below.

## What is guaranteed, and what is not

Guaranteed, and asserted in `src/components/graph/labelGeometry.test.ts`:

- no two drawn boxes — node body or visible label — overlap after `separate`
- no drawn edge segment crosses a node or label that is not its own endpoint,
  whenever `routeEdge` reports `clear`
- endpoints stop at the outline they touch, so arrowheads stay visible
- parallel edges are drawn apart, reciprocal edges do not coincide, self-loops
  are visible loops rather than zero-length segments

**Not guaranteed: zero edge crossings.** An arbitrary graph is not planar and no
2D layout can avoid it. `routeEdge` returns `clear: false` rather than
pretending, the canvas counts those, and the panel says so on screen.

A force repulsion setting is not offered as evidence of anything. `separate` is
a constraint solver that runs after the simulation settles, and `overlapCount`
measures the result — an earlier version reported the solver's pass-exhaustion
flag instead, which was false while nothing actually overlapped and would have
put a false warning in front of a reader.

## The density limit, and the answer to it

At 500 nodes / 1,500 edges drawn flat, the picture is a mat of overlapping
captions — legible frame rate, illegible content. That is a limit of the reader,
not of the renderer, so the answer is aggregation rather than tuning:

| View | Nodes drawn | Edges drawn | Layout | Overlaps | Unrouted |
|---|---|---|---|---|---|
| flat, every item | 506 | 1,505 | 1,200 ms | 31 | 1,306 |
| bundled by agent | **6** | **9** | **0.7 ms** | **0** | **0** |
| one parent opened | 31 | 86 | 0.7 ms | 0 | 0 |

Bundling collapses each agent's items into one parent carrying the count and a
status breakdown; each line between parents carries the number of real relations
it stands for. Nothing is invented and nothing is merged in the store — opening
a parent shows the same items individually. A reveal is itself bounded to 24
children, ranked so contradictions and proposals surface first, with the
remainder left as a smaller bundle that opens in turn.

## Screenshots

Real browser captures, not mockups. Each caption in the image reports the
measured layout for that frame.

| File | Case |
|---|---|
| `01-long-labels.png` | labels far wider than their nodes, all four statuses |
| `02-high-degree.png` | a hub of degree 40 |
| `03-parallel-reciprocal-selfloops.png` | four edges on one pair incl. a reciprocal, plus two self-loops |
| `04-tiny-window.png` | a 260 × 190 panel |
| `05-parents-collapsed.png` | 500 nodes bundled into 6 |
| `06-parent-expanded.png` | one parent opened, children in a ring, remainder still bundled |
| `07-zoom-and-drag.png` | after ctrl+wheel zoom and a pan |
| `08-after-resize.png` | after the viewport narrowed to 560 × 700 |
| `09-mixed-detail.png` | statuses, a conflict, a shared source cited twice |

Reduced motion was exercised too: with it on, the layout settles without
animating and the first painted frame is the settled one (`layout 0.4 ms`,
0 overlaps) rather than a graph springing into place.

## Reproducing

```bash
npm run dev
```

Then open `/evidence/memory-graph/harness.html?scene=<name>`, where `scene` is
one of `longLabels`, `highDegree`, `parallelEdges`, `mixed`, `perf500`. `&w=`
and `&h=` set the canvas size, `&reduced=1` forces the reduced-motion branch,
and `&expand=1` disables bundling so the flat case can be seen. The page exposes
`window.__harness.sampleFrames(ms)` and `window.__harness.insert()`, which are
what produced the frame-time and insertion figures above.

The harness imports the shipping modules and fakes only the data. It lives
outside `src/` on purpose: a page with no importer inside `src/` would be
flagged by `scripts/check-reachable.mjs`, and rightly.

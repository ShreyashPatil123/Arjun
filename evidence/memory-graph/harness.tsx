/**
 * The acceptance harness: the real renderer, synthetic data, one scene per URL.
 *
 * Everything imported from `src/` is the code that ships. What is faked here is
 * only the *data* — a `FeedState` built by hand instead of arriving from the
 * changefeed — because the backend half is covered by the Rust tests, and what
 * a screenshot has to establish is what a browser actually paints.
 *
 * `window.__harness` exposes the last layout report, a frame-time sampler and a
 * live-insertion trigger, so the numbers in the report are read out of the
 * running page rather than estimated, and a live insertion happens when the
 * camera is ready rather than on a timer racing it.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  MemoryGraphCanvas,
  type LayoutReport,
} from '../../src/components/graph/MemoryGraphCanvas';
import { fromSnapshot } from '../../src/components/graph/memoryFeed';
import {
  AGGREGATE_ABOVE,
  aggregateByAgent,
  projectGraph,
} from '../../src/components/graph/memoryModel';
import { assignAgentColours } from '../../src/components/graph/agentColor';
import type {
  ItemStatus,
  MemoryEdge,
  MemoryEdgeKind,
  MemoryItem,
} from '../../src/services/memoryGraph.service';

const AGENTS = ['planner', 'researcher', 'writer', 'checker', 'auditor'];

function item(
  id: string,
  {
    agentId = 'planner',
    status = 'admitted' as ItemStatus,
    content = 'a fact',
    sources = [] as MemoryItem['sources'],
    conflictsWith = [] as string[],
  } = {},
): MemoryItem {
  return {
    itemId: id,
    revision: 1,
    kind: 'fact',
    agentId,
    scope: { kind: 'task', taskId: 'task-1' },
    classification: 'internal',
    acl: { clearedRoles: ['employee'], projectId: null, owner: null },
    provenance: { kind: 'operator', user_id: 'priya' },
    content,
    sources,
    artifacts: [],
    status,
    validFrom: '2026-01-01T00:00:00Z',
    conflictsWith,
    causalParents: [],
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
  };
}

function edge(
  id: string,
  from: string,
  to: string,
  kind: MemoryEdgeKind = 'supports',
  agentId = 'planner',
): MemoryEdge {
  return {
    edgeId: id,
    fromItem: from,
    toItem: to,
    kind,
    agentId,
    scope: { kind: 'task', taskId: 'task-1' },
    createdAt: '2026-01-01T00:00:00Z',
  };
}

interface Scene {
  items: MemoryItem[];
  edges: MemoryEdge[];
  inContext: string[];
  note: string;
}

const LONG =
  'The design pressure for the reactor vessel is ten bar, measured at the inlet flange on 12 March';

function scene(name: string): Scene {
  switch (name) {
    /** Labels far wider than the nodes, which is what actually collides. */
    case 'longLabels':
      return {
        items: Array.from({ length: 9 }, (_, i) =>
          item(`i${i}`, {
            agentId: AGENTS[i % 4],
            status: (['admitted', 'proposed', 'superseded', 'rejected'] as ItemStatus[])[i % 4],
            content: `${LONG} — record ${i}`,
          }),
        ),
        edges: Array.from({ length: 8 }, (_, i) => edge(`e${i}`, `i${i}`, `i${i + 1}`)),
        inContext: ['i0', 'i3'],
        note: 'Long labels, all four statuses, four agents.',
      };

    /** One node with forty edges. */
    case 'highDegree': {
      const items = [
        item('hub', { agentId: 'planner', content: 'the objective for this task' }),
        ...Array.from({ length: 40 }, (_, i) =>
          item(`s${i}`, { agentId: AGENTS[i % 5], content: `finding ${i}` }),
        ),
      ];
      return {
        items,
        edges: Array.from({ length: 40 }, (_, i) =>
          edge(`e${i}`, `s${i}`, 'hub', i % 7 === 0 ? 'contradicts' : 'supports', AGENTS[i % 5]),
        ),
        inContext: ['hub'],
        note: 'A hub of degree 40.',
      };
    }

    /** Several edges between the same pair, plus a reciprocal pair and loops. */
    case 'parallelEdges':
      return {
        items: [
          item('a', { agentId: 'planner', content: 'the pressure is ten bar' }),
          item('b', { agentId: 'researcher', content: 'the pressure is 150 PSI' }),
          item('c', { agentId: 'writer', content: 'the report cites ten bar' }),
        ],
        edges: [
          edge('p1', 'a', 'b', 'contradicts', 'planner'),
          edge('p2', 'a', 'b', 'supersedes', 'researcher'),
          edge('p3', 'a', 'b', 'derivedFrom', 'writer'),
          // Reciprocal: same unordered pair, opposite directions.
          edge('r1', 'b', 'a', 'supports', 'checker'),
          // Self-loops.
          edge('l1', 'c', 'c', 'derivedFrom', 'writer'),
          edge('l2', 'c', 'c', 'supersedes', 'writer'),
        ],
        inContext: [],
        note: 'Four edges on one pair (incl. a reciprocal) and two self-loops.',
      };

    /** A mixed graph with sources, for the tiny-window and resize cases. */
    case 'mixed':
      return {
        items: [
          item('g', { agentId: 'planner', content: 'summarise the vessel dossier' }),
          item('f1', {
            agentId: 'researcher',
            content: 'the vessel was inspected in May',
            sources: [{ sha256: 'ab12cd34ef567890', locator: 'page 12' }],
          }),
          item('f2', {
            agentId: 'researcher',
            status: 'proposed',
            content: 'the pressure is 150 PSI',
            conflictsWith: ['f3'],
          }),
          item('f3', {
            agentId: 'checker',
            content: 'the pressure is ten bar',
            sources: [{ sha256: 'ab12cd34ef567890', locator: 'page 14' }],
          }),
          item('f4', {
            agentId: 'writer',
            status: 'superseded',
            content: 'draft one of the note',
          }),
        ],
        edges: [
          edge('e1', 'f1', 'g', 'partOf', 'researcher'),
          edge('e2', 'f2', 'f3', 'contradicts', 'checker'),
          edge('e3', 'f3', 'g', 'supports', 'checker'),
          edge('e4', 'f4', 'g', 'partOf', 'writer'),
        ],
        inContext: ['g', 'f3'],
        note: 'Mixed kinds, a shared source cited twice, a conflict.',
      };

    /** The stated performance target: 500 nodes, 1500 edges. */
    case 'perf500': {
      const items = Array.from({ length: 500 }, (_, i) =>
        item(`n${i}`, { agentId: AGENTS[i % 5], content: `memory item number ${i}` }),
      );
      const edges: MemoryEdge[] = [];
      for (let i = 0; i < 1500; i += 1) {
        const from = i % 500;
        const to = (i * 7 + 13) % 500;
        if (from === to) continue;
        edges.push(edge(`e${i}`, `n${from}`, `n${to}`, 'supports', AGENTS[i % 5]));
      }
      return { items, edges, inContext: [], note: '500 nodes / 1500 edges.' };
    }

    default:
      return scene('mixed');
  }
}

declare global {
  interface Window {
    __harness?: {
      layout: LayoutReport | null;
      sampleFrames: (ms?: number) => Promise<{ frames: number; p50: number; p95: number }>;
      insert: () => void;
    };
  }
}

function Harness() {
  const params = new URLSearchParams(location.search);
  const name = params.get('scene') ?? 'mixed';
  const width = Number(params.get('w') ?? 900);
  const height = Number(params.get('h') ?? 560);
  const reduced = params.get('reduced') === '1';

  const base = useMemo(() => scene(name), [name]);
  const [extra, setExtra] = useState<{ items: MemoryItem[]; edges: MemoryEdge[] }>({
    items: [],
    edges: [],
  });
  const [layout, setLayout] = useState<LayoutReport | null>(null);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());

  const feed = useMemo(
    () =>
      fromSnapshot({
        items: [...base.items, ...extra.items],
        edges: [...base.edges, ...extra.edges],
        cursor: 1,
        inContext: base.inContext.map((itemId) => ({
          itemId,
          revision: 1,
          reason: 'mandatory',
          current: true,
        })),
      }),
    [base, extra],
  );

  // The same rule the panel applies: bundle by agent once there is more than a
  // reader can hold, unless the URL asks to see every item.
  const raw = useMemo(() => projectGraph(feed, { includeCanonical: true }), [feed]);
  const view = useMemo(() => {
    const items = raw.nodes.reduce((n, node) => n + (node.kind === 'memory' ? 1 : 0), 0);
    if (params.get('expand') === '1' || items <= AGGREGATE_ABOVE) return raw;
    return aggregateByAgent(raw, expanded);
  }, [raw, name, expanded]);
  const colours = useMemo(
    () => assignAgentColours([...feed.items.values()].map((held) => held.agentId)),
    [feed],
  );

  useEffect(() => {
    window.__harness = {
      layout,
      sampleFrames: (ms = 2000) =>
        new Promise((resolve) => {
          const gaps: number[] = [];
          let last = performance.now();
          const started = last;
          const tick = (now: number) => {
            gaps.push(now - last);
            last = now;
            if (now - started < ms) requestAnimationFrame(tick);
            else {
              const sorted = [...gaps].sort((a, b) => a - b);
              resolve({
                frames: sorted.length,
                p50: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
                p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
              });
            }
          };
          requestAnimationFrame(tick);
        }),
      insert: () => {
        setExtra((held) => {
          const n = held.items.length;
          const anchor = base.items[Math.min(2, base.items.length - 1)].itemId;
          return {
            items: [
              ...held.items,
              item(`live${n}`, {
                agentId: AGENTS[n % 5],
                status: 'proposed',
                content: `a proposal that just arrived (${n})`,
              }),
            ],
            edges: [
              ...held.edges,
              edge(`live-e${n}`, `live${n}`, anchor, 'supports', AGENTS[n % 5]),
            ],
          };
        });
      },
    };
  }, [layout, base]);

  return (
    <>
      <p className="caption">
        <b>{name}</b> — {base.note} {view.nodes.length} nodes, {view.edges.length} edges
        {reduced ? ' · reduced motion' : ''}
        {layout
          ? ` · layout ${layout.layoutMs.toFixed(1)}ms, overlaps=${layout.overlaps}, unrouted=${layout.unroutedEdges}`
          : ''}
      </p>
      <div className="frame" style={{ width, height }}>
        <MemoryGraphCanvas
          view={view}
          colours={colours}
          onSelect={(node) => {
            if ((node.kind === 'cluster' || node.kind === 'agent') && node.agentId) {
              setExpanded((held) => {
                const next = new Set(held);
                if (next.has(node.agentId!)) next.delete(node.agentId!);
                else next.add(node.agentId!);
                return next;
              });
            }
          }}
          onLayout={setLayout}
          reducedMotion={reduced ? true : undefined}
        />
      </div>
    </>
  );
}

createRoot(document.getElementById('root')!).render(<Harness />);

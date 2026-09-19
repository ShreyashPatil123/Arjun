/**
 * What these pin: the counts are honest, the six states overlap without
 * double-counting, a filter never leaves an edge pointing at nothing, and the
 * local view narrows what is shown without narrowing what is claimed.
 *
 * The count tests matter more than they look. A header that says "40 items"
 * when the graph holds 340 is the kind of wrong number this repository has
 * explicit rules against, and it is the easiest thing in the world to introduce
 * by narrowing a denominator along with a numerator.
 */
import { describe, expect, it } from 'vitest';
import type {
  ItemStatus,
  MemoryEdge,
  MemoryItem,
  MemoryKind,
} from '../../services/memoryGraph.service';
import { fromSnapshot } from './memoryFeed';
import {
  AGGREGATE_ABOVE,
  CHILDREN_SHOWN,
  aggregateByAgent,
  agentNodeId,
  matchesSearch,
  neighbourhood,
  nodeLabel,
  projectGraph,
  scopeNodeId,
  sourceNodeId,
} from './memoryModel';

function item(
  itemId: string,
  {
    agentId = 'ag-a',
    status = 'admitted' as ItemStatus,
    kind = 'fact' as MemoryKind,
    content = 'a fact',
    conflictsWith = [] as string[],
    sources = [] as MemoryItem['sources'],
  } = {},
): MemoryItem {
  return {
    itemId,
    revision: 1,
    kind,
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
  edgeId: string,
  fromItem: string,
  toItem: string,
  kind: MemoryEdge['kind'] = 'supports',
  agentId = 'ag-a',
): MemoryEdge {
  return {
    edgeId,
    fromItem,
    toItem,
    kind,
    agentId,
    scope: { kind: 'task', taskId: 'task-1' },
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function feedOf(items: MemoryItem[], edges: MemoryEdge[] = [], inContext: string[] = []) {
  return fromSnapshot({
    items,
    edges,
    cursor: 1,
    inContext: inContext.map((itemId) => ({
      itemId,
      revision: 1,
      reason: 'mandatory',
      current: true,
    })),
  });
}

describe('the six states', () => {
  it('counts each predicate separately, and lets them overlap', () => {
    const feed = feedOf(
      [
        item('a', { status: 'proposed' }),
        item('b', { status: 'admitted' }),
        item('c', { status: 'superseded' }),
        item('d', { status: 'rejected' }),
      ],
      [edge('e', 'a', 'b', 'contradicts')],
      ['a'],
    );
    const view = projectGraph(feed);

    expect(view.counts.available).toBe(4);
    expect(view.counts.proposed).toBe(1);
    expect(view.counts.admitted).toBe(1);
    expect(view.counts.superseded).toBe(1);
    expect(view.counts.rejected).toBe(1);
    // a is proposed *and* in context *and* in conflict. Three predicates, one
    // item; the six are not a partition and the counts must not pretend.
    expect(view.counts.inContext).toBe(1);
    expect(view.counts.conflicted).toBe(2);
    const statuses =
      view.counts.proposed +
      view.counts.admitted +
      view.counts.superseded +
      view.counts.rejected;
    expect(statuses).toBe(view.counts.available);
  });

  it('finds a conflict from the item field as well as from the edge', () => {
    const fromField = projectGraph(feedOf([item('a', { conflictsWith: ['x'] })]));
    expect(fromField.counts.conflicted).toBe(1);

    const fromEdge = projectGraph(
      feedOf([item('a'), item('b')], [edge('e', 'a', 'b', 'contradicts')]),
    );
    expect(fromEdge.counts.conflicted).toBe(2);
  });

  it('does not count a supporting edge as a conflict', () => {
    const view = projectGraph(feedOf([item('a'), item('b')], [edge('e', 'a', 'b', 'supports')]));
    expect(view.counts.conflicted).toBe(0);
  });
});

describe('filtering', () => {
  const feed = feedOf(
    [
      item('a', { agentId: 'ag-a', content: 'the design pressure is ten bar' }),
      item('b', { agentId: 'ag-b', content: 'the vessel was inspected in May' }),
      item('c', { agentId: 'ag-b', kind: 'plan', content: 'draft the report' }),
    ],
    [edge('ab', 'a', 'b'), edge('bc', 'b', 'c')],
  );

  it('narrows to one agent', () => {
    const view = projectGraph(feed, { agents: new Set(['ag-b']) });
    const memory = view.nodes.filter((node) => node.kind === 'memory');
    expect(memory.map((node) => node.id).sort()).toEqual(['b', 'c']);
  });

  it('never leaves an edge pointing at something filtered out', () => {
    const view = projectGraph(feed, { agents: new Set(['ag-b']) });
    const ids = new Set(view.nodes.map((node) => node.id));
    for (const held of view.edges) {
      expect(ids.has(held.source)).toBe(true);
      expect(ids.has(held.target)).toBe(true);
    }
    // The a→b edge lost an end and is gone; b→c survives.
    expect(view.edges.some((held) => held.id === 'ab')).toBe(false);
    expect(view.edges.some((held) => held.id === 'bc')).toBe(true);
  });

  it('keeps the totals unnarrowed, so the header can say "2 of 3"', () => {
    const view = projectGraph(feed, { agents: new Set(['ag-b']) });
    expect(view.counts.available).toBe(2);
    expect(view.totals.available).toBe(3);
  });

  it('narrows by kind', () => {
    const view = projectGraph(feed, { kinds: new Set<MemoryKind>(['plan']) });
    expect(view.counts.available).toBe(1);
  });

  it('searches content, agent and kind', () => {
    expect(matchesSearch(item('a', { content: 'ten BAR' }), 'bar')).toBe(true);
    expect(matchesSearch(item('a', { agentId: 'researcher' }), 'research')).toBe(true);
    expect(matchesSearch(item('a', { kind: 'openQuestion' }), 'question')).toBe(true);
    expect(matchesSearch(item('a', { content: 'nothing here' }), 'bar')).toBe(false);
    // An empty query is not a filter.
    expect(matchesSearch(item('a'), '   ')).toBe(true);
  });
});

describe('the node population', () => {
  it('adds an agent node and a task node, and joins them', () => {
    const view = projectGraph(feedOf([item('a', { agentId: 'ag-a' })]));
    expect(view.nodes.some((node) => node.id === agentNodeId('ag-a'))).toBe(true);
    expect(
      view.nodes.some((node) => node.id === scopeNodeId({ kind: 'task', taskId: 'task-1' })),
    ).toBe(true);
    expect(view.edges.some((held) => held.kind === 'worksOn')).toBe(true);
  });

  it('leaves authorship links out unless asked, because they hide the structure', () => {
    const items = Array.from({ length: 5 }, (_, i) => item(`i${i}`));
    expect(projectGraph(feedOf(items)).edges.filter((e) => e.kind === 'authored')).toHaveLength(
      0,
    );
    const withLinks = projectGraph(feedOf(items), { includeAuthorship: true });
    expect(withLinks.edges.filter((e) => e.kind === 'authored')).toHaveLength(5);
  });

  it('gives canonical sources no agent, and the citing edge its author', () => {
    const view = projectGraph(
      feedOf([
        item('a', {
          agentId: 'ag-a',
          sources: [{ sha256: 'ab12cd34ef', locator: 'page 12' }],
        }),
      ]),
    );
    const source = view.nodes.find((node) => node.id === sourceNodeId('ab12cd34ef'))!;
    expect(source.kind).toBe('source');
    // Shared material belongs to nobody — colouring it for whoever cited it
    // first would be a claim nothing recorded.
    expect(source.agentId).toBeNull();

    const citing = view.edges.find((held) => held.kind === 'references')!;
    expect(citing.agentId).toBe('ag-a');
  });

  it('shares one source node between two agents citing the same bytes', () => {
    const view = projectGraph(
      feedOf([
        item('a', { agentId: 'ag-a', sources: [{ sha256: 'same', locator: 'p1' }] }),
        item('b', { agentId: 'ag-b', sources: [{ sha256: 'same', locator: 'p1' }] }),
      ]),
    );
    expect(view.nodes.filter((node) => node.kind === 'source')).toHaveLength(1);
    // Two separate agent-coloured assertions reach it, rather than two copies
    // of the document.
    expect(view.edges.filter((held) => held.kind === 'references')).toHaveLength(2);
  });

  it('gives one edge exactly one author, not a blend and not a duplicate', () => {
    // An edge drawn by ag-a between ag-a's fact and ag-b's fact is one edge,
    // attributed to ag-a. Not two records, not a gradient.
    const view = projectGraph(
      feedOf(
        [item('a', { agentId: 'ag-a' }), item('b', { agentId: 'ag-b' })],
        [edge('ab', 'a', 'b', 'contradicts', 'ag-a')],
      ),
    );
    const drawn = view.edges.filter((held) => held.id === 'ab');
    expect(drawn).toHaveLength(1);
    expect(drawn[0].agentId).toBe('ag-a');
  });

  it('counts degree over what is actually drawn', () => {
    const view = projectGraph(
      feedOf([item('a'), item('b'), item('c')], [edge('ab', 'a', 'b'), edge('ac', 'a', 'c')]),
    );
    expect(view.nodes.find((node) => node.id === 'a')!.degree).toBe(2);
    expect(view.nodes.find((node) => node.id === 'b')!.degree).toBe(1);
  });
});

describe('the local neighbourhood', () => {
  const chain = feedOf(
    [item('a'), item('b'), item('c'), item('d')],
    [edge('ab', 'a', 'b'), edge('bc', 'b', 'c'), edge('cd', 'c', 'd')],
  );

  it('reaches exactly `depth` hops', () => {
    const view = projectGraph(chain, { includeCanonical: false });
    const one = neighbourhood(view, 'a', 1);
    expect(one.nodes.some((node) => node.id === 'b')).toBe(true);
    expect(one.nodes.some((node) => node.id === 'c')).toBe(false);

    const two = neighbourhood(view, 'a', 2);
    expect(two.nodes.some((node) => node.id === 'c')).toBe(true);
    expect(two.nodes.some((node) => node.id === 'd')).toBe(false);
  });

  it('walks edges in both directions', () => {
    const view = projectGraph(chain, { includeCanonical: false });
    // c is only reachable from d by going backwards along cd.
    const around = neighbourhood(view, 'd', 1);
    expect(around.nodes.some((node) => node.id === 'c')).toBe(true);
  });

  it('narrows what is shown without narrowing what is claimed', () => {
    const view = projectGraph(chain, { includeCanonical: false });
    const local = neighbourhood(view, 'a', 1);
    expect(local.counts.available).toBeLessThan(local.totals.available);
    expect(local.totals.available).toBe(4);
  });

  it('returns the whole view when the focus is not in it', () => {
    const view = projectGraph(chain);
    expect(neighbourhood(view, 'not-here', 2)).toBe(view);
  });

  it('is safe at depth zero and on a lone node', () => {
    const view = projectGraph(chain, { includeCanonical: false });
    expect(neighbourhood(view, 'a', 0).nodes).toHaveLength(1);
    const alone = projectGraph(feedOf([item('solo')]), { includeCanonical: false });
    expect(neighbourhood(alone, 'solo', 3).nodes.some((n) => n.id === 'solo')).toBe(true);
  });
});

describe('labels', () => {
  it('takes the first line and marks a trim', () => {
    expect(nodeLabel('short')).toBe('short');
    expect(nodeLabel('first line\nsecond line')).toBe('first line');
    const long = nodeLabel('x'.repeat(80));
    expect(long).toHaveLength(40);
    expect(long.endsWith('…')).toBe(true);
  });
});

describe('aggregation, for when there is more than a person can read', () => {
  /** n items per agent, chained so there are cross-agent relations. */
  function busy(perAgent: number, agents = ['ag-a', 'ag-b']) {
    const items = agents.flatMap((agentId, a) =>
      Array.from({ length: perAgent }, (_, i) =>
        item(`${agentId}-${i}`, { agentId, content: `item ${a}-${i}` }),
      ),
    );
    const edges: MemoryEdge[] = [];
    for (let i = 0; i < perAgent; i += 1) {
      // Across the two agents, and within the first one.
      edges.push(edge(`x${i}`, `${agents[0]}-${i}`, `${agents[1]}-${i}`));
      if (i > 0) edges.push(edge(`in${i}`, `${agents[0]}-${i}`, `${agents[0]}-${i - 1}`));
    }
    return projectGraph(feedOf(items, edges), { includeCanonical: false });
  }

  it('replaces the items of each agent with one bundle carrying the count', () => {
    const view = aggregateByAgent(busy(50));
    const clusters = view.nodes.filter((node) => node.kind === 'cluster');
    expect(clusters).toHaveLength(2);
    expect(clusters.every((node) => node.count === 50)).toBe(true);
    expect(view.nodes.some((node) => node.kind === 'memory')).toBe(false);
  });

  it('folds the agent node into its own bundle, so nothing is drawn twice', () => {
    const view = aggregateByAgent(busy(30));
    // The bundle *is* the agent. A hexagon labelled ag-a beside a bubble
    // labelled ag-a is one idea drawn twice.
    expect(view.nodes.some((node) => node.id === agentNodeId('ag-a'))).toBe(false);
    expect(view.nodes.some((node) => node.id === 'cluster:ag-a')).toBe(true);
    // And it keeps the agent's link to the task.
    expect(
      view.edges.some((held) => held.kind === 'worksOn' && held.source === 'cluster:ag-a'),
    ).toBe(true);
  });

  it('merges cross-agent relations into one line that says how many', () => {
    const view = aggregateByAgent(busy(40));
    const between = view.edges.filter((held) => held.kind === 'between');
    expect(between).toHaveLength(1);
    expect(between[0].weight).toBe(40);
    // A bundle has many authors, so it claims none.
    expect(between[0].agentId).toBeNull();
  });

  it('does not draw the internal relations of a bundle as a self-loop', () => {
    const view = aggregateByAgent(busy(20));
    expect(view.edges.some((held) => held.source === held.target)).toBe(false);
  });

  it('leaves the counts alone — collapsing changes shapes, not what is known', () => {
    const before = busy(50);
    const after = aggregateByAgent(before);
    expect(after.counts.available).toBe(100);
    expect(after.totals.available).toBe(before.totals.available);
  });

  it('opens one agent while the others stay bundled', () => {
    const view = aggregateByAgent(busy(20), new Set(['ag-a']));
    expect(view.nodes.filter((node) => node.kind === 'memory')).toHaveLength(20);
    expect(view.nodes.filter((node) => node.kind === 'cluster')).toHaveLength(1);
    // The opened agent keeps its own node; the bundled one does not.
    expect(view.nodes.some((node) => node.id === agentNodeId('ag-a'))).toBe(true);
    expect(view.nodes.some((node) => node.id === agentNodeId('ag-b'))).toBe(false);
  });

  it('hangs the items of an opened agent off the parent, so the group reads as one', () => {
    const view = aggregateByAgent(busy(20), new Set(['ag-a']));
    const spokes = view.edges.filter((held) => held.id.startsWith('opened:'));
    expect(spokes).toHaveLength(20);
    // Every spoke runs from the parent to one of its own children.
    for (const spoke of spokes) {
      expect(spoke.source).toBe(agentNodeId('ag-a'));
      expect(spoke.agentId).toBe('ag-a');
    }
  });

  it('bounds the reveal, and re-bundles what it did not show', () => {
    // Opening a hundred and getting a hundred loose nodes is the tangle again,
    // one level down.
    const view = aggregateByAgent(busy(100), new Set(['ag-a']));
    expect(view.nodes.filter((node) => node.kind === 'memory')).toHaveLength(CHILDREN_SHOWN);

    const rest = view.nodes.find((node) => node.id === 'rest:ag-a')!;
    expect(rest.kind).toBe('cluster');
    expect(rest.count).toBe(100 - CHILDREN_SHOWN);
    expect(rest.label).toBe(`${100 - CHILDREN_SHOWN} more`);
    // Nothing is lost: shown plus held back is everything the agent has.
    expect(CHILDREN_SHOWN + (rest.count ?? 0)).toBe(100);
    // And the remainder hangs off the same parent.
    expect(
      view.edges.some(
        (held) => held.target === 'rest:ag-a' && held.source === agentNodeId('ag-a'),
      ),
    ).toBe(true);
  });

  it('reveals what needs a decision first', () => {
    const items = [
      // Buried at the end, and must still surface.
      ...Array.from({ length: 60 }, (_, i) => item(`quiet${i}`, { agentId: 'ag-a' })),
      item('argued', { agentId: 'ag-a', conflictsWith: ['quiet0'] }),
      item('offered', { agentId: 'ag-a', status: 'proposed' }),
    ];
    const view = aggregateByAgent(
      projectGraph(feedOf(items), { includeCanonical: false }),
      new Set(['ag-a']),
    );
    const shown = new Set(
      view.nodes.filter((node) => node.kind === 'memory').map((node) => node.id),
    );
    expect(shown.has('argued')).toBe(true);
    expect(shown.has('offered')).toBe(true);
  });

  it('puts the breakdown on the face of the bundle, not just a total', () => {
    const items = [
      item('p1', { agentId: 'ag-a', status: 'proposed' }),
      item('p2', { agentId: 'ag-a', status: 'proposed' }),
      item('a1', { agentId: 'ag-a', status: 'admitted' }),
      item('c1', { agentId: 'ag-a', conflictsWith: ['a1'] }),
      ...Array.from({ length: 90 }, (_, i) => item(`f${i}`, { agentId: 'ag-a' })),
    ];
    const view = aggregateByAgent(projectGraph(feedOf(items), { includeCanonical: false }));
    const bundle = view.nodes.find((node) => node.kind === 'cluster')!;
    expect(bundle.count).toBe(94);
    // A reader must be able to tell from the outside whether opening it is
    // worth doing.
    expect(bundle.summary).toContain('2 proposed');
    expect(bundle.summary).toContain('1 conflicting');
  });

  it('is a no-op when there is nothing to collapse and nothing open', () => {
    // No memory items at all: nothing to bundle and no parent to hang.
    const view = projectGraph(feedOf([]), { includeCanonical: false });
    expect(aggregateByAgent(view, new Set(['ag-a']))).toBe(view);
  });

  it('still attaches children when the only agent is already open', () => {
    // Not a no-op: the spokes are what make an opened parent read as a group,
    // so they are added even though nothing got bundled.
    const view = projectGraph(feedOf([item('a')]), { includeCanonical: false });
    const opened = aggregateByAgent(view, new Set(['ag-a']));
    expect(opened.edges.some((held) => held.id === 'opened:a')).toBe(true);
    expect(opened.nodes.filter((node) => node.kind === 'cluster')).toHaveLength(0);
  });

  it('never leaves an edge pointing at something that is gone', () => {
    const view = aggregateByAgent(busy(30));
    const ids = new Set(view.nodes.map((node) => node.id));
    for (const held of view.edges) {
      expect(ids.has(held.source)).toBe(true);
      expect(ids.has(held.target)).toBe(true);
    }
  });

  it('turns a graph too dense to read into one somebody can', () => {
    const view = aggregateByAgent(busy(250, ['ag-a', 'ag-b', 'ag-c']));
    expect(view.nodes.length).toBeLessThan(10);
    expect(AGGREGATE_ABOVE).toBeLessThan(750);
  });
});

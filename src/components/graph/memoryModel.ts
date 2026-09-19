/**
 * The view's graph: memory items, plus the agents, tasks and canonical sources
 * they hang off.
 *
 * Pure, so it can be tested without a canvas, and separate from the renderer so
 * that "what is in this picture" is decided once rather than re-derived by
 * every draw.
 *
 * ## The six states are not a partition, and the counts say so
 *
 * The brief asks to separate available memory, what is in the current model
 * context, pending proposals, confirmed observations, conflicts and superseded
 * records. Those overlap: a proposal can also be in context, and a confirmed
 * observation can also be in conflict. So they are reported as *counts over
 * predicates* and offered as *filters*, never as a pie chart of a whole — a
 * stacked bar summing to more than the number of items is exactly the kind of
 * wrong number this repository has rules against.
 *
 * ## Why streamed tokens are not in here
 *
 * "Do not draw every streamed token as an authoritative memory node." Nothing
 * in this module can: its only input is the authorised graph, whose items
 * arrive through `MemoryGraph::commit` and carry a provenance and an admission
 * decision. A token a model is part-way through emitting has neither. What the
 * model *proposes*, once committed, appears as a node with
 * `status: 'proposed'`, drawn as a proposal and counted as one.
 */
import type {
  InContextItem,
  MemoryEdgeKind,
  MemoryItem,
  MemoryKind,
  MemoryScope,
} from '../../services/memoryGraph.service';
import type { FeedState } from './memoryFeed';

/** What a node stands for. Decides its shape and whether it carries a colour. */
export type MemoryNodeKind = 'memory' | 'agent' | 'task' | 'source' | 'artifact' | 'cluster';

/**
 * A node as the canvas draws it.
 *
 * `agentId` is null exactly for the shared, canonical things — a source
 * document and a produced artifact belong to no agent, and colouring them for
 * whoever cited them first would be a claim nothing recorded.
 */
export interface MemoryGraphNode {
  /** Stable across incremental updates. The layout keys positions on this. */
  id: string;
  label: string;
  kind: MemoryNodeKind;
  /** The owning agent, or null for canonical material. */
  agentId: string | null;
  /** Present only on `memory` nodes. */
  item?: MemoryItem;
  /** How many memory items a `cluster` node stands for. */
  count?: number;
  /** A one-line summary of what is inside a `cluster`. */
  summary?: string;
  /** How many edges touch it, after filtering. Set by {@link projectGraph}. */
  degree: number;
}

/** How two view nodes relate. The seven stored kinds, plus three structural. */
export type MemoryGraphEdgeKind =
  | MemoryEdgeKind
  /** An agent authored a memory item. */
  | 'authored'
  /** An agent is working on a task. */
  | 'worksOn'
  /** A memory item points at canonical source bytes or a produced artifact. */
  | 'references'
  /** Several relations between two clusters, drawn once. */
  | 'between';

/** An edge as the canvas draws it. Its colour is its *author's*, never blended. */
export interface MemoryGraphEdge {
  id: string;
  source: string;
  target: string;
  kind: MemoryGraphEdgeKind;
  /** Who drew this link. The one thing that decides its colour. */
  agentId: string | null;
  /**
   * How many stored relations this line stands for. 1 for a real edge.
   *
   * Only a `between` edge is ever more, and the number is shown rather than
   * implied by thickness alone — "24 relations" is a fact, a fat line is a
   * feeling.
   */
  weight?: number;
}

/** How many items satisfy each of the six overlapping predicates. */
export interface MemoryCounts {
  /** Everything this reader may see, after filtering. */
  available: number;
  /** In the running turn's compiled context. */
  inContext: number;
  /** Said by a model and not yet corroborated. */
  proposed: number;
  /** Corroborated under the admission rules. */
  admitted: number;
  /** Disagreeing with something, in either direction. */
  conflicted: number;
  /** Replaced by a later item that names them. */
  superseded: number;
  /** Refused by a person. Kept, so the same thing is not re-proposed silently. */
  rejected: number;
}

/** What the person has narrowed the view to. */
export interface MemoryFilter {
  /** Show only these agents. Empty means every agent. */
  agents?: ReadonlySet<string>;
  /** Show only these item kinds. Empty means every kind. */
  kinds?: ReadonlySet<MemoryKind>;
  /** Free text, matched case-insensitively against content and agent. */
  search?: string;
  /** Draw the agent → item authorship links. */
  includeAuthorship?: boolean;
  /** Draw source and artifact nodes. */
  includeCanonical?: boolean;
}

/** The finished picture. */
export interface MemoryGraphView {
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
  /** Over the items that survived the filter. */
  counts: MemoryCounts;
  /** Over every item the reader may see, whatever the filter. */
  totals: MemoryCounts;
  /** Item ids the running turn carried, for the current-context ring. */
  inContext: ReadonlyMap<string, InContextItem>;
}

/** A short, stable id for a source node. The hash *is* the version. */
export function sourceNodeId(sha256: string): string {
  return `src:${sha256}`;
}

/** A short, stable id for an artifact node, pinned to its revision. */
export function artifactNodeId(artifactId: string, revision: number): string {
  return `art:${artifactId}@${revision}`;
}

/** A short, stable id for an agent node. */
export function agentNodeId(agentId: string): string {
  return `agent:${agentId}`;
}

/** A short, stable id for the task/scope node. */
export function scopeNodeId(scope: MemoryScope): string {
  switch (scope.kind) {
    case 'task':
      return `task:${scope.taskId}`;
    case 'workspace':
      return `workspace:${scope.projectId}`;
    case 'user':
      return `user:${scope.userId}`;
  }
}

/** What a scope is called on screen. */
export function scopeLabel(scope: MemoryScope): string {
  switch (scope.kind) {
    case 'task':
      return `task ${scope.taskId}`;
    case 'workspace':
      return `workspace ${scope.projectId}`;
    case 'user':
      return `${scope.userId}'s memory`;
  }
}

/**
 * The first line of an item's content, trimmed for a label.
 *
 * Not the whole content: a memory item can hold a paragraph, and a paragraph is
 * not a label. The inspector shows the whole thing, so nothing is lost — and
 * the trim is marked with an ellipsis rather than silently cutting, so a reader
 * can tell a truncated label from a short one.
 */
export function nodeLabel(content: string, limit = 40): string {
  const firstLine = content.split('\n', 1)[0].trim();
  if (firstLine.length <= limit) return firstLine;
  return `${firstLine.slice(0, limit - 1)}…`;
}

/** Whether an item matches a free-text query. */
export function matchesSearch(item: MemoryItem, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle.length === 0) return true;
  return (
    item.content.toLowerCase().includes(needle) ||
    item.agentId.toLowerCase().includes(needle) ||
    item.kind.toLowerCase().includes(needle)
  );
}

/** Counts the six predicates over a set of items. */
export function countItems(
  items: readonly MemoryItem[],
  inContext: ReadonlySet<string>,
  conflicted: ReadonlySet<string>,
): MemoryCounts {
  const counts: MemoryCounts = {
    available: items.length,
    inContext: 0,
    proposed: 0,
    admitted: 0,
    conflicted: 0,
    superseded: 0,
    rejected: 0,
  };
  for (const item of items) {
    if (inContext.has(item.itemId)) counts.inContext += 1;
    if (conflicted.has(item.itemId)) counts.conflicted += 1;
    if (item.status === 'proposed') counts.proposed += 1;
    else if (item.status === 'admitted') counts.admitted += 1;
    else if (item.status === 'superseded') counts.superseded += 1;
    else if (item.status === 'rejected') counts.rejected += 1;
  }
  return counts;
}

/**
 * Turns the cache into the picture.
 *
 * ## Authorship links are off by default, and that is a density decision
 *
 * Every memory item has exactly one author, so drawing the agent → item edge
 * for all of them adds one edge per node and turns every agent into a hub with
 * degree equal to its entire output. On a task with two hundred items that is
 * two hundred lines converging on four points, which hides the item-to-item
 * structure that the view is *for*.
 *
 * The information is not lost by omitting them: every node is already drawn in
 * its author's colour, which is the same fact rendered without the lines. The
 * toggle exists because "show me everything agent B touched" is a real question
 * and the explicit links answer it better than colour does.
 */
export function projectGraph(feed: FeedState, filter: MemoryFilter = {}): MemoryGraphView {
  const everyItem = [...feed.items.values()];
  const inContext = new Map(feed.inContext.map((entry) => [entry.itemId, entry]));
  const inContextIds = new Set(inContext.keys());

  // Conflict is a property of the pair, so it is read off both the item's own
  // `conflictsWith` and the stored `contradicts` edges. Either alone would
  // undercount: the field records what the writer knew, the edge records what
  // somebody established afterwards.
  const conflicted = new Set<string>();
  for (const item of everyItem) {
    if (item.conflictsWith.length > 0) conflicted.add(item.itemId);
  }
  for (const edge of feed.edges.values()) {
    if (edge.kind === 'contradicts') {
      conflicted.add(edge.fromItem);
      conflicted.add(edge.toItem);
    }
  }

  const totals = countItems(everyItem, inContextIds, conflicted);

  const agents = filter.agents;
  const kinds = filter.kinds;
  const search = filter.search ?? '';
  const kept = everyItem.filter(
    (item) =>
      (!agents || agents.size === 0 || agents.has(item.agentId)) &&
      (!kinds || kinds.size === 0 || kinds.has(item.kind)) &&
      matchesSearch(item, search),
  );
  const keptIds = new Set(kept.map((item) => item.itemId));
  const counts = countItems(kept, inContextIds, conflicted);

  const nodes: MemoryGraphNode[] = [];
  const edges: MemoryGraphEdge[] = [];
  const seen = new Set<string>();

  const push = (node: MemoryGraphNode) => {
    if (seen.has(node.id)) return;
    seen.add(node.id);
    nodes.push(node);
  };

  for (const item of kept) {
    push({
      id: item.itemId,
      label: nodeLabel(item.content),
      kind: 'memory',
      agentId: item.agentId,
      item,
      degree: 0,
    });
  }

  // The stored relations between items, kept only where both ends survived the
  // filter. An edge to something filtered out is not drawn as a stub: a line
  // running off into nothing is a worse answer than no line.
  for (const edge of feed.edges.values()) {
    if (!keptIds.has(edge.fromItem) || !keptIds.has(edge.toItem)) continue;
    edges.push({
      id: edge.edgeId,
      source: edge.fromItem,
      target: edge.toItem,
      kind: edge.kind,
      agentId: edge.agentId,
    });
  }

  // Agents and the task they work on. The agent nodes are drawn whenever any of
  // their work survived the filter, so the legend and the canvas agree.
  const activeAgents = new Set(kept.map((item) => item.agentId));
  if (activeAgents.size > 0) {
    const scope = kept[0].scope;
    const taskId = scopeNodeId(scope);
    push({
      id: taskId,
      label: scopeLabel(scope),
      kind: 'task',
      agentId: null,
      degree: 0,
    });
    for (const agentId of [...activeAgents].sort()) {
      const nodeId = agentNodeId(agentId);
      push({ id: nodeId, label: agentId, kind: 'agent', agentId, degree: 0 });
      edges.push({
        id: `worksOn:${agentId}`,
        source: nodeId,
        target: taskId,
        kind: 'worksOn',
        agentId,
      });
    }
    if (filter.includeAuthorship) {
      for (const item of kept) {
        edges.push({
          id: `authored:${item.itemId}`,
          source: agentNodeId(item.agentId),
          target: item.itemId,
          kind: 'authored',
          agentId: item.agentId,
        });
      }
    }
  }

  // Canonical material: shared, uncoloured, and reached by an agent-coloured
  // assertion. One node per source *version* — the sha is the version, so two
  // items citing different extractions of the same file are two nodes, which is
  // the honest picture.
  if (filter.includeCanonical !== false) {
    for (const item of kept) {
      for (const source of item.sources) {
        const nodeId = sourceNodeId(source.sha256);
        push({
          id: nodeId,
          label: `${source.sha256.slice(0, 8)} · ${source.locator}`,
          kind: 'source',
          agentId: null,
          degree: 0,
        });
        edges.push({
          id: `cites:${item.itemId}:${source.sha256}:${source.locator}`,
          source: item.itemId,
          target: nodeId,
          kind: 'references',
          agentId: item.agentId,
        });
      }
      for (const artifact of item.artifacts) {
        const nodeId = artifactNodeId(artifact.artifactId, artifact.revision);
        push({
          id: nodeId,
          label: `${artifact.artifactId} r${artifact.revision}`,
          kind: 'artifact',
          agentId: null,
          degree: 0,
        });
        edges.push({
          id: `produces:${item.itemId}:${nodeId}`,
          source: item.itemId,
          target: nodeId,
          kind: 'references',
          agentId: item.agentId,
        });
      }
    }
  }

  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (const edge of edges) {
    const from = byId.get(edge.source);
    const to = byId.get(edge.target);
    if (from) from.degree += 1;
    if (to) to.degree += 1;
  }

  return { nodes, edges, counts, totals, inContext };
}

/**
 * Above this many memory items, the default view aggregates instead of drawing
 * every one.
 *
 * Not a performance limit — the canvas holds its frame rate well past this.
 * It is a *legibility* limit, and it was measured by looking: at five hundred
 * nodes and fifteen hundred edges the picture is a mat of overlapping captions
 * with a coloured cloud behind it, and no amount of separation or edge routing
 * rescues it, because the problem is that five hundred things are being shown
 * to somebody who can hold about seven.
 *
 * Eighty is roughly where a force layout stops having visible structure and
 * starts having texture.
 */
export const AGGREGATE_ABOVE = 80;

/**
 * How many of a parent items are revealed when it is opened.
 *
 * Opening a bundle of a hundred and getting a hundred loose nodes is the same
 * problem one level down: the reader clicked to understand something and was
 * handed the tangle again. So the reveal is bounded too, and what is left over
 * stays bundled as a smaller node that can itself be opened.
 *
 * Twenty-four is about what fits in one comfortable ring around a parent at
 * readable label size.
 */
export const CHILDREN_SHOWN = 24;

/**
 * How much a reader is likely to care about one item, for choosing which of a
 * parent children to reveal first.
 *
 * Ordered by what needs a decision: a contradiction is something somebody has
 * to resolve, a proposal is something nobody has corroborated yet, and what the
 * model is actually reading right now is what an answer rests on. Established
 * facts that nothing disagrees with are the least urgent thing on the screen,
 * which is exactly why they are the ones that can stay in the bundle.
 */
function urgency(node: MemoryGraphNode, inContext: ReadonlySet<string>): number {
  const item = node.item;
  if (!item) return 0;
  let score = 0;
  if (item.conflictsWith.length > 0) score += 8;
  if (item.status === 'proposed') score += 4;
  if (inContext.has(item.itemId)) score += 3;
  if (item.status === 'rejected') score += 2;
  if (item.status === 'superseded') score += 1;
  return score;
}

/**
 * Collapses each agent's memory into one node.
 *
 * ## Why by agent
 *
 * Because "who established this" is the question this view exists to answer,
 * and because it is the grouping a person can hold in their head: four or five
 * agents, each with a colour and a count. Grouping by kind would make ten
 * buckets nobody asked about; grouping by cluster analysis would make groups
 * that change shape every time a fact arrives, which is the opposite of a
 * picture you can point at.
 *
 * ## What a collapsed edge means, exactly
 *
 * Every stored relation between two agents' items becomes one line carrying a
 * `weight` — the number of relations it stands for — and that number is drawn
 * as text. It is emphatically not a new semantic record: nothing is invented,
 * nothing is merged in the store, and expanding the cluster shows the same
 * relations individually. Relations *within* one agent's own work are not drawn
 * at all at this level; they appear when the cluster is opened.
 *
 * `expanded` names the agents to leave uncollapsed, so a reader can open one
 * cluster and see its items against the others' summaries.
 */
export function aggregateByAgent(
  view: MemoryGraphView,
  expanded: ReadonlySet<string> = new Set(),
): MemoryGraphView {
  const memory = view.nodes.filter((node) => node.kind === 'memory');
  const collapsing = new Map<string, MemoryGraphNode[]>();
  /** The opened agents, and the items now hanging off them. */
  const opened = new Map<string, MemoryGraphNode[]>();
  for (const node of memory) {
    const agentId = node.agentId;
    if (agentId === null) continue;
    const into = expanded.has(agentId) ? opened : collapsing;
    const held = into.get(agentId);
    if (held) held.push(node);
    else into.set(agentId, [node]);
  }
  if (collapsing.size === 0 && opened.size === 0) return view;

  // Bound each reveal. Anything past the budget goes back into a smaller
  // bundle hanging off the same parent, which can be opened in turn.
  const inContextIds = new Set(view.inContext.keys());
  const overflow = new Map<string, MemoryGraphNode[]>();
  for (const [agentId, group] of opened) {
    if (group.length <= CHILDREN_SHOWN) continue;
    const ranked = [...group].sort(
      (a, b) => urgency(b, inContextIds) - urgency(a, inContextIds) || (a.id < b.id ? -1 : 1),
    );
    opened.set(agentId, ranked.slice(0, CHILDREN_SHOWN));
    overflow.set(agentId, ranked.slice(CHILDREN_SHOWN));
  }

  /**
   * Where each collapsed item now lives.
   *
   * The agent's own node is folded in too. A bundle of an agent's work drawn
   * *beside* a node standing for that same agent is the same thing twice, and
   * the reader has to work out that the pink bubble and the pink hexagon
   * labelled `planner` are one idea. When the work is collapsed, the bundle is
   * the agent, and it inherits the agent's link to the task.
   */
  const clusterOf = new Map<string, string>();
  for (const [agentId, group] of collapsing) {
    const cluster = `cluster:${agentId}`;
    for (const node of group) clusterOf.set(node.id, cluster);
    clusterOf.set(agentNodeId(agentId), cluster);
  }
  for (const [agentId, group] of overflow) {
    const rest = `rest:${agentId}`;
    for (const node of group) clusterOf.set(node.id, rest);
  }

  const nodes: MemoryGraphNode[] = view.nodes.filter((node) => !clusterOf.has(node.id));

  for (const [agentId, group] of [...collapsing].sort(([a], [b]) => (a < b ? -1 : 1))) {
    // The summary says what is *in* the bundle, in the terms the header uses,
    // so opening it is a decision rather than a surprise.
    const status = (want: string) =>
      group.filter((node) => node.item?.status === want).length;
    const admitted = status('admitted');
    const proposed = status('proposed');
    const superseded = status('superseded');
    const rejected = status('rejected');
    const conflicted = group.filter(
      (node) => (node.item?.conflictsWith.length ?? 0) > 0,
    ).length;
    const inContext = group.filter((node) => view.inContext.has(node.id)).length;

    // The breakdown, not just a total. A bundle that says only "100 items"
    // makes the reader open it to find out whether anything in it needs
    // attention; one that says "12 proposed · 3 conflicts" answers that from
    // the outside, which is the entire job of a summary. Zeroes are left out
    // rather than printed, so what is shown is what is there.
    const parts: string[] = [];
    if (admitted > 0) parts.push(`${admitted} established`);
    if (proposed > 0) parts.push(`${proposed} proposed`);
    if (conflicted > 0) parts.push(`${conflicted} conflicting`);
    if (superseded > 0) parts.push(`${superseded} superseded`);
    if (rejected > 0) parts.push(`${rejected} rejected`);
    if (inContext > 0) parts.push(`${inContext} in context`);
    nodes.push({
      id: `cluster:${agentId}`,
      label: agentId,
      kind: 'cluster',
      agentId,
      count: group.length,
      summary: parts.length > 0 ? parts.join(' · ') : `${group.length} items`,
      degree: 0,
    });
  }

  for (const [agentId, group] of [...overflow].sort(([a], [b]) => (a < b ? -1 : 1))) {
    const proposed = group.filter((node) => node.item?.status === 'proposed').length;
    const conflicted = group.filter((node) => (node.item?.conflictsWith.length ?? 0) > 0).length;
    const parts: string[] = [];
    if (proposed > 0) parts.push(`${proposed} proposed`);
    if (conflicted > 0) parts.push(`${conflicted} conflicting`);
    nodes.push({
      id: `rest:${agentId}`,
      // Named for what it is rather than for the agent again: the parent is
      // already on screen with that name, and repeating it would read as a
      // second agent.
      label: `${group.length} more`,
      kind: 'cluster',
      agentId,
      count: group.length,
      summary: parts.length > 0 ? parts.join(' · ') : 'nothing needing attention',
      degree: 0,
    });
  }

  // Edges are re-pointed at whichever cluster now holds each end, then merged.
  const merged = new Map<string, MemoryGraphEdge>();
  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (const edge of view.edges) {
    const source = clusterOf.get(edge.source) ?? edge.source;
    const target = clusterOf.get(edge.target) ?? edge.target;
    // A relation between two items of the same collapsed agent is inside the
    // bundle. Drawing it as a self-loop on the cluster would say something the
    // reader cannot act on until they open it.
    if (source === target) continue;
    if (!byId.has(source) || !byId.has(target)) continue;

    const collapsed = source !== edge.source || target !== edge.target;
    if (!collapsed) {
      merged.set(edge.id, edge);
      continue;
    }
    // The agent's link to its task survives the fold as itself: it is one
    // relation before and after, and calling it a bundle of one would put a
    // meaningless count on it.
    if (edge.kind === 'worksOn') {
      merged.set(`worksOn:${source}`, { ...edge, id: `worksOn:${source}`, source, target });
      continue;
    }
    const key = `between:${source} ${target}`;
    const held = merged.get(key);
    if (held) held.weight = (held.weight ?? 1) + 1;
    else
      merged.set(key, {
        id: key,
        source,
        target,
        kind: 'between',
        // The bundle has many authors, so it claims none. Attributing it to
        // whichever agent happened to draw the first relation would be a
        // statement nothing recorded.
        agentId: null,
        weight: 1,
      });
  }

  // An opened parent keeps its children on a short leash.
  //
  // Without these spokes the items are loose in the layout, and opening a
  // bundle looks like the graph exploding rather than like one thing being
  // unpacked: the items drift wherever the force layout puts them and nothing
  // on screen says which parent they came out of. The spoke is what makes the
  // group read as a group, and it is also what the simulation uses to hold
  // them near their parent.
  for (const [agentId] of overflow) {
    const parent = agentNodeId(agentId);
    const rest = `rest:${agentId}`;
    if (!byId.has(parent) || !byId.has(rest)) continue;
    merged.set(rest, {
      id: `opened:${rest}`,
      source: parent,
      target: rest,
      kind: 'authored',
      agentId,
    });
  }

  for (const [agentId, group] of opened) {
    const parent = agentNodeId(agentId);
    if (!byId.has(parent)) continue;
    for (const child of group) {
      if (!byId.has(child.id)) continue;
      merged.set(`opened:${child.id}`, {
        id: `opened:${child.id}`,
        source: parent,
        target: child.id,
        kind: 'authored',
        agentId,
      });
    }
  }

  const edges = [...merged.values()];
  for (const node of nodes) node.degree = 0;
  for (const edge of edges) {
    const from = byId.get(edge.source);
    const to = byId.get(edge.target);
    if (from) from.degree += 1;
    if (to) to.degree += 1;
  }

  // Counts are untouched: collapsing changes how many *shapes* are drawn, not
  // how many things the task knows, and a header that fell to "5 items"
  // because five bundles were drawn would be a lie.
  return { nodes, edges, counts: view.counts, totals: view.totals, inContext: view.inContext };
}

/**
 * The local neighbourhood of one node, out to `depth` hops.
 *
 * ## Why a local view exists at all
 *
 * A force layout is legible up to a couple of hundred nodes and becomes an
 * undifferentiated tangle after that — the same reason `NotebookGraphPanel`
 * defaults to one node and its neighbours. This is also the honest answer to
 * the parts of the brief that cannot be guaranteed globally: a dense graph
 * cannot be drawn without crossings, and the remedy offered is a filtered view
 * rather than a claim that the tangle is fine.
 *
 * Undirected: "what is near this" does not care which way an edge points, and a
 * reader chasing a contradiction backwards would otherwise hit a wall.
 */
export function neighbourhood(
  view: MemoryGraphView,
  focusId: string,
  depth: number,
): MemoryGraphView {
  if (!view.nodes.some((node) => node.id === focusId)) return view;

  const adjacency = new Map<string, string[]>();
  const join = (from: string, to: string) => {
    const held = adjacency.get(from);
    if (held) held.push(to);
    else adjacency.set(from, [to]);
  };
  for (const edge of view.edges) {
    join(edge.source, edge.target);
    join(edge.target, edge.source);
  }

  const reached = new Set([focusId]);
  let frontier = [focusId];
  for (let hop = 0; hop < Math.max(0, depth); hop += 1) {
    const next: string[] = [];
    for (const id of frontier) {
      for (const neighbour of adjacency.get(id) ?? []) {
        if (reached.has(neighbour)) continue;
        reached.add(neighbour);
        next.push(neighbour);
      }
    }
    if (next.length === 0) break;
    frontier = next;
  }

  const nodes = view.nodes
    .filter((node) => reached.has(node.id))
    .map((node) => ({ ...node, degree: 0 }));
  const edges = view.edges.filter(
    (edge) => reached.has(edge.source) && reached.has(edge.target),
  );
  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (const edge of edges) {
    const from = byId.get(edge.source);
    const to = byId.get(edge.target);
    if (from) from.degree += 1;
    if (to) to.degree += 1;
  }

  const items = nodes.flatMap((node) => (node.item ? [node.item] : []));
  const conflicted = new Set<string>();
  for (const item of items) {
    if (item.conflictsWith.length > 0) conflicted.add(item.itemId);
  }
  for (const edge of edges) {
    if (edge.kind === 'contradicts') {
      conflicted.add(edge.source);
      conflicted.add(edge.target);
    }
  }

  return {
    nodes,
    edges,
    counts: countItems(items, new Set(view.inContext.keys()), conflicted),
    // Totals are deliberately *not* narrowed. The header says "12 of 340
    // shown", and a local view that also shrank the denominator would read as
    // a task that only ever knew twelve things.
    totals: view.totals,
    inContext: view.inContext,
  };
}

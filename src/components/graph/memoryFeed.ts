/**
 * The client half of the changefeed: a cache, and the rules for folding
 * changes into it.
 *
 * Pure, and separate from the panel that uses it, because every interesting
 * property here is a property of a sequence of messages rather than of a
 * rendered thing — and a sequence of messages can be tested without a DOM. The
 * failures this guards against are all silent ones: a duplicate that
 * double-counts, a reordered pair that resurrects an old revision, a gap that
 * leaves a deleted node on screen forever.
 *
 * ## What the transport is allowed to do to us
 *
 * The backend reads the feed from a cursor and answers in order. That is not
 * the same as the *client* seeing an ordered, exactly-once stream: a doorbell
 * can ring twice, a reply can arrive after a retry's reply, and a panel that
 * remounts re-asks from a cursor it already consumed. So this module assumes
 * only that each entry carries a `revision`, and derives everything else:
 *
 * - **Duplicates** are entries at or below the cursor already reached. Ignored,
 *   because applying one again is how a count drifts.
 * - **Out-of-order** is an item arriving at a revision older than the one
 *   cached. Ignored per item, not per batch — a stale copy of one node must not
 *   discard fresh copies of the others in the same message.
 * - **Gaps** are the backend's `reset`, raised when the cursor is below what the
 *   log still covers. Answered by re-snapshotting, never by applying what
 *   happens to have survived.
 *
 * ## Revocation cascades
 *
 * An `itemDropped` removes the node *and every edge touching it*. That is the
 * instruction "permission revocation must remove cached unauthorized nodes,
 * edges, labels and counts" taken literally. Leaving the edges would keep a
 * line on the canvas pointing into nothing, and — worse — would keep the node
 * in the degree counts and in the "3 contradictions" badge, which is a
 * disclosure about a row the reader just lost the right to.
 */
import type {
  ChangeBatch,
  InContextItem,
  MemoryEdge,
  MemoryItem,
  MemorySnapshot,
} from '../../services/memoryGraph.service';

/**
 * Everything the view draws from, and where in the feed it stands.
 *
 * Maps rather than arrays: every operation here is by id, and an array would
 * turn each one into a scan. The panel sorts once per render instead.
 */
export interface FeedState {
  items: ReadonlyMap<string, MemoryItem>;
  edges: ReadonlyMap<string, MemoryEdge>;
  /** Ask for changes after this. */
  cursor: number;
  /**
   * What the running turn carried. Comes from the snapshot only.
   *
   * The feed does not update it, and that is deliberate rather than an
   * omission: "what the model read" is a fact about a compiled turn, not about
   * the graph, and it changes when a turn is compiled — not when somebody
   * writes a memory item. Refreshing it on every edge insertion would make the
   * highlight flicker against a context that had not moved.
   */
  inContext: readonly InContextItem[];
  /**
   * The cursor is behind what can be replayed. The holder must re-snapshot.
   *
   * Kept as state rather than thrown, because the view stays useful while it
   * re-fetches: what is on screen was true, it is merely no longer known to be
   * current, and blanking the canvas would be a worse answer than a stale one
   * that says so.
   */
  stale: boolean;
}

/** An empty cache, before the first snapshot. */
export function emptyFeed(): FeedState {
  return {
    items: new Map(),
    edges: new Map(),
    cursor: 0,
    inContext: [],
    stale: false,
  };
}

/**
 * The cache a snapshot establishes.
 *
 * Replaces rather than merges. A snapshot is the whole authorised truth at its
 * cursor, so anything held that is not in it is something this reader may no
 * longer see — merging would preserve exactly the rows a re-snapshot exists to
 * shed.
 */
export function fromSnapshot(snapshot: MemorySnapshot): FeedState {
  const items = new Map(snapshot.items.map((item) => [item.itemId, item]));
  const edges = new Map(
    snapshot.edges
      // The backend already drops an edge with a hidden end. Filtered again
      // here because this is the invariant the renderer depends on, and a
      // renderer that trusts an invariant it does not enforce is one wire
      // change away from drawing a line to nowhere.
      .filter((edge) => items.has(edge.fromItem) && items.has(edge.toItem))
      .map((edge) => [edge.edgeId, edge] as const),
  );
  return {
    items,
    edges,
    cursor: snapshot.cursor,
    inContext: snapshot.inContext ?? [],
    stale: false,
  };
}

/**
 * Folds one batch in, returning the new cache.
 *
 * Returns the *same* object when nothing applied. The panel holds this in React
 * state, and a new object with identical contents would re-render the canvas
 * and restart its layout for a batch that changed nothing — which is what an
 * idle poll every second would otherwise do.
 */
export function applyBatch(state: FeedState, batch: ChangeBatch): FeedState {
  if (batch.reset) {
    // Nothing is applied. The entries in a reset batch are empty by contract,
    // and even if they were not, applying part of an unreplayable range is how
    // a cache ends up holding a mixture of two different moments.
    return state.stale ? state : { ...state, stale: true };
  }

  const items = new Map(state.items);
  const edges = new Map(state.edges);
  let touched = false;

  for (const entry of batch.entries) {
    // A duplicate delivery, or a reply to a request we have already moved past.
    if (entry.revision <= state.cursor) continue;

    switch (entry.change) {
      case 'itemChanged': {
        const held = items.get(entry.item.itemId);
        // An older revision of something already held is a reordered message,
        // not news. Equal revisions are the same write arriving twice.
        if (held && held.revision >= entry.item.revision) break;
        items.set(entry.item.itemId, entry.item);
        touched = true;
        break;
      }
      case 'itemDropped': {
        if (!items.delete(entry.itemId)) break;
        // The cascade. Every edge that touched it goes, so no line survives
        // pointing into a node this reader may no longer see — and so the
        // degree counts drawn beside the remaining nodes are honest.
        for (const [edgeId, edge] of edges) {
          if (edge.fromItem === entry.itemId || edge.toItem === entry.itemId) {
            edges.delete(edgeId);
          }
        }
        touched = true;
        break;
      }
      case 'edgeChanged': {
        // Both ends must be present. The backend guarantees it for edges it
        // sends, but a coalesced batch can carry an edge *and* a later drop of
        // one of its ends; whichever order they are applied in, the edge must
        // not outlive the node.
        if (!items.has(entry.edge.fromItem) || !items.has(entry.edge.toItem)) break;
        edges.set(entry.edge.edgeId, entry.edge);
        touched = true;
        break;
      }
      case 'edgeDropped': {
        if (edges.delete(entry.edgeId)) touched = true;
        break;
      }
    }
  }

  // A batch can apply nothing and still move the cursor — every entry may have
  // been a duplicate, or about rows this reader cannot see. The cursor must
  // advance anyway, or the next poll asks the same question forever.
  const cursor = Math.max(state.cursor, batch.cursor);
  if (!touched) {
    return cursor === state.cursor ? state : { ...state, cursor };
  }

  // One last sweep: an edge whose end was removed later in this same batch.
  for (const [edgeId, edge] of edges) {
    if (!items.has(edge.fromItem) || !items.has(edge.toItem)) edges.delete(edgeId);
  }

  return { items, edges, cursor, inContext: state.inContext, stale: false };
}

/**
 * Whether a doorbell is worth answering.
 *
 * The event says the graph reached revision N. If the cache is already at or
 * past N, the change was in another scope or was one this reader cannot see,
 * and asking again would be a round trip for nothing. At sixty writes a second
 * across a busy workspace that is the difference between a quiet panel and one
 * that re-fetches continuously.
 */
export function needsFetch(state: FeedState, revision: number): boolean {
  return state.stale || revision > state.cursor;
}

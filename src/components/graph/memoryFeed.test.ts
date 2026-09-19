/**
 * What these pin: the four ways a stream of changes can quietly corrupt a
 * cache, and the one way a permission change has to reach it.
 *
 * Every failure here is silent in production. A double-applied duplicate shows
 * up as a count that is wrong by one; a reordered pair shows up as a node that
 * reverted to an older revision; a gap shows up as a deleted node that stays on
 * screen forever. None of them throws, and none of them looks broken.
 */
import { describe, expect, it } from 'vitest';
import type {
  ChangeBatch,
  MemoryEdge,
  MemoryItem,
  MemorySnapshot,
} from '../../services/memoryGraph.service';
import { applyBatch, emptyFeed, fromSnapshot, needsFetch } from './memoryFeed';

function item(itemId: string, revision = 1, content = 'a fact'): MemoryItem {
  return {
    itemId,
    revision,
    kind: 'fact',
    agentId: 'ag-a',
    scope: { kind: 'task', taskId: 'task-1' },
    classification: 'internal',
    acl: { clearedRoles: ['employee'], projectId: null, owner: null },
    provenance: { kind: 'operator', user_id: 'priya' },
    content,
    sources: [],
    artifacts: [],
    status: 'admitted',
    validFrom: '2026-01-01T00:00:00Z',
    conflictsWith: [],
    causalParents: [],
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
  };
}

function edge(edgeId: string, fromItem: string, toItem: string): MemoryEdge {
  return {
    edgeId,
    fromItem,
    toItem,
    kind: 'contradicts',
    agentId: 'ag-a',
    scope: { kind: 'task', taskId: 'task-1' },
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function snapshot(items: MemoryItem[], edges: MemoryEdge[], cursor = 10): MemorySnapshot {
  return { items, edges, cursor, inContext: [] };
}

function batch(entries: ChangeBatch['entries'], cursor: number): ChangeBatch {
  return { entries, cursor, hasMore: false, reset: false };
}

describe('a snapshot establishes the cache', () => {
  it('replaces rather than merges, so a re-snapshot sheds what was revoked', () => {
    const before = fromSnapshot(snapshot([item('a'), item('b')], []));
    expect(before.items.size).toBe(2);

    // The reader may no longer see b. A merge would keep it forever.
    const after = fromSnapshot(snapshot([item('a')], [], 20));
    expect([...after.items.keys()]).toEqual(['a']);
    expect(after.cursor).toBe(20);
  });

  it('refuses an edge whose ends are not both present', () => {
    const state = fromSnapshot(snapshot([item('a')], [edge('e1', 'a', 'gone')]));
    expect(state.edges.size).toBe(0);
  });
});

describe('duplicate delivery', () => {
  it('ignores an entry at or below the cursor already reached', () => {
    const state = fromSnapshot(snapshot([item('a', 1)], [], 10));
    // The same write arriving again, at a revision already consumed.
    const replayed = applyBatch(
      state,
      batch([{ revision: 9, at: 'x', change: 'itemChanged', item: item('a', 5) }], 10),
    );
    expect(replayed.items.get('a')!.revision).toBe(1);
  });

  it('applying the same batch twice is applying it once', () => {
    const state = fromSnapshot(snapshot([], [], 0));
    const incoming = batch(
      [{ revision: 1, at: 'x', change: 'itemChanged', item: item('a', 1) }],
      1,
    );
    const once = applyBatch(state, incoming);
    const twice = applyBatch(once, incoming);
    expect(twice.items.size).toBe(1);
    expect(twice.cursor).toBe(1);
  });

  it('returns the identical object when nothing applied, so React does not redraw', () => {
    const state = fromSnapshot(snapshot([item('a')], [], 10));
    const unchanged = applyBatch(state, batch([], 10));
    expect(unchanged).toBe(state);
  });
});

describe('out-of-order delivery', () => {
  it('never lets an older revision overwrite a newer one', () => {
    const state = fromSnapshot(snapshot([], [], 0));
    const applied = applyBatch(
      state,
      batch(
        [
          { revision: 2, at: 'x', change: 'itemChanged', item: item('a', 4, 'the new text') },
          { revision: 3, at: 'x', change: 'itemChanged', item: item('a', 2, 'the old text') },
        ],
        3,
      ),
    );
    expect(applied.items.get('a')!.revision).toBe(4);
    expect(applied.items.get('a')!.content).toBe('the new text');
  });

  it('drops only the stale entry, not the rest of its batch', () => {
    const state = fromSnapshot(snapshot([item('a', 4)], [], 1));
    const applied = applyBatch(
      state,
      batch(
        [
          { revision: 2, at: 'x', change: 'itemChanged', item: item('a', 2) },
          { revision: 3, at: 'x', change: 'itemChanged', item: item('b', 1) },
        ],
        3,
      ),
    );
    expect(applied.items.get('a')!.revision).toBe(4);
    expect(applied.items.has('b')).toBe(true);
  });
});

describe('gap recovery', () => {
  it('marks the cache stale on a reset and applies nothing', () => {
    const state = fromSnapshot(snapshot([item('a')], [], 10));
    const after = applyBatch(state, {
      entries: [{ revision: 11, at: 'x', change: 'itemChanged', item: item('b') }],
      cursor: 99,
      hasMore: false,
      reset: true,
    });
    expect(after.stale).toBe(true);
    // The old picture is kept: it was true, it is merely no longer known to be
    // current, and blanking it would be a worse answer than a stale one.
    expect(after.items.has('a')).toBe(true);
    expect(after.items.has('b')).toBe(false);
  });

  it('a fresh snapshot clears the stale flag', () => {
    const stale = applyBatch(fromSnapshot(snapshot([item('a')], [], 10)), {
      entries: [],
      cursor: 99,
      hasMore: false,
      reset: true,
    });
    expect(stale.stale).toBe(true);
    expect(fromSnapshot(snapshot([item('a')], [], 99)).stale).toBe(false);
  });
});

describe('revocation reaches the cache', () => {
  it('removes the node and every edge touching it', () => {
    const state = fromSnapshot(
      snapshot([item('a'), item('b'), item('c')], [edge('ab', 'a', 'b'), edge('bc', 'b', 'c')]),
    );
    expect(state.edges.size).toBe(2);

    const after = applyBatch(
      state,
      batch([{ revision: 11, at: 'x', change: 'itemDropped', itemId: 'b' }], 11),
    );

    expect(after.items.has('b')).toBe(false);
    // Both edges touched b. A line pointing at a node this reader may no longer
    // see is a disclosure as well as a drawing error.
    expect(after.edges.size).toBe(0);
  });

  it('leaves the counts honest — the dropped node is not in any of them', () => {
    const state = fromSnapshot(snapshot([item('a'), item('b')], [edge('ab', 'a', 'b')]));
    const after = applyBatch(
      state,
      batch([{ revision: 11, at: 'x', change: 'itemDropped', itemId: 'a' }], 11),
    );
    expect(after.items.size).toBe(1);
    expect(after.edges.size).toBe(0);
  });

  it('does not keep an edge whose end is dropped later in the same batch', () => {
    const state = fromSnapshot(snapshot([item('a'), item('b')], []));
    const after = applyBatch(
      state,
      batch(
        [
          { revision: 11, at: 'x', change: 'edgeChanged', edge: edge('ab', 'a', 'b') },
          { revision: 12, at: 'x', change: 'itemDropped', itemId: 'b' },
        ],
        12,
      ),
    );
    expect(after.edges.size).toBe(0);
  });

  it('ignores an edge arriving for a node it does not hold', () => {
    const state = fromSnapshot(snapshot([item('a')], []));
    const after = applyBatch(
      state,
      batch([{ revision: 11, at: 'x', change: 'edgeChanged', edge: edge('ax', 'a', 'x') }], 11),
    );
    expect(after.edges.size).toBe(0);
  });
});

describe('the cursor', () => {
  it('advances even when every entry was ignored', () => {
    const state = fromSnapshot(snapshot([item('a', 3)], [], 5));
    const after = applyBatch(
      state,
      batch([{ revision: 6, at: 'x', change: 'itemChanged', item: item('a', 1) }], 40),
    );
    // Nothing applied — the entry was stale — but the cursor must still move,
    // or the next poll asks the same question forever.
    expect(after.cursor).toBe(40);
    expect(after.items.get('a')!.revision).toBe(3);
  });

  it('never goes backwards', () => {
    const state = fromSnapshot(snapshot([], [], 50));
    expect(applyBatch(state, batch([], 10)).cursor).toBe(50);
  });
});

describe('the doorbell', () => {
  it('is worth answering only when the graph is ahead of the cache', () => {
    const state = fromSnapshot(snapshot([], [], 10));
    expect(needsFetch(state, 11)).toBe(true);
    expect(needsFetch(state, 10)).toBe(false);
    expect(needsFetch(state, 3)).toBe(false);
  });

  it('is always worth answering when the cache is stale', () => {
    const stale = { ...emptyFeed(), cursor: 100, stale: true };
    expect(needsFetch(stale, 1)).toBe(true);
  });
});

describe('a reconnect after missing revisions', () => {
  it('catches up across several batches and ends level with the truth', () => {
    // The reader has an old cursor and drains in small batches, exactly as the
    // panel does when it comes back from being disconnected.
    let state = fromSnapshot(snapshot([item('a', 1)], [], 1));
    const arriving = [
      batch([{ revision: 2, at: 'x', change: 'itemChanged', item: item('b', 1) }], 2),
      batch([{ revision: 3, at: 'x', change: 'itemChanged', item: item('c', 1) }], 3),
      batch([{ revision: 4, at: 'x', change: 'edgeChanged', edge: edge('bc', 'b', 'c') }], 4),
      batch([{ revision: 5, at: 'x', change: 'itemDropped', itemId: 'a' }], 5),
    ];
    for (const incoming of arriving) state = applyBatch(state, incoming);

    expect([...state.items.keys()].sort()).toEqual(['b', 'c']);
    expect(state.edges.size).toBe(1);
    expect(state.cursor).toBe(5);
    expect(state.stale).toBe(false);
  });
});

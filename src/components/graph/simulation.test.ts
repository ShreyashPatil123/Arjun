/**
 * What these pin: a graph that draws its nodes on top of one another, and a
 * graph that rearranges itself every time it is looked at.
 *
 * The first is why this module exists. The layout it replaces treated every
 * node as a point, so a notebook of eleven files painted eleven names in a
 * heap and nothing in the picture could be read. The second is the rule the old
 * layout held and this one has to keep: a person must be able to point at a
 * cluster, come back, and find it where they left it.
 */
import { describe, expect, it } from 'vitest';
import { ALPHA_MIN, GraphSimulation, seedFrom, type SimLink } from './simulation';

const BOX = { width: 400, height: 300 };

function nodes(count: number, radius = 8) {
  return Array.from({ length: count }, (_, i) => ({ id: `n${i}`, radius }));
}

/** Runs to rest, or gives up — a simulation that never settles is a bug. */
function settle(simulation: GraphSimulation, limit = 2000): number {
  let ticks = 0;
  while (simulation.step() && ticks < limit) ticks += 1;
  return ticks;
}

describe('GraphSimulation', () => {
  it('comes to rest instead of running forever', () => {
    const simulation = new GraphSimulation(nodes(12), [], BOX);
    const ticks = settle(simulation);
    expect(ticks).toBeLessThan(2000);
    expect(simulation.alpha).toBeLessThan(ALPHA_MIN);
  });

  it('settles to the same arrangement every time', () => {
    const links: SimLink[] = [
      { source: 'n0', target: 'n1' },
      { source: 'n1', target: 'n2' },
    ];
    const first = new GraphSimulation(nodes(6), links, BOX);
    const second = new GraphSimulation(nodes(6), links, BOX);
    settle(first);
    settle(second);
    expect(first.nodes.map((n) => [n.id, n.x, n.y])).toEqual(
      second.nodes.map((n) => [n.id, n.x, n.y]),
    );
  });

  it('does not depend on the order the nodes arrived in', () => {
    const links: SimLink[] = [{ source: 'n0', target: 'n3' }];
    const forward = new GraphSimulation(nodes(6), links, BOX);
    const reversed = new GraphSimulation([...nodes(6)].reverse(), links, BOX);
    settle(forward);
    settle(reversed);
    const sorted = (s: GraphSimulation) =>
      [...s.nodes].sort((a, b) => a.id.localeCompare(b.id)).map((n) => [n.id, n.x, n.y]);
    expect(sorted(reversed)).toEqual(sorted(forward));
  });

  it('keeps nodes at least their radii apart', () => {
    // The whole reason for the collide force. Eight nodes of radius 30 in a
    // 400x300 box have room, and none of them may overlap.
    const simulation = new GraphSimulation(nodes(8, 30), [], BOX);
    settle(simulation);
    for (let i = 0; i < simulation.nodes.length; i += 1) {
      for (let j = i + 1; j < simulation.nodes.length; j += 1) {
        const a = simulation.nodes[i];
        const b = simulation.nodes[j];
        // A pixel of tolerance: the constraint is resolved by relaxation rather
        // than solved exactly, and a pixel is not a legibility problem.
        expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(a.radius + b.radius - 1);
      }
    }
  });

  it('keeps every node inside the box, whole', () => {
    const simulation = new GraphSimulation(nodes(10, 20), [], BOX);
    settle(simulation);
    for (const node of simulation.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(node.radius - 0.001);
      expect(node.x).toBeLessThanOrEqual(BOX.width - node.radius + 0.001);
      expect(node.y).toBeGreaterThanOrEqual(node.radius - 0.001);
      expect(node.y).toBeLessThanOrEqual(BOX.height - node.radius + 0.001);
    }
  });

  it('holds a pinned node exactly where it was put', () => {
    const simulation = new GraphSimulation(nodes(6), [{ source: 'n0', target: 'n1' }], BOX);
    const held = simulation.find('n0');
    expect(held).toBeDefined();
    held!.fx = 123;
    held!.fy = 45;
    simulation.reheat();
    for (let i = 0; i < 100; i += 1) simulation.step();
    expect(held!.x).toBe(123);
    expect(held!.y).toBe(45);
  });

  it('reheats on demand and cools again, so a drag keeps it alive', () => {
    const simulation = new GraphSimulation(nodes(4), [], BOX);
    settle(simulation);
    expect(simulation.running).toBe(false);

    simulation.reheat();
    expect(simulation.running).toBe(true);
    for (let i = 0; i < 500; i += 1) simulation.step();
    // Held warm by the target alone: a drag must not time out under the hand.
    expect(simulation.running).toBe(true);

    simulation.cool();
    expect(settle(simulation)).toBeLessThan(2000);
  });

  it('produces no NaN, whatever it is given', () => {
    // Every node on the same spot, a self-link, and a link to a node that is
    // not in the graph.
    const simulation = new GraphSimulation(
      [
        { id: 'a', radius: 5 },
        { id: 'b', radius: 5 },
        { id: 'c', radius: 5 },
      ],
      [
        { source: 'a', target: 'a' },
        { source: 'a', target: 'missing' },
        { source: 'a', target: 'b', weight: 40 },
      ],
      { width: 1, height: 1 },
    );
    settle(simulation);
    for (const node of simulation.nodes) {
      expect(Number.isFinite(node.x)).toBe(true);
      expect(Number.isFinite(node.y)).toBe(true);
    }
  });

  it('places a lone node without dividing by zero', () => {
    const simulation = new GraphSimulation([{ id: 'only', radius: 4 }], [], BOX);
    settle(simulation);
    const only = simulation.find('only');
    expect(Number.isFinite(only!.x)).toBe(true);
    expect(Number.isFinite(only!.y)).toBe(true);
  });

  it('has nothing to do with an empty graph', () => {
    const simulation = new GraphSimulation([], [], BOX);
    expect(simulation.nodes).toHaveLength(0);
    expect(settle(simulation)).toBeLessThan(2000);
  });
});

describe('seedFrom', () => {
  it('is stable for the same ids and different for different ones', () => {
    expect(seedFrom(['a', 'b'])).toBe(seedFrom(['a', 'b']));
    expect(seedFrom(['a', 'b'])).not.toBe(seedFrom(['a', 'c']));
  });
});

/**
 * The property the memory view depends on: a graph that is rebuilt as facts
 * arrive must not throw away the arrangement each time.
 *
 * Without this, every committed memory item makes the whole picture jump, and
 * nothing stays put long enough for a person to point at it.
 */
describe('positions carried across a rebuild', () => {
  const chain: SimLink[] = [
    { source: 'n0', target: 'n1' },
    { source: 'n1', target: 'n2' },
  ];

  it('leaves surviving nodes exactly where they were', () => {
    const first = new GraphSimulation(nodes(4), chain, BOX);
    settle(first);
    const held = first.positionsById();

    const second = new GraphSimulation(nodes(4), chain, { ...BOX, positions: held });
    for (const node of second.nodes) {
      expect(node.x).toBeCloseTo(held.get(node.id)!.x);
      expect(node.y).toBeCloseTo(held.get(node.id)!.y);
    }
  });

  it('keeps them there when a new node arrives', () => {
    const first = new GraphSimulation(nodes(4), chain, BOX);
    settle(first);
    const held = first.positionsById();

    const second = new GraphSimulation(
      [...nodes(4), { id: 'fresh', radius: 8 }],
      [...chain, { source: 'n1', target: 'fresh' }],
      { ...BOX, positions: held, alpha: 0.3 },
    );

    // Everything that existed is untouched at the moment of the rebuild.
    for (const id of ['n0', 'n1', 'n2', 'n3']) {
      expect(second.find(id)!.x).toBeCloseTo(held.get(id)!.x);
      expect(second.find(id)!.y).toBeCloseTo(held.get(id)!.y);
    }
  });

  it('drops a new node beside its neighbour, not across the canvas', () => {
    const first = new GraphSimulation(nodes(6), chain, BOX);
    settle(first);
    const held = first.positionsById();

    const anchor = held.get('n1')!;
    const second = new GraphSimulation(
      [...nodes(6), { id: 'fresh', radius: 8 }],
      [...chain, { source: 'n1', target: 'fresh' }],
      { ...BOX, positions: held },
    );

    const fresh = second.find('fresh')!;
    // Its neighbour's centre plus a small jitter — not the seeding ring, which
    // would be a third of the canvas away and would drag the picture about as
    // it flew back.
    expect(Math.hypot(fresh.x - anchor.x, fresh.y - anchor.y)).toBeLessThan(20);
  });

  it('falls back to the ring for a new node joined to nothing', () => {
    const first = new GraphSimulation(nodes(4), chain, BOX);
    settle(first);
    const second = new GraphSimulation([...nodes(4), { id: 'orphan', radius: 8 }], chain, {
      ...BOX,
      positions: first.positionsById(),
    });
    const orphan = second.find('orphan')!;
    expect(Number.isFinite(orphan.x)).toBe(true);
    expect(Number.isFinite(orphan.y)).toBe(true);
  });

  it('starts warm rather than hot when asked, so a settled graph barely stirs', () => {
    const warm = new GraphSimulation(nodes(4), chain, { ...BOX, alpha: 0.1 });
    expect(warm.alpha).toBeCloseTo(0.1);
    const cold = new GraphSimulation(nodes(4), chain, BOX);
    expect(cold.alpha).toBe(1);
  });

  it('ignores a remembered position for a node that is no longer present', () => {
    const held = new Map([['ghost', { x: 5, y: 5 }]]);
    const simulation = new GraphSimulation(nodes(3), chain, { ...BOX, positions: held });
    expect(simulation.find('ghost')).toBeUndefined();
    expect(simulation.nodes).toHaveLength(3);
  });
});

describe('the viewport clamp', () => {
  it('holds nodes inside the box by default, as the notebook graph needs', () => {
    const simulation = new GraphSimulation(nodes(20), [], BOX);
    settle(simulation);
    for (const node of simulation.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.x).toBeLessThanOrEqual(BOX.width);
    }
  });

  it('lets the world grow when turned off', () => {
    // Twenty fat nodes cannot fit in a 400×300 box without overlapping. With
    // the clamp off they spread past its edge instead, which is what leaves
    // `labelGeometry.separate` a result it does not have to undo.
    const fat = Array.from({ length: 20 }, (_, i) => ({ id: `n${i}`, radius: 40 }));
    const simulation = new GraphSimulation(fat, [], { ...BOX, clampToViewport: false });
    settle(simulation);
    const escaped = simulation.nodes.some(
      (node) => node.x < 0 || node.x > BOX.width || node.y < 0 || node.y > BOX.height,
    );
    expect(escaped).toBe(true);
  });
});

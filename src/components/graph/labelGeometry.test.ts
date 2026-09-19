/**
 * What these pin: the four claims `labelGeometry.ts` makes, checked as geometry
 * rather than inspected by eye.
 *
 * A screenshot shows one arrangement at one size and a person can miss an
 * overlap in it. These check every pair, at sizes chosen to be awkward — a
 * label longer than the graph is wide, a node with forty edges, a window too
 * small to hold anything — and they fail loudly rather than looking slightly
 * wrong.
 *
 * The one claim deliberately *not* made is zero edge crossings, which is
 * impossible for a non-planar graph in 2D. `routeEdge` reports `clear: false`
 * when it cannot find a path, and the test below checks that it says so rather
 * than asserting the case never happens.
 */
import { describe, expect, it } from 'vitest';
import {
  type Box,
  type LaidOutNode,
  type Point,
  bodyBox,
  collisionRadius,
  drawnBoxes,
  groupParallel,
  labelBox,
  obstaclesFrom,
  overlaps,
  pairKey,
  parallelOffset,
  routeEdge,
  segmentHitsBox,
  selfLoopPath,
  separate,
  worldBounds,
} from './labelGeometry';

function node(
  id: string,
  x: number,
  y: number,
  { radius = 8, label = 'a node', labelled = true } = {},
): LaidOutNode {
  return {
    id,
    x,
    y,
    radius,
    // Roughly the 11px system font: about 6 world units per character.
    labelWidth: labelled ? label.length * 6 : 0,
    labelHeight: 13,
    labelled,
  };
}

/** Every pair of drawn boxes that overlap. Empty is the post-condition. */
function overlappingPairs(nodes: readonly LaidOutNode[]): string[] {
  const clashes: string[] = [];
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      for (const a of drawnBoxes(nodes[i])) {
        for (const b of drawnBoxes(nodes[j])) {
          if (overlaps(a, b)) clashes.push(`${nodes[i].id}×${nodes[j].id}`);
        }
      }
    }
  }
  return [...new Set(clashes)];
}

describe('label bounds', () => {
  it('reserves room for the label, not just the body', () => {
    const wide = node('n', 0, 0, { radius: 6, label: 'a label far wider than the node' });
    expect(collisionRadius(wide)).toBeGreaterThan(wide.radius);
    // The whole label has to fit inside the collision disc, or the collide
    // force is satisfied while two captions sit on top of each other.
    expect(collisionRadius(wide)).toBeGreaterThanOrEqual(wide.labelWidth / 2);
  });

  it('reserves nothing for a label that is not drawn', () => {
    const bare = node('n', 0, 0, { labelled: false });
    expect(labelBox(bare)).toBeNull();
    expect(drawnBoxes(bare)).toHaveLength(1);
    // A node with no caption is a plain disc. Padding it for an invisible
    // label would make the graph needlessly sparse.
    expect(collisionRadius(bare)).toBe(bare.radius);
  });

  it('puts the label where the canvas paints it — under the body', () => {
    const held = node('n', 100, 50, { radius: 10, label: 'name' });
    const box = labelBox(held)!;
    expect(box.y).toBeGreaterThan(bodyBox(held).y + bodyBox(held).height);
    expect(box.x + box.width / 2).toBeCloseTo(held.x);
  });
});

describe('separation after the simulation settles', () => {
  it('leaves nothing overlapping, including labels', () => {
    // Ten nodes stacked almost on top of each other: the state a settled
    // simulation can genuinely leave behind.
    const nodes = Array.from({ length: 10 }, (_, i) =>
      node(`n${i}`, 100 + i * 2, 100 + i, { label: `item number ${i}` }),
    );
    expect(overlappingPairs(nodes).length).toBeGreaterThan(0);

    const result = separate(nodes);
    expect(result.converged).toBe(true);
    expect(overlappingPairs(nodes)).toEqual([]);
  });

  it('separates long labels, which are the case that actually collides', () => {
    const nodes = [
      node('a', 0, 0, { label: 'the design pressure for the vessel is ten bar' }),
      node('b', 12, 6, { label: 'the design pressure for the vessel is 150 PSI' }),
      node('c', 20, 2, { label: 'a correction recorded by an operator at the site' }),
    ];
    separate(nodes);
    expect(overlappingPairs(nodes)).toEqual([]);
  });

  it('separates nodes sharing a position, without dividing by zero', () => {
    const nodes = [node('a', 50, 50), node('b', 50, 50), node('c', 50, 50)];
    separate(nodes);
    expect(overlappingPairs(nodes)).toEqual([]);
    for (const held of nodes) {
      expect(Number.isFinite(held.x)).toBe(true);
      expect(Number.isFinite(held.y)).toBe(true);
    }
  });

  it('is allowed to grow the world rather than re-stacking to fit', () => {
    // Everything crammed into a 40×40 box. The separated result cannot fit in
    // it, and squeezing it back would reintroduce exactly what was just fixed.
    const nodes = Array.from({ length: 8 }, (_, i) =>
      node(`n${i}`, 20 + (i % 3), 20 + i, { label: `a reasonably long label ${i}` }),
    );
    separate(nodes);
    const bounds = worldBounds(nodes);
    expect(bounds.width).toBeGreaterThan(40);
    expect(overlappingPairs(nodes)).toEqual([]);
  });

  it('converges on a graph the size of the stated target', () => {
    // 500 visible nodes, the figure the performance target names.
    const nodes = Array.from({ length: 500 }, (_, i) =>
      node(`n${i}`, (i % 25) * 30, Math.floor(i / 25) * 26, { label: `item ${i}` }),
    );
    const result = separate(nodes);
    expect(result.converged).toBe(true);
    expect(overlappingPairs(nodes)).toEqual([]);
  });
});

describe('edges do not pass through things', () => {
  const obstacle = node('blocker', 100, 100, { radius: 14, label: 'in the way' });

  it('detects a segment clipping the corner of a label box', () => {
    const box = labelBox(obstacle)!;
    const clipsCorner = segmentHitsBox(
      { x: box.x - 10, y: box.y - 10 },
      { x: box.x + 4, y: box.y + 4 },
      box,
    );
    expect(clipsCorner).toBe(true);
  });

  it('routes around a node sitting between the two ends', () => {
    const from = node('a', 20, 100);
    const to = node('b', 180, 100);
    const obstacles = obstaclesFrom([from, to, obstacle]);

    const route = routeEdge(from, to, obstacles, { endpoints: ['a', 'b'] });
    expect(route.clear).toBe(true);
    // Straight would have gone through the blocker, so it must have bent.
    expect(route.points.length).toBeGreaterThan(2);

    for (let i = 0; i + 1 < route.points.length; i += 1) {
      for (const box of drawnBoxes(obstacle)) {
        expect(segmentHitsBox(route.points[i], route.points[i + 1], box)).toBe(false);
      }
    }
  });

  it('leaves a clear edge straight', () => {
    const from = node('a', 0, 0);
    const to = node('b', 200, 0);
    const route = routeEdge(from, to, obstaclesFrom([from, to]), { endpoints: ['a', 'b'] });
    expect(route.clear).toBe(true);
    expect(route.points).toHaveLength(2);
  });

  it('does not treat its own endpoints as obstacles', () => {
    const from = node('a', 0, 0, { radius: 20, label: 'a wide label on the source' });
    const to = node('b', 60, 0, { radius: 20, label: 'a wide label on the target' });
    const route = routeEdge(from, to, obstaclesFrom([from, to]), { endpoints: ['a', 'b'] });
    // The two overlap each other's boxes; an edge between them is still a
    // straight line, because an edge is allowed to touch what it joins.
    expect(route.clear).toBe(true);
  });

  it('reports failure instead of claiming a route it did not find', () => {
    // A wall of blockers with no gap: no single-bend route exists.
    const from = node('a', 0, 0);
    const to = node('b', 300, 0);
    const wall = Array.from({ length: 60 }, (_, i) =>
      node(`w${i}`, 150, -600 + i * 20, { radius: 12, label: 'wall' }),
    );
    const route = routeEdge(from, to, obstaclesFrom([from, to, ...wall]), {
      endpoints: ['a', 'b'],
    });
    expect(route.clear).toBe(false);
    // And it still returns something drawable: an omitted edge is a worse lie
    // than a crossed one.
    expect(route.points.length).toBeGreaterThanOrEqual(2);
  });

  it('keeps a high-degree hub readable — every spoke clears every other node', () => {
    const hub = node('hub', 0, 0, { radius: 18, label: 'the hub' });
    const spokes = Array.from({ length: 40 }, (_, i) => {
      const angle = (Math.PI * 2 * i) / 40;
      return node(`s${i}`, Math.cos(angle) * 220, Math.sin(angle) * 220, {
        label: `spoke ${i}`,
      });
    });
    const all = [hub, ...spokes];
    separate(all);
    const obstacles = obstaclesFrom(all);

    let failures = 0;
    for (const spoke of spokes) {
      const route = routeEdge(hub, spoke, obstacles, {
        endpoints: ['hub', spoke.id],
        startClearance: hub.radius,
        endClearance: spoke.radius,
      });
      if (!route.clear) failures += 1;
    }
    // Radial spokes from a hub have a clear line by construction; any failure
    // here would mean the router is rejecting routes it should accept.
    expect(failures).toBe(0);
  });
});

describe('endpoints are clipped', () => {
  it('stops short of the node it points at, so an arrowhead is visible', () => {
    const from = node('a', 0, 0, { radius: 10 });
    const to = node('b', 100, 0, { radius: 16 });
    const route = routeEdge(from, to, obstaclesFrom([from, to]), {
      endpoints: ['a', 'b'],
      startClearance: from.radius,
      endClearance: to.radius,
    });
    const start = route.points[0];
    const end = route.points[route.points.length - 1];
    expect(start.x).toBeCloseTo(10);
    expect(end.x).toBeCloseTo(84);
    // Neither endpoint is inside the body it touches.
    expect(Math.hypot(start.x - from.x, start.y - from.y)).toBeGreaterThanOrEqual(from.radius);
    expect(Math.hypot(end.x - to.x, end.y - to.y)).toBeGreaterThanOrEqual(to.radius);
  });

  it('does not overshoot when the nodes are closer than their clearances', () => {
    const from: Point = { x: 0, y: 0 };
    const to: Point = { x: 5, y: 0 };
    const route = routeEdge(from, to, [], {
      endpoints: [],
      startClearance: 40,
      endClearance: 40,
    });
    // Clipping past the far end would flip the line around. Both ends collapse
    // onto the target instead, which draws as nothing rather than as backwards.
    for (const point of route.points) {
      expect(point.x).toBeGreaterThanOrEqual(0);
      expect(point.x).toBeLessThanOrEqual(5);
    }
  });
});

describe('parallel, reciprocal and self edges', () => {
  it('spreads parallel edges symmetrically about the centre line', () => {
    expect(parallelOffset(0, 1)).toBe(0);
    const three = [0, 1, 2].map((i) => parallelOffset(i, 3));
    expect(three[1]).toBe(0);
    expect(three[0]).toBe(-three[2]);
    expect(new Set(three).size).toBe(3);
  });

  it('groups a reciprocal pair together, so the two do not coincide', () => {
    const edges = [
      { id: 'forward', source: 'a', target: 'b' },
      { id: 'back', source: 'b', target: 'a' },
    ];
    expect(pairKey('a', 'b')).toBe(pairKey('b', 'a'));

    const placed = groupParallel(edges);
    expect(placed.get(edges[0])!.count).toBe(2);
    const first = parallelOffset(placed.get(edges[0])!.index, 2);
    const second = parallelOffset(placed.get(edges[1])!.index, 2);
    expect(first).not.toBe(second);
  });

  it('draws parallel edges apart on the canvas, not only in the offsets', () => {
    const a = node('a', 0, 0);
    const b = node('b', 200, 0);
    const obstacles = obstaclesFrom([a, b]);
    const first = routeEdge(a, b, obstacles, { endpoints: ['a', 'b'], offset: -9 });
    const second = routeEdge(a, b, obstacles, { endpoints: ['a', 'b'], offset: 9 });

    const midOf = (points: Point[]) => points[Math.floor(points.length / 2)];
    expect(midOf(first.points).y).not.toBeCloseTo(midOf(second.points).y);
  });

  it('draws a self-loop as a visible loop above the node, clear of its label', () => {
    const held = node('n', 100, 100, { radius: 10, label: 'talks about itself' });
    const loop = selfLoopPath(held);
    expect(loop.length).toBeGreaterThan(4);

    // It has real extent — a zero-length segment is the bug this replaces.
    const xs = loop.map((point) => point.x);
    const ys = loop.map((point) => point.y);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(held.radius);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(held.radius);

    // And it stays off the caption, which hangs below.
    const caption = labelBox(held)!;
    for (const point of loop) {
      expect(point.y).toBeLessThan(caption.y);
    }
  });

  it('spreads several self-loops on one node', () => {
    const held = node('n', 0, 0);
    const first = selfLoopPath(held, 0);
    const second = selfLoopPath(held, 1);
    const spread = (points: Point[]) =>
      Math.max(...points.map((p) => p.x)) - Math.min(...points.map((p) => p.x));
    expect(spread(second)).toBeGreaterThan(spread(first));
  });
});

describe('a tiny window', () => {
  it('still produces a legible arrangement, by growing the world', () => {
    // A 180×120 panel — narrower than one of the labels.
    const nodes = Array.from({ length: 6 }, (_, i) =>
      node(`n${i}`, 90, 60, { label: `a label wider than this whole window ${i}` }),
    );
    separate(nodes);
    expect(overlappingPairs(nodes)).toEqual([]);

    const bounds: Box = worldBounds(nodes);
    // The world is bigger than the window. That is the intended outcome: the
    // canvas zooms to fit, rather than the layout stacking things to obey a
    // viewport it was never told about.
    expect(bounds.width).toBeGreaterThan(180);
  });
});

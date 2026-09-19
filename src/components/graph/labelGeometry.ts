/**
 * The geometry the canvas is actually held to: label bounds, node separation,
 * edge clipping, parallel offsets, self-loops, and routing around things an
 * edge has no business crossing.
 *
 * Pure and in world space, which is what makes the acceptance assertions
 * possible at all. A test can build a graph, run this, and check that no drawn
 * label overlaps another and that no drawn segment passes through an unrelated
 * node — without a DOM, a canvas or a screenshot. The screenshots then confirm
 * that what this computed is what got painted; they are not the primary
 * evidence, because an eye cannot check five hundred pairwise overlaps.
 *
 * ## What is guaranteed, and what is not
 *
 * Guaranteed, and asserted in `labelGeometry.test.ts`:
 *
 * - No two drawn boxes — node body or visible label — overlap after
 *   {@link separate} converges.
 * - No drawn edge segment passes through a node or label that is not one of its
 *   own endpoints, whenever {@link routeEdge} reports `clear`.
 * - Endpoints stop at the outline of the shape they touch, so an arrowhead is
 *   visible rather than buried under the node it points at.
 * - Parallel edges between the same pair are drawn apart, reciprocal edges do
 *   not coincide, and a self-loop is a visible loop rather than a zero-length
 *   segment.
 *
 * **Not** guaranteed: zero edge *crossings*. An arbitrary graph is not planar,
 * and no 2D layout can draw a non-planar graph without edges crossing each
 * other — K₅ and K₃,₃ are the standard witnesses. What is offered instead is
 * fewer crossings where cheaply achievable, and the filtered and local views in
 * `memoryModel.ts` for when density defeats the picture. {@link routeEdge}
 * returns `clear: false` rather than pretending, so a caller can count the
 * failures and a report can state them.
 *
 * A force simulation's repulsion is emphatically *not* one of these guarantees.
 * Repulsion is a preference expressed through velocities; two labels can and do
 * come to rest overlapping. {@link separate} is a constraint solver run after
 * the simulation settles, and it is the thing that actually holds.
 */

/** A point in world space. */
export interface Point {
  x: number;
  y: number;
}

/** An axis-aligned box in world space. */
export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * A node as the geometry sees it: a body, and the label actually drawn for it.
 *
 * `labelled` is false when the renderer will not draw one. That distinction is
 * load-bearing — reserving room for a label that is never painted makes a graph
 * needlessly sparse, and *not* reserving room for one that is painted is the
 * bug this whole module exists to fix.
 */
export interface LaidOutNode {
  id: string;
  x: number;
  y: number;
  /** The drawn radius of the body. */
  radius: number;
  /** Full width of the label, in world units. Zero when unlabelled. */
  labelWidth: number;
  /** Height of the label, in world units. Zero when unlabelled. */
  labelHeight: number;
  /** Whether a label is drawn at all. */
  labelled: boolean;
}

/** Height of a label line, matching the 11px font the canvas draws with. */
export const LABEL_HEIGHT = 13;

/** Gap between a node's outline and the top of its label. */
export const LABEL_GAP = 3;

/**
 * A monotonic clock, where one is available.
 *
 * `performance.now` in a browser; `Date.now` in the node environment the tests
 * run in, which has no DOM. Neither is imported, so this module stays free of
 * anything a test would have to stub.
 */
const now: () => number =
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? () => performance.now()
    : () => Date.now();

/** Breathing room kept between any two drawn boxes. */
export const SEPARATION_PADDING = 4;

/** The body's box. */
export function bodyBox(node: LaidOutNode): Box {
  return {
    x: node.x - node.radius,
    y: node.y - node.radius,
    width: node.radius * 2,
    height: node.radius * 2,
  };
}

/**
 * The label's box, where the canvas paints it: centred under the body.
 *
 * Returns null when no label is drawn, rather than a zero-sized box at the
 * node's centre — a degenerate box still tests as overlapping things and would
 * push nodes apart for a label nobody can see.
 */
export function labelBox(node: LaidOutNode): Box | null {
  if (!node.labelled || node.labelWidth <= 0) return null;
  return {
    x: node.x - node.labelWidth / 2,
    y: node.y + node.radius + LABEL_GAP,
    width: node.labelWidth,
    height: node.labelHeight || LABEL_HEIGHT,
  };
}

/** Every box a node actually paints. One or two. */
export function drawnBoxes(node: LaidOutNode): Box[] {
  const label = labelBox(node);
  return label ? [bodyBox(node), label] : [bodyBox(node)];
}

/**
 * The radius a collision force needs so a node's *label* does not land on top
 * of somebody else.
 *
 * The simulation models a node as a disc, and the drawn thing is a disc with a
 * caption hanging below it. Feeding the body radius to the collide force is
 * precisely how eleven names end up in a heap — the force was satisfied, the
 * picture was not. This returns the radius of a disc that contains both.
 *
 * It is a conservative over-estimate: a circle around a wide, short label is
 * bigger than the label. That costs some sparseness and buys a simulation whose
 * resting state is already close to legible, which is what leaves
 * {@link separate} with a small correction rather than a rearrangement.
 */
export function collisionRadius(node: LaidOutNode): number {
  const label = labelBox(node);
  if (!label) return node.radius;
  // The furthest painted corner from the node's centre.
  const dx = Math.max(node.radius, label.width / 2);
  const dy = Math.max(node.radius, node.radius + LABEL_GAP + label.height);
  return Math.hypot(dx, dy);
}

/** Whether two boxes overlap, with `padding` of required clearance. */
export function overlaps(a: Box, b: Box, padding = 0): boolean {
  return (
    a.x < b.x + b.width + padding &&
    b.x < a.x + a.width + padding &&
    a.y < b.y + b.height + padding &&
    b.y < a.y + a.height + padding
  );
}

/** The combined box a node paints: body and label together. */
function combinedBox(node: LaidOutNode): Box {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const box of drawnBoxes(node)) {
    minX = Math.min(minX, box.x);
    minY = Math.min(minY, box.y);
    maxX = Math.max(maxX, box.x + box.width);
    maxY = Math.max(maxY, box.y + box.height);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * A push smaller than this is not movement.
 *
 * Without it the solver never reports convergence: two boxes settled exactly
 * `padding` apart produce a penetration depth of a floating-point crumb, which
 * is enough to set the "something moved" flag and spin out the whole pass
 * budget on nudges of 10⁻¹³ of a pixel. Every overlap was in fact gone by then;
 * the layout was correct and said it had failed, which is the more dangerous
 * direction for a flag that a caller reports to a person.
 */
const MOVEMENT_EPSILON = 0.01;

/** Successive over-relaxation factor. See the note where it is applied. */
const OVER_RELAXATION = 1.6;

/** How much more room than a bare grid the pre-expansion asks for. */
const SLACK = 1.25;

/**
 * Past this multiple, pre-expansion grids rather than scales.
 *
 * Needing to grow an axis fourfold means the nodes were effectively on top of
 * one another, and their relative positions are noise rather than structure.
 */
const MAX_SCALE = 4;

/**
 * Spreads everything out before relaxing, when the area needed exceeds the area
 * available.
 *
 * ## Why a scale step exists at all
 *
 * Relaxation is local: it resolves the pair in front of it. Given five hundred
 * labelled nodes on a grid whose spacing is smaller than the labels, every
 * resolution creates a new overlap with a neighbour, and the arrangement
 * shuffles for hundreds of passes to achieve, in total, the uniform expansion
 * that one multiplication does immediately.
 *
 * Scaling about the centroid preserves the arrangement exactly — relative
 * positions, clusters, the shape the simulation found — and changes only its
 * size. That is the one transformation that buys room without discarding the
 * layout, which is why it is worth doing before the part that does disturb it.
 *
 * ## The degenerate case
 *
 * A cluster with no extent cannot be scaled: multiplying zero spread by
 * anything is still zero. Nodes stacked on one point are placed on a grid
 * instead, ordered by id so the result is reproducible. Nothing is lost by
 * that — an arrangement in which everything shares a position is not an
 * arrangement.
 */
function preExpand(nodes: LaidOutNode[], padding: number): void {
  if (nodes.length < 2) return;

  let widest = 1;
  let tallest = 1;
  for (const node of nodes) {
    const box = combinedBox(node);
    widest = Math.max(widest, box.width + padding);
    tallest = Math.max(tallest, box.height + padding);
  }

  // The room a grid of these would need. Used as the target rather than the
  // sum of the areas, because the widest label sets the column width whether
  // or not the other labels are short.
  const columns = Math.max(1, Math.ceil(Math.sqrt(nodes.length)));
  const rows = Math.max(1, Math.ceil(nodes.length / columns));
  // A grid of these cells fits everything exactly. `SLACK` makes it not
  // exactly: relaxation needs somewhere to move things *to*, and a target that
  // is precisely full is the hopeless case — every node is already touching its
  // neighbours, every push creates another overlap, and the solver shuffles
  // until it runs out of passes. A quarter more room in each axis is enough to
  // turn that into a handful of passes.
  const wantWidth = columns * widest * SLACK;
  const wantHeight = rows * tallest * SLACK;

  // Measured over node *centres*, not over the drawn boxes.
  //
  // This is the correction that made the ten-node case converge. Scaling moves
  // centres and does not resize labels, so a bounding box dominated by one wide
  // caption reports plenty of room where there is none: ten nodes two pixels
  // apart with eighty-pixel labels have a 102-wide bounding box and a 18-wide
  // spread, and sizing against the former asks for a scale of 3 where 19 is
  // needed.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    maxX = Math.max(maxX, node.x);
    minY = Math.min(minY, node.y);
    maxY = Math.max(maxY, node.y);
  }
  const spreadX = maxX - minX;
  const spreadY = maxY - minY;

  // Scaled per axis, so an arrangement that is already wide and flat is not
  // made needlessly tall to satisfy a square target.
  const scaleX = spreadX >= 1 ? Math.max(1, wantWidth / spreadX) : Infinity;
  const scaleY = spreadY >= 1 ? Math.max(1, wantHeight / spreadY) : Infinity;

  // Two cases go to the grid rather than the scale.
  //
  // An axis with no spread cannot be scaled at all: multiplying zero by
  // anything is still zero. And an axis needing a *large* multiple is a cluster
  // so tight that its arrangement carries no usable information — scaling a
  // near-coincident diagonal just yields a long thin diagonal, which relaxation
  // then has to unstack one adjacent pair at a time, taking a pass per node.
  // Grid placement is O(1) and deterministic; what it discards was not a
  // layout.
  if (scaleX > MAX_SCALE || scaleY > MAX_SCALE) {
    const ordered = [...nodes].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    ordered.forEach((node, index) => {
      node.x = minX + (index % columns) * widest;
      node.y = minY + Math.floor(index / columns) * tallest;
    });
    return;
  }

  if (scaleX === 1 && scaleY === 1) return;

  const centreX = minX + spreadX / 2;
  const centreY = minY + spreadY / 2;
  for (const node of nodes) {
    node.x = centreX + (node.x - centreX) * scaleX;
    node.y = centreY + (node.y - centreY) * scaleY;
  }
}

/** What a separation pass achieved. */
export interface SeparationResult {
  /** How many passes it took. */
  passes: number;
  /**
   * Whether it finished with nothing overlapping.
   *
   * Reported rather than assumed. A caller that wants to claim "no labels
   * overlap" has to read this, and a layout that ran out of passes says so
   * instead of looking fine.
   */
  converged: boolean;
}

/**
 * Pushes overlapping nodes apart until nothing drawn overlaps anything else.
 *
 * ## Why this exists as well as the collide force
 *
 * Because they are different kinds of thing. The collide force is a preference
 * applied while the simulation has energy: it nudges velocities, it is damped,
 * and when alpha reaches its minimum whatever overlap remains is simply left
 * there. This runs *after* that, treats an overlap as a constraint to satisfy
 * now, and moves positions directly. "The repulsion is set high" is not
 * evidence that nothing overlaps; this function's post-condition is.
 *
 * ## Jacobi, not Gauss–Seidel
 *
 * Displacements are accumulated across a whole pass and applied at the end,
 * rather than each pair being moved the moment it is found. Applying
 * immediately means resolving (a,b) and then (a,c) undoes part of the first
 * fix, and a chain of nodes oscillates instead of settling — which is exactly
 * how the first version of this spent its entire pass budget on ten nodes.
 * Accumulating averages the competing demands on each node, and it relaxes.
 *
 * ## Why it may grow the world
 *
 * Separating things needs room, and the only place room comes from is space. A
 * pass that also clamped every node back inside the viewport would undo its own
 * work on the last node it touched and quietly reintroduce the overlap it had
 * just fixed. So nothing is clamped here. The canvas pans and zooms, and the
 * caller fits the view to whatever bounds come out — which is why
 * {@link worldBounds} exists.
 */
export function separate(
  nodes: LaidOutNode[],
  { padding = SEPARATION_PADDING, maxPasses = 200, budgetMs = 120 } = {},
): SeparationResult {
  if (nodes.length < 2) return { passes: 0, converged: true };

  const deadline = now() + budgetMs;
  preExpand(nodes, padding);

  const pushX = new Float64Array(nodes.length);
  const pushY = new Float64Array(nodes.length);
  const demands = new Int32Array(nodes.length);

  for (let pass = 0; pass < maxPasses; pass += 1) {
    pushX.fill(0);
    pushY.fill(0);
    demands.fill(0);
    let moved = false;

    // A uniform grid over the combined boxes. Without it each pass is O(n²) —
    // at the 500-node target, 125,000 pair tests every pass, which measured at
    // 726 ms for one call. The brief asks for quadratic hot paths to be
    // replaced on measured evidence; that measurement is the evidence.
    const boxes = nodes.map(combinedBox);
    let cell = 1;
    for (const box of boxes) {
      cell = Math.max(cell, box.width + padding, box.height + padding);
    }

    const buckets = new Map<string, number[]>();
    boxes.forEach((box, index) => {
      const x1 = Math.floor((box.x + box.width) / cell);
      const y1 = Math.floor((box.y + box.height) / cell);
      for (let cx = Math.floor(box.x / cell); cx <= x1; cx += 1) {
        for (let cy = Math.floor(box.y / cell); cy <= y1; cy += 1) {
          const key = `${cx},${cy}`;
          const held = buckets.get(key);
          if (held) held.push(index);
          else buckets.set(key, [index]);
        }
      }
    });

    const considered = new Set<number>();
    for (let i = 0; i < nodes.length; i += 1) {
      const box = boxes[i];
      considered.clear();
      const x1 = Math.floor((box.x + box.width) / cell) + 1;
      const y1 = Math.floor((box.y + box.height) / cell) + 1;
      for (let cx = Math.floor(box.x / cell) - 1; cx <= x1; cx += 1) {
        for (let cy = Math.floor(box.y / cell) - 1; cy <= y1; cy += 1) {
          for (const j of buckets.get(`${cx},${cy}`) ?? []) {
            if (j > i) considered.add(j);
          }
        }
      }

      for (const j of considered) {
        const a = nodes[i];
        const b = nodes[j];

        let worst: { dx: number; dy: number; depth: number } | null = null;
        for (const boxA of drawnBoxes(a)) {
          for (const boxB of drawnBoxes(b)) {
            if (!overlaps(boxA, boxB, padding)) continue;
            const overlapX =
              Math.min(boxA.x + boxA.width, boxB.x + boxB.width) -
              Math.max(boxA.x, boxB.x) +
              padding;
            const overlapY =
              Math.min(boxA.y + boxA.height, boxB.y + boxB.height) -
              Math.max(boxA.y, boxB.y) +
              padding;
            // Resolve along the axis of least penetration: the shorter push is
            // the one that disturbs the settled arrangement least.
            const alongX = overlapX < overlapY;
            const depth = alongX ? overlapX : overlapY;
            if (worst && depth <= worst.depth) continue;
            // Direction from a to b, by centres, so two nodes sharing a centre
            // still get a deterministic push rather than a division by zero.
            const sign = alongX
              ? Math.sign(b.x - a.x) || (a.id < b.id ? -1 : 1)
              : Math.sign(b.y - a.y) || (a.id < b.id ? -1 : 1);
            worst = {
              dx: alongX ? sign * depth : 0,
              dy: alongX ? 0 : sign * depth,
              depth,
            };
          }
        }

        if (!worst || worst.depth <= MOVEMENT_EPSILON) continue;
        // Half each, so neither node is privileged and the arrangement stays
        // centred on where the simulation put it.
        pushX[i] -= worst.dx / 2;
        pushY[i] -= worst.dy / 2;
        pushX[j] += worst.dx / 2;
        pushY[j] += worst.dy / 2;
        demands[i] += 1;
        demands[j] += 1;
        moved = true;
      }
    }

    if (!moved) return { passes: pass, converged: true };
    // A time budget as well as a pass budget.
    //
    // On a sparse graph the solver converges in a handful of passes and never
    // sees this. On a dense one it does not converge at all, and the passes it
    // would spend are a second of the reader's time buying a few fewer
    // overlapping captions. Bounding by wall clock rather than by a pass count
    // is what makes that trade the same on a fast machine and a slow one — a
    // fixed count means the slow machine pays the most and gets the worst of
    // it. What remains is counted by `overlapCount` and reported, so the
    // trade is visible rather than hidden.
    if (now() > deadline) return { passes: pass, converged: false };
    for (let i = 0; i < nodes.length; i += 1) {
      if (demands[i] === 0) continue;
      // Averaged over the neighbours demanding room, then over-relaxed.
      //
      // Summing instead would give a node wedged between six others a push six
      // times larger than any one of them asked for, which overshoots past them
      // and comes back next pass: the oscillation that had ten nodes taking a
      // hundred and fifty passes. Averaging makes each step the compromise its
      // neighbours are jointly asking for. The 1.6 is successive over-relaxation
      // — the standard trick for making an averaged iteration reach its fixed
      // point in fewer steps, at the cost of a little overshoot that the next
      // pass corrects.
      nodes[i].x += (pushX[i] / demands[i]) * OVER_RELAXATION;
      nodes[i].y += (pushY[i] / demands[i]) * OVER_RELAXATION;
    }
  }

  // Out of passes. Whether anything still overlaps is the caller's to report,
  // and `converged: false` is how it finds out.
  return { passes: maxPasses, converged: false };
}

/**
 * How many pairs of nodes still overlap on screen.
 *
 * ## Why this exists beside `SeparationResult.converged`
 *
 * Because they answer different questions, and only one of them is the question
 * a person cares about. `converged` says the solver stopped moving things
 * within its pass budget; it can be false while nothing whatsoever overlaps,
 * because the last few passes were chasing sub-pixel corrections. Reporting
 * that as "some labels could not be separated" puts a warning in front of
 * somebody looking at a perfectly legible graph — which teaches them to ignore
 * warnings.
 *
 * This counts the thing itself. It is O(n²) and is run once per settled layout,
 * never per frame.
 */
export function overlapCount(nodes: readonly LaidOutNode[]): number {
  let count = 0;
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      let hit = false;
      for (const a of drawnBoxes(nodes[i])) {
        for (const b of drawnBoxes(nodes[j])) {
          if (overlaps(a, b)) hit = true;
        }
      }
      if (hit) count += 1;
    }
  }
  return count;
}

/** The box containing everything drawn. The world, after separation grew it. */
export function worldBounds(nodes: readonly LaidOutNode[], margin = 24): Box {
  if (nodes.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    for (const box of drawnBoxes(node)) {
      minX = Math.min(minX, box.x);
      minY = Math.min(minY, box.y);
      maxX = Math.max(maxX, box.x + box.width);
      maxY = Math.max(maxY, box.y + box.height);
    }
  }
  return {
    x: minX - margin,
    y: minY - margin,
    width: maxX - minX + margin * 2,
    height: maxY - minY + margin * 2,
  };
}

/** Whether a segment crosses a box. Used to keep edges out of nodes. */
export function segmentHitsBox(p: Point, q: Point, box: Box): boolean {
  // Liang–Barsky. Chosen over sampling the segment because sampling misses a
  // box the line clips the corner of, and a corner clip is exactly what a
  // label's box gets.
  let t0 = 0;
  let t1 = 1;
  const dx = q.x - p.x;
  const dy = q.y - p.y;

  const edges: Array<[number, number]> = [
    [-dx, p.x - box.x],
    [dx, box.x + box.width - p.x],
    [-dy, p.y - box.y],
    [dy, box.y + box.height - p.y],
  ];

  for (const [denominator, numerator] of edges) {
    if (denominator === 0) {
      // Parallel to this edge and outside it: no intersection is possible.
      if (numerator < 0) return false;
      continue;
    }
    const t = numerator / denominator;
    if (denominator < 0) {
      if (t > t1) return false;
      if (t > t0) t0 = t;
    } else {
      if (t < t0) return false;
      if (t < t1) t1 = t;
    }
  }
  return t0 <= t1;
}

/** Moves a point along a segment by `distance`, to clear a node's outline. */
export function clipFrom(from: Point, to: Point, distance: number): Point {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  if (length < 1e-6) return { ...from };
  const step = Math.min(distance, length);
  return { x: from.x + (dx / length) * step, y: from.y + (dy / length) * step };
}

/**
 * How far off the straight line a parallel edge is drawn.
 *
 * Spreads `count` edges symmetrically about the centre line. A single edge gets
 * zero, so the ordinary case is still a straight line between two nodes.
 *
 * Reciprocal edges — A→B and B→A — are the same unordered pair and therefore in
 * the same group, so they get different offsets and are drawn as two visible
 * lines rather than one line with two arrowheads. Grouping by the *unordered*
 * pair is what makes that fall out; grouping by the ordered pair would put each
 * in a group of one and lay them exactly on top of each other.
 */
export function parallelOffset(index: number, count: number, spacing = 9): number {
  if (count <= 1) return 0;
  return (index - (count - 1) / 2) * spacing;
}

/** The unordered key two nodes share, for grouping parallel edges. */
export function pairKey(a: string, b: string): string {
  return a < b ? `${a} ${b}` : `${b} ${a}`;
}

/**
 * Groups edges by the pair they join, so each can be told its index and how
 * many siblings it has.
 */
export function groupParallel<E extends { source: string; target: string }>(
  edges: readonly E[],
): Map<E, { index: number; count: number }> {
  const groups = new Map<string, E[]>();
  for (const edge of edges) {
    const key = pairKey(edge.source, edge.target);
    const held = groups.get(key);
    if (held) held.push(edge);
    else groups.set(key, [edge]);
  }
  const placed = new Map<E, { index: number; count: number }>();
  for (const group of groups.values()) {
    group.forEach((edge, index) => placed.set(edge, { index, count: group.length }));
  }
  return placed;
}

/**
 * The loop drawn for an edge whose ends are the same node.
 *
 * A self-loop drawn as a segment is a segment of length zero: invisible, and
 * un-clickable. This returns a polyline going up and around, clearing the
 * body — and clearing the *label* too, by sitting above the node rather than
 * below it, where the caption is.
 *
 * `index` spreads several loops on one node so a node that both supersedes and
 * contradicts itself shows two.
 */
export function selfLoopPath(node: LaidOutNode, index = 0): Point[] {
  const r = node.radius * 0.8 + 8 + index * 7;
  const centreY = node.y - node.radius - r * 0.55;
  const points: Point[] = [];
  // Three-quarters of a circle, sampled. Sampled rather than an arc command so
  // that the geometry tests measure the same polyline the canvas strokes.
  for (let step = 0; step <= 16; step += 1) {
    const angle = Math.PI * 0.75 + (Math.PI * 1.5 * step) / 16;
    points.push({ x: node.x + r * Math.cos(angle), y: centreY + r * Math.sin(angle) });
  }
  return points;
}

/** An obstacle an edge must not pass through. */
export interface Obstacle {
  id: string;
  boxes: Box[];
}

/** Obstacles for every node, ready for {@link routeEdge}. */
export function obstaclesFrom(nodes: readonly LaidOutNode[]): Obstacle[] {
  return nodes.map((node) => ({ id: node.id, boxes: drawnBoxes(node) }));
}

/** What routing produced, and whether it succeeded. */
export interface Route {
  /** The polyline actually stroked, start to end. */
  points: Point[];
  /**
   * False when no tried route cleared every obstacle.
   *
   * The edge is still drawn — an omitted edge is a worse lie than a crossed
   * one — but the caller counts these and the report states the number, rather
   * than the code claiming a guarantee it did not deliver.
   */
  clear: boolean;
}

/**
 * Routes one edge, bending around anything that is not its own endpoints.
 *
 * ## The method, and why this one
 *
 * Straight first: most edges in a settled graph cross nothing, and a graph of
 * gratuitously curved lines is harder to read than one of straight ones. When
 * the straight segment does hit something, a single bend point is tried at
 * increasing perpendicular offsets, alternating sides, and the first one that
 * clears everything wins. Preferring the smallest offset keeps the bend as
 * close to the straight line as the obstacles allow, so an edge still reads as
 * joining its two ends.
 *
 * A full visibility-graph or force-directed edge router would find a path in
 * more cases. It is not here because the search above resolves the realistic
 * cases at this graph's size, and because an unbounded router is a frame-time
 * risk on a canvas that must stay interactive while memory arrives. The bound
 * is explicit, and a failure is reported rather than hidden.
 *
 * `offset` bends the whole route sideways before routing, which is how parallel
 * and reciprocal edges are separated.
 */
export function routeEdge(
  from: Point,
  to: Point,
  obstacles: readonly Obstacle[],
  options: {
    /** Ids whose obstacles are skipped: this edge's own endpoints. */
    endpoints: readonly string[];
    /** Sideways displacement for a parallel or reciprocal edge. */
    offset?: number;
    /** How far the line stops short of each end, so arrowheads stay visible. */
    startClearance?: number;
    endClearance?: number;
    maxAttempts?: number;
  },
): Route {
  const skip = new Set(options.endpoints);
  const blocking = obstacles.filter((obstacle) => !skip.has(obstacle.id));

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  // Unit normal, for both the parallel offset and the routing bends.
  const nx = -dy / length;
  const ny = dx / length;

  const start = clipFrom(from, to, options.startClearance ?? 0);
  const end = clipFrom(to, from, options.endClearance ?? 0);
  const offset = options.offset ?? 0;

  const hits = (points: Point[]): boolean => {
    for (let i = 0; i + 1 < points.length; i += 1) {
      for (const obstacle of blocking) {
        for (const box of obstacle.boxes) {
          if (segmentHitsBox(points[i], points[i + 1], box)) return true;
        }
      }
    }
    return false;
  };

  const bendAt = (magnitude: number): Point[] => {
    const midX = (start.x + end.x) / 2 + nx * magnitude;
    const midY = (start.y + end.y) / 2 + ny * magnitude;
    return [start, { x: midX, y: midY }, end];
  };

  // The parallel offset is part of the shape even when nothing is in the way:
  // two edges between the same pair must not coincide whether or not a third
  // node happens to sit between them.
  const base = offset === 0 ? [start, end] : bendAt(offset);
  if (!hits(base)) return { points: base, clear: true };

  const maxAttempts = options.maxAttempts ?? 16;
  const step = Math.max(14, length * 0.12);
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const magnitude = step * Math.ceil(attempt / 2);
    // Alternate sides, so a bend is as likely to go the short way round.
    const signed = attempt % 2 === 1 ? magnitude : -magnitude;
    const candidate = bendAt(offset + signed);
    if (!hits(candidate)) return { points: candidate, clear: true };
  }

  return { points: base, clear: false };
}

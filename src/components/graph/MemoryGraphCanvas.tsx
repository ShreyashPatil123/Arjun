import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GraphSimulation } from './simulation';
import {
  type LaidOutNode,
  type Obstacle,
  type Point,
  type Route,
  collisionRadius,
  groupParallel,
  labelBox,
  obstaclesFrom,
  overlapCount,
  parallelOffset,
  routeEdge,
  selfLoopPath,
  separate,
  worldBounds,
} from './labelGeometry';
import { CANONICAL_COLOUR, colourOf, fade } from './agentColor';
import type { MemoryGraphNode, MemoryGraphView } from './memoryModel';

/**
 * The agent-memory graph, drawn on a black canvas.
 *
 * A sibling of `GraphCanvas`, not a replacement for it. That one draws a
 * notebook's terms and documents and is left alone; this one draws what agents
 * know, which is a different domain with different node kinds, a different
 * colour rule and a different layout contract. Sharing one component would have
 * meant a prop matrix where half the combinations are meaningless.
 *
 * ## Three cues, three channels, no overlap
 *
 * - **Shape** is what a node *is*: a memory item is a circle, an agent a
 *   hexagon, a task a square, a source a document box, an artifact a diamond.
 * - **Outline** is where a claim *stands*: an admitted item is filled and solid,
 *   a proposal is hollow and dashed, a superseded record is dotted and dim, a
 *   rejected one carries a slash. This is the distinction the whole view exists
 *   to make, so it gets the channel that survives being small.
 * - **Colour** is *who*: the agent that owns the node, or that drew the edge.
 *   Nothing else is ever encoded in hue, because a channel carrying two things
 *   carries neither.
 *
 * Every one of them is also named in the legend and spelled out in the
 * inspector, so nothing has to be read off the drawing.
 *
 * ## Two-phase layout, and why
 *
 * While the simulation has energy, edges are drawn as straight clipped lines.
 * When it comes to rest, `separate` removes the label overlaps the force could
 * not, and only then are the edges routed around the things they must avoid.
 *
 * Routing every edge on every frame would be the obvious implementation and the
 * wrong one: it is O(edges × nodes) per frame, and it would spend that budget
 * computing avoidance for an arrangement that is about to move anyway. Doing it
 * once, against the positions that will actually be painted, is both cheaper
 * and more correct — a route computed mid-flight is stale by the next tick.
 */

/** How a node kind is drawn. Shape first; the label always accompanies it. */
type Glyph = 'circle' | 'hexagon' | 'square' | 'document' | 'diamond';

const GLYPH_FOR_KIND: Record<MemoryGraphNode['kind'], Glyph> = {
  memory: 'circle',
  agent: 'hexagon',
  task: 'square',
  source: 'document',
  artifact: 'diamond',
  cluster: 'circle',
};

/**
 * Below this many nodes, every node is labelled.
 *
 * Above it, labels go to the nodes a reader is actually working with — the
 * focus, the selection, whatever is under the cursor — plus the busiest, which
 * are the ones worth orienting by.
 */
const LABEL_EVERY_NODE_BELOW = 60;

/**
 * At most this many labels are drawn on a dense graph.
 *
 * ## Why a budget rather than a degree threshold
 *
 * The first version of this labelled anything of degree four or more, which
 * reads as a sensible rule and is not one: degree is a property of the *graph*,
 * not of the page. At 530 nodes and 1,529 edges almost every node clears four,
 * so almost every node got a caption, and the screenshot of that case was a
 * solid mat of overlapping text with the graph somewhere underneath it.
 *
 * A budget is a property of the page, which is the thing that actually runs out
 * of room. The busiest nodes win it, because those are the ones worth
 * orienting by; everything else keeps its shape, its colour and its outline,
 * and says its name on hover or in the inspector.
 */
const MAX_LABELS = 48;

/** Dash patterns per edge kind. Named in the legend; never the only cue. */
const DASH_FOR_EDGE: Record<string, number[]> = {
  supports: [],
  contradicts: [7, 4],
  supersedes: [11, 4],
  derivedFrom: [3, 3],
  cites: [1, 4],
  partOf: [9, 3, 2, 3],
  answers: [5, 3],
  authored: [1, 5],
  worksOn: [4, 4],
  references: [1, 4],
  between: [],
};

export interface MemoryGraphCanvasProps {
  view: MemoryGraphView;
  /** Agent id to colour, from `assignAgentColours`. */
  colours: ReadonlyMap<string, string>;
  /** The node the local view is centred on, drawn with a ring. */
  focusId?: string | null;
  /** The node the inspector is showing. */
  selectedId?: string | null;
  onSelect?: (node: MemoryGraphNode) => void;
  onFocus?: (node: MemoryGraphNode) => void;
  /**
   * Reports what the layout achieved, so the panel can say so out loud.
   *
   * A layout that could not route some edges, or could not finish separating,
   * is a fact about the picture on screen and belongs in the interface rather
   * than in a console nobody reads.
   */
  onLayout?: (report: LayoutReport) => void;
  /**
   * Overrides the `prefers-reduced-motion` media query.
   *
   * For the acceptance harness, which has to be able to photograph both
   * branches on one machine. Left undefined everywhere in the product, where
   * the person's own setting decides.
   */
  reducedMotion?: boolean;
}

/** What one settled layout achieved, and what it could not. */
export interface LayoutReport {
  nodes: number;
  edges: number;
  /**
   * How many pairs of nodes are still drawn overlapping.
   *
   * Counted rather than inferred from whether the solver ran out of passes: it
   * routinely exhausts its budget on sub-pixel corrections with nothing
   * actually overlapping, and warning about that would be a false alarm.
   */
  overlaps: number;
  /** Edges drawn straight through something because no route was found. */
  unroutedEdges: number;
  /** Wall-clock milliseconds for separation plus routing. */
  layoutMs: number;
}

/**
 * The surface, and the ink on it.
 *
 * ## Why a palette lives here and not in the theme
 *
 * The rest of the product is monochrome and follows the light/dark switch. This
 * view cannot: eight saturated agent colours only hold their relative weight
 * against a dark ground, and a palette that inverted would mean an agent
 * changed colour when somebody flipped a switch. So the surface is fixed, and
 * it is a very dark grey rather than pure black — pure black against saturated
 * dots produces a hard, buzzing edge, and a few points of lift takes it off.
 */
const SURFACE = '#08090C';
const INK = 'rgba(236, 237, 243, 0.94)';
const INK_DIM = 'rgba(236, 237, 243, 0.42)';
/** What a label sits on, so text stays readable where lines pass behind it. */
const LABEL_BACKDROP = 'rgba(8, 9, 12, 0.78)';

/** Rounded rectangle, for label backdrops. */
function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.min(radius, height / 2, width / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function glyphPath(
  context: CanvasRenderingContext2D,
  glyph: Glyph,
  x: number,
  y: number,
  r: number,
) {
  context.beginPath();
  switch (glyph) {
    case 'square':
      // Rounded, not square-cornered. A hard rectangle among soft discs reads
      // as a different material; a squircle reads as the same family.
      roundedRect(context, x - r, y - r, r * 2, r * 2, r * 0.42);
      break;
    case 'diamond':
      context.moveTo(x, y - r);
      context.lineTo(x + r, y);
      context.lineTo(x, y + r);
      context.lineTo(x - r, y);
      context.closePath();
      break;
    case 'hexagon':
      for (let i = 0; i < 6; i += 1) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const px = x + r * Math.cos(angle);
        const py = y + r * Math.sin(angle);
        if (i === 0) context.moveTo(px, py);
        else context.lineTo(px, py);
      }
      context.closePath();
      break;
    case 'document':
      roundedRect(context, x - r * 1.45, y - r * 1.1, r * 2.9, r * 2.2, r * 0.38);
      break;
    default:
      context.arc(x, y, r, 0, Math.PI * 2);
  }
}

/**
 * A soft halo behind a node.
 *
 * The one piece of pure decoration here, and it earns its place: against a dark
 * ground a flat disc has a hard aliased edge, and a halo gives it a little
 * depth so a node reads as sitting *on* the surface rather than being punched
 * out of it. It also separates a node from a line passing behind it without
 * needing an outline in a contrasting colour.
 */
function halo(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  colour: string,
  strength: number,
) {
  const gradient = context.createRadialGradient(x, y, r * 0.6, x, y, r * 2.5);
  gradient.addColorStop(0, fade(colour, 0.34 * strength));
  gradient.addColorStop(0.55, fade(colour, 0.1 * strength));
  gradient.addColorStop(1, fade(colour, 0));
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(x, y, r * 2.5, 0, Math.PI * 2);
  context.fill();
}

/** How big a node is drawn, from how connected it is — or how much it holds. */
function radiusFor(node: MemoryGraphNode): number {
  if (node.kind === 'cluster') {
    // Area, not radius, follows the count: doubling the items should look like
    // twice as much, and a radius that doubled would look like four times.
    return Math.min(46, 16 + Math.sqrt(node.count ?? 1) * 2.6);
  }
  const base = node.kind === 'memory' ? 6 : 10;
  const max = node.kind === 'memory' ? 16 : 22;
  return Math.min(max, base + Math.sqrt(Math.max(0, node.degree)) * 2.1);
}

/** A small, soft head at the far end of a route that already stops short. */
function arrowHead(context: CanvasRenderingContext2D, from: Point, to: Point, size = 5) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const spread = 0.46;
  context.beginPath();
  context.moveTo(to.x, to.y);
  context.lineTo(to.x - Math.cos(angle - spread) * size, to.y - Math.sin(angle - spread) * size);
  context.lineTo(to.x - Math.cos(angle + spread) * size, to.y - Math.sin(angle + spread) * size);
  context.closePath();
  context.fill();
}

/**
 * Strokes a route as a smooth curve rather than a polyline.
 *
 * `routeEdge` returns the bend points that keep the line clear of other nodes.
 * Joining them with straight segments draws a visible kink at each one; a
 * quadratic through the same points passes through the same clear space and
 * reads as one continuous line. The geometry tests assert against the polyline,
 * which is the conservative thing to assert against — the curve stays inside
 * the corridor its control points define.
 */
function strokeRoute(context: CanvasRenderingContext2D, points: Point[]) {
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  if (points.length === 2) {
    context.lineTo(points[1].x, points[1].y);
  } else {
    for (let i = 1; i < points.length - 1; i += 1) {
      const next = points[i + 1];
      const midX = (points[i].x + next.x) / 2;
      const midY = (points[i].y + next.y) / 2;
      context.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
    }
    const last = points[points.length - 1];
    context.lineTo(last.x, last.y);
  }
  context.stroke();
}
/** Whether this machine has been asked to keep motion down. */
function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export const MemoryGraphCanvas: React.FC<MemoryGraphCanvasProps> = ({
  view,
  colours,
  focusId,
  selectedId,
  onSelect,
  onFocus,
  onLayout,
  reducedMotion,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 800, height: 560 });
  const [hovered, setHovered] = useState<string | null>(null);

  const simulation = useRef<GraphSimulation | null>(null);
  const frame = useRef<number | null>(null);
  const drawRef = useRef<() => void>(() => {});

  /**
   * Where every node was last time, by id.
   *
   * The thing that makes an incremental update not throw the picture away. Held
   * in a ref rather than state: it is written by the frame loop, and putting it
   * through React would re-render sixty times a second.
   */
  const positions = useRef<Map<string, { x: number; y: number }>>(new Map());
  /** The settled geometry actually painted, for hit-testing and routing. */
  const laid = useRef<LaidOutNode[]>([]);
  const routes = useRef<Map<string, Route>>(new Map());
  const settled = useRef(false);

  // Pan and zoom live in a ref: they change on every pointer move and the frame
  // loop reads them directly.
  const viewRef = useRef({ scale: 1, x: 0, y: 0 });
  /** Set once per graph identity, so a live insertion does not re-centre. */
  const framedFor = useRef<string>('');
  /** Once the reader has moved the camera it is theirs, and nothing re-fits it. */
  const userMoved = useRef(false);
  const gesture = useRef<
    | { kind: 'pan'; x: number; y: number; moved: boolean }
    | { kind: 'drag'; id: string; moved: boolean }
    | null
  >(null);

  const byId = useMemo(() => new Map(view.nodes.map((node) => [node.id, node])), [view.nodes]);

  /**
   * Which nodes get a caption, decided once for the whole graph.
   *
   * Agents, tasks, sources and artifacts are always named: there are a handful
   * of them, they are the landmarks a reader navigates by, and an unlabelled
   * one is worth nothing. Memory items compete for what is left of the budget,
   * busiest first, with the id breaking ties so the choice is stable across
   * rebuilds rather than depending on map order.
   */
  const labelledIds = useMemo(() => {
    if (view.nodes.length <= LABEL_EVERY_NODE_BELOW) {
      return new Set(view.nodes.map((node) => node.id));
    }
    const chosen = new Set<string>();
    const items: MemoryGraphNode[] = [];
    for (const node of view.nodes) {
      if (node.kind === 'memory') items.push(node);
      else chosen.add(node.id);
    }
    items.sort((a, b) => b.degree - a.degree || (a.id < b.id ? -1 : 1));
    for (const node of items) {
      if (chosen.size >= MAX_LABELS) break;
      chosen.add(node.id);
    }
    return chosen;
  }, [view.nodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!parent) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(200, entry.contentRect.width);
      const height = Math.max(160, entry.contentRect.height);
      setSize((current) =>
        current.width === width && current.height === height ? current : { width, height },
      );
    });
    observer.observe(parent);
    return () => observer.disconnect();
  }, []);

  const toScreen = useCallback(
    (point: Point) => {
      const { scale, x, y } = viewRef.current;
      return {
        x: (point.x - size.width / 2) * scale + size.width / 2 + x,
        y: (point.y - size.height / 2) * scale + size.height / 2 + y,
      };
    },
    [size],
  );

  const toWorld = useCallback(
    (px: number, py: number): Point => {
      const { scale, x, y } = viewRef.current;
      return {
        x: (px - x - size.width / 2) / scale + size.width / 2,
        y: (py - y - size.height / 2) / scale + size.height / 2,
      };
    },
    [size],
  );

  /**
   * Turns the simulation's current positions into the geometry that gets
   * painted: radii, label widths measured in the font they will be drawn in,
   * and whether a label is drawn at all.
   *
   * Measured rather than estimated, because these widths become the collision
   * geometry. Guess short and two captions overlap; guess long and the graph is
   * needlessly sparse.
   */
  const measure = useCallback(
    (context: CanvasRenderingContext2D): LaidOutNode[] => {
      const sim = simulation.current;
      if (!sim) return [];
      context.font = '11px system-ui, sans-serif';
      return view.nodes.flatMap((node) => {
        const at = sim.find(node.id);
        if (!at) return [];
        const radius = radiusFor(node);
        const labelled =
          labelledIds.has(node.id) ||
          node.id === focusId ||
          node.id === selectedId ||
          node.id === hovered;
        return [
          {
            id: node.id,
            x: at.x,
            y: at.y,
            radius,
            labelWidth: labelled ? context.measureText(node.label).width : 0,
            labelHeight: 13,
            labelled,
          },
        ];
      });
    },
    [view.nodes, labelledIds, focusId, selectedId, hovered],
  );

  /**
   * Arranges an opened parent children in a ring around it.
   *
   * ## Why this overrides the simulation
   *
   * Because a force layout is the wrong tool for a set that is already known to
   * belong together. Left to the springs, twenty-four children of one parent
   * drift into whatever gaps the rest of the graph leaves, their spokes cross
   * everything on the way, and opening a bundle looks like the graph exploding
   * rather than like one thing being unpacked.
   *
   * A ring is the arrangement a reader expects from open: the children are
   * obviously that parent children, obviously all of them, and in a stable
   * order that does not reshuffle on the next frame. The radius is derived from
   * how much room the children actually need, so the ring is as tight as it can
   * be without the captions touching.
   *
   * Run before , so anything the ring still leaves overlapping — a
   * child landing on an unrelated node — is fixed by the constraint solver that
   * follows.
   */
  const ringOpenedChildren = useCallback(
    (nodes: LaidOutNode[]) => {
      const children = new Map<string, string[]>();
      for (const edge of view.edges) {
        if (edge.kind !== 'authored' || !edge.id.startsWith('opened:')) continue;
        const held = children.get(edge.source);
        if (held) held.push(edge.target);
        else children.set(edge.source, [edge.target]);
      }
      if (children.size === 0) return;

      const at = new Map(nodes.map((node) => [node.id, node]));
      const opening = new Set([...children.keys(), ...[...children.values()].flat()]);

      // Where the rest of the graph is. The ring is pushed away from it.
      let crowdX = 0;
      let crowdY = 0;
      let crowd = 0;
      for (const node of nodes) {
        if (opening.has(node.id)) continue;
        crowdX += node.x;
        crowdY += node.y;
        crowd += 1;
      }
      if (crowd > 0) {
        crowdX /= crowd;
        crowdY /= crowd;
      }

      for (const [parentId, ids] of children) {
        const parent = at.get(parentId);
        if (!parent) continue;
        // Sorted, so the ring does not reshuffle between frames or sessions.
        const ring = ids
          .map((id) => at.get(id))
          .filter((node): node is LaidOutNode => node !== undefined)
          .sort((a, b) => (a.id < b.id ? -1 : 1));
        if (ring.length === 0) continue;

        // The circumference has to hold every child box side by side.
        let needed = 0;
        let widest = 0;
        for (const child of ring) {
          const width = Math.max(child.radius * 2, child.labelWidth) + 14;
          needed += width;
          widest = Math.max(widest, width);
        }
        const radius = Math.max(
          parent.radius + widest * 0.6,
          needed / (Math.PI * 2),
        );

        // Open outward, into empty space, rather than back over everything
        // else.
        //
        // A ring centred exactly on its parent spends half of itself on the
        // side the rest of the graph is on, and those children land on top of
        // the other bundles — which is what makes opening one thing look like
        // breaking the whole picture. Nudging the ring away from the crowd, and
        // starting the fan on that side, puts the revealed items in the space
        // that is actually free.
        const awayX = parent.x - crowdX;
        const awayY = parent.y - crowdY;
        const awayLength = Math.hypot(awayX, awayY);
        // A parent sitting exactly on the centroid has no outward direction;
        // upward is as good as any, and is deterministic.
        const ux = awayLength > 1 ? awayX / awayLength : 0;
        const uy = awayLength > 1 ? awayY / awayLength : -1;
        const centreX = parent.x + ux * radius * 0.45;
        const centreY = parent.y + uy * radius * 0.45;
        const start = Math.atan2(uy, ux);

        ring.forEach((child, index) => {
          const angle = start + (Math.PI * 2 * index) / ring.length;
          child.x = centreX + Math.cos(angle) * radius;
          child.y = centreY + Math.sin(angle) * radius;
        });
      }
    },
    [view.edges],
  );

  /**
   * Fits the camera to the whole world, once per graph and never after the
   * reader has moved it.
   *
   * ## Why this runs after separation rather than before
   *
   * `separate` is allowed to grow the world — that is the whole point of it,
   * and of `clampToViewport: false`. A camera fitted to the pre-separation
   * bounds is therefore fitted to a world smaller than the one that gets
   * painted, and the difference shows up as captions clipped by the edge of a
   * small panel. Fitting afterwards costs nothing and is simply correct.
   *
   * ## Why it is once
   *
   * Re-fitting whenever the graph changes would move the picture under somebody
   * who is reading it, every time a fact arrives. And once the reader has
   * panned or zoomed, the camera is theirs: `userMoved` is never unset, so
   * nothing here takes it back.
   */
  const frameToWorld = useCallback(
    (nodes: readonly LaidOutNode[]) => {
      const identity = `${nodes.length}:${nodes[0]?.id ?? ''}`;
      if (userMoved.current || framedFor.current === identity || nodes.length === 0) return;
      const bounds = worldBounds(nodes);
      if (bounds.width <= 0 || bounds.height <= 0) return;
      framedFor.current = identity;
      const scale = Math.min(
        2,
        Math.max(0.05, Math.min(size.width / bounds.width, size.height / bounds.height)),
      );
      const centreX = bounds.x + bounds.width / 2;
      const centreY = bounds.y + bounds.height / 2;
      viewRef.current = {
        scale,
        x: (size.width / 2 - centreX) * scale,
        y: (size.height / 2 - centreY) * scale,
      };
    },
    [size],
  );

  /**
   * The expensive half, run once the simulation is at rest.
   *
   * Separation first, then routing against the separated positions — routing
   * against positions that are about to move would produce avoidance for an
   * arrangement that never appears.
   */
  const finishLayout = useCallback(
    (context: CanvasRenderingContext2D) => {
      const started = performance.now();
      const nodes = measure(context);
      ringOpenedChildren(nodes);
      separate(nodes);

      const obstacles: Obstacle[] = obstaclesFrom(nodes);
      const at = new Map(nodes.map((node) => [node.id, node]));
      const parallel = groupParallel(view.edges);
      const computed = new Map<string, Route>();
      const loopsPerNode = new Map<string, number>();
      let unrouted = 0;

      for (const edge of view.edges) {
        const from = at.get(edge.source);
        const to = at.get(edge.target);
        if (!from || !to) continue;

        if (edge.source === edge.target) {
          // A self-loop drawn as a segment is a segment of length zero.
          const index = loopsPerNode.get(edge.source) ?? 0;
          loopsPerNode.set(edge.source, index + 1);
          computed.set(edge.id, { points: selfLoopPath(from, index), clear: true });
          continue;
        }

        const placed = parallel.get(edge) ?? { index: 0, count: 1 };
        const route = routeEdge(from, to, obstacles, {
          endpoints: [edge.source, edge.target],
          offset: parallelOffset(placed.index, placed.count),
          startClearance: from.radius + 2,
          endClearance: to.radius + 4,
        });
        if (!route.clear) unrouted += 1;
        computed.set(edge.id, route);
      }

      laid.current = nodes;
      routes.current = computed;
      settled.current = true;
      for (const node of nodes) positions.current.set(node.id, { x: node.x, y: node.y });
      frameToWorld(nodes);

      onLayout?.({
        nodes: nodes.length,
        edges: view.edges.length,
        overlaps: overlapCount(nodes),
        unroutedEdges: unrouted,
        layoutMs: performance.now() - started,
      });
    },
    [measure, view.edges, onLayout, frameToWorld, ringOpenedChildren],
  );

  const ensureLoop = useCallback(() => {
    if (frame.current !== null) return;
    const tick = () => {
      frame.current = null;
      const sim = simulation.current;
      const running = sim?.step() ?? false;
      if (sim) {
        for (const node of sim.nodes) positions.current.set(node.id, { x: node.x, y: node.y });
      }
      if (!running && !settled.current) {
        const context = canvasRef.current?.getContext('2d');
        if (context) finishLayout(context);
      }
      drawRef.current();
      if (running) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  }, [finishLayout]);

  // A changed graph is a new simulation — seeded from where everything already
  // is, so what survives stays put.
  useEffect(() => {
    const context = canvasRef.current?.getContext('2d');
    if (!context) return;

    context.font = '11px system-ui, sans-serif';
    const known = positions.current.size > 0;

    const simNodes = view.nodes.map((node) => {
      const radius = radiusFor(node);
      const labelled = labelledIds.has(node.id);
      return {
        id: node.id,
        // The collide force is given the radius that covers the *label*, which
        // is what stops the simulation settling into an arrangement that
        // `separate` then has to tear apart.
        radius: collisionRadius({
          id: node.id,
          x: 0,
          y: 0,
          radius,
          labelWidth: labelled ? context.measureText(node.label).width : 0,
          labelHeight: 13,
          labelled,
        }),
      };
    });

    simulation.current = new GraphSimulation(
      simNodes,
      view.edges.map((edge) => ({ source: edge.source, target: edge.target })),
      {
        width: size.width,
        height: size.height,
        positions: known ? positions.current : undefined,
        // Warm rather than hot for an incremental update: an arrangement that
        // was already settled should let the new nodes find room, not re-run
        // three hundred ticks of physics on everything.
        alpha: known ? 0.35 : 1,
        // The world is allowed to grow. See the note on `clampToViewport`.
        clampToViewport: false,
      },
    );
    settled.current = false;

    // Reduced motion: settle without animating, then draw the result once.
    // Somebody who has asked their machine to keep motion down should not be
    // shown a graph springing about; they should be shown the graph.
    if (reducedMotion ?? prefersReducedMotion()) {
      const sim = simulation.current;
      for (let tick = 0; tick < 400 && sim.step(); tick += 1);
      for (const node of sim.nodes) positions.current.set(node.id, { x: node.x, y: node.y });
      finishLayout(context);
      drawRef.current();
      return;
    }
    ensureLoop();
  }, [view.nodes, view.edges, labelledIds, size, ensureLoop, finishLayout, reducedMotion]);

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
    },
    [],
  );

  const nodeAt = useCallback(
    (clientX: number, clientY: number): MemoryGraphNode | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const point = toWorld(clientX - rect.left, clientY - rect.top);
      // Last drawn wins, so the node painted on top is the one picked.
      for (let i = laid.current.length - 1; i >= 0; i -= 1) {
        const held = laid.current[i];
        if (Math.hypot(point.x - held.x, point.y - held.y) <= held.radius + 4) {
          return byId.get(held.id) ?? null;
        }
      }
      return null;
    },
    [byId, toWorld],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Registered natively because React attaches wheel listeners passively, and
    // `preventDefault` from a React handler is ignored with a console warning.
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const current = viewRef.current;
      userMoved.current = true;
      viewRef.current = {
        ...current,
        scale: Math.min(6, Math.max(0.05, current.scale * (event.deltaY < 0 ? 1.1 : 0.9))),
      };
      drawRef.current();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  drawRef.current = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = size.width * ratio;
    canvas.height = size.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.lineCap = 'round';
    context.lineJoin = 'round';

    context.fillStyle = SURFACE;
    context.fillRect(0, 0, size.width, size.height);

    const sim = simulation.current;
    if (!sim) return;

    const nodes: LaidOutNode[] = settled.current ? laid.current : measure(context);
    if (!settled.current) laid.current = nodes;
    const at = new Map(nodes.map((node) => [node.id, node]));
    const { scale } = viewRef.current;

    // A very soft lift under the graph, so the surface reads as lit rather than
    // as a void. Centred on the content, not on the canvas.
    if (nodes.length > 0) {
      const centre = toScreen({
        x: nodes.reduce((total, node) => total + node.x, 0) / nodes.length,
        y: nodes.reduce((total, node) => total + node.y, 0) / nodes.length,
      });
      const spread = Math.max(size.width, size.height) * 0.75;
      const lift = context.createRadialGradient(centre.x, centre.y, 0, centre.x, centre.y, spread);
      lift.addColorStop(0, 'rgba(255,255,255,0.035)');
      lift.addColorStop(1, 'rgba(255,255,255,0)');
      context.fillStyle = lift;
      context.fillRect(0, 0, size.width, size.height);
    }

    // What the pointer is resting on, and everything one hop from it. Hovering
    // is how a reader asks "what does this touch?", and dimming the rest is a
    // far stronger answer than thickening the few.
    // An agent is "open" when the view is showing its items as its children.
    // Derived from the edges rather than passed in, so the canvas has one
    // source of truth about what it is drawing.
    const openParents = new Set<string>();
    for (const edge of view.edges) {
      if (edge.kind === 'authored' && edge.id.startsWith('opened:') && edge.agentId) {
        openParents.add(edge.agentId);
      }
    }

    const near = new Set<string>();
    if (hovered) {
      near.add(hovered);
      for (const edge of view.edges) {
        if (edge.source === hovered) near.add(edge.target);
        else if (edge.target === hovered) near.add(edge.source);
      }
    }

    // ── Edges ────────────────────────────────────────────────────────────
    for (const edge of view.edges) {
      const from = at.get(edge.source);
      const to = at.get(edge.target);
      if (!from || !to) continue;

      const lit = !hovered || near.has(edge.source) || near.has(edge.target);
      const colour = colourOf(colours, edge.agentId);
      const weight = edge.weight ?? 1;
      // A bundled edge thickens with the log of what it stands for, and says
      // the number too. Thickness alone is a feeling; the number is a fact.
      const width = edge.kind === 'between' ? 1 + Math.log2(weight) * 0.9 : 1;
      context.strokeStyle = fade(colour, lit ? (edge.kind === 'between' ? 0.4 : 0.3) : 0.06);
      context.lineWidth = Math.max(0.5, Math.min(3.5, width * scale));
      context.setLineDash(
        edge.kind === 'between' ? [] : (DASH_FOR_EDGE[edge.kind] ?? []).map((n) => n * scale),
      );

      const route = settled.current ? routes.current.get(edge.id) : undefined;
      const points: Point[] = route ? route.points : [from, to];
      const screen = points.map(toScreen);
      strokeRoute(context, screen);
      context.setLineDash([]);

      if (edge.kind !== 'worksOn' && edge.kind !== 'between' && screen.length >= 2) {
        context.fillStyle = fade(colour, lit ? 0.4 : 0.06);
        arrowHead(context, screen[screen.length - 2], screen[screen.length - 1], 5);
      }
    }

    // ── Nodes ────────────────────────────────────────────────────────────
    context.textAlign = 'center';

    for (const held of nodes) {
      const node = byId.get(held.id);
      if (!node) continue;
      const screen = toScreen(held);
      const r = Math.max(3, held.radius * scale);
      const colour = colourOf(colours, node.agentId);
      const status = node.item?.status;
      const lit = !hovered || near.has(node.id);
      const faded = status === 'superseded' || status === 'rejected';
      const strength = (lit ? 1 : 0.22) * (faded ? 0.5 : 1);

      halo(context, screen.x, screen.y, r, colour, strength);

      // Status is the fill and the outline, never the colour: an admitted claim
      // is solid, a proposal is hollow and dashed. "A model said this and
      // nothing corroborated it" must not look like an established fact.
      const hollow = status !== undefined && status !== 'admitted';
      glyphPath(context, GLYPH_FOR_KIND[node.kind], screen.x, screen.y, r);
      context.fillStyle = hollow ? SURFACE : fade(colour, strength);
      context.fill();

      if (hollow || node.kind === 'source' || node.kind === 'artifact') {
        const dash = status === 'proposed' ? [3.5, 3] : status === 'superseded' ? [1.5, 3] : [];
        context.setLineDash(dash.map((n) => n * scale));
        context.lineWidth = Math.max(1, 1.5 * scale);
        context.strokeStyle = fade(
          node.kind === 'source' || node.kind === 'artifact' ? CANONICAL_COLOUR : colour,
          strength,
        );
        context.stroke();
        context.setLineDash([]);
      }

      if (status === 'rejected') {
        context.strokeStyle = fade(colour, strength);
        context.lineWidth = Math.max(1, 1.4 * scale);
        context.beginPath();
        context.moveTo(screen.x - r * 0.7, screen.y + r * 0.7);
        context.lineTo(screen.x + r * 0.7, screen.y - r * 0.7);
        context.stroke();
      }

      // A conflict is a soft second ring: two items disagreeing is the thing an
      // operator most needs to find, so it is drawn and not merely counted.
      const conflicted =
        (node.item?.conflictsWith.length ?? 0) > 0 ||
        view.edges.some(
          (edge) =>
            edge.kind === 'contradicts' && (edge.source === node.id || edge.target === node.id),
        );
      if (conflicted) {
        context.beginPath();
        context.arc(screen.x, screen.y, r + 4 * scale, 0, Math.PI * 2);
        context.strokeStyle = fade(colour, strength * 0.55);
        context.lineWidth = Math.max(0.8, 1.2 * scale);
        context.stroke();
      }

      // In the running turn's context: a white ring, in the one colour that is
      // not an agent's, so it reads as a property of the view rather than of
      // whoever wrote the item. Dashed when the item has been corrected since
      // the turn was compiled — the model read an older revision.
      const inContext = view.inContext.get(node.id);
      if (inContext) {
        context.beginPath();
        context.arc(screen.x, screen.y, r + 7 * scale, 0, Math.PI * 2);
        context.strokeStyle = inContext.current
          ? 'rgba(255,255,255,' + 0.85 * strength + ')'
          : 'rgba(255,255,255,' + 0.4 * strength + ')';
        context.setLineDash(inContext.current ? [] : [2.5, 3].map((n) => n * scale));
        context.lineWidth = Math.max(1, 1.6 * scale);
        context.stroke();
        context.setLineDash([]);
      }

      if (node.id === focusId || node.id === selectedId) {
        context.beginPath();
        context.arc(screen.x, screen.y, r + 11 * scale, 0, Math.PI * 2);
        context.strokeStyle =
          node.id === focusId ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.45)';
        context.lineWidth = Math.max(1, (node.id === focusId ? 2 : 1.2) * scale);
        context.stroke();
      }

      // A bundle says how much it holds, inside itself, and that it opens.
      //
      // The count alone leaves a reader guessing whether the circle is a thing
      // or a door. The ring of dots around the rim is the same convention a
      // folder uses: it says there is more inside without spending a caption
      // saying so, and it disappears once the bundle is open.
      if (node.kind === 'cluster' && r > 12) {
        context.fillStyle = 'rgba(8,9,12,' + 0.88 * strength + ')';
        context.font = '600 ' + Math.max(11, Math.min(19, r * 0.6)) + 'px system-ui, sans-serif';
        context.textBaseline = 'middle';
        context.fillText(String(node.count ?? 0), screen.x, screen.y + 0.5);

        // Three dots on the lower rim: 'opens'.
        context.fillStyle = 'rgba(8,9,12,' + 0.55 * strength + ')';
        for (let i = -1; i <= 1; i += 1) {
          context.beginPath();
          context.arc(screen.x + i * r * 0.22, screen.y + r * 0.58, Math.max(1, r * 0.055), 0, Math.PI * 2);
          context.fill();
        }
      }

      // An opened parent carries a minus, so closing it is as discoverable as
      // opening it was.
      if (node.kind === 'agent' && openParents.has(node.agentId ?? '') && r > 7) {
        context.strokeStyle = 'rgba(8,9,12,' + 0.8 * strength + ')';
        context.lineWidth = Math.max(1.2, r * 0.16);
        context.beginPath();
        context.moveTo(screen.x - r * 0.38, screen.y);
        context.lineTo(screen.x + r * 0.38, screen.y);
        context.stroke();
      }
    }

    // ── Labels, last, so nothing is drawn over them ──────────────────────
    context.textBaseline = 'top';
    for (const held of nodes) {
      const node = byId.get(held.id);
      if (!node || !held.labelled) continue;
      const box = labelBox(held);
      if (!box) continue;
      const screen = toScreen(held);
      const top = toScreen({ x: held.x, y: box.y }).y;
      const lit = !hovered || near.has(node.id);

      const isCluster = node.kind === 'cluster';
      context.font = (isCluster ? '600 12px' : '11px') + ' system-ui, sans-serif';
      const text = node.label;
      const width = context.measureText(text).width;

      // A backdrop behind every caption. This is the single biggest legibility
      // win in the whole view: without it, text sitting over a line is
      // unreadable, and at any density there is always a line.
      context.fillStyle = lit ? LABEL_BACKDROP : 'rgba(8,9,12,0.5)';
      roundedRect(context, screen.x - width / 2 - 4, top - 2, width + 8, 15, 4);
      context.fill();

      context.fillStyle = lit ? INK : INK_DIM;
      context.fillText(text, screen.x, top);

      if (isCluster && node.summary && lit) {
        context.font = '10px system-ui, sans-serif';
        const sub = node.summary;
        const subWidth = context.measureText(sub).width;
        context.fillStyle = LABEL_BACKDROP;
        roundedRect(context, screen.x - subWidth / 2 - 4, top + 15, subWidth + 8, 14, 4);
        context.fill();
        context.fillStyle = INK_DIM;
        context.fillText(sub, screen.x, top + 16);
      }
    }

    // How many relations a bundled line stands for, said in words.
    if (settled.current) {
      context.font = '10px system-ui, sans-serif';
      context.textBaseline = 'middle';
      for (const edge of view.edges) {
        if (edge.kind !== 'between' || (edge.weight ?? 1) < 2) continue;
        const route = routes.current.get(edge.id);
        if (!route) continue;
        const screen = route.points.map(toScreen);
        // The true middle of the line, not `points[length/2]`: a straight route
        // has two points, and index 1 is its far end — which put the count
        // underneath the node it pointed at, where nobody could see it.
        const mid =
          screen.length === 2
            ? { x: (screen[0].x + screen[1].x) / 2, y: (screen[0].y + screen[1].y) / 2 }
            : screen[Math.floor(screen.length / 2)];
        const text = String(edge.weight);
        const width = context.measureText(text).width;
        context.fillStyle = 'rgba(8,9,12,0.85)';
        roundedRect(context, mid.x - width / 2 - 4, mid.y - 7, width + 8, 14, 7);
        context.fill();
        context.fillStyle = INK_DIM;
        context.fillText(text, mid.x, mid.y);
      }
    }
  };
  // Anything that changes the picture without moving a node still needs a
  // frame: a hover, a selection, a changed focus.
  useEffect(() => {
    drawRef.current();
  });

  return (
    <canvas
      ref={canvasRef}
      aria-label="Agent memory graph"
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        background: '#000000',
        cursor: gesture.current?.kind === 'drag' ? 'grabbing' : hovered ? 'grab' : 'default',
      }}
      onMouseDown={(event) => {
        const node = nodeAt(event.clientX, event.clientY);
        const sim = simulation.current;
        if (node && sim) {
          const held = sim.find(node.id);
          if (held) {
            held.fx = held.x;
            held.fy = held.y;
          }
          sim.reheat();
          settled.current = false;
          gesture.current = { kind: 'drag', id: node.id, moved: false };
          ensureLoop();
          return;
        }
        gesture.current = { kind: 'pan', x: event.clientX, y: event.clientY, moved: false };
      }}
      onMouseMove={(event) => {
        const active = gesture.current;
        if (active?.kind === 'drag') {
          const canvas = canvasRef.current;
          const sim = simulation.current;
          if (!canvas || !sim) return;
          const rect = canvas.getBoundingClientRect();
          const point = toWorld(event.clientX - rect.left, event.clientY - rect.top);
          const held = sim.find(active.id);
          if (held) {
            held.fx = point.x;
            held.fy = point.y;
          }
          active.moved = true;
          sim.reheat();
          ensureLoop();
          return;
        }
        if (active?.kind === 'pan') {
          const dx = event.clientX - active.x;
          const dy = event.clientY - active.y;
          if (Math.abs(dx) > 2 || Math.abs(dy) > 2) active.moved = true;
          active.x = event.clientX;
          active.y = event.clientY;
          userMoved.current = true;
          const current = viewRef.current;
          viewRef.current = { ...current, x: current.x + dx, y: current.y + dy };
          drawRef.current();
          return;
        }
        const over = nodeAt(event.clientX, event.clientY)?.id ?? null;
        if (over !== hovered) setHovered(over);
      }}
      onMouseUp={(event) => {
        const active = gesture.current;
        gesture.current = null;
        const sim = simulation.current;
        if (active?.kind === 'drag' && sim) {
          // Dropped nodes are let go rather than left pinned: a graph that
          // quietly accumulated pins would stop being a function of its data.
          const held = sim.find(active.id);
          if (held) {
            held.fx = null;
            held.fy = null;
          }
          sim.cool();
          ensureLoop();
          if (!active.moved) {
            const node = byId.get(active.id);
            if (node) onSelect?.(node);
          }
          return;
        }
        if (active?.moved) return;
        const node = nodeAt(event.clientX, event.clientY);
        if (node) onSelect?.(node);
      }}
      onMouseLeave={() => {
        const active = gesture.current;
        gesture.current = null;
        const sim = simulation.current;
        if (active?.kind === 'drag' && sim) {
          const held = sim.find(active.id);
          if (held) {
            held.fx = null;
            held.fy = null;
          }
          sim.cool();
          ensureLoop();
        }
        setHovered(null);
      }}
      onDoubleClick={(event) => {
        const node = nodeAt(event.clientX, event.clientY);
        if (node) onFocus?.(node);
      }}
    />
  );
};

export default MemoryGraphCanvas;

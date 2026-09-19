/**
 * Which colour belongs to which agent.
 *
 * ## Colour means authorship here, and only authorship
 *
 * The rest of this product is monochrome by rule (see the note at
 * index.css:117), and the notebook graph carries type by *shape* because eight
 * types cannot be told apart on a greyscale ramp. This view adds one thing that
 * genuinely is categorical, unordered, and needs to be legible at a glance
 * across a whole canvas: which agent said this. So colour is spent on exactly
 * that, and on nothing else.
 *
 * Everything else keeps its existing cue. Kind is shape, status is outline and
 * fill, the current context is a ring. An agent's colour never also means
 * "admitted" or "recent", because a channel that carries two things carries
 * neither reliably.
 *
 * ## One author, one colour, one edge
 *
 * An edge is drawn in the colour of the agent that *drew the edge* — which is a
 * different question from who authored each end, and the graph stores it
 * separately for that reason. A link from agent A's fact to agent B's fact,
 * drawn by A, is one line in A's colour.
 *
 * It is deliberately not a gradient between the two ends, and there is
 * deliberately not a second copy of the relation attributed to B. A gradient
 * would invent a provenance nothing recorded; a duplicate record would make
 * "how many times was this contradicted" answer two when it happened once.
 *
 * ## Canonical things are not anybody's
 *
 * A source document and a produced artifact are shared: several agents cite the
 * same PDF, and colouring it for whoever cited it first would be a claim about
 * ownership that is simply false. Those draw in [`CANONICAL_COLOUR`], a
 * deliberately desaturated slate that reads as "not an agent" beside any of the
 * eight. The agent-specific part is the *assertion* — the citing edge — which
 * does carry its author's colour.
 */

/**
 * The eight, in order.
 *
 * Fixed by the brief. They are the Tailwind 400-weight stops, which matters
 * only in that they were chosen to sit at similar luminance: on the black
 * surface this view uses, no one of them reads as louder than the others, so an
 * agent does not look more important because of the colour it drew.
 */
export const AGENT_PALETTE = [
  '#60A5FA',
  '#2DD4BF',
  '#A78BFA',
  '#F472B6',
  '#FBBF24',
  '#FB923C',
  '#4ADE80',
  '#F87171',
] as const;

/**
 * Shared, canonical material: sources and artifacts that belong to no agent.
 *
 * Slate 400. Desaturated on purpose — it has to be visibly *not* one of the
 * eight rather than a ninth agent colour.
 */
export const CANONICAL_COLOUR = '#94A3B8';

/**
 * A stable index from an agent id. FNV-1a, as `simulation.ts` uses for seeds.
 *
 * Exported for the test that pins it: the property that matters is that the
 * same id gives the same number on every machine and every run, and a hash
 * quietly swapped for `Math.random` or for insertion order would pass every
 * other test in this file.
 */
export function agentHash(agentId: string): number {
  let hash = 2166136261;
  for (let i = 0; i < agentId.length; i += 1) {
    hash ^= agentId.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/**
 * Assigns a colour to every agent in a graph.
 *
 * ## Why not simply hash into the palette
 *
 * Because a collision is a UX failure, not a cosmetic one: two agents drawn in
 * the same blue are two agents a person cannot tell apart, and with eight slots
 * and four agents a plain hash collides about forty per cent of the time. So
 * the hash picks a *preferred* slot and a linear probe takes the next free one.
 * Up to eight agents therefore always get eight distinct colours.
 *
 * ## What that costs, stated plainly
 *
 * The assignment depends on the set. Add a ninth agent and some agent must
 * reuse a colour — eight slots, pigeonhole, no cleverness avoids it. Add an
 * agent that probes into a taken slot and a later agent can shift. So colour is
 * a *recognition aid within one view*, not an identity: the legend names every
 * agent beside its swatch, and the inspector states the agent id in text. An
 * operator reading which agent asserted something reads the name, never the
 * hue.
 *
 * Ids are sorted first, so the result depends on which agents are present and
 * not on the order rows happened to arrive in — the same rule the simulation
 * holds for seeding.
 */
export function assignAgentColours(agentIds: Iterable<string>): Map<string, string> {
  const distinct = [...new Set(agentIds)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const taken = new Array<string | null>(AGENT_PALETTE.length).fill(null);
  const assigned = new Map<string, string>();

  for (const agentId of distinct) {
    const preferred = agentHash(agentId) % AGENT_PALETTE.length;
    let slot = preferred;
    for (let probe = 0; probe < AGENT_PALETTE.length; probe += 1) {
      const candidate = (preferred + probe) % AGENT_PALETTE.length;
      if (taken[candidate] === null) {
        slot = candidate;
        break;
      }
    }
    // Past eight agents every slot is taken and the probe finds nothing free.
    // Falling back to the preferred slot is the honest outcome: a reused
    // colour, deterministically chosen, rather than an invented ninth hue.
    taken[slot] = agentId;
    assigned.set(agentId, AGENT_PALETTE[slot]);
  }
  return assigned;
}

/** The colour for one agent, or the canonical slate when it has none. */
export function colourOf(colours: ReadonlyMap<string, string>, agentId: string | null): string {
  if (agentId === null) return CANONICAL_COLOUR;
  return colours.get(agentId) ?? CANONICAL_COLOUR;
}

/**
 * The same colour at an alpha, for a dimmed or unfocused draw.
 *
 * Canvas has no per-shape opacity that survives `globalAlpha` being used for
 * something else, so the alpha is baked into the stroke colour instead. Takes
 * the `#rrggbb` the palette holds and returns `#rrggbbaa`.
 */
export function fade(colour: string, alpha: number): string {
  const clamped = Math.max(0, Math.min(1, alpha));
  const byte = Math.round(clamped * 255)
    .toString(16)
    .padStart(2, '0');
  return `${colour}${byte}`;
}

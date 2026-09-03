/**
 * The itemised layer under the context ledger.
 *
 * ## Why sections were not enough
 *
 * {@link ContextLedger} answers "which *kind* of thing filled the window" —
 * tool schemas, evidence, transcript. That is the right question when the
 * remedy is a configuration change, and it is the wrong one the moment a person
 * attaches two documents and wants to know which of *those* is the expensive
 * one. Both land in the same section, and the section total names neither.
 *
 * So entities sit underneath sections: every entity belongs to exactly one
 * section, and the entities of a section sum to that section's total. That
 * invariant is the whole design. It is checked by {@link reconcileSections} and
 * asserted in the tests, because a breakdown whose parts do not add up to its
 * whole cannot be used to decide anything — which is the only reason to keep
 * one. The same argument is already written out at length in
 * `context-ledger.ts`, where measuring two sections with a usage-aware
 * estimator made exactly that mistake.
 *
 * ## Why the drop order is not invented here
 *
 * A meter that says "this document will be dropped first" and is wrong is worse
 * than a meter that says nothing: the person moves the wrong file. So the
 * ordering below is not a policy this module chose. It mirrors what
 * `RunCompactor.transform` actually does, in that order:
 *
 * 1. `pruneStaleToolResults` clears tool results whose evidence marker is
 *    already durable — every turn, before anything else, compaction or not.
 * 2. If still over budget, the oldest messages up to the cut point are replaced
 *    by a summary. `keepRecentTokens` protects the recent tail.
 * 3. Nothing else is ever evicted. The system prompt, the tool schemas, the
 *    working notes and the reserve stay.
 *
 * If that policy changes, this changes with it, and `evictionPlan`'s tests are
 * where the disagreement shows up.
 */

import type { ContextSection } from "./context-ledger.js";

/**
 * Where an entity stands.
 *
 * `pending` is load-bearing rather than cosmetic: a document still being read
 * by the OCR model has a real place in the next turn's context and an unknown
 * size. Showing it as absent would understate the window; showing it as active
 * with a guessed size would be the estimate-wearing-a-measurement's-name
 * problem again. It is its own state, and it reports no tokens.
 */
export type EntityStatus = "active" | "pending" | "summarised" | "dropped";

/** How an entity's token figure was arrived at. Never inferred. */
export type Measurement = "exact" | "provider" | "estimated";

/** One addressable thing occupying context. */
export interface ContextEntity {
  /**
   * Stable across turns. A file uses its content hash, so re-attaching the same
   * document does not produce a second row; a turn uses its message id.
   */
  id: string;
  /** The section this rolls into. Every entity has exactly one. */
  section: ContextSection;
  /** What to show. The file's name, "System prompt", "Turn 4" — not an id. */
  label: string;
  /** Tokens occupied. Zero while `pending`, because nothing has been counted. */
  tokens: number;
  measurement: Measurement;
  status: EntityStatus;
  /**
   * Protected from eviction by the person.
   *
   * Honoured by the compactor, not only drawn: a pin the summariser ignores is
   * a promise the screen makes and the run breaks.
   */
  pinned: boolean;
  /** Ordering hint within a section — a turn index, or arrival order. */
  sequence: number;
  /** Extra facts worth showing on the row. OCR reads carry theirs here. */
  detail?: Readonly<Record<string, string | number | boolean | null>>;
}

/** Sections whose contents the compactor will never evict. */
export const IMMOVABLE_SECTIONS: ReadonlySet<ContextSection> = new Set<ContextSection>([
  "system",
  "toolSchema",
  "notes",
  "reserve",
  "compaction",
]);

/**
 * Sums entities per section.
 *
 * `pending` entities contribute nothing, which is the point: their size is not
 * yet known, and a roll-up that guessed would make the ledger disagree with the
 * projection it claims to describe. `dropped` ones contribute nothing because
 * they are no longer there.
 */
export function rollUp(entities: readonly ContextEntity[]): Record<string, number> {
  const totals: Record<string, number> = {};
  for (const entity of entities) {
    if (entity.status === "pending" || entity.status === "dropped") continue;
    totals[entity.section] = (totals[entity.section] ?? 0) + entity.tokens;
  }
  return totals;
}

/** Total tokens currently occupied by entities. */
export function occupiedByEntities(entities: readonly ContextEntity[]): number {
  return Object.values(rollUp(entities)).reduce((sum, n) => sum + n, 0);
}

/**
 * Whether the itemised rows agree with the section totals they claim to explain.
 *
 * Returns the sections that disagree, so a caller can name them. An empty array
 * means the breakdown is trustworthy.
 *
 * Exported because it is worth asserting in production and not only in tests: a
 * silent divergence here turns every number downstream into fiction, and the
 * cheapest place to catch it is where both halves are still in hand.
 */
export function reconcileSections(
  entities: readonly ContextEntity[],
  sections: Readonly<Record<string, number>>,
): { section: string; fromEntities: number; fromSection: number }[] {
  const rolled = rollUp(entities);
  const names = new Set([...Object.keys(rolled), ...Object.keys(sections)]);
  const mismatched: { section: string; fromEntities: number; fromSection: number }[] = [];
  for (const name of names) {
    const fromEntities = rolled[name] ?? 0;
    const fromSection = sections[name] ?? 0;
    if (fromEntities !== fromSection) {
      mismatched.push({ section: name, fromEntities, fromSection });
    }
  }
  return mismatched;
}

/**
 * Completes an itemisation so that it adds up to the section totals.
 *
 * ## Why a remainder row rather than a silent gap
 *
 * The section totals are authoritative: they are measured from the projection
 * that is actually sent. The entities are an itemisation of that, and the
 * itemisation is never guaranteed to be complete — the ledger knows the name of
 * every attached document, but the transcript is a stream of messages nobody
 * registered one at a time.
 *
 * There are three things one can do about the difference, and two of them are
 * bad. Dropping it makes the rows sum to less than the bar above them, so the
 * bar looks wrong. Scaling the rows up to fit makes every individual number
 * false. What is left is to show the difference as what it is: a row saying
 * "the rest of the conversation", with the tokens it really holds.
 *
 * So this function is the one that keeps {@link reconcileSections} returning an
 * empty array, and it does it by adding truth rather than by hiding a mismatch.
 * A negative remainder — entities claiming more than the section holds — is a
 * real defect and is surfaced as a `dropped` row rather than clamped away,
 * because clamping it is what would make it invisible.
 */
export function withRemainders(
  entities: readonly ContextEntity[],
  sections: Readonly<Record<string, number>>,
  labels: Readonly<Record<string, string>> = {},
): ContextEntity[] {
  const rolled = rollUp(entities);
  const completed = [...entities];

  for (const [section, total] of Object.entries(sections)) {
    const itemised = rolled[section] ?? 0;
    const remainder = total - itemised;
    if (remainder === 0) continue;
    completed.push({
      id: `${section}:remainder`,
      section: section as ContextSection,
      label: labels[section] ?? `Rest of ${section}`,
      tokens: remainder,
      // The section total it came from was measured the same way the section
      // was; calling it exact would overstate what the estimator knows.
      measurement: "estimated",
      // A negative remainder means the itemisation over-claims. It is shown,
      // not swallowed — see this function's note.
      status: remainder > 0 ? "active" : "dropped",
      pinned: false,
      // Sorts after anything explicitly registered, so the eviction plan names
      // a document a person recognises before it names an anonymous remainder.
      sequence: Number.MAX_SAFE_INTEGER,
    });
  }

  return completed;
}

/**
 * How readily a section gives up its contents. Lower goes first.
 *
 * Only two ranks, because the compactor only has two passes. Inventing finer
 * gradations here would describe an eviction policy that does not exist.
 */
function rank(entity: ContextEntity): number {
  return entity.section === "evidence" ? 0 : 1;
}

/**
 * The entities the compactor would reclaim, in the order it would reclaim them.
 *
 * Mirrors `RunCompactor.transform`; see this module's header for the mapping.
 * The first element is what goes first.
 *
 * `protectedTailSequence` is the sequence number at which the recent tail
 * begins — everything at or after it sits inside `keepRecentTokens` and is not
 * a candidate.
 *
 * `null` means the caller does not know where that boundary falls, and the
 * answer is then an empty plan. This is deliberately *not* spelled `Infinity`:
 * the two facts are different, and conflating them is how a meter ends up
 * naming a turn the compactor was never going to touch. `Infinity` says "I know
 * the boundary, and nothing is protected"; `null` says "I do not know", and the
 * honest response to not knowing is to promise nothing — the same rule
 * `explainLedger` follows when it returns `null` instead of a hedge.
 */
export function evictionPlan(
  entities: readonly ContextEntity[],
  protectedTailSequence: number | null = null,
): ContextEntity[] {
  if (protectedTailSequence === null) return [];

  const candidates = entities.filter(
    (entity) =>
      !entity.pinned &&
      entity.status === "active" &&
      !IMMOVABLE_SECTIONS.has(entity.section) &&
      entity.sequence < protectedTailSequence,
  );

  return candidates.sort((a, b) => {
    // Stale evidence first: it is retrievable by reference, so reclaiming it
    // loses nothing at all. This is the pass that runs every turn.
    const staleness = rank(a) - rank(b);
    if (staleness !== 0) return staleness;
    // Then oldest first, which is the direction the cut point moves.
    return a.sequence - b.sequence;
  });
}

/** How close the window is to full, and what to do about it. */
export type PressureLevel = "ok" | "tight" | "critical" | "over";

export interface Pressure {
  level: PressureLevel;
  /** 0–1 against the window. `null` when the window is unknown. */
  ratio: number | null;
  /**
   * What goes first if the limit is reached, or `null` when nothing can be
   * reclaimed — which is the case worth interrupting somebody for, because it
   * means the next turn fails rather than degrades.
   */
  firstToGo: ContextEntity | null;
}

/**
 * Reads the pressure on the window.
 *
 * An unknown window (zero) reports `ok` with a `null` ratio rather than
 * guessing. Not `critical`: an unmeasured window is not evidence of a full one,
 * and a meter that shows red on every run with an unconfigured model teaches
 * people to ignore red.
 */
export function pressure(
  committed: number,
  window: number,
  entities: readonly ContextEntity[],
  protectedTailSequence: number | null = null,
): Pressure {
  const plan = evictionPlan(entities, protectedTailSequence);
  const firstToGo = plan[0] ?? null;
  if (window <= 0) {
    return { level: "ok", ratio: null, firstToGo };
  }
  const ratio = committed / window;
  const level: PressureLevel =
    ratio > 1 ? "over" : ratio >= 0.9 ? "critical" : ratio >= 0.7 ? "tight" : "ok";
  return { level, ratio, firstToGo };
}

/**
 * The share of the committed total one entity holds, 0–1.
 *
 * Zero rather than `NaN` when nothing is committed. A bar of width `NaN`
 * renders full, which would read as "this one entity filled the window" — the
 * same trap `ledgerRows` documents.
 */
export function shareOf(entity: ContextEntity, committed: number): number {
  return committed > 0 ? entity.tokens / committed : 0;
}

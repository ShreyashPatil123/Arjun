/**
 * One measurement, and how much is known about it.
 *
 * ## Why this exists
 *
 * The Audit page held its three sources as plain values — `mode`, `events`,
 * `observed` — initialised to `null` and `[]`. Nothing in those types can say
 * "nobody has looked yet", so the page rendered the initial value as a
 * finding: `{observed ? observed.connections.length : 0}` put a green **0** on
 * screen next to the words *none of them leave this machine*, before any
 * measurement had been taken and again every time one failed.
 *
 * That is the defect `securityClaims.ts` was written for, one floor down. A
 * badge reading "Audit chain: intact" without checking and a panel reading
 * "0 connections" without looking are the same lie — on the page whose entire
 * job is to be the evidence.
 *
 * ## The four states are all different
 *
 * - `loading` — nobody has looked yet. Resolves.
 * - `unavailable` — somebody looked and could not find out. Does not resolve
 *   on its own, and is *not* a zero.
 * - `measured` — a real reading, with the moment it was taken.
 * - stale — a real reading that has stopped being refreshed. Not a separate
 *   variant, because it is a property of *when* rather than of *what*: see
 *   [`isStale`], which asks the question against a clock the caller supplies.
 *
 * A measured zero is a finding and says so. The other three are not zero, and
 * none of them may borrow its words.
 *
 * ## Why the timestamp is not optional
 *
 * A reading with no time on it cannot go stale, so a page polling a source
 * that started failing ten minutes ago keeps showing the last good answer as
 * though it were current. `at` is what makes "this was true once" and "this is
 * true now" different sentences.
 */

/** A measurement of `T`, or an honest account of why there isn't one. */
export type Reading<T> =
  | { state: 'loading' }
  | { state: 'unavailable'; reason: string; at: number }
  | { state: 'measured'; value: T; at: number };

/** Nobody has looked yet. */
export const LOADING = { state: 'loading' } as const;

/** A real reading, taken at `at` (epoch milliseconds). */
export function measured<T>(value: T, at: number = Date.now()): Reading<T> {
  return { state: 'measured', value, at };
}

/**
 * Somebody looked and could not find out.
 *
 * Carries a time for the same reason a measurement does: "the connection table
 * could not be read" is more useful when a person can see it has been saying
 * that for a minute rather than for a moment.
 */
export function unavailable<T = never>(reason: string, at: number = Date.now()): Reading<T> {
  return { state: 'unavailable', reason, at };
}

/** The value if there is one, and `null` otherwise. Never a default. */
export function valueOf<T>(reading: Reading<T>): T | null {
  return reading.state === 'measured' ? reading.value : null;
}

/**
 * Whether a reading is old enough that it should not be read as current.
 *
 * `loading` is never stale — it has not claimed anything yet. An `unavailable`
 * reading can be, and that matters: a source that failed once and a source
 * that has been failing for ten minutes are different situations.
 */
export function isStale<T>(
  reading: Reading<T>,
  maxAgeMs: number,
  now: number = Date.now(),
): boolean {
  if (reading.state === 'loading') return false;
  return now - reading.at > maxAgeMs;
}

/**
 * How long ago, in words, for a reader glancing at a panel.
 *
 * Deliberately coarse. The question is "is this current?", not the exact age,
 * and a ticking seconds counter would invite precision into a number whose
 * accuracy is bounded by the poll interval anyway.
 */
export function describeAge(at: number, now: number = Date.now()): string {
  const seconds = Math.max(0, Math.round((now - at) / 1000));
  if (seconds < 2) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.round(minutes / 60)}h ago`;
}

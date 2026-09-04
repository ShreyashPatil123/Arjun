/**
 * The four states a measurement can be in, and why they may not be merged.
 *
 * ## The defect this models against
 *
 * The Audit page held `observed: ObservationReport | null`, seeded `null`, and
 * rendered `{observed ? observed.connections.length : 0}` beside the words
 * *none of them leave this machine*. `null` meant three different things —
 * nobody has looked yet, the probe failed, the poll threw — and all three came
 * out as a green **0**, which is the one thing they definitely did not mean.
 *
 * Every test here is a pair the old `| null` could not tell apart.
 */
import { describe, expect, it } from 'vitest';
import {
  LOADING,
  describeAge,
  isStale,
  measured,
  unavailable,
  valueOf,
  type Reading,
} from './reading';

/** A fixed clock. Nothing here reads the real one. */
const T0 = 1_700_000_000_000;
const seconds = (n: number) => T0 + n * 1000;

describe('a measurement and the absence of one are different values', () => {
  it('gives no value for a reading nobody has taken', () => {
    expect(valueOf(LOADING as Reading<number>)).toBeNull();
  });

  it('gives no value for a reading that could not be taken', () => {
    expect(valueOf(unavailable<number>('the table could not be read', T0))).toBeNull();
  });

  it('distinguishes a measured zero from both of them', () => {
    // The defect, in one assertion. All three used to render as `0`.
    const zero = measured(0, T0);
    expect(valueOf(zero)).toBe(0);
    expect(zero.state).toBe('measured');
    expect(LOADING.state).not.toBe('measured');
    expect(unavailable('x', T0).state).not.toBe('measured');
  });

  it('keeps the reason a reading could not be taken', () => {
    const reading = unavailable('GetExtendedTcpTable returned 1244', T0);
    expect(reading.state === 'unavailable' && reading.reason).toBe(
      'GetExtendedTcpTable returned 1244',
    );
  });

  it('records when every non-loading reading happened', () => {
    // Without this a page cannot tell "true now" from "true once".
    //
    // Narrowed rather than reached into: `loading` has no `at`, and the union
    // is what stops a caller assuming every reading has a time on it.
    const taken = measured(3, T0);
    expect(taken.state === 'measured' && taken.at).toBe(T0);
    const failed = unavailable('x', T0);
    expect(failed.state === 'unavailable' && failed.at).toBe(T0);
  });
});

describe('staleness is asked against a clock, not assumed', () => {
  it('does not call a fresh reading stale', () => {
    expect(isStale(measured(0, T0), 8000, seconds(3))).toBe(false);
  });

  it('calls a reading stale once it stops being refreshed', () => {
    expect(isStale(measured(0, T0), 8000, seconds(9))).toBe(true);
  });

  it('never calls a loading reading stale', () => {
    // It has not claimed anything yet, so there is nothing to go out of date.
    expect(isStale(LOADING, 1, seconds(3600))).toBe(false);
  });

  it('lets an unavailable reading go stale too', () => {
    // A source that failed once and one that has been failing for ten minutes
    // are different situations, and only the second is alarming.
    expect(isStale(unavailable('x', T0), 8000, seconds(3))).toBe(false);
    expect(isStale(unavailable('x', T0), 8000, seconds(600))).toBe(true);
  });

  it('treats the boundary as not yet stale', () => {
    expect(isStale(measured(0, T0), 8000, seconds(8))).toBe(false);
  });
});

describe('ages read the way somebody glancing at a panel would say them', () => {
  it('rounds the recent past to a phrase, not a number', () => {
    expect(describeAge(T0, T0)).toBe('just now');
    expect(describeAge(T0, seconds(1))).toBe('just now');
  });

  it('counts seconds, then minutes, then hours', () => {
    expect(describeAge(T0, seconds(4))).toBe('4s ago');
    expect(describeAge(T0, seconds(90))).toBe('2m ago');
    expect(describeAge(T0, seconds(3600))).toBe('1h ago');
  });

  it('never reports a negative age from a clock that stepped back', () => {
    expect(describeAge(seconds(10), T0)).toBe('just now');
  });
});

/**
 * The Audit page may not claim a measurement it does not have.
 *
 * ## The defect
 *
 *     const [observed, setObserved] = useState<ObservationReport | null>(null);
 *     ...
 *     const [nextMode, nextEvents, nextObserved] = await Promise.all([...]);
 *     ...
 *     <strong>{observed ? observed.connections.length : 0}</strong> TCP connection
 *     ... <strong>none of them leave this machine</strong>.
 *
 * Three faults, compounding:
 *
 *  1. `null` rendered as `0`. Before the first poll, and after every failed
 *     one, the page put a green zero next to the strongest claim it makes.
 *  2. The `Promise.all` gave the three sources one fate. A connection table
 *     that could not be read rejected the batch, so the mode and the broker's
 *     log kept their last values with nothing saying they had stopped being
 *     refreshed.
 *  3. No timestamps, so a reading from ten minutes ago and one from this
 *     second looked identical.
 *
 * The page whose whole purpose is to be the evidence was the page asserting
 * things nobody had measured.
 *
 * ## Why these are source assertions
 *
 * `vitest.config.mjs` runs `environment: 'node'` and this repository vendors no
 * DOM, so a rendering test is not available. `src/contexts/activeRun.test.ts`
 * sets the precedent of pinning a wiring invariant by reading the file. The
 * behaviour underneath is covered by `src/services/reading.test.ts`.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

/** The page with its comments removed, so prose cannot satisfy a check. */
const source = readFileSync('src/pages/AuditNetwork.tsx', 'utf8')
  .split('\n')
  .filter(line => {
    const trimmed = line.trim();
    return !trimmed.startsWith('//') && !trimmed.startsWith('*') && !trimmed.startsWith('/*');
  })
  .join('\n');

describe('nothing unmeasured is rendered as a measurement', () => {
  it('never falls back to zero for a reading it does not have', () => {
    // The exact regression.
    expect(source).not.toMatch(/observed\s*\?\s*observed\.connections\.length\s*:\s*0/);
    // Nor any other shape of the same idea.
    expect(source).not.toMatch(/:\s*0\}<\/strong>/);
  });

  it('holds each source as a reading rather than a nullable value', () => {
    expect(source).not.toMatch(/useState<ObservationReport \| null>\(null\)/);
    expect(source).toContain('useState<Reading<ObservationReport>>(LOADING)');
    expect(source).toContain('useState<Reading<EgressEvent[]>>(LOADING)');
    expect(source).toContain('useState<Reading<OperatingMode>>(LOADING)');
  });

  it('renders the loading and unavailable states as themselves', () => {
    expect(source).toContain("observedReading.state === 'loading'");
    expect(source).toContain("observedReading.state === 'unavailable'");
    // The broker's log had the same defect: an empty list because the log
    // could not be read is not "nothing attempted to leave this machine".
    expect(source).toContain("eventsReading.state === 'loading'");
    expect(source).toContain("eventsReading.state === 'unavailable'");
  });

  it('reads the measured value through the narrowed union', () => {
    // Not through a nullable alias with a `!` on it, which is the same
    // assumption the bug rested on, spelled as a compiler override.
    expect(source).toContain('observedReading.value.connections.length');
    expect(source).not.toContain('observed!');
  });
});

describe('the sources are refreshed independently', () => {
  it('does not settle all three probes as one batch', () => {
    // The `Promise.all` that gave them a shared fate destructured its results.
    expect(source).not.toMatch(/const \[nextMode, nextEvents, nextObserved\]/);
  });

  it('gives each source its own success and failure path', () => {
    // One helper, applied per source, so a failing probe marks only itself.
    expect(source).toContain('setModeReading');
    expect(source).toContain('setEventsReading');
    expect(source).toContain('setObservedReading');
    expect(source).toMatch(/set\(measured\(value\)\)/);
    expect(source).toMatch(/set\(unavailable\(/);
  });

  it('does not funnel every probe failure into one page-level error', () => {
    // A single `error` string could not say *which* source failed, so the page
    // showed one message while three panels went on looking live.
    expect(source).not.toMatch(/const \[error, setError\]/);
  });
});

describe('every reading says when it was taken', () => {
  it('shows the age of what it is claiming', () => {
    expect(source).toContain('describeAge');
  });

  it('calls out a reading that has stopped being refreshed', () => {
    expect(source).toContain('isStale');
    expect(source).toContain('STALE_AFTER_MS');
    expect(source).toMatch(/observedStale/);
    expect(source).toMatch(/eventsStale/);
  });
});

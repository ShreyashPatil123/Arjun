import { describe, expect, it } from 'vitest';
import { computeTokenMetrics } from './useTokenMetrics';
import { formatTokens } from './format';

/**
 * The counter that was blank.
 *
 * Every assistant message in the chat showed a duration and no token count.
 * Two things caused that together: the run's own completion write erased the
 * usage the front-end had just recorded (covered by a Rust test), and this
 * side then had no reported count and no duration it trusted, so it reported
 * a rate of zero — which the status pill reads as "nothing to show".
 */
describe('computeTokenMetrics', () => {
  const settled = {
    isLive: false,
    contentLength: 0,
    liveElapsed: 0,
  };

  it('reports what the model reported, exactly, when the model reported it', () => {
    const metrics = computeTokenMetrics({
      ...settled,
      contentLength: 4000,
      tokensIn: 512,
      tokensOut: 64,
      elapsedMs: 2000,
    });
    expect(metrics.tokensIn).toBe(512);
    // Not the 1000 the text would have been estimated at.
    expect(metrics.tokensOut).toBe(64);
    expect(metrics.approx).toBe(false);
    expect(metrics.speed).toBe(32);
  });

  it('estimates from the text when the server sends no usage, and says so', () => {
    const metrics = computeTokenMetrics({
      ...settled,
      contentLength: 400,
      elapsedMs: 2000,
    });
    expect(metrics.tokensOut).toBe(100);
    expect(metrics.approx).toBe(true);
    expect(metrics.speed).toBe(50);
  });

  it('times a settled message by the run, not by when the cell was mounted', () => {
    // The case that produced a blank counter on reload: the component has
    // only just mounted, so a clock started here has measured nothing.
    const metrics = computeTokenMetrics({
      ...settled,
      contentLength: 400,
      elapsedMs: 8400,
      liveElapsed: 0,
    });
    expect(metrics.elapsedMs).toBe(8400);
    expect(metrics.speed).toBe(12);
  });

  it('times a live message by the stream, ignoring any duration already on the row', () => {
    const metrics = computeTokenMetrics({
      isLive: true,
      contentLength: 400,
      elapsedMs: 999_999,
      liveElapsed: 1000,
    });
    expect(metrics.elapsedMs).toBe(1000);
    expect(metrics.speed).toBe(100);
  });

  it('reports no rate rather than a division by zero when nothing has elapsed', () => {
    const metrics = computeTokenMetrics({ ...settled, contentLength: 400, elapsedMs: 0 });
    expect(metrics.speed).toBe(0);
    expect(metrics.tokensOut).toBe(100);
  });

  it('reports nothing at all for a message with no text and no usage', () => {
    const metrics = computeTokenMetrics({ ...settled, elapsedMs: 5000 });
    expect(metrics.tokensOut).toBe(0);
    expect(metrics.speed).toBe(0);
  });
});

describe('formatTokens', () => {
  it('keeps small counts exact and abbreviates the ones the pill cannot fit', () => {
    expect(formatTokens(0)).toBe('0');
    expect(formatTokens(64)).toBe('64');
    expect(formatTokens(999)).toBe('999');
    expect(formatTokens(1000)).toBe('1.0k');
    expect(formatTokens(1240)).toBe('1.2k');
    expect(formatTokens(12_400)).toBe('12k');
  });
});

import { describe, it, expect } from 'vitest';
import { relativeTime } from './relativeTime';

describe('relativeTime', () => {
  const NOW = new Date('2026-03-04T12:00:00Z').getTime();

  it('returns "just now" for times within the last minute', () => {
    expect(relativeTime(new Date(NOW - 5_000).toISOString(), NOW)).toBe('just now');
    expect(relativeTime(new Date(NOW - 59_000).toISOString(), NOW)).toBe('just now');
  });

  it('returns minutes for the last hour', () => {
    expect(relativeTime(new Date(NOW - 60_000).toISOString(), NOW)).toBe('1m');
    expect(relativeTime(new Date(NOW - 5 * 60_000).toISOString(), NOW)).toBe('5m');
    expect(relativeTime(new Date(NOW - 59 * 60_000).toISOString(), NOW)).toBe('59m');
  });

  it('returns hours for the last day', () => {
    expect(relativeTime(new Date(NOW - 60 * 60_000).toISOString(), NOW)).toBe('1h');
    expect(relativeTime(new Date(NOW - 23 * 60 * 60_000).toISOString(), NOW)).toBe('23h');
  });

  it('returns days for the last week', () => {
    expect(relativeTime(new Date(NOW - 24 * 60 * 60_000).toISOString(), NOW)).toBe('1d');
    expect(relativeTime(new Date(NOW - 6 * 24 * 60 * 60_000).toISOString(), NOW)).toBe('6d');
  });

  it('falls back to a locale date for older times', () => {
    const out = relativeTime(new Date(NOW - 30 * 24 * 60 * 60_000).toISOString(), NOW);
    // We don't pin a locale-specific month name in CI; assert the shape
    // (non-empty, contains a numeric day).
    expect(out).toMatch(/\d+/);
  });

  it('returns empty string for invalid input', () => {
    expect(relativeTime('not-a-date', NOW)).toBe('');
    expect(relativeTime('', NOW)).toBe('');
  });

  it('uses Date.now() when no reference is supplied', () => {
    const out = relativeTime(new Date(Date.now() - 30_000).toISOString());
    expect(out).toBe('just now');
  });
});

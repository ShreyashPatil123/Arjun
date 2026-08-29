/**
 * Compact relative-time labels for short lists.
 *
 * Used by the AppMenu dropdown to show how long ago a conversation was
 * last active. Kept separate from any date-fns / dayjs import so the
 * offline build does not need to vendor a larger library for a single
 * formatter.
 */

/** Render a date as a compact relative label: "just now", "5m", "2h", "3d", "Mar 4". */
export function relativeTime(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return '';
  const diff = now - then;
  if (diff < 60_000) return 'just now';
  const mins = Math.floor(diff / 60_000);
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/**
 * A span of milliseconds the way a person would say it.
 *
 * Kept in its own file so the chat components and the run inspector can
 * share one definition. Below one second we say "X ms" (so a 200 ms
 * calculation does not round to "0 s"); below one minute we use one
 * decimal under ten seconds, no decimal otherwise; above a minute we
 * split into minutes and seconds.
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  if (ms < 60_000) {
    const seconds = ms / 1000;
    return `${seconds.toFixed(seconds < 10_000 ? 1 : 0)} s`;
  }
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

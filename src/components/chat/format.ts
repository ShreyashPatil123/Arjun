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

/**
 * A token count at the size a status line has room for.
 *
 * Exact below a thousand, because the difference between 180 and 240 tokens
 * is something an operator reads; abbreviated above it, because the
 * difference between 12,400 and 12,600 is not, and the pill is narrow.
 */
export function formatTokens(tokens: number): string {
  if (tokens < 1000) return `${Math.max(0, Math.round(tokens))}`;
  const thousands = tokens / 1000;
  return `${thousands < 10 ? thousands.toFixed(1) : Math.round(thousands)}k`;
}

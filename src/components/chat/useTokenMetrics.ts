import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Characters per token, for the estimate used when the model reports nothing.
 *
 * Four is the usual rough figure for English prose. Code runs denser than
 * that, so the estimate reads low on a code answer — which is why anything
 * derived from it is shown with a `~` rather than presented as a count.
 */
const CHARS_PER_TOKEN = 4;

export interface TokenMetrics {
  tokensIn: number;
  tokensOut: number;
  /** Output tokens per second. */
  speed: number;
  elapsedMs: number;
  /**
   * True when `tokensOut` was estimated from the text rather than reported by
   * the model. A local server that does not send `usage` is common enough
   * that showing nothing at all would leave the counter permanently blank.
   */
  approx: boolean;
}

function estimate(chars: number): number {
  return chars > 0 ? Math.max(1, Math.round(chars / CHARS_PER_TOKEN)) : 0;
}

function rate(tokens: number, elapsedMs: number): number {
  if (tokens <= 0 || elapsedMs <= 0) return 0;
  return Math.round((tokens / elapsedMs) * 1000);
}

/**
 * The counter's whole decision, with no React in it.
 *
 * Split out because this repository tests the frontend's pure modules and
 * vendors no DOM to render a hook into. Everything that can be got wrong here
 * — which duration to divide by, what to do when the model reports no usage —
 * is in this function, and the hook around it only supplies a clock.
 */
export function computeTokenMetrics(input: {
  isLive: boolean;
  contentLength: number;
  tokensIn?: number | null;
  tokensOut?: number | null;
  /** The duration the run recorded. Present once the message has settled. */
  elapsedMs?: number | null;
  /** Time since this stream started, as measured while it is running. */
  liveElapsed: number;
}): TokenMetrics {
  const reportedOut = input.tokensOut ?? null;
  const out = reportedOut ?? estimate(input.contentLength);
  // A settled message is timed by what the run recorded, not by a clock this
  // side started: a cell rendering a message read back from disk mounted
  // milliseconds ago, and dividing the token count by that reports nonsense.
  const span = input.isLive ? input.liveElapsed : (input.elapsedMs ?? input.liveElapsed);
  return {
    tokensIn: input.tokensIn ?? 0,
    tokensOut: out,
    speed: rate(out, span),
    elapsedMs: span,
    approx: reportedOut === null,
  };
}

/**
 * The token counter behind an assistant cell.
 *
 * While a message streams the count is measured from the start of the stream
 * and refreshed four times a second. Once it settles the reading comes from
 * what the run recorded. See {@link computeTokenMetrics} for the rules; this
 * hook is the clock they are applied to.
 */
export function useTokenMetrics(
  isLive: boolean,
  contentLength: number,
  tokensIn?: number | null,
  tokensOut?: number | null,
  elapsedMs?: number | null,
): TokenMetrics {
  const startRef = useRef<number>(Date.now());
  const [liveElapsed, setLiveElapsed] = useState(0);

  useEffect(() => {
    if (!isLive) return;
    // Depends on `isLive` alone, so the clock is rewound on the edge into
    // streaming and not every time a delta lands.
    startRef.current = Date.now();
    setLiveElapsed(0);
    const id = window.setInterval(() => {
      setLiveElapsed(Date.now() - startRef.current);
    }, 200);
    return () => window.clearInterval(id);
  }, [isLive]);

  return useMemo(
    () =>
      computeTokenMetrics({
        isLive,
        contentLength,
        tokensIn,
        tokensOut,
        elapsedMs,
        liveElapsed,
      }),
    [isLive, contentLength, tokensIn, tokensOut, elapsedMs, liveElapsed],
  );
}

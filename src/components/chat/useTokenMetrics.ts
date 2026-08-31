import { useEffect, useRef, useState } from 'react';

interface TokenMetrics {
  tokensIn: number;
  tokensOut: number;
  speed: number; // tokens per second
  elapsedMs: number;
}

export function useTokenMetrics(
  isLive: boolean,
  contentLength: number,
  tokensIn?: number | null,
  tokensOut?: number | null
) {
  const startRef = useRef<number>(Date.now());
  const [metrics, setMetrics] = useState<TokenMetrics>({
    tokensIn: tokensIn ?? 0,
    tokensOut: tokensOut ?? 0,
    speed: 0,
    elapsedMs: 0,
  });

  // Reset timer when a new live stream starts
  useEffect(() => {
    if (isLive) {
      startRef.current = Date.now();
      setMetrics({ tokensIn: tokensIn ?? 0, tokensOut: tokensOut ?? 0, speed: 0, elapsedMs: 0 });
    }
  }, [isLive, tokensIn, tokensOut]);

  // Live update every 200ms while streaming
  useEffect(() => {
    if (!isLive) {
      // On completion, freeze to provided final counts
      if (tokensIn != null && tokensOut != null) {
        const elapsed = Math.max(1, Date.now() - startRef.current);
        setMetrics({
          tokensIn,
          tokensOut,
          speed: Math.round((tokensOut / elapsed) * 1000),
          elapsedMs: elapsed,
        });
      }
      return;
    }

    const id = setInterval(() => {
      const now = Date.now();
      const elapsed = now - startRef.current;
      // Approximate live tokens: 1 token ≈ 4 chars for display only
      const approxOut = Math.max(1, Math.floor(contentLength / 4));
      const speed = elapsed > 0 ? Math.round((approxOut / elapsed) * 1000) : 0;
      setMetrics({
        tokensIn: tokensIn ?? 0,
        tokensOut: approxOut,
        speed,
        elapsedMs: elapsed,
      });
    }, 200);

    return () => clearInterval(id);
  }, [isLive, contentLength, tokensIn, tokensOut]);

  return metrics;
}
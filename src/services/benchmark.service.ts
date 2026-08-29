/**
 * Service for the System Health page's benchmark section.
 *
 * The page renders the most recent `run_benchmark` result. The
 * `synthetic_benchmark` command is the source for the "what the
 * SIH pitch quotes" row, which the page shows before any real
 * run has been recorded.
 */

import { getBackendService } from './api';

export interface BenchmarkResult {
  modelId: string;
  promptTokens: number;
  replyTokens: number;
  ttftMs: number;
  totalMs: number;
  tokensPerSecond: number;
  vramPeakMib: number;
  accuracyPct: number;
  at: string;
  hardwareTier: string;
}

export interface BenchmarkRow extends BenchmarkResult {
  synthetic: boolean;
}

export const benchmarkService = {
  /** Returns a synthetic row for the SIH pitch (Tier 1 / RTX 5060 4GB). */
  synthetic(): Promise<BenchmarkRow> {
    return getBackendService().invoke<BenchmarkRow>('synthetic_benchmark');
  },

  /** Returns the most recent rows, newest first. */
  recent(limit?: number): Promise<BenchmarkRow[]> {
    return getBackendService().invoke<BenchmarkRow[]>('recent_benchmarks', {
      limit: limit ?? 5,
    });
  },
};

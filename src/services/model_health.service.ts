// filepath: src/services/model_health.service.ts
/**
 * Read the per-model telemetry aggregate the backend keeps in memory.
 *
 * The backend stores one row per model id, with the most recent calls
 * collapsed. The audit log is the source of truth for full history;
 * this is the "what is right now" view the Model Health page renders.
 *
 * No periodic timer here: the page polls. If the page is closed, the
 * backend keeps recording calls into the audit log, so reopening the
 * page later shows recent activity rather than a stale snapshot.
 */
import { invoke } from '@tauri-apps/api/core';

export interface ModelAggregate {
  modelId: string;
  calls: number;
  ok: number;
  refused: number;
  timeouts: number;
  oom: number;
  otherFailures: number;
  fallbacksUsed: number;
  totalLatencyMs: number;
  maxLatencyMs: number;
  totalTokensIn: number;
  totalTokensOut: number;
  /** RFC 3339, UTC. null until the first call has been recorded. */
  lastSeen: string | null;
  /** Average tokens-per-second over the observed window, when available. */
  avgTokensPerSecond: number | null;
}

export const modelHealthService = {
  snapshot(): Promise<ModelAggregate[]> {
    return invoke<ModelAggregate[]>('model_health_snapshot');
  },
};

import { getBackendService } from './api';

/**
 * How one reading stands.
 *
 * `unknown` is the one that matters. A panel showing green because a probe
 * failed converts an unknown into a reassurance, so "could not be checked" is
 * a state of its own and is never rendered as if it were fine.
 */
export type Reading = 'ok' | 'attention' | 'unknown';

/** One line on the panel. */
export interface HealthItem {
  name: string;
  state: Reading;
  /** The number or short phrase shown large. */
  value: string;
  /** One line explaining what the value means, in a person's words. */
  note: string;
}

export interface HealthSnapshot {
  /** ISO-8601, from the backend clock. */
  takenAt: string;
  items: HealthItem[];
  /**
   * Always 0. Carried on the response so the constraint is visible on the panel
   * itself rather than only in the code — ARJUN design rule 34 requires that no health
   * check call anything external.
   */
  externalCallsMade: number;
}

export const healthService = {
  /**
   * Reads GPU, model, index, queue, network and approval state.
   *
   * Every source is local: a DXGI query, a COUNT against the on-disk index, the
   * broker's in-memory log, and the OS's own socket table. Nothing is fetched.
   */
  snapshot(): Promise<HealthSnapshot> {
    return getBackendService().invoke<HealthSnapshot>('health_snapshot');
  },
};

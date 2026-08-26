import { getBackendService } from './api';

/**
 * ARJUN runs in one of two modes, and the two never overlap:
 *
 * - `provisioning` — the network is reachable, but only for the model catalog
 *   and weight download, and no confidential document may be opened.
 * - `work` — every outbound call is refused. All confidential work happens here.
 *
 * The safety comes from the pair, not from either one alone.
 */
export type OperatingMode = 'provisioning' | 'work';

/** One decision the network broker made about an outbound attempt. */
export interface EgressEvent {
  /** ISO-8601, from the backend clock. */
  at: string;
  /** Host as parsed from the URL — never a raw string fragment. */
  host: string;
  mode: OperatingMode;
  permitted: boolean;
  reason: string;
  /** True when this was the app deliberately testing its own controls. */
  canary: boolean;
}

/** One TCP connection the operating system attributes to the ARJUN process. */
export interface ObservedConnection {
  local: string;
  remote: string;
  /** False when the remote address leaves this machine — the thing that matters. */
  loopback: boolean;
}

/**
 * What the OS independently reports, as opposed to what the broker says it did.
 *
 * `unavailableReason` being set means the query could not run — which must be
 * shown as "unknown", never as a clean result.
 */
export interface ObservationReport {
  connections: ObservedConnection[];
  externalCount: number;
  unavailableReason: string | null;
}

export const sovereigntyService = {
  getMode(): Promise<OperatingMode> {
    return getBackendService().invoke<OperatingMode>('get_operating_mode');
  },

  /** Returns the previous mode, so the caller can record the transition. */
  setMode(mode: OperatingMode): Promise<OperatingMode> {
    return getBackendService().invoke<OperatingMode>('set_operating_mode', { mode });
  },

  /** Newest first. Includes permitted calls, not only refusals. */
  recentEvents(): Promise<EgressEvent[]> {
    return getBackendService().invoke<EgressEvent[]>('recent_egress_events');
  },

  /**
   * Deliberately attempts an external connection that must fail.
   * In Work mode the returned event must come back `permitted: false`.
   */
  runCanary(): Promise<EgressEvent> {
    return getBackendService().invoke<EgressEvent>('run_egress_canary');
  },

  /** Asks Windows which connections this process owns. Does not consult the broker. */
  observeConnections(): Promise<ObservationReport> {
    return getBackendService().invoke<ObservationReport>('observe_process_connections');
  },

  /**
   * Checks whether confidential material may be handled right now.
   *
   * Rejects with the refusal text when ARJUN is in Provisioning mode — the
   * network is reachable then, so nothing confidential may be opened.
   */
  assertConfidentialAllowed(operation: string): Promise<void> {
    return getBackendService().invoke<void>('assert_confidential_allowed', { operation });
  },
};

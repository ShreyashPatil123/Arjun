/**
 * What the security badges are allowed to say.
 *
 * ## Why this exists
 *
 * The SIH dashboard's badges were JSX string literals: "Sovereign", "Audit
 * intact", "Work mode", "Zero egress", "Audit chain: intact". They said the
 * same thing on a machine with a broken audit chain, on a machine in
 * provisioning mode with the network open, and on a machine where nobody had
 * checked. This repository ships evidence to judges; a badge that reads
 * "Audit chain: intact" without having verified one is the single worst thing
 * in it.
 *
 * Every claim here is derived from state the backend actually measured, and
 * every one of them can say it does not know. The rule the whole module exists
 * to enforce: **no positive claim without supporting state.**
 *
 * ## Why "zero egress" needs an interval
 *
 * "Zero egress" as a standing badge is unfalsifiable — it is equally true of a
 * machine that blocked a thousand attempts and one whose broker is not running.
 * A measured zero is a count over a window: *no outbound connection was
 * permitted among the attempts recorded in the last N minutes*. That is a claim
 * with a denominator, and it goes wrong loudly when the denominator is missing.
 */

import type { EgressEvent, OperatingMode } from './sovereignty.service';
import type { ChainVerification } from './governance.service';

/**
 * How much is known about one security property.
 *
 * `loading` and `unknown` are different: the first is a state that resolves,
 * the second is one that has resolved to "nobody can say". A surface that
 * conflated them would spin forever on a broken probe.
 */
export type ClaimLevel = 'loading' | 'unknown' | 'verified' | 'degraded' | 'failed';

export interface SecurityClaim {
  level: ClaimLevel;
  /** The words on the badge. Never a positive claim unless `verified`. */
  label: string;
  /** What the claim rests on, for a tooltip or an inspector. */
  detail: string;
}

/** Nothing has come back yet. */
const loading = (label: string): SecurityClaim => ({
  level: 'loading',
  label,
  detail: 'Checking.',
});

/**
 * The sovereignty-mode claim.
 *
 * `work` is the mode in which confidential material may be handled and the
 * network is closed; provisioning opens the network deliberately, and saying
 * "Sovereign" during it would be false at exactly the moment it matters.
 */
export function sovereigntyClaim(mode: OperatingMode | null | undefined): SecurityClaim {
  if (mode === undefined) return loading('Sovereignty');
  if (mode === null) {
    return {
      level: 'unknown',
      label: 'Sovereignty unknown',
      detail: 'The operating mode could not be read, so no claim is made about it.',
    };
  }
  if (mode === 'work') {
    return {
      level: 'verified',
      label: 'Sovereign · Work mode',
      detail: 'Confidential material may be handled and outbound connections are refused.',
    };
  }
  return {
    level: 'degraded',
    label: 'Provisioning · network open',
    detail:
      'The network is deliberately reachable in this mode, so confidential material must not ' +
      'be handled. This is a working state, not a fault.',
  };
}

/**
 * The egress claim, as a measured count over a stated window.
 *
 * `events` is the broker's own record, newest first. `null` means the broker
 * could not be reached — which is precisely when a standing "Zero egress"
 * badge would be most misleading and least justified.
 */
export function egressClaim(
  events: EgressEvent[] | null | undefined,
  windowMinutes: number,
): SecurityClaim {
  if (events === undefined) return loading('Egress');
  if (events === null) {
    return {
      level: 'unknown',
      label: 'Egress unknown',
      detail:
        'The egress broker could not be reached, so nothing is claimed about outbound ' +
        'connections.',
    };
  }

  const permitted = events.filter((event) => event.permitted);
  if (permitted.length > 0) {
    return {
      level: 'degraded',
      label: `${permitted.length} permitted in ${windowMinutes} min`,
      detail:
        `${permitted.length} outbound connection(s) were permitted among the attempts ` +
        `recorded in the last ${windowMinutes} minutes. Permitted is not the same as wrong — ` +
        'provisioning permits them deliberately — but it is not zero.',
    };
  }

  return {
    level: 'verified',
    label: `Zero egress · ${events.length} checked / ${windowMinutes} min`,
    detail:
      `${events.length} outbound attempt(s) were recorded in the last ${windowMinutes} ` +
      'minutes and none was permitted. A measured zero, not a standing assertion.',
  };
}

/**
 * The audit-chain claim.
 *
 * The one badge that must never be optimistic. "Intact" is a cryptographic
 * result: every row's seal recomputed and agreeing. Anything else — not yet
 * checked, unreadable, or a broken seal — says so.
 */
export function auditChainClaim(
  verification: ChainVerification | null | undefined,
): SecurityClaim {
  if (verification === undefined) return loading('Audit chain');
  if (verification === null) {
    return {
      level: 'unknown',
      label: 'Audit chain unchecked',
      detail: 'The audit chain could not be verified, so nothing is claimed about it.',
    };
  }
  if (!verification.intact) {
    return {
      level: 'failed',
      label: `Audit chain broken at #${verification.firstBrokenSeq ?? '?'}`,
      detail:
        verification.detail ||
        'A seal did not recompute. Everything before the break remains verifiable; ' +
          'everything after it must not be relied on.',
    };
  }
  if (verification.entriesChecked === 0) {
    // Zero rows verify vacuously. Presenting that as "intact" is how an empty
    // installation would claim the strongest property in the product.
    return {
      level: 'unknown',
      label: 'Audit chain empty',
      detail: 'There are no entries to verify yet, so nothing is claimed about the chain.',
    };
  }
  return {
    level: 'verified',
    label: `Audit chain intact · ${verification.entriesChecked} entries`,
    detail: `${verification.entriesChecked} entries were re-sealed and every seal agreed.`,
  };
}

/** Whether a claim asserts something good. Used by the tests, and by styling. */
export function isPositiveClaim(claim: SecurityClaim): boolean {
  return claim.level === 'verified';
}

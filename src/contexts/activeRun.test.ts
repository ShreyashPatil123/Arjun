/**
 * Every surface follows one run.
 *
 * ## The defect
 *
 * `useRun` is a hook, so every caller gets its own state: its own run id, its
 * own event subscription, its own filter. `SIHDashboard` called it and never
 * called `start`, so its instance had no run id and discarded every event. Its
 * routing, plan, activity, verification and security panes showed the idle
 * state for the whole of a run somebody had launched from the chat — and a
 * pane that renders "idle" honestly is indistinguishable from one wired to
 * nothing at all.
 *
 * Two things are pinned here. That the panes derive from one state, so they
 * cannot disagree about which run they are describing; and that no page has
 * gone back to constructing a private instance, which is the only way they
 * could start disagreeing again.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { paneViews } from './ActiveRunContext';
import {
  auditChainClaim,
  egressClaim,
  isPositiveClaim,
  sovereigntyClaim,
} from '../services/securityClaims';
import type { RunViewState } from '../components/run/useRun';

/** A run in flight, with something for each pane. */
function runningState(runId: string): RunViewState {
  return {
    phase: 'running',
    state: 'running',
    runId,
    prompt: 'Specify the seal for pump P-101',
    plan: {
      steps: [],
      stepsTaken: 2,
      maxSteps: 12,
      stoppedBecause: null,
    },
    activity: [{ tool: 'knowledge.search_authorized', outcome: 'succeeded' }],
    summary: null,
    seq: 3,
    error: null,
    stopped: null,
  } as unknown as RunViewState;
}

/** Nothing running. */
function idleState(): RunViewState {
  return {
    phase: 'idle',
    state: null,
    runId: null,
    prompt: '',
    plan: null,
    activity: [],
    summary: null,
    seq: 0,
    error: null,
    stopped: null,
  } as unknown as RunViewState;
}

describe('paneViews: every pane describes the same run', () => {
  it('gives all five panes one matching run id', () => {
    const views = paneViews(runningState('run-abc'));
    const ids = Object.values(views).map((pane) => pane.runId);
    expect(new Set(ids).size, `panes disagreed about the run: ${ids.join(', ')}`).toBe(1);
    expect(ids[0]).toBe('run-abc');
  });

  it('names every pane, so none can be quietly left on its own source', () => {
    const views = paneViews(runningState('run-abc'));
    expect(Object.keys(views).sort()).toEqual([
      'activity',
      'plan',
      'routing',
      'security',
      'verification',
    ]);
  });

  it('distinguishes a pane with no data from a pane on a different run', () => {
    // The distinction the dashboard could not previously make. A routing pane
    // with nothing to show is still describing this run; it is not describing
    // no run.
    const views = paneViews(runningState('run-abc'));
    expect(views.routing.hasData).toBe(false);
    expect(views.routing.runId).toBe('run-abc');
    expect(views.plan.hasData).toBe(true);
    expect(views.activity.hasData).toBe(true);
  });

  it('reports no run at all when nothing is running', () => {
    const views = paneViews(idleState());
    for (const [name, pane] of Object.entries(views)) {
      expect(pane.runId, name).toBeNull();
    }
  });

  it('follows the run it is given, not one captured earlier', () => {
    expect(paneViews(runningState('run-1')).plan.runId).toBe('run-1');
    expect(paneViews(runningState('run-2')).plan.runId).toBe('run-2');
  });
});

/**
 * The structural half.
 *
 * `paneViews` can only keep the panes in agreement if the page hands it one
 * state. A page that called `useRun()` for itself would have a second state
 * and the tests above would still pass, which is exactly how the defect
 * survived: every pane read `state`, and `state` was simply the wrong one.
 */
describe('no page constructs a private run instance', () => {
  const PAGES = ['src/pages/SIHDashboard.tsx', 'src/pages/Demo.tsx', 'src/pages/Tasks.tsx'];

  it('every run surface reads the shared active run', () => {
    for (const page of PAGES) {
      const source = readFileSync(page, 'utf8');
      expect(source, `${page} does not read the shared run`).toContain('useActiveRun()');
    }
  });

  it('no run surface calls useRun() for itself', () => {
    for (const page of PAGES) {
      // Comment lines are dropped first: these files explain the defect they
      // used to have, and a check that failed on its own explanation is a
      // check nobody can keep.
      const code = readFileSync(page, 'utf8')
        .split('\n')
        .filter((line) => {
          const trimmed = line.trim();
          return !trimmed.startsWith('//') && !trimmed.startsWith('*');
        })
        .join('\n');
      // `useRun()` with no arguments is the hook being instantiated. Helpers
      // exported from the same module — `isBusy`, `hasSummary` — are fine and
      // are not matched.
      expect(code, `${page} constructs its own run state`).not.toMatch(/\buseRun\(\)/);
    }
  });

  it('the chat surface hands its run to the shared source', () => {
    // The chat issues its own run id before the backend replies, so the
    // provider cannot learn about a chat-launched run on its own.
    const source = readFileSync('src/components/chat/ChatSurface.tsx', 'utf8');
    expect(source).toContain('useShareActiveRun');
    expect(source).toContain('useActiveRun');
  });
});

/**
 * What the security badges are allowed to say.
 *
 * ## The defect
 *
 * They were JSX string literals: "Sovereign", "Audit intact", "Work mode",
 * "Zero egress", "Audit chain: intact". They said the same thing on a machine
 * with a broken audit chain, on a machine in provisioning mode with the network
 * open, and on a machine where nobody had checked. This repository ships
 * evidence to judges; a badge reading "Audit chain: intact" without having
 * verified one is the worst thing in it.
 *
 * The rule every test below enforces: **no positive claim without supporting
 * state.**
 */
describe('security claims: nothing positive without evidence for it', () => {
  it('claims nothing while a probe has not answered', () => {
    for (const claim of [
      sovereigntyClaim(undefined),
      egressClaim(undefined, 60),
      auditChainClaim(undefined),
    ]) {
      expect(claim.level).toBe('loading');
      expect(isPositiveClaim(claim)).toBe(false);
    }
  });

  it('claims nothing when a probe answered that it could not tell', () => {
    // Distinct from loading. A spinner that never resolves and a probe that
    // failed are different things to the person looking at them.
    for (const claim of [
      sovereigntyClaim(null),
      egressClaim(null, 60),
      auditChainClaim(null),
    ]) {
      expect(claim.level).toBe('unknown');
      expect(isPositiveClaim(claim)).toBe(false);
      expect(claim.label.toLowerCase()).toMatch(/unknown|unchecked/);
    }
  });

  it('does not call provisioning mode sovereign', () => {
    // The moment the old badge was most wrong: the network is deliberately
    // open, and it still read "Sovereign".
    const claim = sovereigntyClaim('provisioning');
    expect(isPositiveClaim(claim)).toBe(false);
    expect(claim.level).toBe('degraded');
    expect(claim.label.toLowerCase()).toContain('network open');
  });

  it('calls work mode sovereign, because it is', () => {
    const claim = sovereigntyClaim('work');
    expect(claim.level).toBe('verified');
    expect(claim.label).toContain('Work mode');
  });

  it('states an interval and a count for zero egress, never a bare assertion', () => {
    const claim = egressClaim(
      [
        { permitted: false, host: 'api.example.com', reason: 'work mode', at: 'x' },
        { permitted: false, host: 'cdn.example.com', reason: 'work mode', at: 'x' },
      ] as never,
      60,
    );
    expect(claim.level).toBe('verified');
    // A denominator. "Zero egress" alone is equally true of a machine whose
    // broker is not running.
    expect(claim.label).toContain('2 checked');
    expect(claim.label).toContain('60 min');
  });

  it('does not claim zero egress when something was permitted', () => {
    const claim = egressClaim(
      [{ permitted: true, host: 'huggingface.co', reason: 'provisioning', at: 'x' }] as never,
      60,
    );
    expect(isPositiveClaim(claim)).toBe(false);
    expect(claim.level).toBe('degraded');
    expect(claim.label).toContain('1 permitted');
  });

  it('reports a broken audit chain as failed, and says where', () => {
    const claim = auditChainClaim({
      entriesChecked: 120,
      intact: false,
      firstBrokenSeq: 87,
      detail: 'the seal at 87 did not recompute',
    });
    expect(claim.level).toBe('failed');
    expect(isPositiveClaim(claim)).toBe(false);
    expect(claim.label).toContain('87');
  });

  it('does not call an empty audit chain intact', () => {
    // Zero rows verify vacuously. Presenting that as "intact" is how a fresh
    // installation would claim the strongest property in the product.
    const claim = auditChainClaim({
      entriesChecked: 0,
      intact: true,
      firstBrokenSeq: null,
      detail: 'nothing to check',
    });
    expect(isPositiveClaim(claim)).toBe(false);
    expect(claim.level).toBe('unknown');
  });

  it('calls a verified chain intact, with the number of entries behind it', () => {
    const claim = auditChainClaim({
      entriesChecked: 120,
      intact: true,
      firstBrokenSeq: null,
      detail: 'all seals agree',
    });
    expect(claim.level).toBe('verified');
    expect(claim.label).toContain('120 entries');
  });

  it('gives every claim a detail explaining what it rests on', () => {
    const claims = [
      sovereigntyClaim(undefined),
      sovereigntyClaim(null),
      sovereigntyClaim('work'),
      sovereigntyClaim('provisioning'),
      egressClaim(undefined, 60),
      egressClaim(null, 60),
      egressClaim([], 60),
      auditChainClaim(undefined),
      auditChainClaim(null),
    ];
    for (const claim of claims) {
      expect(claim.detail.length, claim.label).toBeGreaterThan(0);
    }
  });

  it('no page renders a hard-coded security assertion any more', () => {
    // The structural half. The literals that used to be there would pass every
    // test above, because they never went through this module at all.
    const source = readFileSync('src/pages/SIHDashboard.tsx', 'utf8')
      .split('\n')
      .filter((line) => {
        const trimmed = line.trim();
        return !trimmed.startsWith('//') && !trimmed.startsWith('*');
      })
      .join('\n');
    expect(source).not.toContain('Audit chain: intact');
    expect(source).not.toContain('>Zero egress');
    expect(source).toContain('ClaimBadge');
  });
});

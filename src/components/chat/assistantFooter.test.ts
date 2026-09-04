/**
 * What the assistant run footer is allowed to claim.
 *
 * ## The defect
 *
 * The footer rendered a shield icon and the word **verified**, unconditionally,
 * for every turn that had a run id:
 *
 *     {message.runId && onOpenInspector && (
 *       ...
 *       <ShieldCheck size={10} />
 *       <span>verified</span>
 *
 * Nothing about that was derived from anything. A run that failed said
 * verified. A run a person stopped part way said verified. A run still
 * streaming said verified. A run the verifier never looked at said verified,
 * and so did one where it looked and found blocking problems.
 *
 * What makes it worse than an ordinary bug is that this component had already
 * been fixed once. `messageStatus` exists precisely because the status pill
 * used to do this, and its comment says so. The pill was rewritten to derive
 * from persisted state; the footer a few lines further down was left alone.
 * One component, two rules, one of them true.
 *
 * This repository ships evidence to judges. A badge reading "verified" over
 * work nothing checked is the same class of claim as `scripts/bench.py`
 * returning a hardcoded 38 tok/s — a plausible-looking answer from something
 * that never measured.
 *
 * ## The rule
 *
 * No positive claim without supporting state. `src/contexts/activeRun.test.ts`
 * already enforces it for the security badges; this enforces it for the run
 * footer.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  MESSAGE_STATUS_LABELS,
  messageStatus,
  type MessageStatusInput,
  type RunOutcomeKind,
} from '../../services/agent.service';

/** Exactly what the footer renders: the status label, lowercased. */
function footerClaim(input: MessageStatusInput): string {
  return MESSAGE_STATUS_LABELS[messageStatus(input)].toLowerCase();
}

/** A finished turn carrying an answer. */
const finished: MessageStatusInput = {
  isStreaming: false,
  contentLength: 240,
  runningTools: 0,
};

/** Every way a run can end. Listed so a new one fails this file. */
const EVERY_OUTCOME: readonly RunOutcomeKind[] = [
  'completed',
  'failed',
  'aborted',
  'lengthLimited',
  'budgetStopped',
  'policyStopped',
];

describe('the footer claims only what was recorded', () => {
  it('says verified for exactly one combination, across every ending', () => {
    // The whole defect in one table. Only a completed run whose verifier ran
    // and passed may carry the word.
    const verified: string[] = [];
    for (const outcome of EVERY_OUTCOME) {
      for (const verification of ['ready', 'needsReview', null] as const) {
        if (footerClaim({ ...finished, outcome, verification }) === 'verified') {
          verified.push(`${outcome}/${verification}`);
        }
      }
    }
    expect(verified).toEqual(['completed/ready']);
  });

  it('never says verified for a run that did not complete', () => {
    for (const outcome of EVERY_OUTCOME.filter(o => o !== 'completed')) {
      // Even when the verifier passed: a run cut off at its budget can leave a
      // fragment that verifies perfectly well, and certifying half an answer
      // is the failure this prevents.
      expect(footerClaim({ ...finished, outcome, verification: 'ready' }), outcome).not.toBe(
        'verified',
      );
    }
  });

  it('reports each ending in the words that ending deserves', () => {
    expect(footerClaim({ ...finished, outcome: 'failed' })).toBe('failed');
    for (const outcome of ['aborted', 'budgetStopped', 'policyStopped', 'lengthLimited'] as const) {
      expect(footerClaim({ ...finished, outcome }), outcome).toBe('stopped');
    }
    expect(footerClaim({ ...finished, outcome: 'completed', verification: 'needsReview' })).toBe(
      'needs review',
    );
    expect(footerClaim({ ...finished, outcome: 'completed', verification: null })).toBe(
      'unverified',
    );
    expect(
      footerClaim({ ...finished, contentLength: 0, outcome: 'completed', verification: null }),
    ).toBe('completed');
  });

  it('says nothing conclusive while the turn is still going', () => {
    // The footer renders as soon as a run id exists, which is long before the
    // run has ended. It used to say "verified" throughout.
    expect(footerClaim({ ...finished, isStreaming: true })).toBe('composing…');
    expect(
      footerClaim({ ...finished, isStreaming: true, contentLength: 0, runningTools: 2 }),
    ).toBe('using a tool…');
    expect(footerClaim({ ...finished, isStreaming: true, contentLength: 0 })).toBe('thinking');
  });

  it('claims nothing on a turn with no recorded ending at all', () => {
    // A cell rendered from a record written before the typed ending existed.
    // Absence of an outcome is not success.
    expect(footerClaim(finished)).toBe('unverified');
  });
});

describe('the footer is wired to that state, not to a literal', () => {
  /** The component with its comments removed, so prose cannot pass a check. */
  const source = readFileSync('src/components/chat/AssistantMessageCell.tsx', 'utf8')
    .split('\n')
    .filter(line => {
      const trimmed = line.trim();
      return !trimmed.startsWith('//') && !trimmed.startsWith('*') && !trimmed.startsWith('/*');
    })
    .join('\n');

  it('renders no hard-coded verdict', () => {
    // The exact regression: a claim written into the markup.
    expect(source).not.toMatch(/<span>verified<\/span>/i);
    expect(source).not.toContain('ShieldCheck');
  });

  it('derives the footer claim from the status the rest of the cell uses', () => {
    expect(source).toContain('MESSAGE_STATUS_LABELS[status]');
    expect(source).toContain('<StatusIcon state={status}');
  });

  it('draws the pill and the footer from one icon rule', () => {
    // They drifted once. A single `StatusIcon` is what stops it happening
    // again — two copies of the tick rule is how the footer kept its shield.
    expect(source).toContain('<StatusIcon state={state}');
    expect(source.match(/function StatusIcon/g) ?? []).toHaveLength(1);
  });
});

/**
 * Tests for the context meter's rows.
 *
 * The interesting cases are all orderings. A document's cost arrives from two
 * places, and which one is on screen depends on whether the OCR read finished
 * before or after the run's first model call. Both orders have to produce a row
 * a person can trust, and neither may produce a number that looks measured when
 * it is not.
 */

import { describe, expect, it } from 'vitest';

import {
  driftSummary,
  entityRows,
  firstToGo,
  hasUnmeasuredTurns,
  mergeAttachments,
} from './context-entities';
import type {
  AttachmentContextEvent,
  ContextEntity,
  ContextLedgerRecord,
} from '../../services/agent.service';

function entity(over: Partial<ContextEntity> = {}): ContextEntity {
  return {
    id: 'e1',
    section: 'transcript',
    label: 'Turn 1',
    tokens: 100,
    measurement: 'estimated',
    status: 'active',
    pinned: false,
    sequence: 0,
    ...over,
  };
}

function attachment(over: Partial<AttachmentContextEvent> = {}): AttachmentContextEvent {
  return {
    name: 'invoice.pdf',
    sha256: 'sha-abc',
    pages: 3,
    documentTokens: 2_000,
    injectedTokens: 2_000,
    strategy: 'full',
    explanation: 'The whole document was included.',
    ...over,
  };
}

function ledger(over: Partial<ContextLedgerRecord> = {}): ContextLedgerRecord {
  return {
    system: 0,
    skill: 0,
    toolSchema: 0,
    evidence: 0,
    notes: 0,
    transcript: 0,
    compaction: 0,
    reserve: 0,
    occupied: 0,
    committed: 10_000,
    window: 32_000,
    headroom: 22_000,
    ...over,
  };
}

describe('OCR finishing before the first model call', () => {
  it('shows the document with no ledger at all', () => {
    // The first turn, in full: the read is done, the decision is taken, and the
    // run has not called a model yet so there is nothing else to show. A meter
    // that waits for the ledger is blank for the whole of this window.
    const rows = entityRows(null, [attachment()]);
    expect(rows).toHaveLength(1);
    expect(rows[0].label).toBe('invoice.pdf');
    expect(rows[0].tokens).toBe(2_000);
  });

  it('does not claim the pre-call figure was measured', () => {
    // It came from a character count in the prompt composer. Marking it
    // measured would give an estimate a measurement's authority, which is the
    // exact defect the reconciliation layer exists to prevent.
    const rows = entityRows(null, [attachment()]);
    expect(rows[0].measured).toBe(false);
  });

  it('reports what the turn actually carries, not the file size', () => {
    // A chunked document's file is 40,000 tokens and its cost is 14,000. The
    // meter is about the window, so it shows the 14,000 — showing the file size
    // would overstate the window by the part that was left out.
    const rows = entityRows(null, [
      attachment({ documentTokens: 40_000, injectedTokens: 14_000, strategy: 'chunked' }),
    ]);
    expect(rows[0].tokens).toBe(14_000);
  });

  it('carries the explanation only when something was left out', () => {
    const whole = entityRows(null, [attachment({ strategy: 'full' })]);
    expect(whole[0].note).toBeUndefined();

    const partial = entityRows(null, [
      attachment({ strategy: 'chunked', explanation: 'roughly 35% of it' }),
    ]);
    expect(partial[0].note).toContain('35%');
  });
});

describe('OCR finishing after the run has a ledger', () => {
  it('prefers the runtime measurement over the pre-call estimate', () => {
    // Once the runtime has seen the document its figure is the better one, and
    // the row must move to it. Keeping the estimate would pin the row to a
    // character count for the life of the run.
    const rows = entityRows(
      ledger({
        entities: [
          entity({
            id: 'sha-abc',
            section: 'evidence',
            label: 'invoice.pdf',
            tokens: 2_450,
            measurement: 'provider',
          }),
        ],
      }),
      [attachment({ injectedTokens: 2_000 })],
    );
    const row = rows.find(r => r.id === 'sha-abc');
    expect(row?.tokens).toBe(2_450);
    expect(row?.measured).toBe(true);
  });

  it('does not draw the same document twice', () => {
    // The two sources describe one file. Keyed by content hash so they converge
    // rather than accumulate.
    const rows = entityRows(
      ledger({ entities: [entity({ id: 'sha-abc', section: 'evidence', tokens: 2_450 })] }),
      [attachment({ sha256: 'sha-abc' })],
    );
    expect(rows.filter(r => r.id === 'sha-abc')).toHaveLength(1);
  });

  it('shows a document still being read as unsized rather than free', () => {
    // `pending` with a zero would draw an empty bar next to a file about to
    // cost thousands. Null renders as "reading", which is the truth.
    const rows = entityRows(
      ledger({
        entities: [
          entity({
            id: 'sha-x',
            section: 'evidence',
            label: 'scan.pdf',
            status: 'pending',
            tokens: 0,
          }),
        ],
      }),
    );
    expect(rows[0].tokens).toBeNull();
    expect(rows[0].status).toBe('pending');
  });
});

describe('rows', () => {
  it('puts the largest first, because that is the question being asked', () => {
    const rows = entityRows(
      ledger({
        entities: [
          entity({ id: 'small', tokens: 100 }),
          entity({ id: 'big', tokens: 9_000 }),
          entity({ id: 'mid', tokens: 500 }),
        ],
      }),
    );
    expect(rows.map(r => r.id)).toEqual(['big', 'mid', 'small']);
  });

  it('gives a zero share rather than NaN when nothing is committed', () => {
    const rows = entityRows(ledger({ committed: 0, entities: [entity({ tokens: 0 })] }));
    expect(rows[0].share).toBe(0);
    expect(Number.isNaN(rows[0].share)).toBe(false);
  });

  it('sums to the committed total across every row', () => {
    const rows = entityRows(
      ledger({
        committed: 1_000,
        entities: [entity({ id: 'a', tokens: 250 }), entity({ id: 'b', tokens: 750 })],
      }),
    );
    expect(rows.reduce((sum, r) => sum + r.share, 0)).toBeCloseTo(1);
  });
});

describe('what goes first', () => {
  it('names retrievable evidence ahead of conversation', () => {
    const rows = entityRows(
      ledger({
        entities: [
          entity({ id: 'turn', section: 'transcript', tokens: 5_000 }),
          entity({ id: 'doc', section: 'evidence', tokens: 900 }),
        ],
      }),
    );
    // The larger row is the transcript, and it is still not the answer: the
    // compactor clears retrievable evidence first, and a meter that named the
    // transcript would send the person to move the wrong thing.
    expect(firstToGo(rows)?.id).toBe('doc');
  });

  it('never names the system prompt or the reserve', () => {
    const rows = entityRows(
      ledger({
        entities: [
          entity({ id: 'sys', section: 'system', tokens: 9_000 }),
          entity({ id: 'reserve', section: 'reserve', tokens: 4_000 }),
          entity({ id: 'turn', section: 'transcript', tokens: 100 }),
        ],
      }),
    );
    expect(firstToGo(rows)?.id).toBe('turn');
  });

  it('skips a pinned document', () => {
    const rows = entityRows(
      ledger({
        entities: [
          entity({ id: 'pinned', section: 'evidence', tokens: 9_000, pinned: true }),
          entity({ id: 'turn', section: 'transcript', tokens: 100 }),
        ],
      }),
    );
    expect(firstToGo(rows)?.id).toBe('turn');
  });

  it('reports nothing reclaimable when everything is protected', () => {
    // The next turn fails outright rather than degrading. Worth saying plainly.
    const rows = entityRows(
      ledger({
        entities: [
          entity({ id: 'sys', section: 'system', tokens: 9_000 }),
          entity({ id: 'doc', section: 'evidence', tokens: 900, pinned: true }),
        ],
      }),
    );
    expect(firstToGo(rows)).toBeNull();
  });
});

describe('drift', () => {
  it('says nothing when no call reported usage', () => {
    // Not "drift unknown". A hedge occupies the space a real figure would.
    expect(driftSummary(ledger())).toBeNull();
    expect(
      driftSummary(
        ledger({
          reconciliations: [
            {
              turn: 1,
              at: '2026-09-02T00:00:00Z',
              estimatedIn: 100,
              actualIn: null,
              actualOut: null,
              driftRatio: null,
            },
          ],
        }),
      ),
    ).toBeNull();
  });

  it('names the direction of a meaningful drift', () => {
    const low = driftSummary(
      ledger({
        reconciliations: [
          {
            turn: 1,
            at: '2026-09-02T00:00:00Z',
            estimatedIn: 100,
            actualIn: 130,
            actualOut: 10,
            driftRatio: 1.3,
          },
        ],
      }),
    );
    expect(low).toContain('30% low');
  });

  it('says so when estimates have been matching', () => {
    const matched = driftSummary(
      ledger({
        reconciliations: [
          {
            turn: 1,
            at: '2026-09-02T00:00:00Z',
            estimatedIn: 100,
            actualIn: 101,
            actualOut: 10,
            driftRatio: 1.01,
          },
        ],
      }),
    );
    expect(matched).toContain('matched');
  });

  it('flags a run whose totals were never confirmed', () => {
    // A local server that sends no usage leaves the meter on estimates alone.
    // That is a different number to trust, and the screen should say so.
    expect(
      hasUnmeasuredTurns(
        ledger({
          reconciliations: [
            {
              turn: 1,
              at: '2026-09-02T00:00:00Z',
              estimatedIn: 100,
              actualIn: null,
              actualOut: null,
              driftRatio: null,
            },
          ],
        }),
      ),
    ).toBe(true);
  });
});

describe('merging', () => {
  it('keeps ledger rows that have no attachment', () => {
    const rows = mergeAttachments([entity({ id: 'turn' })], []);
    expect(rows.map(r => r.id)).toEqual(['turn']);
  });

  it('keeps attachments that have no ledger row', () => {
    const rows = mergeAttachments([], [attachment()]);
    expect(rows.map(r => r.id)).toEqual(['sha-abc']);
  });
});

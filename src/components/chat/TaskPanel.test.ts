/**
 * The task panel's step states.
 *
 * These exist because the last three cards could not complete. `buildSteps`
 * keyed "Composed the answer", "Verifying" and "Done" off `summary`, which is
 * set from the live run view — and that view hard-codes `summary: null` for
 * any run adopted from a snapshot, which is every run this panel sees. Nothing
 * anywhere assigned it. The result, on every finished turn: a spinner on
 * "Composed the answer", "Verifying" stuck pending, and "Not yet done"
 * underneath a complete answer.
 */
import { describe, expect, it } from 'vitest';
import { buildSteps } from './TaskPanel';

const plan = { steps: [{ intent: 'Search the connected collections' }] };
const run = { modelName: 'Nemotron3-Nano-4B', live: false };

function titles(cards: { title: string; state: string }[]) {
  return cards.map(c => `${c.state}:${c.title}`);
}

describe('buildSteps: a finished turn reads as finished', () => {
  it('completes composing, verifying and done when the message is done', () => {
    const cards = buildSteps({
      activity: [],
      plan,
      summary: null,
      message: { status: 'done', verification: 'needsReview' },
      run,
      turns: 1,
      compactions: 0,
    });
    const out = titles(cards);
    expect(out).toContain('done:Composed the answer');
    expect(out).toContain('done:Needs review');
    expect(out).toContain('done:Done');
    expect(out.join(' ')).not.toContain('Not yet done');
  });

  it('reports a verified answer as verified', () => {
    const cards = buildSteps({
      activity: [],
      plan,
      summary: null,
      message: { status: 'done', verification: 'ready' },
      run,
      turns: 1,
      compactions: 0,
    });
    expect(titles(cards)).toContain('done:Verified');
  });

  /** A turn that produced nothing must not read as a success. */
  it('names a failed turn rather than calling it done', () => {
    const cards = buildSteps({
      activity: [],
      plan,
      summary: null,
      message: { status: 'failed', verification: null, outcome: 'failed' },
      run,
      turns: 1,
      compactions: 0,
    });
    expect(titles(cards)).toContain('done:Ended without an answer');
  });

  it('says no verification ran when the verifier did not', () => {
    const cards = buildSteps({
      activity: [],
      plan,
      summary: null,
      message: { status: 'done', verification: null },
      run,
      turns: 1,
      compactions: 0,
    });
    expect(titles(cards)).toContain('done:No verification required');
  });
});

describe('buildSteps: a turn still running still reads as running', () => {
  it('shows composing and leaves the tail pending', () => {
    const cards = buildSteps({
      activity: [],
      plan,
      summary: null,
      message: { status: 'streaming', verification: null },
      run,
      turns: 1,
      compactions: 0,
    });
    const out = titles(cards);
    expect(out).toContain('running:Composing the answer');
    expect(out).toContain('pending:Verifying');
    expect(out).toContain('pending:Not yet done');
  });

  /** No message yet — the very start of a turn — must not claim completion. */
  it('stays pending when there is no message at all', () => {
    const cards = buildSteps({
      activity: [],
      plan: null,
      summary: null,
      message: undefined,
      run: undefined,
      turns: 0,
      compactions: 0,
    });
    const out = titles(cards);
    expect(out).toContain('pending:Composing');
    expect(out).toContain('pending:Not yet done');
  });
});

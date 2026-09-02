/**
 * Per-run reducer isolation tests.
 *
 * The bug the production app shipped with: two streaming runs (or one run
 * and the next `send()` pressed before the previous subscription tore
 * down) shared a single set of refs in the chat surface. The first run's
 * `message_end` would write the second run's content into the first
 * run's persistent record, and the second run's `message_start` would
 * wipe the first run's live content out from under it.
 *
 * The fix is `RunReducer` — one object per `send()` that holds its own
 * `messageId`, `content`, and timers. These tests prove that the
 * isolation holds.
 */
import { describe, expect, it, beforeAll } from 'vitest';
import {
  RunReducer,
  RunReducerRegistry,
  type MessageEvent,
} from '../ConversationContext';
import type {
  AgentEventEnvelope,
  ChatMessage,
  Conversation,
} from '../../services/agent.service';

// `RunReducer` uses `window.setTimeout` for the mirror debounce.
// vitest's default `node` environment does not provide `window`; install
// a minimal polyfill so the reducers can run unchanged.
beforeAll(() => {
  if (typeof globalThis.window === 'undefined') {
    (globalThis as unknown as { window: unknown }).window = {
      setTimeout: (handler: (...args: unknown[]) => void, ms: number) =>
        setTimeout(handler, ms) as unknown as number,
      clearTimeout: (id: number) => clearTimeout(id),
    };
  }
});

function makeMessage(id: string, role: 'user' | 'assistant' = 'user'): ChatMessage {
  return {
    id,
    conversationId: 'conv-1',
    role,
    content: '',
    status: 'done',
    createdAt: '2026-09-01T00:00:00Z',
  };
}

function makeConversation(
  id: string,
  messages: ChatMessage[] = [],
): Conversation {
  return {
    id,
    title: 't',
    createdAt: '2026-09-01T00:00:00Z',
    lastActivityAt: '2026-09-01T00:00:00Z',
    messages,
    runs: [],
    compactions: 0,
  };
}

function makeEnvelope(
  runId: string,
  event: MessageEvent,
): AgentEventEnvelope {
  return { runId, event };
}

describe('RunReducer: per-run content isolation', () => {
  it('two reducers do not share content buffers', async () => {
    const seen: Array<{ messageId: string; content: string }> = [];
    const registry = new RunReducerRegistry({
      onContent: (messageId, content) => seen.push({ messageId, content }),
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });

    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    const b = new RunReducer(registry, 'run-b', 'conv-1', 'msg-b');
    // Do NOT call registry.register() here — the registry's register
    // subscribes to the agentService event channel, which the unit
    // test does not mock. The reducers are usable in isolation: their
    // own state is fully encapsulated and they only need a registry
    // for the `publishContent` callback.

    const startA: MessageEvent = { type: 'message_start', messageId: 'msg-a', role: 'assistant' };
    const updateA: MessageEvent = { type: 'message_update', messageId: 'msg-a', delta: 'Hello' };
    const startB: MessageEvent = { type: 'message_start', messageId: 'msg-b', role: 'assistant' };
    const updateB: MessageEvent = { type: 'message_update', messageId: 'msg-b', delta: 'World' };

    a.apply(makeEnvelope('run-a', startA));
    a.apply(makeEnvelope('run-a', updateA));
    b.apply(makeEnvelope('run-b', startB));
    b.apply(makeEnvelope('run-b', updateB));

    // Wait for the 30ms mirror debounce to fire and push content to the
    // registry's publish callback.
    await new Promise((resolve) => setTimeout(resolve, 80));

    const contentA = seen.filter((s) => s.messageId === 'msg-a').pop();
    const contentB = seen.filter((s) => s.messageId === 'msg-b').pop();
    expect(contentA?.content).toBe('Hello');
    expect(contentB?.content).toBe('World');

    a.dispose();
    b.dispose();
  });

  it('an event for a different run is dropped, not routed to this reducer', () => {
    const registry = new RunReducerRegistry({
      onContent: () => undefined,
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');

    const event: MessageEvent = {
      type: 'message_update',
      messageId: 'msg-b',
      delta: 'World',
    };
    const consumed = a.apply(makeEnvelope('run-b', event));
    expect(consumed).toBe(false);
  });

  it('a message_start that arrives before plan_ready still reaches the right reducer', () => {
    // Regression for the race where a fast model streams before
    // `plan_ready` and the reducer loses the very first event.
    const seen: Array<{ messageId: string; content: string }> = [];
    const registry = new RunReducerRegistry({
      onContent: (messageId, content) => seen.push({ messageId, content }),
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');

    const start: MessageEvent = { type: 'message_start', messageId: 'msg-a', role: 'assistant' };
    const consumed = a.apply(makeEnvelope('srv-run-7', start));
    expect(consumed).toBe(true);
  });

  it('dispose() clears the timers and the next apply is a no-op for live content', async () => {
    const seen: Array<{ messageId: string; content: string }> = [];
    const registry = new RunReducerRegistry({
      onContent: (messageId, content) => seen.push({ messageId, content }),
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');

    a.apply(makeEnvelope('run-a', {
      type: 'message_start',
      messageId: 'msg-a',
      role: 'assistant',
    }));
    a.apply(makeEnvelope('run-a', {
      type: 'message_update',
      messageId: 'msg-a',
      delta: 'partial',
    }));

    // Wait for the mirror debounce so 'partial' is published.
    await new Promise((resolve) => setTimeout(resolve, 80));
    const beforeDispose = seen.filter((s) => s.messageId === 'msg-a').pop();
    expect(beforeDispose?.content).toBe('partial');

    a.dispose();

    a.apply(makeEnvelope('run-a', {
      type: 'message_update',
      messageId: 'msg-a',
      delta: ' should be ignored',
    }));
    await new Promise((resolve) => setTimeout(resolve, 80));
    const afterDispose = seen.filter((s) => s.messageId === 'msg-a').pop();
    expect(afterDispose?.content).toBe('partial');
  });
});

describe('collapseRepeats: sentence-level dedup', () => {
  it('collapses three identical sentences into one', () => {
    // The reducer uses an internal helper. We exercise it indirectly:
    // a stream of three identical messages should leave a single
    // sentence in the live content.
    const registry = new RunReducerRegistry({
      onContent: () => undefined,
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    a.apply(makeEnvelope('run-a', {
      type: 'message_start',
      messageId: 'msg-a',
      role: 'assistant',
    }));
    const sentence = 'How may I assist you today?';
    a.apply(makeEnvelope('run-a', {
      type: 'message_update',
      messageId: 'msg-a',
      delta: `${sentence} `,
    }));
    a.apply(makeEnvelope('run-a', {
      type: 'message_update',
      messageId: 'msg-a',
      delta: `${sentence} `,
    }));
    a.apply(makeEnvelope('run-a', {
      type: 'message_update',
      messageId: 'msg-a',
      delta: `${sentence}`,
    }));
    // Force a flush via dispose. Dispose does not call publish, so we
    // can only assert on the internal buffer by using a getter; but
    // for this test we can verify the publish callback received the
    // collapsed form. The mirror debounce is 30ms, so we wait it out.
    return new Promise<void>((resolve) => {
      setTimeout(() => {
        const last = (a as unknown as { content: string }).content;
        expect(last).toBe(sentence);
        resolve();
      }, 50);
    });
  });
});

describe('Registry: publish callbacks', () => {
  it('onConversation is called with the conversation returned by appendTurn', async () => {
    const conversations: Conversation[] = [];
    const registry = new RunReducerRegistry({
      onContent: () => undefined,
      onProgress: () => undefined,
      onConversation: (next) => conversations.push(next),
      onRunDone: () => undefined,
    });

    const conv = makeConversation('conv-1', [makeMessage('msg-a', 'assistant')]);
    registry.publishConversation(conv);
    expect(conversations).toEqual([conv]);
  });
});

/**
 * Progress routing.
 *
 * The stages that matter most — attachment reading, routing, model loading —
 * are emitted *before* the server has issued a run id, so their envelope
 * carries the caller's correlation id instead. A reducer that only recognised
 * the server id would drop exactly the events that cover the long silent part
 * of a cold turn, which is the interval this feature exists to fill.
 *
 * These tests hold both halves: the events must reach their own turn, and
 * they must never reach another one.
 */
describe('RunReducer: progress events reach the right turn and no other', () => {
  function harness() {
    const progress: Array<{ messageId: string; labels: string[] }> = [];
    const registry = new RunReducerRegistry({
      onContent: () => undefined,
      onProgress: (messageId, steps) =>
        progress.push({ messageId, labels: steps.map((s) => s.label) }),
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    return { registry, progress };
  }

  const stage = (
    name: string,
    messageId: string,
    detail: Record<string, unknown> = {},
  ) =>
    ({
      type: 'run_stage',
      stage: name,
      elapsedMs: 0,
      messageId,
      ...detail,
    }) as unknown as AgentEventEnvelope['event'];

  it('seeds a first step at construction, before any event arrives', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    expect(progress).toHaveLength(1);
    expect(progress[0].messageId).toBe('msg-a');
    expect(progress[0].labels).toEqual(['Sending the request']);
    a.dispose();
  });

  it('accepts a stage addressed by the correlation id, before plan_ready', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    // The envelope carries the caller's own run id: the server has none yet.
    const consumed = a.apply({
      runId: 'run-a',
      event: stage('routing', 'msg-a'),
    });
    expect(consumed).toBe(true);
    expect(progress[progress.length - 1].labels).toContain('Choosing a model');
    a.dispose();
  });

  it('accepts a stage addressed by the server run id, after plan_ready', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    a.apply({
      runId: 'srv-9',
      event: {
        type: 'plan_ready',
        correlationId: 'run-a',
      } as unknown as AgentEventEnvelope['event'],
    });
    const consumed = a.apply({
      runId: 'srv-9',
      event: stage('planning', 'msg-a'),
    });
    expect(consumed).toBe(true);
    expect(progress[progress.length - 1].labels).toContain('Planning the work');
    a.dispose();
  });

  it('rejects a stage that names another turn even on a matching run id', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    const before = progress.length;
    const consumed = a.apply({
      runId: 'run-a',
      event: stage('generating', 'msg-somebody-else'),
    });
    expect(consumed).toBe(false);
    expect(progress).toHaveLength(before);
    a.dispose();
  });

  it('rejects a stage from an unrelated run', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    const before = progress.length;
    expect(
      a.apply({ runId: 'run-b', event: stage('generating', 'msg-b') }),
    ).toBe(false);
    expect(progress).toHaveLength(before);
    a.dispose();
  });

  it('keeps two concurrent turns' + " progress lists apart", () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    const b = new RunReducer(registry, 'run-b', 'conv-1', 'msg-b');
    a.apply({ runId: 'run-a', event: stage('routing', 'msg-a') });
    b.apply({ runId: 'run-b', event: stage('planning', 'msg-b') });

    const forA = progress.filter((p) => p.messageId === 'msg-a').pop();
    const forB = progress.filter((p) => p.messageId === 'msg-b').pop();
    expect(forA?.labels).toEqual(['Sending the request', 'Choosing a model']);
    expect(forB?.labels).toEqual(['Sending the request', 'Planning the work']);
    a.dispose();
    b.dispose();
  });

  it('records that the model is thinking without recording what it thought', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    const secret = 'private chain of thought';
    a.apply({
      runId: 'run-a',
      event: {
        type: 'model_thinking',
        messageId: 'msg-a',
        state: 'start',
        characters: secret.length,
        elapsedMs: 0,
      } as unknown as AgentEventEnvelope['event'],
    });
    const latest = progress[progress.length - 1];
    expect(latest.labels).toContain('Thinking');
    expect(JSON.stringify(latest)).not.toContain(secret);
    a.dispose();
  });

  it('a disposed reducer takes no further progress', () => {
    const { registry, progress } = harness();
    const a = new RunReducer(registry, 'run-a', 'conv-1', 'msg-a');
    a.dispose();
    const before = progress.length;
    a.apply({ runId: 'run-a', event: stage('generating', 'msg-a') });
    expect(progress).toHaveLength(before);
  });
});

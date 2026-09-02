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
      onConversation: (next) => conversations.push(next),
      onRunDone: () => undefined,
    });

    const conv = makeConversation('conv-1', [makeMessage('msg-a', 'assistant')]);
    registry.publishConversation(conv);
    expect(conversations).toEqual([conv]);
  });
});

/**
 * Verify `ConversationContext` exposes the `useConversation` hook and
 * the `ConversationProvider` component, and that the `RunReducer` and
 * `RunReducerRegistry` classes are exported for the rest of the app.
 *
 * The actual shared-state contract is exercised by the `useConversation`
 * consumers in the chat surface; the reducer isolation contract is
 * covered by `RunReducer.test.ts`.
 */
import { describe, expect, it } from 'vitest';
import * as ctx from '../ConversationContext';

describe('ConversationContext: public surface', () => {
  it('exports useConversation, ConversationProvider, and the reducer classes', () => {
    expect(typeof ctx.useConversation).toBe('function');
    expect(typeof ctx.ConversationProvider).toBe('function');
    expect(typeof ctx.RunReducer).toBe('function');
    expect(typeof ctx.RunReducerRegistry).toBe('function');
  });

  it('exports the message-event type guard', () => {
    expect(typeof ctx.isMessageEvent).toBe('function');
    // A non-message event is not a message event. Cast through unknown
    // because the guard's job is precisely to narrow an `AgentEvent`
    // union member; tests cover both branches.
    const notMessage = { type: 'plan_ready' } as unknown as Parameters<
      typeof ctx.isMessageEvent
    >[0];
    expect(ctx.isMessageEvent(notMessage)).toBe(false);
    const isMessage = {
      type: 'message_start',
      messageId: 'm',
      role: 'assistant',
    } as unknown as Parameters<typeof ctx.isMessageEvent>[0];
    expect(ctx.isMessageEvent(isMessage)).toBe(true);
  });
});

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  agentService,
  type AgentEvent,
  type AgentEventEnvelope,
  type ChatMessage,
  type Classification,
  type Conversation,
  type RunSummary,
} from '../../services/agent.service';

/**
 * Driving one chat conversation.
 *
 * The chat surface owns a `Conversation` and a list of `Message`s. Each user
 * submission is a turn: the front-end calls `send(prompt)`, which (a) creates
 * a fresh assistant cell in `Streaming` state, (b) calls `agent_start_run`
 * with the right `conversationId`/`messageId`, and (c) subscribes to the run's
 * events on `agent://event` to fill the cell token-by-token.
 *
 * The hook keeps three independent things in sync:
 *
 *  - **The on-disk `Conversation`**, re-read after every mutation. The disk
 *    is the source of truth: a remount that lands here reads from it, not
 *    from in-memory state, and the durable record is what the audit chain
 *    reads.
 *  - **The live `streamingContent` for the active assistant cell**, held in a
 *    ref so rapid `message_update` deltas do not thrash React state. Written
 *    through `updateStreamingContent` on a debounce so the durable record
 *    catches up without a per-token file write.
 *  - **The `Activity` list for the active run**, lifted from the existing
 *    `useRun` infrastructure. The chat surface does not own the activity
 *    log; the run does, and the hook subscribes to it through
 *    `agentService.subscribe`.
 *
 * Multiple conversations can be opened in sequence; only the active one is
 * watched live. The `conversationId` is persisted in `sessionStorage` the
 * way `useRun` persists `LAST_RUN_KEY`, so a reload reattaches to the
 * conversation the user was last in.
 *
 * The hook deliberately does not own the routing preview or the
 * per-message-cell timing — those are concerns of the components above it.
 */

const LAST_CONVERSATION_KEY = 'arjun.conversation.last';

function rememberConversation(id: string | null) {
  try {
    if (id) sessionStorage.setItem(LAST_CONVERSATION_KEY, id);
    else sessionStorage.removeItem(LAST_CONVERSATION_KEY);
  } catch {
    // A browser with storage disabled loses reattachment across a reload and
    // nothing else. Not worth failing a chat over.
  }
}

function lastConversation(): string | null {
  try {
    return sessionStorage.getItem(LAST_CONVERSATION_KEY);
  } catch {
    return null;
  }
}

function newMessageId(role: 'user' | 'assistant'): string {
  const prefix = role === 'user' ? 'u' : 'a';
  return `${prefix}-${crypto.randomUUID()}`;
}

const STREAMING_PERSIST_DEBOUNCE_MS = 400;

/**
 * The hook's surface. Designed to be the only thing the chat components
 * need to import.
 */
export interface UseConversation {
  /** The active conversation, or `null` while the list is loading. */
  conversation: Conversation | null;
  /** Every conversation on disk, newest first. Used by the sidebar. */
  conversations: Conversation[];
  /** True while a turn is being submitted or streamed. */
  isStreaming: boolean;
  /** The id of the assistant message currently being streamed, if any. */
  activeMessageId: string | null;
  /** The id of the run currently being streamed, if any. */
  activeRunId: string | null;
  /** Live streaming content of the active message, kept here for components
   *  that want the latest text without subscribing to state directly. */
  streamingContent: string;
  /**
   * Send a new user message in the current conversation. If there is no
   * active conversation, one is created with a default welcome.
   */
  send: (prompt: string, classification?: Classification) => Promise<void>;
  /** Switch to a different conversation. */
  open: (conversationId: string) => Promise<void>;
  /** Create a fresh conversation and switch to it. */
  newConversation: () => Promise<void>;
  /** Refresh the list of conversations (e.g. on remount or after a new chat). */
  refresh: () => Promise<void>;
  /**
   * Mark the active assistant message as done with the given final content.
   * Called by the chat surface on `message_end` or run completion; the
   * server may also write the final state itself when `start` resolves.
   */
  complete: (args: {
    finalContent: string;
    elapsedMs: number;
    modelName?: string;
    modelRole?: string;
    usedFallback?: boolean;
    error?: string;
    failed?: boolean;
  }) => Promise<void>;
  /** Re-emit a previous user message (Replay). Re-runs the same prompt in the
   *  same conversation, producing a new assistant message. */
  replay: (userMessage: ChatMessage) => Promise<void>;
}

export function useConversation(): UseConversation {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState('');

  // Live content kept in a ref so rapid deltas do not queue renders. The
  // React state mirrors the ref at a debounce, which is what the chat
  // surface reads.
  const liveContentRef = useRef('');
  const debounceRef = useRef<number | null>(null);
  const persistDebounceRef = useRef<number | null>(null);
  // The conversation id of the run we are subscribed to. In a ref because
  // the subscriber closure is created once and must filter against the
  // current value.
  const subscribedConversationId = useRef<string | null>(null);
  const subscribedMessageId = useRef<string | null>(null);
  // The run id we *expect* to see, which is the client-generated id sent
  // to `appendTurn` and as the `correlationId` on `start`. Used to lock
  // onto the real (server-generated) runId when the first `plan_ready`
  // event arrives — see `subscribeToRun` for why this indirection exists.
  const subscribedRunId = useRef<string | null>(null);
  // The actual server-issued runId, populated from the first matching
  // `plan_ready` event. Once set, every other event is filtered against
  // this and the `correlationId` match is no longer needed.
  const subscribedActualRunId = useRef<string | null>(null);
  const isStreamingRef = useRef(false);
  // BUG FIX: track start time for elapsed calculation
  const startTimeRef = useRef<number>(Date.now());

  /** Refresh the conversation list. */
  const refresh = useCallback(async () => {
    try {
      const list = await agentService.listConversations();
      setConversations(list);
    } catch {
      // A failing list call should not break the chat. The sidebar will
      // simply show what we have.
    }
  }, []);

  /** Reload the active conversation from disk. */
  const reloadActive = useCallback(async (id: string | null) => {
    if (!id) {
      setConversation(null);
      return;
    }
    try {
      const next = await agentService.getConversation(id);
      setConversation(next);
    } catch {
      setConversation(null);
    }
  }, []);

  /** Open a specific conversation by id. */
  const open = useCallback(
    async (conversationId: string) => {
      rememberConversation(conversationId);
      await reloadActive(conversationId);
      await refresh();
    },
    [reloadActive, refresh],
  );

  /**
   * Subscribe to `agent://event` for one run. Resolves to the unlisten
   * function so the caller can tear it down.
   *
   * The server generates its own `runId` for a run and does not accept
   * one from the client, so the client cannot pre-filter events by
   * `runId`. Instead it subscribes without a filter, locks onto the
   * real `runId` the first time a `plan_ready` event arrives whose
   * `correlationId` matches the id the client sent on
   * `StartRunRequest.correlationId`, and then filters against the
   * server-issued id from that point on. The same pattern is used in
   * `useRun` for the inspector.
   */
  const subscribeToRun = useCallback(
    async (
      runId: string,
      conversationId: string,
      messageId: string,
    ): Promise<() => void> => {
      subscribedConversationId.current = conversationId;
      subscribedMessageId.current = messageId;
      // `runId` here is the client-generated id. The real (server-issued)
      // id is captured the first time we see a `plan_ready` event with
      // this `correlationId`.
      subscribedRunId.current = runId;
      subscribedActualRunId.current = null;
      const unlisten = await agentService.subscribe(
        (envelope: AgentEventEnvelope) => {
          // Lock-on: the first `plan_ready` event whose `correlationId`
          // matches the id we sent tells us the server's runId. After
          // that, every event from any other run is dropped.
          if (
            subscribedActualRunId.current === null &&
            envelope.event.type === 'plan_ready' &&
            envelope.event.correlationId === runId
          ) {
            subscribedActualRunId.current = envelope.runId;
          }
          if (envelope.runId !== subscribedActualRunId.current) return;
          const event = envelope.event;
          // We only care about the assistant message stream for the cell
          // we just reserved. Other events are routed to the activity log
          // by the existing useRun path; the chat surface reads that log
          // when it renders the assistant cell's tool activity.
          if (!isMessageEvent(event)) return;
          if (event.messageId !== messageId) return;
          applyMessageEvent(event);
        },
      );
      return () => {
        unlisten();
        if (subscribedRunId.current === runId) {
          subscribedRunId.current = null;
          subscribedActualRunId.current = null;
          subscribedMessageId.current = null;
          subscribedConversationId.current = null;
        }
      };
    },
    // We intentionally do not depend on `applyMessageEvent`; the closure
    // reaches `liveContentRef` and the ref-based state setters which are
    // stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  /**
   * Collapse runs of repeated content into a single instance.
   *
   * Small models tend to fall into loops and emit the same sentence
   * two or three times in a row ("How may I assist you today?
   * How may I assist you today? How may I assist you today?"). The
   * raw stream faithfully reflects the model, but the duplication
   * makes the answer look broken in the UI.
   *
   * Two passes:
   *  1. Sentence-level: if the last N characters of the stream end
   *     with the same sentence that already appears earlier in the
   *     stream, strip the earlier copy.
   *  2. Word-level: if the last 200 chars contain an immediate
   *     repeat of a chunk (8..200 chars), strip one copy.
   *
   * Only exact matches are collapsed, so a genuine restatement of a
   * key phrase is preserved.
   */
  const collapseRepeats = (s: string): string => {
    if (s.length < 20) return s;

    // Pass 1: sentence-level repeat. The most common pattern from
    // small models is the same sentence appended two or three times
    // with no whitespace between. We look for sentences that appear
    // at the very end AND somewhere earlier, separated only by a
    // space, period, or nothing.
    const sentenceRe = /[.!?]\s+([^.!?]{8,200})[.!?]/g;
    const tail1 = s.slice(-500);
    const tailSentences: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = sentenceRe.exec(tail1)) !== null) {
      tailSentences.push(m[1]);
    }
    if (tailSentences.length > 0) {
      const last = tailSentences[tailSentences.length - 1];
      // See if the same sentence appears earlier in the body.
      const escaped = last.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const dupRe = new RegExp(`(?:^|[.!?]\\s+)${escaped}[.!?](?:\\s+${escaped}[.!?])+`, 'g');
      if (dupRe.test(s)) {
        return s.replace(dupRe, (match) => {
          // Keep only the last occurrence (the trailing one).
          const parts = match.split(new RegExp(`(?<=[.!?])\\s+(?=${escaped}[.!?])`));
          return parts[parts.length - 1] ?? match;
        });
      }
    }

    // Pass 2: immediate word-level repeat.
    const tail = s.slice(-400);
    for (let len = Math.min(200, Math.floor(tail.length / 2)); len >= 8; len -= 1) {
      const last = tail.slice(-len);
      const before = tail.slice(-len * 2, -len);
      if (last === before) {
        return s.slice(0, s.length - len);
      }
    }
    return s;
  };

  /** Apply one message-event to the live content ref + state. */
  const applyMessageEvent = useCallback(
    (event: Extract<AgentEvent, { type: 'message_start' | 'message_update' | 'message_end' }>) => {
      switch (event.type) {
        case 'message_start':
          // Reset the live content for a fresh message.
          liveContentRef.current = '';
          setStreamingContent('');
          break;
        case 'message_update': {
          // Append the new delta, then collapse any immediate repeats
          // the model just produced. The collapse only ever trims the
          // tail, so previously-rendered characters are not affected.
          const next = liveContentRef.current + event.delta;
          const collapsed = collapseRepeats(next);
          liveContentRef.current = collapsed;
          // Debounce the React state mirror so a rapid token stream does
          // not queue a render per token. The ref is the truth for
          // subscribers that need the latest content.
          if (debounceRef.current === null) {
            debounceRef.current = window.setTimeout(() => {
              setStreamingContent(liveContentRef.current);
              debounceRef.current = null;
            }, 30);
          }
          schedulePersist();
          break;
        }
        case 'message_end':
          // Flush any pending state mirror and persist immediately.
          if (debounceRef.current !== null) {
            window.clearTimeout(debounceRef.current);
            debounceRef.current = null;
          }
          setStreamingContent(liveContentRef.current);
          if (persistDebounceRef.current !== null) {
            window.clearTimeout(persistDebounceRef.current);
            persistDebounceRef.current = null;
          }

          // ─── BUG FIX: mark message as done so composing stops ───
          if (subscribedConversationId.current && subscribedMessageId.current) {
            const convId = subscribedConversationId.current;
            const msgId = subscribedMessageId.current;
            const runId = subscribedActualRunId.current ?? subscribedRunId.current;
            const finalContent = liveContentRef.current;
            const elapsed = Date.now() - startTimeRef.current;

            // 1. Persist final content
            void agentService
              .updateStreamingContent(convId, msgId, finalContent)
              .then(() => reloadActive(convId))
              .catch(() => undefined);

            // 2. MARK MESSAGE AS DONE — this was the missing call
            if (runId) {
              void agentService.completeMessage({
                conversationId: convId,
                messageId: msgId,
                runId,
                finalContent,
                elapsedMs: elapsed,
                failed: false,
                tokensIn: event.tokensIn,
                tokensOut: event.tokensOut,
              }).then(() => reloadActive(convId))
                .catch(() => undefined);
            }
          }

          // 3. Reset streaming flags immediately
          isStreamingRef.current = false;
          setIsStreaming(false);
          setActiveMessageId(null);
          setActiveRunId(null);
          break;
      }
    },
    [reloadActive],
  );

  /** Schedule a debounced persist of the streaming content. */
  const schedulePersist = useCallback(() => {
    if (persistDebounceRef.current !== null) return;
    persistDebounceRef.current = window.setTimeout(() => {
      persistDebounceRef.current = null;
      if (!subscribedConversationId.current || !subscribedMessageId.current) return;
      const conversationId = subscribedConversationId.current;
      const messageId = subscribedMessageId.current;
      const content = liveContentRef.current;
      void agentService
        .updateStreamingContent(conversationId, messageId, content)
        .catch(() => undefined);
    }, STREAMING_PERSIST_DEBOUNCE_MS);
  }, []);

  /**
   * Send a new user message in the current conversation. Creates the
   * conversation on first call.
   */
  const send = useCallback(
    async (
      prompt: string,
      classification?: Classification,
      options?: { systemPrompt?: string },
    ) => {
      if (isStreamingRef.current) return;
      if (!prompt.trim()) return;

      let conv = conversation;
      if (!conv) {
        const title =
          prompt
            .split('\n')
            .map(line => line.trim())
            .find(line => line.length > 0) ?? 'New conversation';
        const created = await agentService.createConversation(title.slice(0, 80));
        conv = created;
        rememberConversation(created.id);
      }

      const runId = crypto.randomUUID();
      const messageId = newMessageId('assistant');

      // Reserve the user message and the assistant cell on the
      // conversation, and bind the run id so the runtime can route
      // streaming events to the right cell.
      const updated = await agentService.appendTurn(conv.id, runId, messageId, prompt);
      if (updated) setConversation(updated);
      await refresh();

      // BUG FIX: reset start time
      startTimeRef.current = Date.now();

      isStreamingRef.current = true;
      setIsStreaming(true);
      setActiveMessageId(messageId);
      setActiveRunId(runId);
      liveContentRef.current = '';
      setStreamingContent('');

      // Subscribe to the run's events BEFORE we start it, so no early
      // tokens are lost.
      let unsubscribe: (() => void) | undefined;
      try {
        unsubscribe = await subscribeToRun(runId, conv.id, messageId);
      } catch {
        // A failing subscription is not a reason to refuse the run; the
        // user will simply see the final answer when `start` resolves.
      }

      try {
        const summary: RunSummary | null = await agentService.start({
          prompt,
          classification,
          systemPrompt: options?.systemPrompt,
          conversationId: conv.id,
          messageId,
          // The server echoes this on the first `plan_ready` event, which
          // is how the subscriber above identifies the real runId. See
          // the comment on `subscribeToRun` for why this is needed.
          correlationId: runId,
        });
        // Final write-through: if the back-end did not stream, the cell is
        // now updated to its final state by the server; if it did, this
        // is a no-op (the cell is already `done`).
        if (summary && summary.messageId) {
          isStreamingRef.current = false;
          setIsStreaming(false);
          setActiveMessageId(null);
          setActiveRunId(null);
          await reloadActive(conv.id);
          await refresh();
        }
      } catch (error) {
        // Surface the failure to the conversation as a failed message.
        const message = error instanceof Error ? error.message : String(error);
        if (conv && messageId) {
          await agentService
            .completeMessage({
              conversationId: conv.id,
              messageId,
              runId,
              finalContent: liveContentRef.current || message,
              elapsedMs: 0,
              error: message,
              failed: true,
            })
            .catch(() => undefined);
          await reloadActive(conv.id);
        }
        isStreamingRef.current = false;
        setIsStreaming(false);
        setActiveMessageId(null);
        setActiveRunId(null);
      } finally {
        unsubscribe?.();
      }
    },
    [conversation, refresh, reloadActive, subscribeToRun],
  );

  /** Mark the active message as complete. */
  const complete = useCallback(
    async (args: {
      finalContent: string;
      elapsedMs: number;
      modelName?: string;
      modelRole?: string;
      usedFallback?: boolean;
      error?: string;
      failed?: boolean;
    }) => {
      if (!conversation || !activeMessageId || !activeRunId) return;
      await agentService.completeMessage({
        conversationId: conversation.id,
        messageId: activeMessageId,
        runId: activeRunId,
        finalContent: args.finalContent,
        elapsedMs: args.elapsedMs,
        modelName: args.modelName,
        modelRole: args.modelRole,
        usedFallback: args.usedFallback,
        error: args.error,
        failed: args.failed ?? false,
      });
      await reloadActive(conversation.id);
      await refresh();
      isStreamingRef.current = false;
      setIsStreaming(false);
      setActiveMessageId(null);
      setActiveRunId(null);
    },
    [conversation, activeMessageId, activeRunId, reloadActive, refresh],
  );

  /** Replay: re-send a previous user message as a fresh turn. */
  const replay = useCallback(
    async (userMessage: ChatMessage) => {
      if (userMessage.role !== 'user') return;
      if (!conversation) return;
      await send(userMessage.content);
    },
    [conversation, send],
  );

  /** Create a fresh conversation. */
  const newConversation = useCallback(async () => {
    const created = await agentService.createConversation('New conversation');
    rememberConversation(created.id);
    await reloadActive(created.id);
    await refresh();
  }, [refresh, reloadActive]);

  /**
   * On mount, restore the last conversation the user was in.
   *
   * Falls back to a fresh conversation if the remembered one is gone
   * (a clean install, or a profile that was reset). The new conversation
   * is created and adopted immediately so the chat surface has somewhere
   * to write the first user message.
   *
   * Also listens for `arjun:trigger-send` window events. Other parts of
   * the app (the SIH demo buttons, the Tasks replay button) dispatch
   * these to ask the chat surface to send a prompt in its current
   * conversation without having to thread the hook through props.
   */
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const remembered = lastConversation();
      if (remembered) {
        const conv = await agentService.getConversation(remembered);
        if (!cancelled) {
          if (conv) {
            setConversation(conv);
          } else {
            rememberConversation(null);
          }
        }
      }
      await refresh();
      if (cancelled) return;
      if (!conversation) {
        // No remembered conversation and no current one — create one so
        // the user lands on an empty chat rather than a placeholder.
        const created = await agentService.createConversation('New conversation');
        if (cancelled) return;
        rememberConversation(created.id);
        setConversation(created);
        await refresh();
      }
    })();

    const onTrigger = (event: Event) => {
      const detail = (event as CustomEvent<{
        prompt: string;
        title?: string;
        systemPrompt?: string;
        classification?: Classification;
      }>).detail;
      if (!detail || !detail.prompt) return;
      void (async () => {
        // A demo event can ask for a brand-new conversation by giving
        // it a title. The chat surface will create one before sending.
        if (detail.title) {
          const created = await agentService.createConversation(detail.title);
          rememberConversation(created.id);
          await reloadActive(created.id);
          await refresh();
        }
        await send(detail.prompt, detail.classification, {
          systemPrompt: detail.systemPrompt,
        });
      })();
    };
    window.addEventListener('arjun:trigger-send', onTrigger);

    return () => {
      cancelled = true;
      window.removeEventListener('arjun:trigger-send', onTrigger);
    };
    // Mount-only effect; `send` and friends are intentionally not in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    conversation,
    conversations,
    isStreaming,
    activeMessageId,
    activeRunId,
    streamingContent,
    send,
    open,
    newConversation,
    refresh,
    complete,
    replay,
  };
}

function isMessageEvent(
  event: AgentEvent,
): event is Extract<AgentEvent, { type: 'message_start' | 'message_update' | 'message_end' }> {
  return (
    event.type === 'message_start' ||
    event.type === 'message_update' ||
    event.type === 'message_end'
  );
}

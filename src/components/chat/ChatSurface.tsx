import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { useConversation } from '../run/useConversation';
import { ChatComposer } from './ChatComposer';
import { AssistantMessageCell } from './AssistantMessageCell';
import { RunView } from '../run/RunView';
import { useAdoptedRun, useContextLedger, useTaskRecord } from '../run/runAdopt';
import { ChatHeader } from './ChatHeader';
import {
  type ChatMessage,
  type Classification,
  type RunSummary,
} from '../../services/agent.service';
import type { Activity, RunViewState } from '../run/recovery';
import styles from './ChatSurface.module.css';

/**
 * The chat surface: message log + sticky composer.
 *
 * Owns no business state — all data flows through `useConversation`. The
 * per-run state used by the inspector and by the assistant cell's
 * tool-activity list is read through `useAdoptedRun`, a thin hook over
 * the same snapshot/event-replay machinery the workbench uses.
 *
 * Conversation history lives in the ARJUN dropdown in the application
 * shell, not in a sidebar inside this surface, so a chat is focused on
 * the current conversation.
 */
export interface ChatSurfaceProps {
  /** Optional: a system prompt to prepend to every user message. */
  systemPrompt?: string;
  /** Optional: a custom classification to apply to every user message. */
  classification?: Classification;
  /** Whether to show the conversation sidebar. Reserved for future
   *  layouts; the current UI does not render a sidebar. */
  showSidebar?: boolean;
}

export function ChatSurface({
  classification,
  showSidebar: _showSidebar = false,
}: ChatSurfaceProps) {
  const {
    conversation,
    isStreaming,
    activeMessageId,
    streamingContent,
    send,
    replay,
  } = useConversation();

  const [inspectorRunId, setInspectorRunId] = useState<string | null>(null);
  const adopted = useAdoptedRun(inspectorRunId);
  const taskSummary = useTaskRecord(inspectorRunId);

  // runsByMessageId: a stable map of `messageId → runId` for the
  // current conversation, used to find which run a message belongs to
  // when the user clicks "View details" or the retry button.
  const runsByMessageId = useMemo(() => {
    const map = new Map<string, string>();
    if (conversation) {
      for (const r of conversation.runs) {
        map.set(r.messageId, r.runId);
      }
    }
    return map;
  }, [conversation]);

  // The assistant cell that is currently streaming: the one matching
  // the active message id, or the run that is still live.
  const streamingMessageId = useMemo(() => {
    if (!conversation) return null;
    if (activeMessageId) return activeMessageId;
    for (const r of conversation.runs) {
      if (r.live) return r.messageId;
    }
    return null;
  }, [conversation, activeMessageId]);

  // The most recent run in the conversation. Used to feed the chat
  // header's context chip; the inspector, if open, takes precedence.
  const latestRunId = useMemo(() => {
    if (!conversation || conversation.runs.length === 0) return null;
    return conversation.runs[conversation.runs.length - 1].runId;
  }, [conversation]);
  const headerContext = useContextLedger(
    inspectorRunId ?? latestRunId,
  );

  // Auto-scroll to the bottom on new content, but only if the user has
  // not scrolled upward. The "↓ N new messages" pill lets them catch
  // up without losing their place.
  const scrollRef = useRef<HTMLDivElement>(null);
  const [unseenCount, setUnseenCount] = useState(0);
  const [stuckAtBottom, setStuckAtBottom] = useState(true);

  const messages = conversation?.messages ?? [];

  useEffect(() => {
    if (!stuckAtBottom) {
      setUnseenCount(c => c + 1);
      return;
    }
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, [messages.length, streamingContent, stuckAtBottom]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distance < 8;
    setStuckAtBottom(atBottom);
    if (atBottom) setUnseenCount(0);
  };

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setUnseenCount(0);
    setStuckAtBottom(true);
  };

  const handleSubmit = useCallback(
    async (text: string) => {
      await send(text, classification);
    },
    [send, classification],
  );

  return (
    <div className={styles.chatRoot}>
      <div className={styles.chatMain}>
        <ChatHeader
          conversation={conversation}
          ledger={headerContext.ledger}
          compactions={headerContext.compactions.length}
          lastCompaction={
            headerContext.compactions.length > 0
              ? headerContext.compactions[headerContext.compactions.length - 1]
              : null
          }
        />
        <div
          className={styles.chatScroll}
          ref={scrollRef}
          onScroll={onScroll}
        >
          {messages.length === 0 && (
            <p className={styles.emptyState}>
              Ask Arjun anything. The conversation persists across turns.
            </p>
          )}

          {messages.map(m => (
            <MessageRow
              key={m.id}
              message={m}
              isLive={m.id === streamingMessageId}
              liveContent={
                m.id === streamingMessageId ? streamingContent : undefined
              }
              runId={runsByMessageId.get(m.id) ?? null}
              activity={
                inspectorRunId &&
                runsByMessageId.get(m.id) === inspectorRunId
                  ? adopted?.activity
                  : undefined
              }
              runSummary={
                inspectorRunId && runsByMessageId.get(m.id) === inspectorRunId
                  ? taskSummary ?? null
                  : null
              }
              onOpenInspector={runId => setInspectorRunId(runId)}
              onRetry={() => {
                if (m.role === 'user') {
                  void replay(m);
                } else {
                  // The retry on a failed assistant cell re-sends the
                  // previous user message. Find the user message just
                  // before this assistant message.
                  const idx = messages.findIndex(x => x.id === m.id);
                  for (let i = idx - 1; i >= 0; i--) {
                    if (messages[i].role === 'user') {
                      void replay(messages[i]);
                      return;
                    }
                  }
                }
              }}
              composerDisabled={isStreaming}
            />
          ))}
        </div>

        {unseenCount > 0 && !stuckAtBottom && (
          <button
            type="button"
            className={styles.unseenPill}
            onClick={scrollToBottom}
            aria-label={`${unseenCount} new messages`}
          >
            <ArrowDown size={13} />
            <span>
              {unseenCount} new message{unseenCount === 1 ? '' : 's'}
            </span>
          </button>
        )}

        <div className={styles.composerWrap}>
          <ChatComposer
            disabled={isStreaming}
            disabledReason="Arjun is answering…"
            onSubmit={handleSubmit}
          />
        </div>
      </div>

      {inspectorRunId && adopted && (
        <div className={styles.inspectorOverlay} role="dialog" aria-modal="true">
          <div
            className={styles.inspectorBackdrop}
            onClick={() => setInspectorRunId(null)}
          />
          <div className={styles.inspectorPanel}>
            <div className={styles.inspectorHeader}>
              <h3 className={styles.inspectorTitle}>Run details</h3>
              <button
                type="button"
                className={styles.inspectorClose}
                onClick={() => setInspectorRunId(null)}
              >
                Close
              </button>
            </div>
            <div className={styles.inspectorBody}>
              <RunView
                state={adopted.view}
                onAbort={() => setInspectorRunId(null)}
                onNewTask={() => setInspectorRunId(null)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface MessageRowProps {
  message: ChatMessage;
  isLive?: boolean;
  liveContent?: string;
  runId: string | null;
  activity?: Activity[];
  runSummary?: RunSummary | null;
  onOpenInspector: (runId: string) => void;
  onRetry: () => void;
  composerDisabled?: boolean;
}

function MessageRow({
  message,
  isLive,
  liveContent,
  runId,
  activity,
  runSummary,
  onOpenInspector,
  onRetry,
  composerDisabled,
}: MessageRowProps) {
  if (message.role === 'user') {
    return (
      <div className={styles.userRow}>
        <div className={styles.userBubble}>
          <p className={styles.userText}>{message.content}</p>
        </div>
      </div>
    );
  }
  if (message.role === 'system') {
    return (
      <div className={styles.systemRow}>
        <p className={styles.systemText}>{message.content}</p>
      </div>
    );
  }
  return (
    <AssistantMessageCell
      message={message}
      isLive={isLive}
      liveContent={liveContent}
      activity={runId ? activity : undefined}
      runSummary={runSummary ?? null}
      onOpenInspector={runId ? () => onOpenInspector(runId) : undefined}
      onRetry={composerDisabled ? undefined : onRetry}
      composerDisabled={composerDisabled}
    />
  );
}

/* `useAdoptedRun` and `useTaskRecord` are imported from `../run/runAdopt`
 * above; the local placeholder that used to live here was removed once
 * those hooks were extracted. */

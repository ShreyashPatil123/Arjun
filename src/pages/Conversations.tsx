import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, MessageSquare, Trash2 } from 'lucide-react';
import { agentService, type Conversation } from '../services/agent.service';
import styles from './Conversations.module.css';

/**
 * A list of every conversation on this machine.
 *
 * Each row is a clickable card showing the conversation's title, the
 * first user message as a one-line preview, the most recent activity
 * time, and a one-line summary of how many turns and runs it contains.
 * Clicking a row opens the conversation in the workbench (`/`) and
 * switches the active conversation via `useConversation`'s last-id
 * memory so a reload lands here too.
 *
 * The delete button on each row is the affordance for the
 * "delete from history" action: it asks for confirmation, calls
 * `agent_delete_conversation`, and re-loads the list. A delete of the
 * conversation that is currently being read is refused (the workbench
 * is the only place that should hold the last-opened id, so the
 * sign-out flow is unaffected).
 */
export const Conversations = () => {
  const navigate = useNavigate();
  const [list, setList] = useState<Conversation[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setList(await agentService.listConversations());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setList([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const open = useCallback(
    (id: string) => {
      try {
        sessionStorage.setItem('arjun.conversation.last', id);
      } catch {
        // ignored
      }
      navigate('/');
    },
    [navigate],
  );

  const remove = useCallback(
    async (id: string, title: string) => {
      const ok = window.confirm(
        `Delete "${title || 'Untitled chat'}" from the conversation history?\n\nThis removes the on-disk file. It cannot be undone.`,
      );
      if (!ok) return;
      setBusyId(id);
      setError(null);
      try {
        await agentService.deleteConversation(id);
        // If the user just deleted the conversation they were reading,
        // clear the last-opened pointer so the workbench lands on a
        // fresh chat on next visit.
        try {
          const last = sessionStorage.getItem('arjun.conversation.last');
          if (last === id) sessionStorage.removeItem('arjun.conversation.last');
        } catch {
          // ignored
        }
        await load();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    },
    [load],
  );

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className={styles.title}>Conversations</h1>
        <p className={styles.subtitle}>
          Every chat on this machine. Open one to continue where it left off.
        </p>
      </header>

      {error && (
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </p>
      )}

      {list === null ? (
        <p className={styles.dim}>Reading the conversation log…</p>
      ) : list.length === 0 ? (
        <p className={styles.empty}>
          Nothing yet. Ask Arjun something on the workbench; the conversation
          will appear here.
        </p>
      ) : (
        <ul className={styles.list}>
          {list.map(conv => {
            const firstUser = conv.messages.find(m => m.role === 'user');
            const lastAssistant = [...conv.messages]
              .reverse()
              .find(m => m.role === 'assistant');
            const isBusy = busyId === conv.id;
            return (
              <li key={conv.id} className={styles.item}>
                <button
                  className={styles.itemBtn}
                  onClick={() => open(conv.id)}
                  disabled={isBusy}
                >
                  <span className={styles.itemTitle}>{conv.title}</span>
                  {firstUser && (
                    <span className={styles.itemPreview}>
                      {firstUser.content.slice(0, 120)}
                      {firstUser.content.length > 120 ? '…' : ''}
                    </span>
                  )}
                  <span className={styles.itemMeta}>
                    <MessageSquare size={11} />
                    <span>
                      {conv.messages.length} message
                      {conv.messages.length === 1 ? '' : 's'}
                      {' · '}
                      {conv.runs.length} run
                      {conv.runs.length === 1 ? '' : 's'}
                      {lastAssistant?.elapsedMs != null && (
                        <>
                          {' · '}
                          {formatDuration(lastAssistant.elapsedMs)}
                        </>
                      )}
                    </span>
                    <span className={styles.itemWhen}>
                      {when(conv.lastActivityAt)}
                    </span>
                  </span>
                </button>
                <button
                  className={styles.itemDelete}
                  onClick={e => {
                    e.stopPropagation();
                    void remove(conv.id, conv.title);
                  }}
                  disabled={isBusy}
                  aria-label={`Delete "${conv.title || 'Untitled chat'}"`}
                  title="Delete from history"
                >
                  <Trash2 size={14} />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

function when(iso: string): string {
  const at = new Date(iso);
  return Number.isNaN(at.getTime())
    ? iso
    : at.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

/** A span of milliseconds the way a person would say it. */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  if (ms < 60_000) {
    const seconds = ms / 1000;
    return `${seconds.toFixed(seconds < 10_000 ? 1 : 0)} s`;
  }
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export default Conversations;

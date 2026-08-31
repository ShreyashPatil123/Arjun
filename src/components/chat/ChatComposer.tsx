import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, Plus, Square, X } from 'lucide-react';
import { sovereigntyService } from '../../services/sovereignty.service';
import { agentService } from '../../services/agent.service';
import { RoutingPreview } from '../routing/RoutingPreview';
import { ContextChip } from './ContextChip';
import { useConversation } from '../run/useConversation';
import styles from './ChatSurface.module.css';

/**
 * The bottom-of-chat composer.
 *
 * Two visible layers:
 *  1. **Routing hint** — a single quiet line above the composer that
 *     shows which model will take the next message. This was previously
 *     below the composer; the redesign lifts it up so the user can see
 *     the model before they press Enter.
 *  2. **Composer** — the input row. Same Enter-to-send and auto-grow
 *     behaviour as before; the right side now has a context chip and a
 *     single send/stop button (the send button morphs into a stop
 *     button while a run is streaming).
 *
 * The composer is always present in a chat surface — the user must be
 * able to send a follow-up after every completed response, including
 * during a run that is still streaming. While a run is in flight, the
 * input itself is *not* disabled (the user can type a follow-up that
 * will queue once the run finishes); only the send button morphs into
 * a stop.
 */
const MAX_COMPOSER_HEIGHT = 220;

export interface ChatComposerProps {
  /** True while a run is in flight; the send button becomes stop. */
  streaming?: boolean;
  /** When streaming, the active run's id (used to call abort). */
  activeRunId?: string | null;
  /** Optional placeholder override. */
  placeholder?: string;
  onSubmit: (prompt: string, attachmentNames: string[]) => Promise<void> | void;
}

export function ChatComposer({
  streaming,
  activeRunId,
  placeholder,
  onSubmit,
}: ChatComposerProps) {
  const { conversation } = useConversation();
  const [prompt, setPrompt] = useState('');
  const [attachments, setAttachments] = useState<string[]>([]);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_COMPOSER_HEIGHT)}px`;
  }, []);

  useEffect(() => {
    resize();
  }, [prompt, resize]);

  // Keep the textarea at a sensible height when the chat's content
  // changes around it (e.g. a new assistant message that pushes the
  // composer down).
  useEffect(() => {
    const t = window.setTimeout(resize, 0);
    return () => window.clearTimeout(t);
  }, [conversation?.messages.length, resize]);

  const hasContent = prompt.trim().length > 0 || attachments.length > 0;
  const canSubmit = !streaming && hasContent;

  const submit = useCallback(async () => {
    if (!canSubmit) return;
    const text = prompt.trim();
    const atts = [...attachments];
    setPrompt('');
    setAttachments([]);
    setRefusal(null);
    try {
      await onSubmit(text, atts);
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    }
  }, [canSubmit, prompt, attachments, onSubmit]);

  const stop = useCallback(async () => {
    if (!activeRunId || stopping) return;
    setStopping(true);
    try {
      await agentService.abort(activeRunId);
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
    } finally {
      setStopping(false);
    }
  }, [activeRunId, stopping]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (streaming) return; // never queue a second run while one is in flight
      void submit();
    }
  };

  const onFilesPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []).map(f => f.name);
    e.target.value = '';
    if (!picked.length) return;
    try {
      await sovereigntyService.assertConfidentialAllowed('attaching a document');
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
      return;
    }
    setRefusal(null);
    setAttachments(prev => [...prev, ...picked]);
  };

  return (
    <div className={styles.composerOuter}>
      <div className={styles.composerHintRow}>
        <RoutingPreview prompt={prompt} />
      </div>

      <div className={styles.composerWrap}>
        <div className={styles.composer} data-streaming={streaming || undefined}>
          {attachments.length > 0 && (
            <ul className={styles.attachmentList}>
              {attachments.map((name, i) => (
                <li key={`${name}-${i}`} className={styles.attachmentChip}>
                  <span className={styles.attachmentName}>{name}</span>
                  <button
                    type="button"
                    className={styles.attachmentRemove}
                    aria-label={`Remove ${name}`}
                    onClick={() =>
                      setAttachments(prev => prev.filter((_, j) => j !== i))
                    }
                  >
                    <X size={12} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <textarea
            ref={textareaRef}
            className={styles.composerInput}
            placeholder={
              placeholder ??
              (streaming
                ? 'Type your follow-up — it will send when the run finishes…'
                : 'Ask Arjun — text, images, tables')
            }
            value={prompt}
            rows={1}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={onKeyDown}
            aria-label="Message"
          />

          <div className={styles.composerControls}>
            <button
              type="button"
              className={styles.iconBtn}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach a document, drawing or photograph"
            >
              <Plus size={18} />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              hidden
              onChange={onFilesPicked}
            />
            <div className={styles.composerRight}>
              <ContextChip />
              {streaming ? (
                <button
                  type="button"
                  className={styles.stopBtn}
                  onClick={() => void stop()}
                  disabled={stopping}
                  aria-label="Stop generating"
                  title="Stop generating"
                >
                  <Square size={12} />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="button"
                  className={styles.sendBtn}
                  onClick={() => void submit()}
                  disabled={!canSubmit}
                  aria-label="Send"
                  title="Send (Enter)"
                >
                  <ArrowUp size={16} />
                </button>
              )}
            </div>
          </div>

          {refusal && (
            <p className={styles.refusalLine} role="alert">
              <span>{refusal}</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

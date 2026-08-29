import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, Loader2, Plus, X } from 'lucide-react';
import { sovereigntyService } from '../../services/sovereignty.service';
import { RoutingPreview } from '../routing/RoutingPreview';
import styles from './ChatSurface.module.css';

/**
 * The bottom-of-chat composer.
 *
 * The composer is always present in a chat surface — the user must be
 * able to send a follow-up after every completed response, including one
 * in a conversation that is still streaming. While a run is in flight,
 * the composer is disabled with a one-line reason.
 *
 * Behaviour:
 *  - **Enter** sends; **Shift+Enter** inserts a newline. Matches the
 *    assistants this is meant to replace.
 *  - The textbox auto-grows up to a cap; beyond it, it scrolls.
 *  - The pre-send routing hint (which model will take this) sits
 *    underneath, below the controls. Reused from the existing
 *    `RoutingPreview` so the model choice is consistent with the
 *    workbench.
 *  - Attachments are accepted the same way the workbench accepts them:
 *    the sovereignty gateway is asked whether confidential material is
 *    allowed, and the names are held in state. The actual document is
 *    not sent up the wire here — that flows through the run.
 */
const MAX_COMPOSER_HEIGHT = 220;

export interface ChatComposerProps {
  /** Disabled while a run is in flight, with a one-line reason. */
  disabled?: boolean;
  disabledReason?: string;
  onSubmit: (prompt: string, attachmentNames: string[]) => Promise<void> | void;
  /** Optional placeholder override. */
  placeholder?: string;
}

export function ChatComposer({
  disabled,
  disabledReason,
  onSubmit,
  placeholder,
}: ChatComposerProps) {
  const [prompt, setPrompt] = useState('');
  const [attachments, setAttachments] = useState<string[]>([]);
  const [refusal, setRefusal] = useState<string | null>(null);
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

  const canSubmit =
    !disabled && (prompt.trim().length > 0 || attachments.length > 0);

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
      // Surface the error in the composer, so the user can see the
      // reason rather than losing what they typed.
      setRefusal(error instanceof Error ? error.message : String(error));
    }
  }, [canSubmit, prompt, attachments, onSubmit]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
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
    <div className={styles.composer} data-disabled={disabled || undefined}>
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
          (disabled
            ? disabledReason ?? 'Arjun is answering…'
            : 'Ask Arjun — text, images, tables')
        }
        value={prompt}
        rows={1}
        onChange={e => setPrompt(e.target.value)}
        onKeyDown={onKeyDown}
        aria-label="Message"
        disabled={disabled}
      />

      <div className={styles.composerControls}>
        <button
          type="button"
          className={styles.iconBtn}
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach a document, drawing or photograph"
          disabled={disabled}
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
          <button
            type="button"
            className={styles.sendBtn}
            onClick={() => void submit()}
            disabled={!canSubmit}
            aria-label="Send"
            title={disabled ? disabledReason : 'Send (Enter)'}
          >
            {disabled ? <Loader2 size={16} className={styles.spin} /> : <ArrowUp size={16} />}
          </button>
        </div>
      </div>

      {refusal && (
        <p className={styles.refusalLine} role="alert">
          <span>{refusal}</span>
        </p>
      )}

      <RoutingPreview prompt={prompt} />
    </div>
  );
}

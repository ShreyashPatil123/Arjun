import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Brain, ChevronRight } from 'lucide-react';
import styles from './ChatSurface.module.css';
// Defined where it is produced rather than redeclared here: the reducer owns
// the buffer, and a second copy of the shape is a second thing to keep in step.
import type { LiveReasoning } from '../../contexts/ConversationContext';

export type { LiveReasoning };

interface ReasoningStreamProps {
  reasoning?: LiveReasoning;
  /** True while the turn is still running. */
  isLive: boolean;
  /** True once the answer has begun, which is what closes the panel. */
  hasAnswer: boolean;
}

/**
 * The model's reasoning, as it is produced.
 *
 * ## Why this exists
 *
 * A reasoning model is silent for as long as it thinks, and on local hardware
 * that is not a moment. Measured here: Qwen3.5-9B at 3 tok/s spending two
 * minutes forty on one turn, of which the visible answer was the last few
 * seconds. For all of that time the surface showed a single static label.
 * There was no way to tell a model working from a model wedged — and the
 * honest signal that had been chosen instead, a character counter ticking up,
 * turned out to answer a question nobody was asking. People do not want to be
 * told that thinking is happening. They want to see it.
 *
 * ## Open, then closed
 *
 * The panel opens itself while the reasoning is the only thing happening and
 * closes itself the moment the answer starts, because from then on the answer
 * is what the person came for and the thinking is something they may want to
 * check rather than read. A click pins it either way: once somebody has
 * expressed a preference this stops moving it. That is the whole interaction,
 * and it is the rule `ThinkingTree` already follows, so two panels in one cell
 * never disagree about what a collapse means.
 *
 * ## It is not the answer, and does not look like it
 *
 * Rendered as plain monospace text, deliberately: no Markdown, no headings, no
 * code blocks. Reasoning is a draft the model is talking itself through, and
 * formatting it the way the answer is formatted invites reading it as a
 * conclusion. It is also live only — nothing here is stored, so reopening the
 * conversation later shows the answer alone.
 */
export function ReasoningStream({ reasoning, isLive, hasAnswer }: ReasoningStreamProps) {
  const text = reasoning?.text ?? '';
  const streaming = isLive && !hasAnswer;

  const [open, setOpen] = useState(true);
  /** Set once the reader has chosen, after which this stops choosing for them. */
  const [pinned, setPinned] = useState(false);
  const bodyRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (pinned) return;
    setOpen(streaming);
  }, [streaming, pinned]);

  // Follows the newest text, and only while it is already near the bottom: a
  // reader who has scrolled up to re-read something is not dragged back down
  // by the next delta.
  useLayoutEffect(() => {
    const body = bodyRef.current;
    if (!body || !open) return;
    const distanceFromBottom = body.scrollHeight - body.scrollTop - body.clientHeight;
    if (distanceFromBottom < 80) body.scrollTop = body.scrollHeight;
  }, [text, open]);

  // Nothing to show and nothing coming. A model that does not reason should
  // not leave an empty panel behind on every turn.
  if (text.length === 0) return null;

  const characters = text.length;
  const summary = streaming
    ? 'thinking…'
    : `${characters.toLocaleString()} character${characters === 1 ? '' : 's'}`;

  return (
    <section className={styles.thinkingSummary} data-live={streaming || undefined}>
      <button
        type="button"
        className={styles.thinkingSummaryHeader}
        onClick={() => {
          setPinned(true);
          setOpen(current => !current);
        }}
        aria-expanded={open}
      >
        <ChevronRight
          size={12}
          className={styles.progressChevron}
          data-open={open || undefined}
          aria-hidden="true"
        />
        <Brain size={12} aria-hidden="true" />
        <span>Thinking</span>
        <span className={styles.progressSummary}>{summary}</span>
      </button>

      {open && (
        <div className={styles.thinkingSummaryBody}>
          {reasoning?.trimmed && (
            <p className={styles.reasoningTrimmed} role="status">
              The start of this reasoning has scrolled out of the live buffer.
              What follows is the most recent part of it.
            </p>
          )}
          <pre className={styles.thinkingReasoningText} ref={bodyRef}>
            {text}
            {streaming && <span className={styles.caret} aria-hidden="true" />}
          </pre>
        </div>
      )}
    </section>
  );
}

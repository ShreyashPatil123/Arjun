import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronRight, Loader2 } from 'lucide-react';
import {
  humanMs,
  summariseProgress,
  type ProgressStep,
} from './runProgress';
import styles from './ChatSurface.module.css';

/**
 * The collapsible account of what a turn is doing, above its answer.
 *
 * ## What it is for
 *
 * The chat surface used to show a single motionless "Thinking" pill from the
 * moment enter was pressed until the first token arrived — through attachment
 * reading, routing, a cold model load and the model's own reasoning pass. On
 * a measured run that was 122 seconds of a window that looked frozen. This
 * panel is the same interval, told honestly.
 *
 * ## What it never contains
 *
 * The model's private reasoning. The thinking row is built from the
 * `model_thinking` event, which carries a duration and a character count and
 * no text; there is no code path from the reasoning stream to this component.
 * See `agent-runtime/src/run.ts`, where that stream is read and discarded.
 *
 * ## Why it collapses itself
 *
 * While there is no answer yet, the steps *are* the content and the panel is
 * open. Once the answer starts arriving the answer is the content, and a
 * ten-row timeline above it competes with the thing the person asked for — so
 * it folds to one line. A person who opens or closes it by hand owns it from
 * then on: `touched` stops the automatic rule from overriding a deliberate
 * choice.
 */
export interface RunProgressPanelProps {
  steps: ProgressStep[];
  /** Whether this turn is still running. Drives the spinner and the summary. */
  isLive: boolean;
  /** True once the answer has visible text. Triggers the automatic collapse. */
  hasAnswer: boolean;
}

/** How often the open panel re-renders to advance the live step's timer. */
const TICK_MS = 500;

export function RunProgressPanel({ steps, isLive, hasAnswer }: RunProgressPanelProps) {
  const [touched, setTouched] = useState(false);
  const [open, setOpen] = useState(true);

  // A clock, not a progress bar. It only ever reports elapsed time, which is
  // a fact; nothing here estimates what fraction of the work is done.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isLive) return;
    const id = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, [isLive]);

  const wasAnswering = useRef(false);
  useEffect(() => {
    if (hasAnswer && !wasAnswering.current) {
      wasAnswering.current = true;
      if (!touched) setOpen(false);
    }
  }, [hasAnswer, touched]);

  const summary = useMemo(
    () => summariseProgress(steps, isLive, now),
    [steps, isLive, now],
  );

  if (steps.length === 0) return null;

  const running = isLive && steps.some(step => step.endedAt === undefined);

  return (
    <div className={styles.progressPanel} data-live={running || undefined}>
      <button
        type="button"
        className={styles.progressHeader}
        aria-expanded={open}
        onClick={() => {
          setTouched(true);
          setOpen(value => !value);
        }}
      >
        <ChevronRight
          size={12}
          className={styles.progressChevron}
          data-open={open || undefined}
          aria-hidden="true"
        />
        {running ? (
          <Loader2 size={11} className={styles.spin} aria-hidden="true" />
        ) : (
          <Check size={11} className={styles.progressDoneIcon} aria-hidden="true" />
        )}
        <span className={styles.progressTitle}>Thinking</span>
        <span className={styles.progressSummary}>{summary}</span>
      </button>

      {open && (
        <ol className={styles.progressList}>
          {steps.map(step => {
            const active = step.endedAt === undefined && isLive;
            const took =
              step.endedAt !== undefined
                ? step.endedAt - step.startedAt
                : now - step.startedAt;
            return (
              <li
                key={step.id}
                className={styles.progressStep}
                data-active={active || undefined}
              >
                <span className={styles.progressDot} aria-hidden="true" />
                <span className={styles.progressLabel}>
                  {step.label}
                  {step.detail && (
                    <span className={styles.progressDetail}> · {step.detail}</span>
                  )}
                </span>
                {/* Sub-second steps get no time: "0.0s" is noise, and a
                  * rounded zero next to real durations reads as a stall. */}
                {took >= 1000 && (
                  <span className={styles.progressTime}>{humanMs(took)}</span>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

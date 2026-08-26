import React, { useEffect, useRef, useState } from 'react';
import { Cpu, AlertTriangle } from 'lucide-react';
import {
  registryService,
  ROLE_LABELS,
  type Classification,
  type RoutingDecision,
} from '../../services/registry.service';
import styles from './RoutingPreview.module.css';

interface RoutingPreviewProps {
  prompt: string;
  classification?: Classification;
}

/** Long enough that routing does not run on every keystroke. */
const DEBOUNCE_MS = 400;
/** Below this, a prompt has too little in it to classify meaningfully. */
const MIN_PROMPT_LENGTH = 12;

/**
 * Which model will handle this, decided before anything runs.
 *
 * PS 26117 asks for automatic model selection *demonstrated* across task types.
 * A choice made silently at execution time cannot be demonstrated, so the
 * decision is surfaced while the prompt is still being written — type a coding
 * question and a summarisation question, and the answer visibly changes.
 *
 * The reasons are shown verbatim from the router rather than re-worded here.
 * Two copies of an explanation drift apart, and the one in the trace is the one
 * that has to be true.
 */
export const RoutingPreview = ({ prompt, classification }: RoutingPreviewProps) => {
  const [decision, setDecision] = useState<RoutingDecision | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const requestRef = useRef(0);

  useEffect(() => {
    const trimmed = prompt.trim();
    if (trimmed.length < MIN_PROMPT_LENGTH) {
      setDecision(null);
      setProblem(null);
      return;
    }

    // Each run claims a ticket; only the newest is allowed to set state, so a
    // slow earlier request cannot overwrite a newer answer.
    const ticket = ++requestRef.current;
    const timer = window.setTimeout(async () => {
      try {
        const next = await registryService.previewRouting(trimmed, classification);
        if (requestRef.current !== ticket) return;
        setDecision(next);
        setProblem(null);
      } catch (e) {
        if (requestRef.current !== ticket) return;
        setDecision(null);
        setProblem(e instanceof Error ? e.message : String(e));
      }
    }, DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [prompt, classification]);

  if (problem) {
    return (
      <div className={styles.problem}>
        <AlertTriangle size={14} />
        <span>{problem}</span>
      </div>
    );
  }

  if (!decision) return null;

  return (
    <div className={styles.panel}>
      <button
        className={styles.summary}
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
      >
        <Cpu size={14} className={styles.icon} />
        <span className={styles.line}>
          <strong>{decision.modelName}</strong>
          <span className={styles.role}>{ROLE_LABELS[decision.role] ?? decision.role}</span>
          {decision.usedFallback && <span className={styles.fallback}>fallback</span>}
          {!decision.fullyOnGpu && <span className={styles.partial}>partly on CPU</span>}
        </span>
        <span className={styles.toggle}>{expanded ? 'Hide why' : 'Why?'}</span>
      </button>

      {expanded && (
        <ol className={styles.reasons}>
          {decision.reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ol>
      )}
    </div>
  );
};

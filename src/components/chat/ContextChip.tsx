import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronUp, Database, X } from 'lucide-react';
import { useContextLedger } from '../run/runAdopt';
import { useConversation } from '../run/useConversation';
import {
  explainLedger,
  fitted,
  ledgerRows,
} from '../run/context-ledger';
import type { CompactionRecord } from '../../services/agent.service';
import styles from './ChatSurface.module.css';

/**
 * The compact context meter that lives in the right side of the
 * composer.
 *
 * Re-uses the same shape as the legacy `ContextPanel` (a chip +
 * popover with a per-section breakdown) so a person switching between
 * workbench and chat sees the same numbers. Three states, distinguished
 * by luminance rather than colour:
 *  - **ok** (< 70% of window) — quiet grey
 *  - **tight** (70-90%) — amber
 *  - **critical** (≥ 90%) — red
 */
export function ContextChip() {
  const { conversation, activeRunId } = useConversation();
  const latestRunId = useMemo(() => {
    if (activeRunId) return activeRunId;
    if (!conversation || conversation.runs.length === 0) return null;
    return conversation.runs[conversation.runs.length - 1].runId;
  }, [activeRunId, conversation]);

  const { ledger, compactions } = useContextLedger(latestRunId);
  const [open, setOpen] = useState(false);

  const lastCompaction: CompactionRecord | null =
    compactions.length > 0 ? compactions[compactions.length - 1] : null;

  if (!ledger || ledger.committed === 0) {
    return (
      <button
        type="button"
        className={styles.contextChipIdle}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        title="Context usage"
      >
        <Database size={11} />
        <span>No context yet</span>
      </button>
    );
  }

  const pct = Math.min(
    100,
    Math.round((ledger.occupied / Math.max(1, ledger.window)) * 100),
  );
  const diagnosis = explainLedger(ledger) ?? 'Context is within budget.';
  const willFit = fitted(ledger);

  return (
    <div className={styles.contextChipWrap}>
      <button
        type="button"
        className={styles.contextChip}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        title="Context usage"
        data-state={pct >= 90 ? 'critical' : pct >= 70 ? 'tight' : 'ok'}
      >
        <Database size={11} />
        <span>
          {pct}% of{' '}
          {ledger.window > 0
            ? `${(ledger.window / 1000).toFixed(0)}K`
            : `${(ledger.committed / 1000).toFixed(0)}K`}
        </span>
        {compactions.length > 0 && (
          <span className={styles.contextCompactCount}>×{compactions.length}</span>
        )}
        {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
      </button>
      {open && (
        <div className={styles.contextCard} role="dialog" aria-label="Context breakdown">
          <div className={styles.contextCardHeader}>
            <strong>Context</strong>
            <button
              type="button"
              className={styles.contextCardClose}
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              <X size={12} />
            </button>
          </div>
          <p className={styles.contextDiagnosis}>{diagnosis}</p>
          {willFit === false && (
            <p className={styles.contextWillNotFit}>
              The next turn would not fit in this model&rsquo;s window.
            </p>
          )}
          {compactions.length > 0 && (
            <p className={styles.contextCompactionLine}>
              {compactions.length} compaction
              {compactions.length === 1 ? '' : 's'} so far
              {lastCompaction && (
                <>
                  {' '}
                  · last: {lastCompaction.tokensBefore.toLocaleString()} →{' '}
                  {lastCompaction.tokensAfter.toLocaleString()} tokens
                </>
              )}
            </p>
          )}
          <ul className={styles.contextList}>
            {ledgerRows(ledger).map(row => (
              <li
                key={row.section}
                className={
                  row.committedNotOccupied
                    ? `${styles.contextRow} ${styles.contextRowReserved}`
                    : styles.contextRow
                }
              >
                <span className={styles.contextRowLabel}>{row.label}</span>
                <span className={styles.contextRowBar}>
                  <span
                    className={styles.contextRowFill}
                    style={{ width: `${Math.round(row.share * 100)}%` }}
                  />
                </span>
                <span className={styles.contextRowTokens}>
                  {row.tokens.toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

import React, { useState } from 'react';
import { Database, X } from 'lucide-react';
import {
  explainLedger,
  fitted,
  ledgerRows,
  type LedgerSection,
} from '../run/context-ledger';
import type { CompactionRecord, ContextLedgerRecord } from '../../services/agent.service';
import styles from './ChatSurface.module.css';

/**
 * A compact summary of how full the model's context window is, and
 * which section is taking the most room. Clicking the chip expands a
 * card with the full breakdown.
 *
 * The narrative is the same one the existing `Tasks` page already
 * uses — `explainLedger()` returns a one-sentence diagnosis that names
 * the largest section and its share, which is the only number a person
 * reading the trace can act on. A full table is one click away.
 */

export interface ContextPanelProps {
  /** The most recent ledger entry, if any. */
  ledger?: ContextLedgerRecord | null;
  /** Times the run's older history was replaced by a summary. */
  compactions?: number;
  /** Last compaction, used to surface the before/after tokens. */
  lastCompaction?: CompactionRecord | null;
}

export function ContextPanel({ ledger, compactions, lastCompaction }: ContextPanelProps) {
  const [open, setOpen] = useState(false);

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

  const pct = Math.min(100, Math.round((ledger.occupied / Math.max(1, ledger.window)) * 100));
  const diagnosis = explainLedger(ledger) ?? 'Context is within budget.';
  const willFit = fitted(ledger);
  const compact = compactions ?? 0;

  return (
    <>
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
        {compact > 0 && <span className={styles.contextCompactCount}>×{compact}</span>}
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
          {compact > 0 && (
            <p className={styles.contextCompactionLine}>
              {compact} compaction{compact === 1 ? '' : 's'} so far
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
    </>
  );
}

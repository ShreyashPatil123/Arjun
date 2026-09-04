import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronUp, Database, Pin, X } from 'lucide-react';
import { useContextLedger } from '../run/runAdopt';
import { useConversation } from '../run/useConversation';
import {
  explainLedger,
  fitted,
  ledgerRows,
} from '../run/context-ledger';
import {
  driftSummary,
  entityRows,
  firstToGo,
  hasUnmeasuredTurns,
  type EntityRow,
} from '../run/context-entities';
import type { CompactionRecord } from '../../services/agent.service';
import styles from './ChatSurface.module.css';

/**
 * How a row's size reads.
 *
 * A document still being read shows "reading" rather than a number. It has a
 * real place in the next turn and an unknown size, and a zero beside it would
 * read as "this is free" — the opposite of what is about to be true.
 */
function tokenLabel(row: EntityRow): string {
  if (row.tokens === null) return 'reading…';
  // The tilde is the whole contract with the reader: it marks a figure nothing
  // has confirmed. Dropping it on a measured row and keeping it on an estimated
  // one is the only way a person can tell which totals to trust.
  return row.measured ? row.tokens.toLocaleString() : `~${row.tokens.toLocaleString()}`;
}

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

  const { ledger, compactions, attachments } = useContextLedger(latestRunId);
  const [open, setOpen] = useState(false);
  /**
   * Rows the person has protected from eviction.
   *
   * Held here rather than on the ledger because the ledger is rebuilt from the
   * runtime on every turn, and a pin has to outlive that. The set is passed
   * down into the rows below so the "what goes first" line accounts for it.
   */
  const [pinned, setPinned] = useState<ReadonlySet<string>>(new Set());

  const rows = useMemo(() => {
    const merged = entityRows(ledger, attachments);
    return merged.map(row => (pinned.has(row.id) ? { ...row, pinned: true } : row));
  }, [ledger, attachments, pinned]);

  const nextToGo = useMemo(() => firstToGo(rows), [rows]);
  const drift = useMemo(() => driftSummary(ledger), [ledger]);
  const unmeasured = useMemo(() => hasUnmeasuredTurns(ledger), [ledger]);

  const togglePin = (id: string) =>
    setPinned(current => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const lastCompaction: CompactionRecord | null =
    compactions.length > 0 ? compactions[compactions.length - 1] : null;

  // A run whose documents are still being read has no ledger yet but does have
  // rows worth showing — that is the whole of the first turn, and a meter that
  // stays idle through it is blank exactly while somebody is watching it.
  if ((!ledger || ledger.committed === 0) && rows.length === 0) {
    // Expands, because it used to only pretend to.
    //
    // This branch returned the button alone while still toggling `open`, and
    // the card that reads `open` lives past the guard below — so clicking the
    // idle chip flipped a flag that nothing rendered. It looked frozen, which
    // is worse than looking disabled: a control that does nothing when pressed
    // reads as a broken application rather than as an empty state.
    return (
      <div className={styles.contextChipWrap}>
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
        {open && (
          <div className={styles.contextCard} role="dialog" aria-label="Context breakdown">
            <div className={styles.contextCardHeader}>
              <span>Context</span>
              <button
                type="button"
                className={styles.contextCardClose}
                onClick={() => setOpen(false)}
                aria-label="Close"
              >
                <X size={13} />
              </button>
            </div>
            <p className={styles.contextEmptyNote}>
              Nothing has been measured yet. The window is itemised from the
              first model call of a turn, so this fills in once you send a
              message — and stays filled for the rest of the conversation.
            </p>
          </div>
        )}
      </div>
    );
  }

  // Past the guard the ledger may still be absent — documents can be read
  // before the run has made a model call. The chip then shows the rows it has
  // and no percentage, because a percentage of an unknown window is not a
  // number anybody can act on.
  const pct = ledger
    ? Math.min(100, Math.round((ledger.occupied / Math.max(1, ledger.window)) * 100))
    : null;
  const diagnosis = ledger ? explainLedger(ledger) : null;
  const willFit = ledger ? fitted(ledger) : null;

  return (
    <div className={styles.contextChipWrap}>
      <button
        type="button"
        className={styles.contextChip}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        title="Context usage"
        data-state={
          pct === null ? 'ok' : pct >= 90 ? 'critical' : pct >= 70 ? 'tight' : 'ok'
        }
      >
        <Database size={11} />
        <span>
          {/* Raw counts as well as the percentage: a share tells somebody how
              worried to be, and the count is what they need to compare against
              a document they are about to attach. */}
          {ledger && pct !== null ? (
            <>
              {pct}% ·{' '}
              {ledger.occupied.toLocaleString()}
              {ledger.window > 0 && ` / ${ledger.window.toLocaleString()}`}
            </>
          ) : (
            'Reading documents…'
          )}
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
          {/* Only when there is something to say. `explainLedger` returns null
              rather than a hedge, and printing "unclear" in its place would
              occupy the line a real diagnosis needs. */}
          {diagnosis && <p className={styles.contextDiagnosis}>{diagnosis}</p>}
          {willFit === false && (
            <p className={styles.contextWillNotFit}>
              The next turn would not fit in this model&rsquo;s window.
            </p>
          )}
          {/* What the compactor takes first, so the person moves the right
              thing. `null` is its own message: nothing can be reclaimed, so the
              next turn fails rather than degrades — which is the one case here
              worth interrupting somebody for. */}
          {willFit !== null &&
            (nextToGo ? (
              <p className={styles.contextCompactionLine}>
                If the window fills, <strong>{nextToGo.label}</strong> goes first.
              </p>
            ) : (
              <p className={styles.contextWillNotFit}>
                Nothing here can be reclaimed — everything is either structural or
                pinned. The next turn will fail rather than shorten.
              </p>
            ))}
          {/* The estimate-against-actual line. Absent when no call has reported
              usage, because "drift unknown" is not worth a line. */}
          {drift && <p className={styles.contextCompactionLine}>{drift}</p>}
          {unmeasured && (
            <p className={styles.contextCompactionLine}>
              Some turns reported no usage, so part of this total is estimated
              rather than confirmed.
            </p>
          )}
          {/* Loud, because it means the rows below do not explain the bar
              above. Silent in normal operation. */}
          {(ledger?.itemisationErrors?.length ?? 0) > 0 && (
            <p className={styles.contextWillNotFit}>
              These rows do not add up to the totals they describe
              ({ledger?.itemisationErrors?.map(e => e.section).join(', ')}). Treat
              the breakdown as unreliable.
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
          {/* The itemisation when there is one, the section breakdown when
              there is not. A record written before entities existed still has
              sections, and falling back to them beats an empty list under a
              full bar. */}
          <ul className={styles.contextList}>
            {rows.length > 0
              ? rows.map(row => (
                  <li
                    key={row.id}
                    className={
                      row.status === 'dropped' || row.status === 'summarised'
                        ? `${styles.contextRow} ${styles.contextRowReserved}`
                        : styles.contextRow
                    }
                    title={row.note ?? undefined}
                  >
                    <span className={styles.contextRowLabel}>
                      {row.label}
                      {/* A document that entered in part is the one thing on
                          this panel that changes how much an answer can be
                          trusted, so it is marked on the row itself and not
                          only in a tooltip. */}
                      {row.note && ' ⚠'}
                    </span>
                    <span className={styles.contextRowBar}>
                      <span
                        className={styles.contextRowFill}
                        style={{ width: `${Math.round(row.share * 100)}%` }}
                      />
                    </span>
                    <span className={styles.contextRowTokens}>{tokenLabel(row)}</span>
                    {row.evictable && (
                      <button
                        type="button"
                        className={styles.contextCardClose}
                        onClick={() => togglePin(row.id)}
                        aria-pressed={row.pinned}
                        aria-label={
                          row.pinned
                            ? `Allow ${row.label} to be dropped`
                            : `Keep ${row.label} when the window fills`
                        }
                        title={
                          row.pinned
                            ? 'Pinned — kept when the window fills'
                            : 'Keep this when the window fills'
                        }
                      >
                        <Pin
                          size={10}
                          style={{ opacity: row.pinned ? 1 : 0.35 }}
                        />
                      </button>
                    )}
                  </li>
                ))
              : ledger &&
                ledgerRows(ledger).map(row => (
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

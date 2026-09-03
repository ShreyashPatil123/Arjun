/**
 * Turning a ledger into the rows the context meter draws.
 *
 * Kept apart from the component and free of JSX for the same reason as
 * `context-ledger.ts` and `recovery.ts`: this repository tests its frontend's
 * pure modules and vendors no DOM to render a hook into. Everything that can be
 * got wrong here — which of two sources wins for one document, what a row says
 * when nothing has been counted yet — is in this file, and the component only
 * draws what it returns.
 *
 * ## The two sources, and why neither can simply win
 *
 * A document's cost reaches the screen twice, from different places and at
 * different times:
 *
 * - `attachment:context`, emitted by `agent_start_run` the moment the
 *   injection decision is taken. This is the earliest anything is known, and
 *   for the whole of the first turn it is the *only* thing known — the run has
 *   not made a model call yet, so no ledger exists.
 * - the run's own ledger, which arrives per turn and carries the document as an
 *   `evidence` entity once the runtime has seen it.
 *
 * The ledger is the better number once it exists, because it is what the
 * runtime actually measured. So the ledger wins where the two overlap, and the
 * attachment event fills the gap before it. Merging the other way round would
 * pin a document's row to the pre-call estimate for the life of the run.
 */

import type {
  AttachmentContextEvent,
  ContextEntity,
  ContextLedgerRecord,
} from '../../services/agent.service';

/** Sections the compactor will never evict. Mirrors the runtime's set. */
const IMMOVABLE = new Set(['system', 'toolSchema', 'notes', 'reserve', 'compaction']);

/**
 * One row of the itemised breakdown.
 *
 * `tokens` is `null` rather than `0` while a document is still being read. Zero
 * would draw an empty bar beside a file that is about to cost thousands of
 * tokens, which reads as "this is free" — the opposite of what is true.
 */
export interface EntityRow {
  id: string;
  label: string;
  section: string;
  tokens: number | null;
  /** Share of the committed total, 0–1. Zero when nothing is committed. */
  share: number;
  status: ContextEntity['status'];
  pinned: boolean;
  /** True when the figure is a measurement rather than a guess. */
  measured: boolean;
  /** Set for a document that did not enter the turn whole. Shown verbatim. */
  note?: string;
  /** False for rows the compactor cannot touch. */
  evictable: boolean;
}

/**
 * Folds the attachment events into the ledger's entities.
 *
 * Documents the ledger already knows about keep the ledger's figure and gain
 * the attachment's explanation, which the ledger has no field for. Documents it
 * does not yet know about become rows of their own, so a file shows its cost
 * during the very first turn, before any ledger exists.
 */
export function mergeAttachments(
  entities: readonly ContextEntity[],
  attachments: readonly AttachmentContextEvent[],
): EntityRow[] {
  const byId = new Map(entities.map(e => [e.id, e]));
  const rows: EntityRow[] = [];
  const claimed = new Set<string>();

  for (const attachment of attachments) {
    const existing = byId.get(attachment.sha256);
    claimed.add(attachment.sha256);
    const partial = attachment.strategy !== 'full';
    rows.push({
      id: attachment.sha256,
      label: attachment.name,
      section: existing?.section ?? 'evidence',
      // The ledger's number when it has one; otherwise what the composer
      // decided to inject. Never `documentTokens` — that is the size of the
      // file, not the space it takes in the window, and for a chunked document
      // the two differ by the whole point of chunking it.
      tokens: existing?.tokens ?? attachment.injectedTokens,
      share: 0,
      status: existing?.status ?? 'active',
      pinned: existing?.pinned ?? false,
      measured: existing?.measurement === 'exact' || existing?.measurement === 'provider',
      // Only when something was left out. A note on a document that went in
      // whole is noise, and noise on every row is how a real warning gets
      // skipped.
      note: partial ? attachment.explanation : undefined,
      evictable: !IMMOVABLE.has(existing?.section ?? 'evidence'),
    });
  }

  for (const entity of entities) {
    if (claimed.has(entity.id)) continue;
    rows.push({
      id: entity.id,
      label: entity.label,
      section: entity.section,
      tokens: entity.status === 'pending' ? null : entity.tokens,
      share: 0,
      status: entity.status,
      pinned: entity.pinned,
      measured: entity.measurement === 'exact' || entity.measurement === 'provider',
      evictable: !IMMOVABLE.has(entity.section),
    });
  }

  return rows;
}

/**
 * The rows for one ledger, largest first, with shares filled in.
 *
 * Sorted by size because the question a person opens this panel with is "what
 * is taking up the room", and the answer is almost always the top row. A
 * ledger-order list makes them read eight numbers to find it.
 */
export function entityRows(
  ledger: ContextLedgerRecord | null,
  attachments: readonly AttachmentContextEvent[] = [],
): EntityRow[] {
  const entities = ledger?.entities ?? [];
  // A pending attachment is worth showing even with no ledger at all — that is
  // the whole of the first turn, and a meter that stays blank until the first
  // token arrives is blank exactly when somebody is watching it.
  const rows = mergeAttachments(entities, attachments);
  const committed = ledger?.committed ?? 0;
  return rows
    .map(row => ({
      ...row,
      // Zero rather than NaN when nothing is committed: a NaN width renders as
      // a full bar, which reads as "this one row filled the window".
      share: committed > 0 && row.tokens !== null ? row.tokens / committed : 0,
    }))
    .sort((a, b) => (b.tokens ?? 0) - (a.tokens ?? 0));
}

/**
 * What the compactor would reclaim first, or `null` when nothing can be.
 *
 * `null` is the case worth interrupting somebody for: everything left is either
 * structural or pinned, so the next turn fails rather than degrades.
 */
export function firstToGo(rows: readonly EntityRow[]): EntityRow | null {
  const candidates = rows.filter(
    row => row.evictable && !row.pinned && row.status === 'active',
  );
  if (candidates.length === 0) return null;
  // Evidence before conversation, mirroring `pruneStaleToolResults` running
  // ahead of summarisation. Within that, the largest — it is the one whose
  // removal actually buys the run something.
  const evidence = candidates.filter(row => row.section === 'evidence');
  const pool = evidence.length > 0 ? evidence : candidates;
  return pool.reduce((worst, row) => ((row.tokens ?? 0) > (worst.tokens ?? 0) ? row : worst));
}

/**
 * How far the estimate has been drifting, in one sentence.
 *
 * `null` when there is nothing to say — no calls yet, or none that reported
 * usage. A line reading "drift unknown" occupies the space where a real figure
 * would have gone, and this repository's rule is to say nothing rather than
 * hedge (see `explainLedger`).
 */
export function driftSummary(ledger: ContextLedgerRecord | null): string | null {
  const records = (ledger?.reconciliations ?? []).filter(r => r.driftRatio !== null);
  if (records.length === 0) return null;

  const mean = records.reduce((sum, r) => sum + (r.driftRatio ?? 0), 0) / records.length;
  const percent = Math.round((mean - 1) * 100);
  const turns = `${records.length} turn${records.length === 1 ? '' : 's'}`;
  if (Math.abs(percent) < 5) {
    return `Estimates have matched what the model charged, across ${turns}.`;
  }
  return percent > 0
    ? `Estimates have read ${percent}% low against what the model charged, across ${turns}.`
    : `Estimates have read ${Math.abs(percent)}% high against what the model charged, across ${turns}.`;
}

/**
 * Whether any model call went unmeasured.
 *
 * Shown so a person reading a total knows whether it was confirmed. A server
 * that reports no usage leaves the meter running on estimates alone, and that
 * is a materially different number to trust than one the model agreed with.
 */
export function hasUnmeasuredTurns(ledger: ContextLedgerRecord | null): boolean {
  return (ledger?.reconciliations ?? []).some(r => r.actualIn === null);
}

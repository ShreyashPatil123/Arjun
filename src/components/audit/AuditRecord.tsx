import React, { useCallback, useEffect, useState } from 'react';
import { FileCheck2, FileWarning, ShieldQuestion } from 'lucide-react';
import {
  governanceService,
  AUDIT_KIND_LABELS,
  type AuditEntry,
  type ChainVerification,
  type MerkleVerification,
} from '../../services/governance.service';
import styles from './AuditRecord.module.css';

const formatTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

/**
 * The permanent record, and the button that proves it has not been edited.
 *
 * Verification is a deliberate action rather than something that runs on load:
 * it re-reads and re-hashes the whole chain, and a result the reader asked for
 * carries more weight than a green tick that was already on the screen when
 * they arrived.
 */
export const AuditRecord = () => {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [verification, setVerification] = useState<ChainVerification | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [merkle, setMerkle] = useState<MerkleVerification | null>(null);
  const [sealing, setSealing] = useState(false);
  /** Set when the signed-in user may not read the record — not an error. */
  const [denied, setDenied] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setEntries(await governanceService.recentEntries(200));
      setDenied(null);
    } catch (e) {
      setDenied(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const verify = async () => {
    setVerifying(true);
    try {
      setVerification(await governanceService.verifyChain());
      await load();
    } catch (e) {
      setDenied(e instanceof Error ? e.message : String(e));
    } finally {
      setVerifying(false);
    }
  };

  /**
   * A different question from {@link verify}.
   *
   * Recomputing the chain proves each row still seals its own contents and
   * position. It cannot prove nothing was *removed* — a chain re-sealed after a
   * deletion is internally consistent. The Merkle root, taken at a point in
   * time, is what catches that, so the two checks are offered separately rather
   * than folded into one reassuring tick.
   */
  const checkSeal = async () => {
    setSealing(true);
    try {
      setMerkle(await governanceService.verifyMerkle());
    } catch (e) {
      setDenied(e instanceof Error ? e.message : String(e));
    } finally {
      setSealing(false);
    }
  };

  if (denied) {
    return (
      <section className={styles.section}>
        <div className={styles.header}>
          <h2 className={styles.title}>The record</h2>
        </div>
        <p className={styles.denied}>
          <ShieldQuestion size={16} />
          <span>{denied}</span>
        </p>
      </section>
    );
  }

  return (
    <section className={styles.section}>
      <div className={styles.header}>
        <h2 className={styles.title}>The record</h2>
        <button className={styles.verifyBtn} onClick={verify} disabled={verifying}>
          <FileCheck2 size={15} />
          {verifying ? 'Checking…' : 'Verify the record'}
        </button>
        <button className={styles.verifyBtn} onClick={checkSeal} disabled={sealing}>
          <FileCheck2 size={15} />
          {sealing ? 'Checking…' : 'Check the seal'}
        </button>
      </div>

      <p className={styles.explainer}>
        Append-only in the database, and hash-chained, so each entry seals both its
        contents and its position. Verifying recomputes every seal from the first entry
        onward and names the first one that disagrees. Checking the seal asks the other
        question — whether anything has been removed since the log was last sealed, which
        a re-hashed chain on its own cannot tell you.
      </p>

      {verification && (
        <div className={verification.intact ? styles.intact : styles.broken} role="status">
          {verification.intact ? <FileCheck2 size={18} /> : <FileWarning size={18} />}
          <span>{verification.detail}</span>
        </div>
      )}

      {merkle && (
        <div className={merkle.intact ? styles.intact : styles.broken} role="status">
          {merkle.intact ? <FileCheck2 size={18} /> : <FileWarning size={18} />}
          <span>
            {merkle.detail}
            {merkle.snapshot
              ? ` Sealed at entry #${merkle.snapshot.upToSeq} on ${formatTime(
                  merkle.snapshot.takenAt,
                )}; ${merkle.eventsSinceSnapshot} entr${
                  merkle.eventsSinceSnapshot === 1 ? 'y has' : 'ies have'
                } been added since.`
              : ' No seal has been taken yet, so there is nothing to compare against — this says the log is new, not that it is sound.'}
          </span>
        </div>
      )}

      {entries.length === 0 ? (
        <p className={styles.empty}>Nothing has been recorded yet.</p>
      ) : (
        <ul className={styles.list}>
          {entries.map(entry => (
            <li
              key={entry.seq}
              className={
                verification?.firstBrokenSeq != null && entry.seq >= verification.firstBrokenSeq
                  ? styles.rowSuspect
                  : styles.row
              }
            >
              <span className={styles.seq}>#{entry.seq}</span>
              <span className={styles.kind}>{AUDIT_KIND_LABELS[entry.kind] ?? entry.kind}</span>
              <span className={styles.summary}>{entry.summary}</span>
              <span className={styles.actor}>{entry.actor}</span>
              <span className={styles.time}>{formatTime(entry.at)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};

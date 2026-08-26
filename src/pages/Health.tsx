import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, CircleCheck, CircleHelp, RefreshCw } from 'lucide-react';
import { healthService, type HealthItem, type HealthSnapshot, type Reading } from '../services/health.service';
import styles from './Health.module.css';

/**
 * How often the panel re-reads. Every source is local — a GPU query, a COUNT
 * against the on-disk index, an in-memory event log, the OS socket table — so
 * this costs nothing outside this machine, which is the point.
 */
const POLL_INTERVAL_MS = 5000;

const ICONS: Record<Reading, React.ReactNode> = {
  ok: <CircleCheck size={16} />,
  attention: <AlertTriangle size={16} />,
  unknown: <CircleHelp size={16} />,
};

/* Unknown gets its own treatment rather than borrowing "ok"'s. The whole
 * purpose of the state is that it cannot be mistaken for health. */
const STATE_CLASS: Record<Reading, string> = {
  ok: styles.stateOk,
  attention: styles.stateAttention,
  unknown: styles.stateUnknown,
};

const STATE_LABEL: Record<Reading, string> = {
  ok: 'Checked',
  attention: 'Needs attention',
  unknown: 'Could not check',
};

const Card = ({ item }: { item: HealthItem }) => (
  <article className={`${styles.card} ${STATE_CLASS[item.state]}`}>
    <header className={styles.cardHead}>
      <span className={styles.cardName}>{item.name}</span>
      <span className={styles.cardState} title={STATE_LABEL[item.state]}>
        {ICONS[item.state]}
      </span>
    </header>
    <p className={styles.cardValue}>{item.value}</p>
    <p className={styles.cardNote}>{item.note}</p>
  </article>
);

export const Health = () => {
  const [snapshot, setSnapshot] = useState<HealthSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await healthService.snapshot();
      if (!mountedRef.current) return;
      setSnapshot(next);
      setError(null);
    } catch (e) {
      if (!mountedRef.current) return;
      // A panel that silently stopped updating would be the worst outcome
      // here — it would look like steady health.
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
    };
  }, [refresh]);

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const attention = snapshot?.items.filter((i) => i.state === 'attention') ?? [];
  const unknown = snapshot?.items.filter((i) => i.state === 'unknown') ?? [];

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Health</h1>
        <p className={styles.subtitle}>
          What this machine can currently do, read from this machine. No check here contacts
          anything outside it — no version check, no licence check, no status beacon. Each of
          those would appear in ARJUN&rsquo;s own network monitor a moment after it fired.
        </p>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={18} />
          <div>
            <strong>The panel could not be read.</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      <div className={styles.summary}>
        <div className={styles.summaryMain}>
          <Activity size={18} />
          <div>
            <strong>
              {attention.length === 0 && unknown.length === 0
                ? 'Everything checked is in order'
                : [
                    attention.length > 0 && `${attention.length} needing attention`,
                    unknown.length > 0 && `${unknown.length} could not be checked`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
            </strong>
            <p>
              {snapshot
                ? `Read ${new Date(snapshot.takenAt).toLocaleTimeString()} · ${snapshot.externalCallsMade} external calls made`
                : 'Reading…'}
            </p>
          </div>
        </div>
        <button
          type="button"
          className={styles.refresh}
          onClick={() => void onRefresh()}
          disabled={refreshing}
        >
          <RefreshCw size={15} className={refreshing ? styles.spinning : undefined} />
          Refresh
        </button>
      </div>

      <div className={styles.grid}>
        {snapshot?.items.map((item) => (
          <Card key={item.name} item={item} />
        ))}
      </div>
    </div>
  );
};

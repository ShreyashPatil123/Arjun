// filepath: src/pages/ModelHealth.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CircleCheck,
  Clock,
  Cpu,
  Hourglass,
  RefreshCw,
  ShieldX,
} from 'lucide-react';
import { modelHealthService, type ModelAggregate } from '../services/model_health.service';
import styles from './Health.module.css';

/**
 * The Model Health page is read-only: it shows the per-model aggregate
 * the backend keeps in memory. Each card is one model; the bars and
 * counts are derived from the same row.
 *
 * Polled, not pushed: the backend has no need to know which page is
 * open, and the audit log is the durable record the page does not
 * duplicate.
 */
const POLL_INTERVAL_MS = 5_000;

const fmtMs = (ms: number): string => {
  if (ms < 1_000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1_000).toFixed(1)} s`;
  return `${Math.round(ms / 60_000)} min`;
};

const fmtTokens = (n: number): string => {
  if (n < 1_000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
};

const fmtTime = (iso: string | null): string => {
  if (!iso) return 'never';
  // Show as a short, local time. A reviewer wants to know "how recent",
  // not the full RFC 3339.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString();
};

const Card = ({ row }: { row: ModelAggregate }) => {
  const okRatio = row.calls === 0 ? 0 : row.ok / row.calls;
  const failures = row.refused + row.timeouts + row.oom + row.otherFailures;
  // Empty page: a model with no calls yet is shown as "no data" rather
  // than as "ok", because the lack of calls is itself a signal worth
  // surfacing.
  const state =
    row.calls === 0
      ? 'unknown'
      : okRatio < 0.5 || row.oom > 0
        ? 'attention'
        : 'ok';

  return (
    <article className={`${styles.card} ${stateClass(state)}`}>
      <header className={styles.cardHead}>
        <span className={styles.cardName}>{row.modelId}</span>
        <span className={styles.cardState} title={stateLabel(state)}>
          {stateIcon(state)}
        </span>
      </header>

      <p className={styles.cardValue}>
        {row.calls === 0
          ? 'No calls yet'
          : `${row.ok} of ${row.calls} ok (${Math.round(okRatio * 100)}%)`}
      </p>

      {row.calls > 0 && (
        <div className={styles.row}>
          <Stat icon={<Clock size={12} />} label="Avg latency" value={fmtMs(avg(row))} />
          <Stat
            icon={<Hourglass size={12} />}
            label="Max latency"
            value={fmtMs(row.maxLatencyMs)}
          />
        </div>
      )}

      {row.calls > 0 && (
        <div className={styles.row}>
          <Stat icon={<Cpu size={12} />} label="In" value={fmtTokens(row.totalTokensIn)} />
          <Stat icon={<Cpu size={12} />} label="Out" value={fmtTokens(row.totalTokensOut)} />
        </div>
      )}

      {failures > 0 && (
        <p className={styles.cardNote}>
          <ShieldX size={12} /> {failures} failure{pl(failures)} · {row.oom} OOM ·{' '}
          {row.timeouts} timeout{pl(row.timeouts)}
          {row.fallbacksUsed > 0 && ` · ${row.fallbacksUsed} on fallback`}
        </p>
      )}

      <p className={styles.cardNote}>Last call: {fmtTime(row.lastSeen)}</p>
    </article>
  );
};

const avg = (row: ModelAggregate) =>
  row.calls === 0 ? 0 : Math.round(row.totalLatencyMs / row.calls);
const pl = (n: number) => (n === 1 ? '' : 's');

const stateClass = (s: 'ok' | 'attention' | 'unknown') =>
  s === 'ok' ? styles.stateOk : s === 'attention' ? styles.stateAttention : styles.stateUnknown;

const stateLabel = (s: 'ok' | 'attention' | 'unknown') =>
  s === 'ok' ? 'Healthy' : s === 'attention' ? 'Needs attention' : 'No data yet';

const stateIcon = (s: 'ok' | 'attention' | 'unknown') =>
  s === 'ok' ? (
    <CircleCheck size={16} />
  ) : s === 'attention' ? (
    <AlertTriangle size={16} />
  ) : (
    <Hourglass size={16} />
  );

const Stat = ({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) => (
  <span className={styles.stat}>
    {icon} <span className={styles.statLabel}>{label}</span> <strong>{value}</strong>
  </span>
);

export const ModelHealth = () => {
  const [rows, setRows] = useState<ModelAggregate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await modelHealthService.snapshot();
      if (!mounted.current) return;
      setRows(next);
      setError(null);
    } catch (e) {
      if (!mounted.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, [refresh]);

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const attention = rows.filter((r) => r.calls > 0 && (r.ok / r.calls < 0.5 || r.oom > 0));

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Model Health</h1>
        <p className={styles.subtitle}>
          Per-model calls the router has made, and how each one ended. The numbers come from the
          in-memory aggregate the backend keeps; the full record is in the audit log.
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
          <Cpu size={18} />
          <div>
            <strong>
              {rows.length === 0
                ? 'No models have been called yet'
                : attention.length === 0
                  ? `${rows.length} model${pl(rows.length)} healthy`
                  : `${attention.length} of ${rows.length} need attention`}
            </strong>
            <p>Polled every {POLL_INTERVAL_MS / 1000}s · source: in-memory aggregate</p>
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
        {rows.map((row) => (
          <Card key={row.modelId} row={row} />
        ))}
      </div>
    </div>
  );
};

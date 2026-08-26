import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ShieldCheck, ShieldAlert, Radio, Play, Binoculars } from 'lucide-react';
import {
  sovereigntyService,
  type EgressEvent,
  type ObservationReport,
  type OperatingMode,
} from '../services/sovereignty.service';
import { AuditRecord } from '../components/audit/AuditRecord';
import styles from './AuditNetwork.module.css';

/** How often the monitor re-reads the broker's decision log. */
const POLL_INTERVAL_MS = 2000;

const MODE_COPY: Record<OperatingMode, { label: string; detail: string }> = {
  work: {
    label: 'Work mode',
    detail: 'Every outbound call is refused. Confidential work is permitted.',
  },
  provisioning: {
    label: 'Provisioning mode',
    detail:
      'The network is reachable for model download only. Confidential documents are refused while this mode is active.',
  },
};

const formatTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString();
};

export const AuditNetwork = () => {
  const [mode, setMode] = useState<OperatingMode | null>(null);
  const [events, setEvents] = useState<EgressEvent[]>([]);
  const [observed, setObserved] = useState<ObservationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [firingCanary, setFiringCanary] = useState(false);

  // Held in a ref so the poll callback never closes over a stale value and the
  // interval does not need tearing down on every render.
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const [nextMode, nextEvents, nextObserved] = await Promise.all([
        sovereigntyService.getMode(),
        sovereigntyService.recentEvents(),
        sovereigntyService.observeConnections(),
      ]);
      if (!mountedRef.current) return;
      setMode(nextMode);
      setEvents(nextEvents);
      setObserved(nextObserved);
      setError(null);
    } catch (e) {
      if (!mountedRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(id);
    };
  }, [refresh]);

  const fireCanary = async () => {
    setFiringCanary(true);
    try {
      await sovereigntyService.runCanary();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setFiringCanary(false);
    }
  };

  // A permitted call while in Work mode means the controls did not hold. It is
  // the one thing on this page that must be impossible to overlook.
  const breach = events.find(e => e.permitted && e.mode === 'work');
  const copy = mode ? MODE_COPY[mode] : null;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Audit &amp; Network</h1>
        <p className={styles.subtitle}>
          Every connection this machine attempted, and what the broker decided about it.
          Permitted calls are listed alongside refusals — a log of blocks alone could not
          show that nothing was sent.
        </p>
      </header>

      {breach && (
        <div className={styles.breach} role="alert">
          <ShieldAlert size={18} />
          <div>
            <strong>Egress controls did not hold.</strong>
            <p>{breach.reason}</p>
          </div>
        </div>
      )}

      <section className={`${styles.status} ${mode === 'work' ? styles.statusSealed : styles.statusOpen}`}>
        <div className={styles.statusIcon}>
          {mode === 'work' ? <ShieldCheck size={22} /> : <Radio size={22} />}
        </div>
        <div className={styles.statusText}>
          <span className={styles.statusLabel}>{copy?.label ?? 'Checking…'}</span>
          <span className={styles.statusDetail}>{copy?.detail ?? ''}</span>
        </div>
        <button className={styles.canaryBtn} onClick={fireCanary} disabled={firingCanary}>
          <Play size={15} />
          {firingCanary ? 'Testing…' : 'Test the controls'}
        </button>
      </section>

      {error && <p className={styles.error}>{error}</p>}

      <section className={styles.logSection}>
        <div className={styles.logHeader}>
          <h2 className={styles.logTitle}>What the operating system sees</h2>
          <span className={styles.logMeta}>Independent of ARJUN</span>
        </div>

        {observed?.unavailableReason ? (
          <p className={styles.empty}>{observed.unavailableReason}</p>
        ) : observed && observed.externalCount > 0 ? (
          <div className={styles.breach} role="alert">
            <Binoculars size={18} />
            <div>
              <strong>
                Windows reports {observed.externalCount} connection
                {observed.externalCount === 1 ? '' : 's'} leaving this machine.
              </strong>
              <p>
                This is measured at the operating-system level, so it holds regardless of what
                ARJUN&rsquo;s own broker reports.
              </p>
            </div>
          </div>
        ) : (
          <div className={styles.observed}>
            <ShieldCheck size={18} className={styles.observedIcon} />
            <span>
              Windows attributes{' '}
              <strong>{observed ? observed.connections.length : 0}</strong> TCP connection
              {observed && observed.connections.length === 1 ? '' : 's'} to this process, and{' '}
              <strong>none of them leave this machine</strong>.
            </span>
          </div>
        )}

        {observed && observed.connections.length > 0 && (
          <ul className={styles.log}>
            {observed.connections.map((c, i) => (
              <li key={`${c.local}-${c.remote}-${i}`} className={styles.entry}>
                <span className={c.loopback ? styles.badgeRefused : styles.badgeAllowed}>
                  {c.loopback ? 'loopback' : 'external'}
                </span>
                <span className={styles.entryHost}>{c.remote}</span>
                <span className={styles.entryReason}>from {c.local}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className={styles.logSection}>
        <div className={styles.logHeader}>
          <h2 className={styles.logTitle}>What ARJUN decided</h2>
          <span className={styles.logMeta}>
            {events.length === 0 ? 'No attempts recorded' : `${events.length} recorded`}
          </span>
        </div>

        {events.length === 0 ? (
          <p className={styles.empty}>
            Nothing has attempted to leave this machine. Use <em>Test the controls</em> to
            make ARJUN deliberately try — it should be refused, and appear here.
          </p>
        ) : (
          <ul className={styles.log}>
            {events.map((e, i) => (
              <li key={`${e.at}-${i}`} className={styles.entry}>
                <span className={styles.entryTime}>{formatTime(e.at)}</span>
                <span
                  className={e.permitted ? styles.badgeAllowed : styles.badgeRefused}
                >
                  {e.permitted ? 'permitted' : 'refused'}
                </span>
                <span className={styles.entryHost}>{e.host}</span>
                {e.canary && <span className={styles.badgeCanary}>self-test</span>}
                <span className={styles.entryReason}>{e.reason}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <AuditRecord />

      <p className={styles.footnote}>
        Two vantage points, deliberately. The lower panel is ARJUN vouching for itself; the upper
        one is Windows answering <code>GetExtendedTcpTable</code> for this process ID, which needs
        no administrator rights and no agent installed. They should agree &mdash; and if they ever
        disagree, that disagreement is the finding. The OS view covers TCP for this process; it
        does not see UDP or a connection opened and closed between two polls.
      </p>
    </div>
  );
};

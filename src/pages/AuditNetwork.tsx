import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ShieldCheck, ShieldAlert, Radio, Play, Binoculars } from 'lucide-react';
import {
  sovereigntyService,
  type EgressEvent,
  type ObservationReport,
  type OperatingMode,
} from '../services/sovereignty.service';
import {
  LOADING,
  describeAge,
  isStale,
  measured,
  unavailable,
  valueOf,
  type Reading,
} from '../services/reading';
import { AuditRecord } from '../components/audit/AuditRecord';
import styles from './AuditNetwork.module.css';

/** How often the monitor re-reads the broker's decision log. */
const POLL_INTERVAL_MS = 2000;

/**
 * How old a reading may be before the page stops presenting it as current.
 *
 * Several poll intervals, so one slow round trip does not flap the label, and
 * short enough that a source which has genuinely stopped answering is called
 * out while somebody is still looking at the screen.
 */
const STALE_AFTER_MS = POLL_INTERVAL_MS * 4;

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
  // Three sources, three readings.
  //
  // Each of these used to be a plain value seeded with `null` or `[]`, and the
  // page rendered that seed as a finding — a green zero beside the words "none
  // of them leave this machine", before anything had been measured. A
  // `Reading` cannot be read that way: there is no value to render until one
  // was taken.
  const [modeReading, setModeReading] = useState<Reading<OperatingMode>>(LOADING);
  const [eventsReading, setEventsReading] = useState<Reading<EgressEvent[]>>(LOADING);
  const [observedReading, setObservedReading] = useState<Reading<ObservationReport>>(LOADING);
  const [firingCanary, setFiringCanary] = useState(false);
  const [canaryError, setCanaryError] = useState<string | null>(null);
  /** Advanced on each poll so the "4s ago" labels stay honest. */
  const [now, setNow] = useState(() => Date.now());

  // Held in a ref so the poll callback never closes over a stale value and the
  // interval does not need tearing down on every render.
  const mountedRef = useRef(true);

  /**
   * Refresh all three sources, independently.
   *
   * This was one `Promise.all`, which made the three share a fate: a
   * connection table that could not be read rejected the whole batch, so the
   * mode and the broker's log silently kept whatever they had from the last
   * successful poll — with nothing on screen to say they had stopped being
   * refreshed. One failing probe froze the page while it went on looking live.
   *
   * Settled separately, a source that fails says so and the other two carry
   * on. That is the only arrangement in which the two vantage points this page
   * closes with can honestly be called independent.
   */
  const refresh = useCallback(async () => {
    const read = async <T,>(take: () => Promise<T>, set: (reading: Reading<T>) => void) => {
      try {
        const value = await take();
        if (mountedRef.current) set(measured(value));
      } catch (e) {
        if (mountedRef.current) {
          set(unavailable(e instanceof Error ? e.message : String(e)));
        }
      }
    };

    await Promise.all([
      read(() => sovereigntyService.getMode(), setModeReading),
      read(() => sovereigntyService.recentEvents(), setEventsReading),
      read(() => sovereigntyService.observeConnections(), setObservedReading),
    ]);
    if (mountedRef.current) setNow(Date.now());
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
    setCanaryError(null);
    try {
      await sovereigntyService.runCanary();
      await refresh();
    } catch (e) {
      setCanaryError(e instanceof Error ? e.message : String(e));
    } finally {
      setFiringCanary(false);
    }
  };

  const mode = valueOf(modeReading);
  const events = valueOf(eventsReading) ?? [];
  const observed = valueOf(observedReading);

  // A permitted call while in Work mode means the controls did not hold. It is
  // the one thing on this page that must be impossible to overlook.
  //
  // Read only from a measurement: a breach that cannot be seen because the log
  // could not be read is not the absence of a breach, and the log panel below
  // says so rather than showing an empty list.
  const breach = events.find(e => e.permitted && e.mode === 'work');
  const copy = mode ? MODE_COPY[mode] : null;

  const observedStale = isStale(observedReading, STALE_AFTER_MS, now);
  const eventsStale = isStale(eventsReading, STALE_AFTER_MS, now);

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
          <span className={styles.statusLabel}>
            {copy?.label ??
              (modeReading.state === 'unavailable' ? 'Mode unknown' : 'Checking…')}
          </span>
          <span className={styles.statusDetail}>
            {copy?.detail ??
              (modeReading.state === 'unavailable'
                ? // The banner's colour keys off `mode === 'work'`, so an
                  // unreadable mode already renders in the open//unsealed
                  // style rather than the sealed one. Saying so is what stops
                  // that being read as "provisioning".
                  `The operating mode could not be read, so no claim is made about it: ${modeReading.reason}`
                : '')}
          </span>
        </div>
        <button className={styles.canaryBtn} onClick={fireCanary} disabled={firingCanary}>
          <Play size={15} />
          {firingCanary ? 'Testing…' : 'Test the controls'}
        </button>
      </section>

      {canaryError && <p className={styles.error}>{canaryError}</p>}

      <section className={styles.logSection}>
        <div className={styles.logHeader}>
          <h2 className={styles.logTitle}>What the operating system sees</h2>
          <span className={styles.logMeta}>Independent of ARJUN</span>
        </div>

        {observedReading.state === 'loading' ? (
          // Nobody has looked yet. This used to render as a green zero.
          <p className={styles.empty}>Reading the operating system&rsquo;s connection table…</p>
        ) : observedReading.state === 'unavailable' ? (
          // Somebody looked and could not find out. Not a zero, and it must
          // not borrow a zero's words.
          <p className={styles.empty}>
            The connection table could not be read, so nothing is claimed about it:{' '}
            {observedReading.reason} (tried {describeAge(observedReading.at, now)})
          </p>
        ) : observedReading.value.unavailableReason ? (
          // The probe ran and reported that it could not look — different
          // again from the transport failing, and equally not a zero.
          <p className={styles.empty}>{observedReading.value.unavailableReason}</p>
        ) : observedReading.value.externalCount > 0 ? (
          <div className={styles.breach} role="alert">
            <Binoculars size={18} />
            <div>
              <strong>
                Windows reports {observedReading.value.externalCount} connection
                {observedReading.value.externalCount === 1 ? '' : 's'} leaving this machine.
              </strong>
              <p>
                This is measured at the operating-system level, so it holds regardless of what
                ARJUN&rsquo;s own broker reports.
              </p>
            </div>
          </div>
        ) : (
          // A measured reading, read through the narrowed union rather than
          // through a nullable alias, so the count below is a number somebody
          // actually counted and the compiler agrees.
          <div className={styles.observed}>
            <ShieldCheck size={18} className={styles.observedIcon} />
            <span>
              Windows attributes{' '}
              <strong>{observedReading.value.connections.length}</strong> TCP connection
              {observedReading.value.connections.length === 1 ? '' : 's'} to this process
              {observedReading.value.connections.length > 0 ? (
                <>
                  , and <strong>none of them leave this machine</strong>.
                </>
              ) : (
                // A measured zero is its own finding, and it is a stronger one
                // than "none of them leave" — there was nothing to leave.
                <> at all.</>
              )}{' '}
              <span className={styles.readingAge}>
                Measured {describeAge(observedReading.at, now)}
                {observedStale && ' — not refreshed since; treat as out of date'}
              </span>
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
            {eventsReading.state === 'loading'
              ? 'Reading…'
              : eventsReading.state === 'unavailable'
                ? 'Could not be read'
                : events.length === 0
                  ? 'No attempts recorded'
                  : `${events.length} recorded`}
          </span>
        </div>

        {eventsReading.state === 'loading' ? (
          <p className={styles.empty}>Reading the broker&rsquo;s decision log…</p>
        ) : eventsReading.state === 'unavailable' ? (
          // The same rule as the panel above. An empty list because the log
          // could not be read is not "nothing attempted to leave", and the
          // difference is the entire point of this page.
          <p className={styles.empty}>
            The broker&rsquo;s log could not be read, so nothing is claimed about what was
            attempted: {eventsReading.reason} (tried{' '}
            {describeAge(eventsReading.at, now)})
          </p>
        ) : events.length === 0 ? (
          <p className={styles.empty}>
            Nothing has attempted to leave this machine. Use <em>Test the controls</em> to
            make ARJUN deliberately try — it should be refused, and appear here.{' '}
            <span className={styles.readingAge}>
              Checked {describeAge(eventsReading.at, now)}
              {eventsStale && ' — not refreshed since; treat as out of date'}
            </span>
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

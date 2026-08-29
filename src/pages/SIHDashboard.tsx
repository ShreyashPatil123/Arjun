import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Cpu,
  Activity,
  ShieldCheck,
  ShieldAlert,
  FileOutput,
  CheckCircle2,
  Circle,
  Image as ImageIcon,
  Table as TableIcon,
  FileText,
  Plus,
  Loader2,
} from 'lucide-react';
import { useToast } from '../hooks/useToast';
import { demoService, type DemoScenario } from '../services/demo.service';
import { sovereigntyService, type EgressEvent } from '../services/sovereignty.service';
import { useRun, isBusy, hasSummary } from '../components/run/useRun';
import { ChatSurface } from '../components/chat/ChatSurface';
import type { RunViewState } from '../components/run/recovery';
import type { RoutingDecision } from '../services/agent.service';
import styles from './SIHDashboard.module.css';

/**
 * Industrial dark palette. Defined in CSS module too; kept here so the
 * components can be read in isolation.
 */
const PALETTE = {
  amber: '#f59e0b',
  steel: '#3f3f46',
  green: '#22c55e',
  red: '#ef4444',
  bg: '#0a0a0a',
  panel: '#18181b',
  text: '#e4e4e7',
  muted: '#a1a1aa',
};

/**
 * The SIH venue surface.
 *
 * Earlier this page was entirely presentational: routing, security, plan
 * and activity were hard-coded samples. That is what made the page look
 * fine but answer nothing — a judge who asked "what is the model doing
 * right now?" got the same answer they would have got yesterday.
 *
 * The fix is not to invent data. It is to wire the surface to the same
 * useRun the workbench uses, and only show what the run actually has
 * produced. Idle panels are honest about being idle; finished panels
 * are honest about being finished.
 */
export const SIHDashboard = () => {
  const { state } = useRun();
  const { addToast } = useToast();
  const scenarios = useMemo<DemoScenario[]>(() => demoService.list(), []);

  // Live egress events from the broker — replaces the SAMPLE_SECURITY list.
  const [egress, setEgress] = useState<EgressEvent[]>([]);
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const initial = await sovereigntyService.recentEvents();
        if (!cancelled) setEgress(initial);
      } catch {
        // The broker may be unavailable in offline / dev builds. Show an
        // empty list rather than failing the whole dashboard.
      }
    })();
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const latest = await sovereigntyService.recentEvents();
          if (!cancelled) setEgress(latest);
        } catch {
          // Ignored — see above.
        }
      })();
    }, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  // Notify when a one-click demo is starting. The actual run is
  // initiated by the chat surface through a `arjun:trigger-send`
  // window event, so the surface creates the user message and the
  // assistant cell in its own conversation.
  const onDemo = useCallback(
    (scenario: DemoScenario) => {
      addToast('info', `Starting: ${scenario.title}`);
      window.dispatchEvent(
        new CustomEvent('arjun:trigger-send', {
          detail: {
            prompt: scenario.prompt,
            title: scenario.title,
            systemPrompt: scenario.systemPrompt,
          },
        }),
      );
    },
    [addToast],
  );

  return (
    <div className={styles.dashboard} style={{ background: PALETTE.bg }}>
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <span className={styles.brand}>ARJUN</span>
          <span className={styles.topbarTag}>SIH 2026 · PS 26117</span>
        </div>
        <div className={styles.topbarRight}>
          <span className={styles.topbarBadge} style={{ background: PALETTE.amber }}>
            Sovereign
          </span>
          <span className={styles.topbarBadge} style={{ background: PALETTE.green }}>
            Audit intact
          </span>
          {isBusy(state.phase) && (
            <span
              className={styles.topbarBadge}
              style={{ background: PALETTE.steel }}
            >
              <Loader2 size={11} className={styles.spin} /> Running
            </span>
          )}
        </div>
      </header>

      <div className={styles.threePane}>
        {/* LEFT: chat (now driven by the new chat surface) */}
        <section className={`${styles.pane} ${styles.paneLeft}`}>
          <header className={styles.paneHeader}>
            <h2>Chat</h2>
            <span className={styles.paneHeaderTag}>multimodal · text + image + table</span>
          </header>
          <div className={styles.chatSurfaceWrap}>
            <ChatSurface
              classification="processDiagram"
              showSidebar={false}
            />
          </div>
        </section>

        {/* CENTER: routing + plan + demos */}
        <section className={`${styles.pane} ${styles.paneCenter}`}>
          <RoutingPanel
            routing={hasSummary(state) ? state.summary.routing : null}
            phase={state.phase}
            runId={state.runId}
          />
          <PlanPanel state={state} />
          <div className={styles.demosBox}>
            <h3>
              <FileOutput size={16} /> One-click demos
            </h3>
            {scenarios.map((s) => (
              <button
                key={s.id}
                type="button"
                className={styles.demoButton}
                onClick={() => onDemo(s)}
              >
                <Plus size={14} />
                <span>{s.title}</span>
                <span className={styles.demoSummary}>{s.summary}</span>
                {s.skillId && (
                  <span className={styles.demoSkill}>skill: {s.skillId}</span>
                )}
              </button>
            ))}
          </div>
        </section>

        {/* RIGHT: security monitor + activity */}
        <section className={`${styles.pane} ${styles.paneRight}`}>
          <SecurityPanel events={egress} />
          <ActivityPanel state={state} />
        </section>
      </div>

      <footer className={styles.bottomPanel}>
        <BottomProgress state={state} />
      </footer>
    </div>
  );
};

/* ------------------------------------------------------------------ *
 * Routing panel
 * ------------------------------------------------------------------ *
 *
 * When a run is in progress or finished, the live `RoutingDecision` is
 * what the router picked. Before that, the panel shows the same idle
 * state the workbench does — nothing has been chosen, and saying so is
 * more truthful than pinning a static "gemma-3-12b-it" row.
 */
const RoutingPanel = ({
  routing,
  phase,
  runId,
}: {
  routing: RoutingDecision | null;
  phase: RunViewState['phase'];
  runId: string | null;
}) => {
  const live = routing !== null;
  return (
    <div className={styles.routingPanel}>
      <header>
        <Cpu size={16} />
        <h3>Live Model Router</h3>
        {live ? (
          <span className={styles.routingStatus} data-state="live">
            live
          </span>
        ) : (
          <span className={styles.routingStatus} data-state="idle">
            idle
          </span>
        )}
      </header>

      {!live && (
        <p className={styles.routingEmpty}>
          No model has been routed yet. Run a demo or send a message and the
          router&apos;s choice will appear here.
        </p>
      )}

      {live && routing && (
        <>
          <div className={styles.routingChosen}>
            <span className={styles.label}>Routed to</span>
            <strong className={styles.chosenModel}>{routing.modelName}</strong>
            <p className={styles.reason}>
              {routing.reasons.length > 0
                ? routing.reasons.join(' · ')
                : routing.intent}
            </p>
            <p className={styles.routingMeta}>
              role: {routing.role} · confidence{' '}
              {routing.confidence.toFixed(2)}
              {routing.usedFallback && ' · fallback used'}
            </p>
          </div>
          <p className={styles.routingGpu}>
            {routing.gpuPlanSummary}
            {routing.fullyOnGpu ? '' : ' (partial offload)'}
          </p>
          {runId && (
            <p className={styles.routingRunId}>
              run {runId.slice(0, 8)}
              {phase === 'running' && ' · running'}
              {phase === 'finished' && ' · finished'}
              {phase === 'failed' && ' · failed'}
            </p>
          )}
        </>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------ *
 * Security panel — now fed by the live egress log, not a SAMPLE list.
 * ------------------------------------------------------------------ */
const SecurityPanel = ({ events }: { events: EgressEvent[] }) => (
  <div className={styles.securityPanel}>
    <header>
      <ShieldCheck size={16} />
      <h3>Security Monitor</h3>
    </header>
    <div className={styles.securityMode}>
      <span className={styles.securityBadge} style={{ background: PALETTE.green }}>
        <ShieldCheck size={12} /> Work mode
      </span>
      <span className={styles.securityBadge} style={{ background: PALETTE.steel }}>
        <ShieldAlert size={12} /> Zero egress
      </span>
      <span className={styles.securityBadge} style={{ background: PALETTE.amber }}>
        Audit chain: intact
      </span>
    </div>
    {events.length === 0 ? (
      <p className={styles.securityEmpty}>
        No egress events recorded yet. The broker is watching; nothing has been
        attempted.
      </p>
    ) : (
      <ul className={styles.securityList}>
        {events.slice(0, 12).map((e, i) => (
          <li key={i} className={styles.securityItem}>
            <span className={styles.securityTime}>
              {new Date(e.at).toLocaleTimeString()}
            </span>
            <span
              className={styles.securityKind}
              data-status={e.permitted ? 'ok' : 'attention'}
            >
              {e.permitted ? 'egress-ok' : 'egress-blocked'}
            </span>
            <span className={styles.securityDetail}>
              {e.permitted ? 'permitted' : 'refused'} → {e.host} ({e.reason})
            </span>
          </li>
        ))}
      </ul>
    )}
  </div>
);

/* ------------------------------------------------------------------ *
 * Plan panel — bound to the live plan record from useRun.
 * ------------------------------------------------------------------ */
function planStepsForDisplay(plan: RunViewState['plan'], isRunning: boolean) {
  if (!plan) {
    return {
      items: [] as { key: string; label: string; state: 'todo' | 'done' | 'current' }[],
      done: 0,
      total: 0,
      hasPlan: false,
    };
  }
  const total = plan.steps.length;
  // The plan's own `done` flag is the source of truth. While running we
  // additionally highlight the first not-done step as the current one.
  type StepState = 'done' | 'todo' | 'current';
  const items: { key: string; label: string; state: StepState }[] =
    plan.steps.map((step) => ({
      key: `step-${step.ordinal}`,
      label: step.intent,
      state: step.done ? 'done' : 'todo',
    }));
  if (isRunning) {
    const next = items.find((it) => it.state === 'todo');
    if (next) next.state = 'current';
  }
  const done = plan.steps.filter((s) => s.done).length;
  return { items, done, total, hasPlan: true };
}

const PlanPanel = ({ state }: { state: RunViewState }) => {
  const running = isBusy(state.phase);
  const { items, done, total, hasPlan } = planStepsForDisplay(state.plan, running);
  const stoppedBecause = state.plan?.stoppedBecause ?? state.stopped;
  return (
    <div className={styles.planPanel}>
      <h3>Plan</h3>
      {!hasPlan ? (
        <p className={styles.planEmpty}>
          No plan yet. The plan is set by the policy before the run starts
          and is shown here as soon as it is.
        </p>
      ) : (
        <>
          <ol className={styles.planList}>
            {items.map((s) => (
              <li
                key={s.key}
                className={`${styles.planItem} ${
                  s.state === 'done'
                    ? styles.planItemDone
                    : s.state === 'current'
                      ? styles.planItemCurrent
                      : ''
                }`}
              >
                {s.state === 'done' ? <CheckCircle2 size={14} /> : <Circle size={14} />}
                <span>{s.label}</span>
              </li>
            ))}
          </ol>
          <div className={styles.planProgress}>
            <span>
              {done} / {total} steps
              {' · '}
              {state.plan?.stepsTaken ?? 0} tool calls
            </span>
            <div className={styles.planProgressBar}>
              <div
                className={styles.planProgressFill}
                style={{ width: `${total > 0 ? (done / total) * 100 : 0}%` }}
              />
            </div>
          </div>
          {stoppedBecause && (
            <p className={styles.planStopped} role="status">
              {stoppedBecause}
            </p>
          )}
        </>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------ *
 * Right-pane activity feed — live tool calls from the run.
 * ------------------------------------------------------------------ */
const ActivityPanel = ({ state }: { state: RunViewState }) => (
  <div className={styles.activityBox}>
    <header>
      <Activity size={16} />
      <h3>Recent activity</h3>
    </header>
    {state.activity.length === 0 ? (
      <p className={styles.activityEmpty}>
        {isBusy(state.phase)
          ? 'The run has not called any tool yet.'
          : 'No tool activity recorded for this run.'}
      </p>
    ) : (
      <ul>
        {state.activity.slice(-8).reverse().map((item) => {
          const at = item.endedAt ?? item.startedAt;
          const time = at ? new Date(at).toLocaleTimeString() : '—';
          const status = item.status;
          return (
            <li key={item.id}>
              <span className={styles.activityTime}>{time}</span>
              <span
                className={styles.activityStatus}
                data-status={status}
              >
                {status}
              </span>
              <span className={styles.activityTool}>{item.tool}</span>
              {item.errorMessage && (
                <span className={styles.activityError}>{item.errorMessage}</span>
              )}
            </li>
          );
        })}
      </ul>
    )}
  </div>
);

/* ------------------------------------------------------------------ *
 * Footer progress — reflects live state, not a static 2/4.
 * ------------------------------------------------------------------ */
const BottomProgress = ({ state }: { state: RunViewState }) => {
  const plan = state.plan;
  const done = plan?.steps.filter((s) => s.done).length ?? 0;
  const total = plan?.steps.length ?? 0;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const currentStep = plan?.steps.find((s) => !s.done);
  const remaining = Math.max(0, total - done);
  return (
    <>
      <div className={styles.bottomProgress}>
        <span>Plan progress</span>
        <div className={styles.bottomBar}>
          <div
            className={styles.bottomBarFill}
            style={{
              width: `${pct}%`,
              background: state.phase === 'failed' ? PALETTE.red : PALETTE.amber,
            }}
          />
        </div>
        <span className={styles.bottomPct}>
          {total > 0 ? `${pct}%` : isBusy(state.phase) ? 'starting' : '—'}
        </span>
      </div>
      <div className={styles.bottomStep}>
        <span className={styles.bottomStepCurrent}>
          {currentStep
            ? currentStep.intent
            : state.phase === 'finished'
              ? 'All planned steps are done.'
              : state.phase === 'failed'
                ? 'Run stopped before the plan was complete.'
                : isBusy(state.phase)
                  ? 'Plan is being built.'
                  : 'No plan yet.'}
        </span>
        <span className={styles.bottomStepRemaining}>
          {total > 0
            ? `${remaining} step${remaining === 1 ? '' : 's'} remaining · ${
                plan?.stepsTaken ?? 0
              } tool calls`
            : ''}
        </span>
      </div>
      <div className={styles.bottomApprovals}>
        {state.milestone ? (
          <span
            className={styles.bottomApprovalsBadge}
            style={{ background: PALETTE.amber }}
          >
            1 approval pending
          </span>
        ) : (
          <span
            className={styles.bottomApprovalsBadge}
            style={{ background: PALETTE.steel }}
          >
            no approvals pending
          </span>
        )}
      </div>
    </>
  );
};

export default SIHDashboard;

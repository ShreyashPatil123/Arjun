import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Briefcase,
  FileText,
  Image as ImageIcon,
  Play,
  RotateCcw,
  ScrollText,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { demoService, type DemoId, type DemoScenario } from '../services/demo.service';
import { useToast } from '../hooks/useToast';
import { useActiveRun } from '../contexts/ActiveRunContext';
import { RunView } from '../components/run/RunView';
import styles from './Demo.module.css';

/**
 * The SIH 2026 demo surface.
 *
 * One source of truth: `useRun`. The same hook the workbench runs drives
 * the demo. When a scenario is clicked, the page starts a *real* run
 * (no setTimeout, no fake stepper) and renders the result in the
 * standard `RunView` — same Prompt / Plan / Work / Produced / Checked /
 * Answer sections, same artifact cards, same verification.
 *
 * Two states:
 *  - **Idle**: three scenario cards, recent-runs list.
 *  - **Active / done**: full RunView, with a "back to scenarios" button.
 *
 * The history is a teaser — the real history lives in `/tasks` and the
 * user can re-open a finished run there. Showing a small list here is
 * just so the demo page is not a dead end after one run.
 */
export const Demo: React.FC = () => {
  const navigate = useNavigate();
  const scenarios = demoService.list();
  // Shared, so a scenario launched here is the run the SIH dashboard shows.
  const { state, start, abort, reset } = useActiveRun();
  const { addToast } = useToast();
  const [history, setHistory] = useState<Array<{ runId: string; title: string; when: string }>>([]);

  const isIdle = state.phase === 'idle';
  const isActive =
    state.phase === 'starting' ||
    state.phase === 'running' ||
    state.phase === 'awaiting_milestone';
  const isDone =
    state.phase === 'finished' || state.phase === 'failed';

  // Load a tiny recent-runs list on mount, refresh after each run.
  const refreshHistory = useCallback(async () => {
    try {
      const { agentService } = await import('../services/agent.service');
      const rows = await agentService.history();
      setHistory(
        rows.slice(0, 5).map(r => ({
          runId: r.runId,
          title: r.prompt.split('\n')[0].slice(0, 80) || '(no prompt)',
          when: r.finishedAt,
        })),
      );
    } catch {
      // Service not reachable: leave the history empty rather than failing
      // the page. The run surface still works.
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    if (isDone) void refreshHistory();
  }, [isDone, refreshHistory]);

  const runScenario = useCallback(
    (s: DemoScenario) => {
      // A fresh scenario click is a fresh run. Reset before starting so a
      // previously-displayed `RunView` does not flash while the new one's
      // first event arrives.
      reset();
      // The scenario's own documents, read before the run starts. A scenario
      // whose fixtures cannot be read does not start: its prompt says
      // "attached", and a run begun without them would be answering about a
      // drawing it never saw.
      void (async () => {
        try {
          const launch = await demoService.launch(s.id);
          await start(launch.prompt, launch.classification, {
            correlationId: `demo-${s.id}-${Date.now()}`,
            scenarioInstructions: launch.scenarioInstructions,
            attachments: launch.attachments,
          });
        } catch (error) {
          addToast('error', error instanceof Error ? error.message : String(error));
        }
      })();
    },
    [start, reset],
  );

  const backToScenarios = useCallback(() => {
    reset();
  }, [reset]);

  if (!isIdle) {
    return (
      <div className={styles.runPage}>
        <button
          type="button"
          className={styles.backButton}
          onClick={backToScenarios}
          disabled={isActive}
        >
          <ArrowLeft size={16} />
          {isActive ? 'Run in progress…' : 'Back to scenarios'}
        </button>
        <RunView
          state={state}
          onAbort={() => void abort()}
          onNewTask={backToScenarios}
          onRerun={
            isDone && state.prompt
              ? () => {
                  reset();
                  void start(state.prompt);
                }
              : undefined
          }
        />
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerText}>
          <h1>SIH 2026 · Scenarios</h1>
          <p className={styles.subtitle}>
            One click runs an end-to-end task on synthetic data. Each
            scenario uses a different industrial skill, produces a real
            artifact, and writes a verifiable row to the audit log.
          </p>
        </div>
        <div className={styles.headerActions}>
          <button
            type="button"
            className={styles.headerButton}
            onClick={() => navigate('/sih')}
          >
            <Sparkles size={16} /> Presentation dashboard
          </button>
          <button
            type="button"
            className={styles.headerButton}
            onClick={() => navigate('/tasks')}
          >
            <ScrollText size={16} /> Run history
          </button>
        </div>
      </header>

      <section className={styles.assurance}>
        <ShieldCheck size={18} />
        <div>
          <strong>What the judges will see</strong>
          <p>
            Every scenario runs against local models and local documents.
            The same run is shown on the workbench and the presentation
            dashboard — one source of truth, not a separate mock.
          </p>
        </div>
      </section>

      <section className={styles.scenarios}>
        {scenarios.map(s => (
          <ScenarioCard key={s.id} scenario={s} onRun={runScenario} />
        ))}
      </section>

      <section className={styles.historyBox}>
        <h2>
          <Briefcase size={16} /> Recent runs
        </h2>
        {history.length === 0 ? (
          <p className={styles.historyEmpty}>
            No runs yet. Pick a scenario above to see ARJUN execute it
            end-to-end.
          </p>
        ) : (
          <ul className={styles.historyList}>
            {history.map(h => (
              <li key={h.runId}>
                <code className={styles.historyId}>{h.runId.slice(0, 8)}</code>
                <span className={styles.historyTitle}>{h.title}</span>
                <span className={styles.historyWhen}>{h.when}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};

/**
 * A scenario card. Compact, focused on the action.
 *
 * Shows: the kind of task (an icon from the scenario's primary tool),
 * the scenario summary, the skill the agent will use, and the run
 * button. The plan checklist that used to live here is now in the
 * Plan section of the real RunView when the run starts.
 */
const ScenarioCard: React.FC<{
  scenario: DemoScenario;
  onRun: (s: DemoScenario) => void;
}> = ({ scenario, onRun }) => {
  const Icon = iconFor(scenario.id);
  return (
    <article className={styles.card}>
      <header className={styles.cardHeader}>
        <span className={styles.cardIcon}>
          <Icon size={20} />
        </span>
        <div>
          <h3>{scenario.title}</h3>
          <p>{scenario.summary}</p>
        </div>
      </header>
      {scenario.skillId && (
        <div className={styles.cardSkill}>
          <code>{scenario.skillId}</code>
          <span>industrial skill</span>
        </div>
      )}
      <div className={styles.cardAudit}>
        <ShieldCheck size={14} /> Writes to the audit log
      </div>
      <button
        type="button"
        className={styles.runButton}
        onClick={() => onRun(scenario)}
      >
        <Play size={16} />
        Run scenario
        <RotateCcw size={14} className={styles.runHint} aria-hidden />
      </button>
    </article>
  );
};

function iconFor(id: DemoId): React.ComponentType<{ size?: number }> {
  if (id === 'pid-analysis') return ImageIcon;
  if (id === 'vendor-quote') return FileText;
  return ScrollText;
}

export default Demo;

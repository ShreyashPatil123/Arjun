import React from 'react';
import { CheckCircle2, CircleDashed, Loader2, ShieldCheck } from 'lucide-react';
import { useAdoptedRun } from '../run/runAdopt';
import { useConversation } from '../run/useConversation';
import type { Activity } from '../run/recovery';
import type { RunSummary, VerificationReport } from '../../services/agent.service';
import { formatDuration } from './format';
import { labelFor } from '../run/useRun';
import styles from './ChatSurface.module.css';

/**
 * Right-side orchestration panel.
 *
 * Only rendered when the latest run is a multi-step / tool-heavy
 * task — the empty state lives in the chat column, and a single-step
 * answer does not need this surface. The panel shows the steps the
 * run went through (Planning → Selecting model → Reading / Tools →
 * Composing → Verifying → Done) as compact cards, each with the
 * activity that produced it.
 */
export function TaskPanel({ runId }: { runId: string | null }) {
  const adopted = useAdoptedRun(runId);
  const { conversation } = useConversation();
  const activity: Activity[] | undefined = adopted?.activity;

  if (!runId) return null;

  // The most recent run's model/role comes from the conversation's
  // `ChatRunMeta` (we use the routing decision when present).
  const run = conversation?.runs.find(r => r.runId === runId);
  const plan = adopted?.view?.plan;
  const turns = adopted?.view?.turns ?? 0;
  const compactions = adopted?.view?.compactions ?? 0;
  const summary = adopted?.view?.summary;

  // Don't show this panel at all when the run is still just an idle
  // chat — the chat column already shows the answer.
  if (!plan && (!activity || activity.length === 0) && !summary) {
    return null;
  }

  const stepCards: StepCard[] = buildSteps({
    activity,
    plan: plan ?? null,
    summary: summary ?? null,
    run,
    turns,
    compactions,
  });

  return (
    <aside className={styles.taskPanel} aria-label="Task orchestration">
      <div className={styles.taskPanelHeader}>
        <ShieldCheck size={12} className={styles.taskPanelHeaderIcon} />
        <span>Task</span>
      </div>
      <ol className={styles.taskSteps}>
        {stepCards.map((s, i) => (
          <li
            key={i}
            className={styles.taskStep}
            data-state={s.state}
          >
            <div className={styles.taskStepIcon}>
              {s.state === 'running' ? (
                <Loader2 size={12} className={styles.spin} />
              ) : s.state === 'done' ? (
                <CheckCircle2 size={12} />
              ) : (
                <CircleDashed size={12} />
              )}
            </div>
            <div className={styles.taskStepBody}>
              <div className={styles.taskStepTitle}>{s.title}</div>
              {s.detail && (
                <div className={styles.taskStepDetail}>{s.detail}</div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </aside>
  );
}

interface StepCard {
  state: 'pending' | 'running' | 'done';
  title: string;
  detail?: string;
}

function buildSteps({
  activity,
  plan,
  summary,
  run,
  turns,
  compactions,
}: {
  activity?: Activity[];
  plan: { steps: { intent: string }[] } | null;
  summary: Pick<RunSummary, 'verification'> | null;
  run: { modelName?: string | null; live?: boolean } | undefined;
  turns: number;
  compactions: number;
}): StepCard[] {
  const steps: StepCard[] = [];

  // 1. Plan
  if (plan) {
    steps.push({
      state: 'done',
      title: 'Planned the answer',
      detail: plan.steps.map(s => s.intent).join(' · '),
    });
  } else {
    steps.push({ state: 'pending', title: 'Planning' });
  }

  // 2. Model
  if (run?.modelName) {
    steps.push({
      state: 'done',
      title: `Selected ${run.modelName}`,
    });
  } else {
    steps.push({ state: 'pending', title: 'Selecting model' });
  }

  // 3. Tools
  if (activity && activity.length > 0) {
    for (const a of activity) {
      const duration =
        a.startedAt && a.endedAt
          ? formatDuration(Math.max(0, a.endedAt - a.startedAt))
          : null;
      steps.push({
        state: a.status === 'running' ? 'running' : a.status === 'done' || a.status === 'replayed' ? 'done' : 'pending',
        title: labelFor(a.tool),
        detail: duration ?? undefined,
      });
    }
  }

  // 4. Composing
  if (turns > 0 || summary) {
    steps.push({
      state: summary ? 'done' : 'running',
      title: 'Composed the answer',
    });
  } else {
    steps.push({ state: 'pending', title: 'Composing' });
  }

  // 5. Verifying
  if (summary?.verification) {
    const v: VerificationReport = summary.verification;
    const ready = v.standing.standing === 'ready';
    steps.push({
      state: 'done',
      title: ready ? 'Verified' : 'Needs review',
      detail: v.findings.length
        ? `${v.findings.length} finding${v.findings.length === 1 ? '' : 's'}`
        : 'all checks passed',
    });
  } else if (summary) {
    steps.push({ state: 'done', title: 'No verification required' });
  } else {
    steps.push({ state: 'pending', title: 'Verifying' });
  }

  // 6. Compaction (only if it happened)
  if (compactions > 0) {
    steps.push({
      state: 'done',
      title: `Compacted ${compactions} time${compactions === 1 ? '' : 's'}`,
    });
  }

  // 7. Done
  steps.push({
    state: summary ? 'done' : 'pending',
    title: summary ? 'Done' : 'Not yet done',
  });

  return steps;
}

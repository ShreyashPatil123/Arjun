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

  // The assistant cell this run produced, found through the run's own
  // `messageId` rather than by position. It is the only durable record of how
  // the turn ended: `status`, `verification` and `outcome` are all written to
  // it when the run completes and survive a restart, where the run's events do
  // not.
  const message = conversation?.messages.find(m => m.id === run?.messageId);

  const stepCards: StepCard[] = buildSteps({
    activity,
    plan: plan ?? null,
    summary: summary ?? null,
    message,
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

export function buildSteps({
  activity,
  plan,
  summary,
  message,
  run,
  turns,
  compactions,
}: {
  activity?: Activity[];
  plan: { steps: { intent: string }[] } | null;
  summary: Pick<RunSummary, 'verification'> | null;
  /**
   * The assistant message this run wrote, when the conversation has it.
   *
   * The last three cards used to key off `summary`, which is set from a live
   * run view — and that view hard-codes `summary: null` for anything adopted
   * from a snapshot, which is every run this panel ever sees. Nothing anywhere
   * assigned it. So "Composed the answer" spun for ever, "Verifying" never
   * left pending, and the final card read "Not yet done" under a finished
   * answer, on every run.
   *
   * The message is the durable answer to the same question. `status` says
   * whether the turn is still writing, and `verification` says what the
   * verifier concluded; both are persisted with the conversation.
   */
  message: { status?: string; verification?: 'ready' | 'needsReview' | null; outcome?: string | null } | undefined;
  run: { modelName?: string | null; live?: boolean } | undefined;
  turns: number;
  compactions: number;
}): StepCard[] {
  // "Finished" is a property of the turn, not of the panel's own state.
  const streaming = message?.status === 'streaming';
  const finished = message ? !streaming : Boolean(summary);
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
  if (finished) {
    steps.push({ state: 'done', title: 'Composed the answer' });
  } else if (streaming || turns > 0) {
    steps.push({ state: 'running', title: 'Composing the answer' });
  } else {
    steps.push({ state: 'pending', title: 'Composing' });
  }

  // 5. Verifying
  //
  // The message's verdict first: it is written when the run completes and is
  // still there after a restart, which is when this panel is most often read.
  if (message?.verification) {
    const ready = message.verification === 'ready';
    steps.push({
      state: 'done',
      title: ready ? 'Verified' : 'Needs review',
    });
  } else if (finished) {
    steps.push({ state: 'done', title: 'No verification required' });
  } else if (summary?.verification) {
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
  //
  // Named for what happened rather than only that it stopped. A turn the
  // operator interrupted and one that failed are both "not running", and
  // showing them the same word is how a failure goes unnoticed.
  if (finished) {
    const failed = message?.status === 'failed' || message?.outcome === 'failed';
    steps.push({ state: 'done', title: failed ? 'Ended without an answer' : 'Done' });
  } else {
    steps.push({ state: 'pending', title: 'Not yet done' });
  }

  return steps;
}

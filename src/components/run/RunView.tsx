import React, { useState } from 'react';
import {
  AlertTriangle,
  CircleSlash,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  ShieldCheck,
  X,
} from 'lucide-react';
import {
  agentService,
  type ArtifactReport,
  type PlanRecord,
  type VerificationReport,
} from '../../services/agent.service';
import { labelFor, type RunState } from './useRun';
import styles from './RunView.module.css';

/**
 * One run, as it happens and afterwards.
 *
 * The order of the sections is the order somebody checks work in: what it was
 * asked, what it planned, what it did, what it produced, whether that holds up,
 * and only then the answer. Putting the answer last is deliberate — an answer
 * read before its provenance is an answer taken on trust, and the point of this
 * screen is that it need not be.
 *
 * Nothing here decides anything. Every judgement it shows — whether a file is
 * sound, whether a claim resolves to a source, why the run stopped — was made
 * in Rust against the file or the passage itself, not in the browser against
 * the model's description of them.
 */

const KIND_ICONS = {
  document: FileText,
  workbook: FileSpreadsheet,
  text: FileText,
} as const;

/** Bytes as somebody would say them. */
function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function Plan({ plan, stopped }: { plan: PlanRecord; stopped: string | null }) {
  const minutes = Math.round(plan.maxDurationSeconds / 60);
  return (
    <section className={styles.section}>
      <header className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>Plan</h2>
        <span className={styles.budget}>
          {plan.stepsTaken} of {plan.maxSteps} tool calls &middot; {minutes} min limit
        </span>
      </header>

      {/* Shown as what the run set out to do, with no per-step tick. One
        * planned step can take several tool calls, so nothing here knows that
        * a given step is finished — the artifacts and the check below are the
        * evidence for what was actually achieved, and a checklist ticking
        * itself off on call count would contradict them. */}
      <ol className={styles.steps}>
        {plan.steps.map(step => (
          <li key={step.ordinal} className={styles.step}>
            <span className={styles.stepMark} aria-hidden>
              {step.ordinal}
            </span>
            <span>{step.intent}</span>
          </li>
        ))}
      </ol>

      <p className={styles.tools}>Allowed to use: {plan.permittedTools.join(', ')}.</p>

      {/* Shown while it is still the freshest thing known — the final stop
        * reason arrives with the summary and is shown under the answer. */}
      {stopped && (
        <p className={styles.stopped} role="status">
          <CircleSlash size={14} />
          <span>{stopped}</span>
        </p>
      )}
    </section>
  );
}

function Verification({ report }: { report: VerificationReport }) {
  const standing = report.standing;
  const ready = standing.standing === 'ready';

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>Checked</h2>

      <p className={ready ? styles.verdictReady : styles.verdictReview}>
        {ready ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
        <span>
          {standing.standing === 'ready'
            ? 'Every claim in this answer resolves to a passage the task retrieved, and its figures match the recorded calculations.'
            : `This is a draft, not a finished answer. ${standing.blocking} thing(s) need checking before it is relied on, and ${standing.advisory} are worth a look.`}
        </span>
      </p>

      <p className={styles.counts}>
        {report.citationsResolved} citation(s) resolved &middot; {report.figuresChecked} figure(s)
        matched to a calculation
      </p>

      {report.findings.length > 0 && (
        <ul className={styles.findings}>
          {report.findings.map((finding, i) => (
            <li
              key={i}
              className={
                finding.severity === 'blocking'
                  ? `${styles.finding} ${styles.findingBlocking}`
                  : styles.finding
              }
            >
              <span className={styles.findingBadge}>
                {finding.severity === 'blocking' ? 'Needs checking' : 'Worth a look'}
              </span>
              <span className={styles.findingText}>{finding.detail}</span>
              {finding.excerpt && <code className={styles.excerpt}>{finding.excerpt}</code>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Artifacts({ artifacts, runId }: { artifacts: ArtifactReport[]; runId: string | null }) {
  const [problem, setProblem] = useState<string | null>(null);

  const reveal = async (name: string) => {
    if (!runId) return;
    try {
      setProblem(null);
      await agentService.revealArtifact(runId, name);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>Produced</h2>
      <ul className={styles.artifacts}>
        {artifacts.map(artifact => {
          const Icon = KIND_ICONS[artifact.kind];
          return (
            <li key={artifact.path} className={styles.artifact}>
              <Icon size={17} className={styles.artifactIcon} />
              <div className={styles.artifactBody}>
                <div className={styles.artifactName}>
                  <strong>{artifact.name}</strong>
                  <span className={styles.artifactSize}>{size(artifact.bytes)}</span>
                  {/* Re-opened and checked by the backend, not inferred from
                    * the fact that a write returned without error. */}
                  <span className={artifact.sound ? styles.tagSound : styles.tagUnsound}>
                    {artifact.sound ? 'opens and checks out' : 'did not pass its check'}
                  </span>
                </div>
                <p className={styles.artifactDetail}>{artifact.detail}</p>
                {artifact.problems.length > 0 && (
                  <ul className={styles.problems}>
                    {artifact.problems.map((text, i) => (
                      <li key={i}>{text}</li>
                    ))}
                  </ul>
                )}
              </div>
              <button
                className={styles.revealBtn}
                onClick={() => void reveal(artifact.name)}
                aria-label={`Show ${artifact.name} in the file manager`}
              >
                <FolderOpen size={15} />
              </button>
            </li>
          );
        })}
      </ul>
      {problem && (
        <p className={styles.stopped} role="alert">
          <AlertTriangle size={14} />
          <span>{problem}</span>
        </p>
      )}
    </section>
  );
}

interface Props {
  state: RunState;
  onAbort: () => void;
  onNewTask: () => void;
}

export const RunView = ({ state, onAbort, onNewTask }: Props) => {
  const running = state.phase === 'starting' || state.phase === 'running';
  const summary = state.summary;

  return (
    <article className={styles.run}>
      <header className={styles.head}>
        <p className={styles.prompt}>{state.prompt}</p>
        {running ? (
          <button className={styles.stopBtn} onClick={onAbort}>
            <X size={14} />
            <span>Stop</span>
          </button>
        ) : (
          <button className={styles.newBtn} onClick={onNewTask}>
            New task
          </button>
        )}
      </header>

      {/* Which model took it and why. The routing decision is shown with the
        * work rather than buried in the audit log, because "why this model"
        * is the question asked when an answer looks wrong. */}
      {summary && (
        <p className={styles.routing}>
          <strong>{summary.routing.modelName}</strong> took this as{' '}
          {summary.routing.intent.toLowerCase()} work
          {summary.routing.usedFallback ? ', after the first choice did not fit' : ''} &middot;{' '}
          {summary.routing.reasons[0]} &middot;{' '}
          {summary.endpoint.runtime === 'llamaCpp' ? 'llama.cpp' : 'Python sidecar'} on{' '}
          {summary.endpoint.baseUrl}
        </p>
      )}

      {state.plan && <Plan plan={state.plan} stopped={state.stopped} />}

      <section className={styles.section}>
        <header className={styles.sectionHead}>
          <h2 className={styles.sectionTitle}>Work</h2>
          {running && (
            <span className={styles.live}>
              <Loader2 size={13} className={styles.spin} />
              <span>{state.turns > 0 ? `turn ${state.turns + 1}` : 'starting'}</span>
            </span>
          )}
        </header>

        {state.activity.length === 0 ? (
          <p className={styles.quiet}>
            {running
              ? 'Reading the task. Nothing has been called yet.'
              : state.phase === 'failed'
                ? // Distinguished from a run that simply needed no tools: a run
                  // that never started did not answer anything, and saying it
                  // did would be the misleading half of an already bad outcome.
                  'This task stopped before it could call anything.'
                : 'This task was answered without calling a tool.'}
          </p>
        ) : (
          <ol className={styles.activity}>
            {state.activity.map(item => (
              <li key={item.id} className={styles.activityItem}>
                <span className={styles[`dot_${item.status}`]} aria-hidden />
                <span className={styles.activityLabel}>{labelFor(item.tool)}</span>
                <span className={styles.activityStatus}>
                  {item.status === 'running' && 'running'}
                  {item.status === 'done' && 'done'}
                  {item.status === 'failed' && 'failed'}
                  {/* Not a failure of the tool: the policy or a person said no,
                    * and the model reads that and carries on. */}
                  {item.status === 'refused' && 'not permitted'}
                </span>
              </li>
            ))}
          </ol>
        )}

        {state.compactions > 0 && (
          <p className={styles.note}>
            Earlier turns were replaced by a summary {state.compactions} time(s) so the task could
            continue. Anything answered after that point rests on the summary rather than on the
            turns themselves.
          </p>
        )}
      </section>

      {summary && summary.artifacts.length > 0 && (
        <Artifacts artifacts={summary.artifacts} runId={state.runId} />
      )}

      {summary?.verification && <Verification report={summary.verification} />}

      {summary && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Answer</h2>
          {summary.text.trim() ? (
            <div className={styles.answer}>{summary.text}</div>
          ) : (
            <p className={styles.quiet}>This task ended without an answer.</p>
          )}
          <p className={styles.note}>
            {summary.plan.stoppedBecause} {summary.turns} turn(s).
          </p>
        </section>
      )}

      {state.phase === 'failed' && state.error && (
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{state.error}</span>
        </p>
      )}
    </article>
  );
};

// filepath: src/components/run/MilestoneGate.tsx
import React, { useState } from 'react';
import { Flag, Loader2, ShieldCheck, X } from 'lucide-react';
import { agentService } from '../../services/agent.service';
import type { MilestoneGate as Gate } from './recovery';
import styles from './RunView.module.css';

/**
 * A gate that pauses a run for human sign-off.
 *
 * The model finished a step that the plan flagged as a milestone —
 * a deliberate decision point, the kind of step where PS 26117 wants
 * a person to confirm the work before the next leg of it starts.
 * The component shows the intent, the checkpoint id (the stable
 * name the resume path uses), and two buttons: approve to continue,
 * reject to stop the run cleanly.
 *
 * Approving writes a `MilestoneRecord` to the durable run notes
 * through the agent service, so the next time the same run is
 * resumed, the gate list shows what was acknowledged and by whom.
 */
export const MilestoneGate = ({
  runId,
  gate,
  onAcknowledged,
}: {
  runId: string;
  gate: Gate;
  onAcknowledged: (gate: Gate, decision: 'approved' | 'rejected') => void;
}) => {
  const [busy, setBusy] = useState<'approved' | 'rejected' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const decide = async (decision: 'approved' | 'rejected') => {
    setBusy(decision);
    setError(null);
    try {
      // The backend is the only place that knows who is signed in;
      // the run is recorded against that actor. The service call
      // is fire-and-forget on the UI side: the live stream will
      // emit a `milestone_acknowledged` event that clears the
      // gate, so we do not clear it locally first.
      await agentService.acknowledgeMilestone(runId, gate.checkpointId, decision);
      onAcknowledged(gate, decision);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The gate could not be decided');
      setBusy(null);
    }
  };

  return (
    <section className={styles.section} data-state="milestone" aria-live="polite">
      <header className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>
          <Flag aria-hidden /> Milestone reached
        </h2>
        <span className={styles.budget}>checkpoint {gate.checkpointId}</span>
      </header>
      <p className={styles.prompt}>{gate.summary}</p>
      <p className={styles.tools}>
        Step {gate.ordinal} finished. Approve to continue, or reject to stop here and keep the
        work so far.
      </p>
      {error && (
        <p className={styles.error}>
          <X aria-hidden /> {error}
        </p>
      )}
      <div className={styles.controls}>
        <button
          type="button"
          className={styles.primary}
          onClick={() => decide('approved')}
          disabled={busy !== null}
          data-testid="milestone-approve"
        >
          {busy === 'approved' ? <Loader2 className={styles.spin} /> : <ShieldCheck />}
          Approve and continue
        </button>
        <button
          type="button"
          className={styles.secondary}
          onClick={() => decide('rejected')}
          disabled={busy !== null}
          data-testid="milestone-reject"
        >
          {busy === 'rejected' ? <Loader2 className={styles.spin} /> : <X />}
          Stop the run here
        </button>
      </div>
    </section>
  );
};

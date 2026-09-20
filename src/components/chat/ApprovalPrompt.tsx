/**
 * The approval a run is waiting on, asked in the conversation.
 *
 * ## Why this exists
 *
 * Every tool that is not read-only carries `needs_approval` and
 * `ApprovalClass::PersonBeforeEffect` — the effect does not happen until a
 * person says so. The backend does that part correctly: the request is written
 * to `run_approvals` durably, the run moves to `RunState::AwaitingApproval`,
 * and both the Tasks and Approvals screens render it.
 *
 * The conversation could not. Its `MessageStatusKind` has no state for "waiting
 * on a person", so a blocked run showed `usingTool` with a spinner — for ever,
 * with nothing on screen saying a decision was owed. Measured on the machine
 * this was written for: three approvals sat pending across eight days, every
 * one of them `workspace.write_text`, while `artifact.create_diagram` requests
 * raised on a screen that *did* show them were answered in four seconds. The
 * tool was never the difference; being visible was.
 *
 * So the question is asked where the person already is.
 *
 * ## Why a rejection insists on a reason
 *
 * `decide_approval` refuses a rejection with no `because`, and that is right:
 * an approval is evidence, and "no" with no reason is not reviewable later.
 * The button stays disabled rather than sending a request the backend will
 * refuse, so the refusal never reaches the person as a mysterious error.
 */
import { useCallback, useState } from 'react';
import { AlertTriangle, Check, CheckCheck, X } from 'lucide-react';

import type { ApprovalItem } from '../../services/approvals.service';
import { approvalsService } from '../../services/approvals.service';
import { labelForTool } from '../../services/toolNames';
import styles from './ApprovalPrompt.module.css';

interface Props {
  /** Pending requests for the run being watched, oldest first. */
  pending: ApprovalItem[];
  /** Called after a decision lands, so the caller can refresh. */
  onDecided: () => void;
}

export function ApprovalPrompt({ pending, onDecided }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [rejecting, setRejecting] = useState<string | null>(null);
  const [because, setBecause] = useState('');
  const [failure, setFailure] = useState<string | null>(null);

  const decide = useCallback(
    async (id: string, approve: boolean, reason?: string, always?: boolean) => {
      setBusy(id);
      setFailure(null);
      try {
        await approvalsService.decide(id, approve, reason, always);
        setRejecting(null);
        setBecause('');
        onDecided();
      } catch (error) {
        // Shown rather than swallowed: a decision that silently failed leaves
        // the run blocked and the person believing they unblocked it.
        setFailure(error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(null);
      }
    },
    [onDecided],
  );

  if (pending.length === 0) return null;

  // Oldest first: the run is blocked on the first one, and answering them in
  // the order they were asked is the order the run will consume them.
  const [next, ...queued] = pending;
  const request = next.request;
  const working = busy === request.id;

  return (
    <div className={styles.card} role="alertdialog" aria-live="assertive">
      <div className={styles.head}>
        <AlertTriangle size={15} className={styles.icon} aria-hidden="true" />
        <span className={styles.title}>Your approval is needed before this can run</span>
        {queued.length > 0 && (
          <span className={styles.queued}>{queued.length} more after this</span>
        )}
      </div>

      <dl className={styles.detail}>
        <dt>Action</dt>
        <dd>{labelForTool(request.tool)}</dd>
        {request.target && (
          <>
            <dt>On</dt>
            <dd className={styles.target} title={request.target}>
              {request.target}
            </dd>
          </>
        )}
        {request.consequences && (
          <>
            <dt>Effect</dt>
            <dd>{request.consequences}</dd>
          </>
        )}
      </dl>

      {failure && (
        <p className={styles.failure} role="status">
          {failure}
        </p>
      )}

      {rejecting === request.id ? (
        <div className={styles.rejectBox}>
          <label className={styles.rejectLabel} htmlFor={`why-${request.id}`}>
            Why are you rejecting this? It is recorded with the decision.
          </label>
          <input
            id={`why-${request.id}`}
            className={styles.rejectInput}
            value={because}
            autoFocus
            placeholder="e.g. wrong path, or not needed"
            onChange={(event) => setBecause(event.target.value)}
          />
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.reject}
              disabled={working || because.trim().length === 0}
              onClick={() => void decide(request.id, false, because.trim())}
            >
              <X size={14} /> Reject
            </button>
            <button
              type="button"
              className={styles.cancel}
              disabled={working}
              onClick={() => {
                setRejecting(null);
                setBecause('');
              }}
            >
              Back
            </button>
          </div>
        </div>
      ) : (
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.approve}
            disabled={working}
            onClick={() => void decide(request.id, true)}
          >
            <Check size={14} /> {working ? 'Approving…' : 'Approve'}
          </button>
          {/*
            Approve everything of this kind for the rest of the conversation.

            Building a small application is a dozen files, and asking per file
            turns one decision the person already made into a dozen prompts.
            The scope is the conversation rather than the account: "yes, write
            the files for this" is not "yes, write files whenever you like".

            Deliberately the middle button and not the default. The title says
            the scope outright, because a control that widens a permission has
            to say how far — and "always" on its own reads as forever.
          */}
          <button
            type="button"
            className={styles.approveAlways}
            disabled={working}
            title={`Approve this and every later ${request.tool} in this conversation, without asking again`}
            onClick={() => void decide(request.id, true, undefined, true)}
          >
            <CheckCheck size={14} /> Always approve
          </button>
          <button
            type="button"
            className={styles.reject}
            disabled={working}
            onClick={() => setRejecting(request.id)}
          >
            <X size={14} /> Reject
          </button>
        </div>
      )}
    </div>
  );
}

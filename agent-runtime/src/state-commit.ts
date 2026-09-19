/**
 * Handing the run's state to Rust at the points it is safe to be interrupted.
 *
 * ## What this replaces
 *
 * Nothing sent the notes to Rust during a run. They were built here, kept here,
 * and returned in the `RunResult` when the run finished — so Rust checkpointed
 * after every tool result with an empty `RunMemory`, because an empty one was
 * the only one it had. A run that died in the middle left a resume point
 * describing a run that had done nothing, and the resumption re-did the work.
 *
 * ## Why after the tool and not before
 *
 * The commit carries `completed` — the side effects that must not happen twice.
 * Recording one before the gateway and the tool have both agreed to it would
 * tell a resumed run not to repeat something that never happened, which is the
 * same failure in the other direction and a worse one. `observeToolResult`
 * already runs only after `tool.execute` resolves; this runs immediately after
 * it, in the same place, for the same reason.
 *
 * ## Why a failure here does not fail the turn
 *
 * The tool has already run. Throwing would lose a result that has already been
 * paid for and, for a side-effecting tool, has already changed something. What
 * a failed commit costs is a resume point one step behind — which is the
 * situation every run was in before this existed. So it is reported and the
 * turn continues.
 *
 * The one thing that is *not* swallowed is a correction: Rust dropping a claim
 * means this side believes something the record does not support, and the
 * outcome carries the notes as written so this side can stop believing it.
 */

import { RpcError, type RpcPeer } from "./peer.js";
import type { WorkingNotes, WorkingNotesState } from "./working-notes.js";

/**
 * The layout of a state commit.
 *
 * Must match `STATE_COMMIT_VERSION` in
 * `src-tauri/src/agent_runtime/state_commit.rs`. Rust refuses a version it does
 * not know rather than applying half a record.
 */
export const STATE_COMMIT_VERSION = 1;

/**
 * Where the run is, in Rust's own lifecycle vocabulary.
 *
 * Only the states this side is in a position to assert. The rest of `RunState`
 * is reached from Rust and is not ours to claim.
 */
export type CommitState =
  | "running"
  | "toolResultRecorded"
  | "compacting"
  | "verifying";

/** What Rust changed about a proposal before writing it. */
export interface Correction {
  correction:
    | "unbackedEffect"
    | "unbackedEvidence"
    | "unbackedArtifact"
    | "unbackedCalculation"
    | "milestonesAreNotProposable"
    | "effectWouldHaveBeenLost";
  tool?: string;
  target?: string;
  marker?: string;
  id?: string;
  offered?: number;
}

/** What Rust decided. */
export interface CommitOutcome {
  accepted: boolean;
  refusedBecause?: string;
  lastEventSeq: number;
  corrections?: Correction[];
  /** The notes as written. Authoritative. */
  notes: WorkingNotesState;
}

/**
 * Commits the run's state, and converges this side's notes on what was written.
 *
 * Returns the outcome, or `undefined` when the commit could not be made at all.
 * Callers do not have to look: the convergence has already happened by then.
 */
export async function commitState(
  peer: RpcPeer,
  options: {
    runId: string;
    attemptId: string;
    notes: WorkingNotes;
    state: CommitState;
    /** Where to send the one-line report. Defaults to stderr. */
    log?: (line: string) => void;
  },
): Promise<CommitOutcome | undefined> {
  const { runId, attemptId, notes, state } = options;
  const log =
    options.log ??
    ((line: string) => {
      process.stderr.write(`[agent-runtime:log] ${line}\n`);
    });

  // An attempt this side was never told about cannot be committed under.
  // Sending an empty string would make Rust refuse every commit for the whole
  // run, and the only symptom would be a resume point that never advanced.
  if (!attemptId) {
    log(
      `[state] run=${runId} no attempt id was supplied, so nothing can be committed; ` +
        `this run's resume point will not advance`,
    );
    return undefined;
  }

  try {
    const outcome = (await peer.request("state.commit", {
      commitVersion: STATE_COMMIT_VERSION,
      runId,
      attemptId,
      state,
      notes: notes.state,
    })) as CommitOutcome;

    // Rust is the authority on what was written. Adopting it here is what stops
    // this side proposing a dropped claim again on every commit, and stops the
    // model being handed a note the record does not support.
    if (outcome?.notes) notes.replaceWith(outcome.notes);

    for (const correction of outcome?.corrections ?? []) {
      log(`[state] run=${runId} ${describe(correction)}`);
    }
    return outcome;
  } catch (cause) {
    // Reported, never thrown. See the note at the top of this file: the tool
    // has already run, and losing its result would be the larger failure.
    const detail =
      cause instanceof RpcError
        ? `${cause.code}: ${cause.message}`
        : cause instanceof Error
          ? cause.message
          : String(cause);
    log(
      `[state] run=${runId} the state commit was not accepted (${detail}); the resume point ` +
        `for this run is one step behind and a recovery would repeat from there`,
    );
    return undefined;
  }
}

/** One correction, in the words an operator reading the log needs. */
function describe(correction: Correction): string {
  switch (correction.correction) {
    case "unbackedEffect":
      return (
        `the notes claimed ${correction.tool} had already been done to ` +
        `${JSON.stringify(correction.target)} and the record does not corroborate it; dropped`
      );
    case "effectWouldHaveBeenLost":
      return (
        `${correction.tool} on ${JSON.stringify(correction.target)} was already accepted and ` +
        `this commit no longer listed it; it was kept`
      );
    case "unbackedEvidence":
      return `the notes cited ${correction.marker}, which no search on this run handed out; dropped`;
    case "unbackedArtifact":
      return (
        `the notes named artifact ${JSON.stringify(correction.id)}, which this run did not ` +
        `produce; dropped`
      );
    case "unbackedCalculation":
      return (
        `the notes named calculation ${JSON.stringify(correction.id)}, which the engine did not ` +
        `issue; dropped`
      );
    case "milestonesAreNotProposable":
      return (
        `the proposal carried ${correction.offered ?? 0} milestone(s), which are written when a ` +
        `person signs one off; replaced`
      );
    default:
      return `the record disagreed with the notes: ${JSON.stringify(correction)}`;
  }
}

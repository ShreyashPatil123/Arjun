import { useEffect, useState } from 'react';
import {
  agentService,
  type AgentEventEnvelope,
  type CompactionRecord,
  type ContextLedgerRecord,
  type RunSummary,
} from '../../services/agent.service';
import {
  applyDurableEvent,
  applyLiveEvent,
  fromSnapshot,
  receive,
  IDLE,
  type Activity,
  type RunViewState,
} from './recovery';

/**
 * Adopt a single run by id, without going through `useRun`.
 *
 * The chat surface needs the activity list and the `RunViewState` for
 * one run at a time, and spinning up the full `useRun` for that would
 * also subscribe to durable/snapshot reconciliation for the *last* run
 * the user started. This module does the minimum: read the snapshot,
 * apply any events after it, subscribe to the live + durable channels
 * while the run is in flight, and hand the resulting `RunViewState`
 * back to the caller.
 */

export interface AdoptedRun {
  view: RunViewState;
  activity: Activity[];
}

export async function adoptRun(
  runId: string,
  onUpdate?: (next: AdoptedRun) => void,
): Promise<AdoptedRun | null> {
  const snapshot = await agentService.snapshot(runId).catch(() => null);
  if (!snapshot) return null;

  let state: RunViewState = fromSnapshot(snapshot);
  let activity: Activity[] = state.activity;
  const emit = () => onUpdate?.({ view: state, activity });
  emit();

  try {
    const page = await agentService.events(runId, snapshot.seq);
    for (const event of page.events) {
      state = applyDurableEvent(state, event);
      activity = state.activity;
    }
    emit();
  } catch {
    // Snapshot alone is fine.
  }

  // Live + durable subscribers. These do not block adoption: the run
  // may have finished before the chat surface opens the inspector,
  // and in that case the subscribers just no-op.
  const live = await agentService.subscribe(
    ({ runId: r, event }: AgentEventEnvelope) => {
      if (r !== runId) return;
      state = applyLiveEvent(state, event);
      activity = state.activity;
      emit();
    },
    runId,
  );
  const durable = await agentService.subscribeDurable(event => {
    if (event.runId !== runId) return;
    if (receive(state.seq, event.seq).action !== 'apply') return;
    state = applyDurableEvent(state, event);
    activity = state.activity;
    emit();
  }, runId);

  // The caller is expected to call this when unmounting. Returning the
  // teardown so the chat surface can wire it to its own effect.
  (state as unknown as { _adoptTeardown?: () => void })._adoptTeardown = () => {
    live();
    durable();
  };

  return { view: state, activity };
}

/**
 * React hook wrapper around `adoptRun`. Returns `null` when no run is
 * being adopted so the caller can render a placeholder cheaply.
 */
export function useAdoptedRun(runId: string | null) {
  const [state, setState] = useState<AdoptedRun | null>(null);

  useEffect(() => {
    if (!runId) {
      setState(null);
      return;
    }
    let cancelled = false;
    let teardown: (() => void) | null = null;
    void (async () => {
      const adopted = await adoptRun(runId, next => {
        if (cancelled) return;
        setState(next);
      });
      if (cancelled) return;
      if (adopted) {
        setState(adopted);
        teardown = (adopted.view as unknown as { _adoptTeardown?: () => void })
          ._adoptTeardown ?? null;
      }
    })();
    return () => {
      cancelled = true;
      teardown?.();
    };
  }, [runId]);

  return state;
}

/**
 * Fetch the `TaskRecord` for one run, in a hook. Used by the inspector
 * to read the final answer + plan + verification + artifacts that
 * `RunView` shows.
 */
export function useTaskRecord(runId: string | null) {
  const [record, setRecord] = useState<RunSummary | null>(null);
  useEffect(() => {
    if (!runId) {
      setRecord(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const task = await agentService.task(runId);
        if (cancelled) return;
        setRecord({
          runId: task.runId,
          text: task.answer,
          turns: task.turns,
          routing: task.routing,
          endpoint: task.endpoint,
          plan: task.plan,
          verification: task.verification,
          artifacts: task.artifacts,
        });
      } catch {
        if (!cancelled) setRecord(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);
  return record;
}

/**
 * Read the context ledger + the list of compactions for one run.
 * Used by the chat header's `ContextPanel` chip.
 */
export function useContextLedger(runId: string | null) {
  const [ledger, setLedger] = useState<ContextLedgerRecord | null>(null);
  const [compactions, setCompactions] = useState<CompactionRecord[]>([]);
  useEffect(() => {
    if (!runId) {
      setLedger(null);
      setCompactions([]);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const task = await agentService.task(runId);
        if (cancelled) return;
        setLedger(task.contextLedger ?? null);
        setCompactions(task.compactions ?? []);
      } catch {
        if (!cancelled) {
          setLedger(null);
          setCompactions([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);
  return { ledger, compactions };
}

/**
 * Re-export the `IDLE` state so a chat surface can use the same
 * defaults as `useRun` without reaching into the reducer module.
 */
export { IDLE };

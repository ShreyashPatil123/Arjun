import { useEffect, useRef, useState } from 'react';
import {
  agentService,
  listenAttachmentContext,
  type AgentEventEnvelope,
  type AttachmentContextEvent,
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
 * Activity for every run in a conversation, keyed by run id.
 *
 * The chat surface needs each assistant cell to show what its own run
 * actually did, which is more than `useAdoptedRun` was built for. Two
 * different costs are involved, so this hook pays them differently:
 *
 *  - The run in flight is adopted in full (snapshot, catch-up events,
 *    then live + durable subscriptions) so its rows appear as the tools
 *    run.
 *  - Runs that already finished are read once from their snapshot. No
 *    subscriptions, and `fetched` makes it once per run for the life of
 *    the surface rather than once per render.
 *
 * A snapshot that fails to load leaves the run's entry alone instead of
 * writing an empty list, so a transient backend error shows the rows we
 * already had rather than blanking the turn.
 */
export function useConversationActivity(
  runIds: string[],
  liveRunId: string | null,
): Map<string, Activity[]> {
  const [byRun, setByRun] = useState<Map<string, Activity[]>>(new Map());
  const fetched = useRef<Set<string>>(new Set());

  // `runIds` is a fresh array every render; the joined key is what
  // actually changes when the conversation gains or loses a run.
  const runKey = runIds.join(',');

  useEffect(() => {
    const pending = runIds.filter(
      id => id !== liveRunId && !fetched.current.has(id),
    );
    if (pending.length === 0) return;
    for (const id of pending) fetched.current.add(id);

    let cancelled = false;
    void (async () => {
      const results = await Promise.all(
        pending.map(async id => {
          const snapshot = await agentService.snapshot(id).catch(() => null);
          return [id, snapshot ? fromSnapshot(snapshot).activity : null] as const;
        }),
      );
      if (cancelled) return;
      setByRun(prev => {
        const next = new Map(prev);
        let changed = false;
        for (const [id, activity] of results) {
          if (!activity) continue;
          next.set(id, activity);
          changed = true;
        }
        return changed ? next : prev;
      });
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey, liveRunId]);

  useEffect(() => {
    if (!liveRunId) return;
    let cancelled = false;
    let teardown: (() => void) | null = null;
    void (async () => {
      const adopted = await adoptRun(liveRunId, next => {
        if (cancelled) return;
        setByRun(prev => new Map(prev).set(liveRunId, next.activity));
      });
      if (cancelled) return;
      if (adopted) {
        setByRun(prev => new Map(prev).set(liveRunId, adopted.activity));
        teardown =
          (adopted.view as unknown as { _adoptTeardown?: () => void })
            ._adoptTeardown ?? null;
      }
    })();
    return () => {
      cancelled = true;
      teardown?.();
    };
  }, [liveRunId]);

  return byRun;
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
          // Records written before the typed ending existed carry only the
          // failure sentence. Read back as `failed` when there is one and
          // `completed` when there is not — which is what the record actually
          // says, rather than a state it never recorded.
          outcome:
            task.outcome ??
            (task.failure
              ? { kind: 'failed', detail: task.failure }
              : { kind: 'completed' }),
          routing: task.routing,
          endpoint: task.endpoint,
          plan: task.plan,
          verification: task.verification,
          artifacts: task.artifacts,
          // The record was read back off disk, so the store this run needed
          // was working. That is a fact about the read that just succeeded,
          // not an assumption about the installation now.
          audit: { state: 'durable' },
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
  /**
   * Per-attachment costs, keyed by content hash.
   *
   * Held beside the ledger rather than inside it because the two arrive from
   * different places at different times: an attachment's cost is known while
   * the OCR model is still finishing and the run has not made a model call yet,
   * so there is no ledger to put it in.
   */
  const [attachments, setAttachments] = useState<AttachmentContextEvent[]>([]);

  useEffect(() => {
    if (!runId) {
      setLedger(null);
      setCompactions([]);
      setAttachments([]);
      return;
    }
    let cancelled = false;
    const unsubscribers: (() => void)[] = [];

    // The stored reading first, so a finished run opened from the Tasks screen
    // shows its ledger without waiting for events that will never come.
    void (async () => {
      try {
        const task = await agentService.task(runId);
        if (cancelled) return;
        // Only as a starting point. A live event that has already landed
        // describes a later moment than this fetch does, so it must not be
        // overwritten by a reply that was in flight when it arrived.
        setLedger(current => current ?? task.contextLedger ?? null);
        setCompactions(current => (current.length > 0 ? current : task.compactions ?? []));
      } catch {
        // A run with no stored record yet is the normal case for one that has
        // only just started. The live events below are what populate it, so
        // this is not an error worth clearing state for.
      }
    })();

    // Live: every turn and every compaction.
    void agentService
      .subscribe(({ event }: AgentEventEnvelope) => {
        if (cancelled) return;
        if (event.type === 'context_ledger') {
          setLedger(event.ledger);
        }
      }, runId)
      .then(un => {
        if (cancelled) un();
        else unsubscribers.push(un);
      });

    // Live: what each attached document cost, known before the first model
    // call and therefore before any ledger exists.
    void listenAttachmentContext(payload => {
      if (cancelled) return;
      setAttachments(current => {
        // Keyed by content hash, so re-reading the same file replaces its row
        // rather than adding a second one for the same document.
        const at = current.findIndex(a => a.sha256 === payload.sha256);
        if (at === -1) return [...current, payload];
        const next = current.slice();
        next[at] = payload;
        return next;
      });
    }).then(un => {
      if (cancelled) un();
      else unsubscribers.push(un);
    });

    return () => {
      cancelled = true;
      for (const un of unsubscribers) un();
    };
  }, [runId]);

  return { ledger, compactions, attachments };
}

/**
 * Re-export the `IDLE` state so a chat surface can use the same
 * defaults as `useRun` without reaching into the reducer module.
 */
export { IDLE };

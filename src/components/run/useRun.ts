import { useCallback, useEffect, useRef, useState } from 'react';
import {
  agentService,
  type AgentEvent,
  type Classification,
  type PlanRecord,
  type RunSummary,
} from '../../services/agent.service';

/**
 * Driving one agent run, and holding what it has done so far.
 *
 * The shape of this hook follows the shape of the backend call, which is
 * deliberately two halves: `agent_start_run` resolves once, at the end, with
 * the answer; everything in between arrives on the `agent://event` stream. So
 * the hook subscribes first, starts second, and treats the resolved summary as
 * the authority over anything the events implied.
 *
 * ## Why the events are not the source of truth
 *
 * An event can be dropped — the backend emits best-effort, so that a slow
 * listener cannot stall a run. That is the right trade, but it means a trace
 * built only from events may be missing a line. The summary is complete, so
 * when it lands it replaces the plan and supplies the artifacts and the
 * verification. Until then the events are the best available account, and the
 * interface says which of the two it is showing rather than letting the live
 * view pass for the final one.
 */

/** One thing the run did, in the order it did it. */
export interface Activity {
  id: string;
  tool: string;
  /** `running` until the call comes back. */
  status: 'running' | 'done' | 'failed' | 'refused';
}

export type RunPhase = 'idle' | 'starting' | 'running' | 'finished' | 'failed';

export interface RunState {
  phase: RunPhase;
  prompt: string;
  runId: string | null;
  plan: PlanRecord | null;
  activity: Activity[];
  /** Set when the plan stopped the run before the loop was done. */
  stopped: string | null;
  /** Times the context was summarised so the run could continue. */
  compactions: number;
  turns: number;
  summary: RunSummary | null;
  error: string | null;
  /** True while a correction is being applied, so the control can say so. */
  steering: boolean;
}

const IDLE: RunState = {
  phase: 'idle',
  prompt: '',
  runId: null,
  plan: null,
  activity: [],
  stopped: null,
  compactions: 0,
  turns: 0,
  summary: null,
  error: null,
  steering: false,
};

/** How a tool name reads in the trace. Follows `ToolName::describe` in Rust. */
const TOOL_LABELS: Record<string, string> = {
  search_documents: 'Searching the documents',
  read_scoped_file: 'Reading a file',
  write_scoped_file: 'Writing a file',
  run_calculation: 'Calculating',
  create_docx: 'Producing a Word document',
  create_xlsx: 'Producing a workbook',
  execute_code: 'Running code',
  validate_artifact: 'Checking a produced file',
};

export const labelFor = (tool: string) => TOOL_LABELS[tool] ?? tool;

export function useRun() {
  const [state, setState] = useState<RunState>(IDLE);

  /** Our run's id, for filtering events. In a ref because the subscriber
   *  closure is created once and has to see the current value rather than the
   *  one that existed when it was made. */
  const runIdRef = useRef<string | null>(null);
  const correlationRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    runIdRef.current = null;
    correlationRef.current = null;
    setState(IDLE);
  }, []);

  const apply = useCallback((event: AgentEvent) => {
    setState(previous => {
      switch (event.type) {
        case 'plan_ready':
          return { ...previous, plan: event.plan, phase: 'running' };

        case 'plan_step':
          // The plan's own count, not one this side keeps: a dropped event
          // would leave a locally incremented counter permanently wrong.
          return previous.plan
            ? { ...previous, plan: { ...previous.plan, stepsTaken: event.stepsTaken } }
            : previous;

        case 'plan_stopped':
          return { ...previous, stopped: event.reason };

        case 'turn_end':
          return { ...previous, turns: previous.turns + 1 };

        case 'context_compacted':
          return { ...previous, compactions: previous.compactions + 1 };

        case 'tool_execution_start':
          return {
            ...previous,
            activity: [
              ...previous.activity,
              { id: event.toolCallId, tool: event.toolName, status: 'running' },
            ],
          };

        case 'tool_execution_end': {
          // A call the gateway stopped before it ran is a different outcome
          // from one that ran and failed, and somebody reading the trace needs
          // to be able to tell them apart.
          const status: Activity['status'] = !event.isError
            ? 'done'
            : event.executionStarted === false
              ? 'refused'
              : 'failed';
          return {
            ...previous,
            activity: previous.activity.map(item =>
              item.id === event.toolCallId ? { ...item, status } : item,
            ),
          };
        }

        default:
          return previous;
      }
    });
  }, []);

  /** Subscribed for the life of the component rather than per run: the backend
   *  emits for the whole session, and re-subscribing between runs would drop
   *  whatever arrived in the gap. */
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    void agentService
      .subscribe(({ runId, event }) => {
        // Lock onto our run the first time it identifies itself, then ignore
        // everything else. Without this, a second window's run would write
        // into this one's trace.
        if (
          runIdRef.current === null &&
          event.type === 'plan_ready' &&
          event.correlationId &&
          event.correlationId === correlationRef.current
        ) {
          runIdRef.current = runId;
          setState(previous => ({ ...previous, runId }));
        }
        if (runIdRef.current !== runId) return;
        apply(event);
      })
      .then(fn => {
        // Unmounted before the listener was registered: tear it down at once
        // rather than leave it updating a component that is gone.
        if (cancelled) fn();
        else unlisten = fn;
      });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [apply]);

  const start = useCallback(async (prompt: string, classification?: Classification) => {
    const correlationId = crypto.randomUUID();
    correlationRef.current = correlationId;
    runIdRef.current = null;

    setState({ ...IDLE, phase: 'starting', prompt });

    try {
      const summary = await agentService.start({ prompt, classification, correlationId });
      // The summary is complete where the event stream is best-effort, so it
      // wins: the plan it carries is the one that was actually enforced.
      setState(previous => ({
        ...previous,
        phase: 'finished',
        runId: summary.runId,
        plan: summary.plan,
        turns: summary.turns,
        summary,
      }));
      return summary;
    } catch (error) {
      setState(previous => ({
        ...previous,
        phase: 'failed',
        error: error instanceof Error ? error.message : String(error),
      }));
      return null;
    }
  }, []);

  const abort = useCallback(async () => {
    const runId = runIdRef.current;
    if (!runId) return;
    // A run that finished just before the button was pressed resolves `false`.
    // That is an ordinary race rather than a failure, so nothing is surfaced.
    await agentService.abort(runId).catch(() => undefined);
  }, []);

  const steer = useCallback(async (text: string) => {
    const runId = runIdRef.current;
    if (!runId || !text.trim()) return false;
    setState(previous => ({ ...previous, steering: true }));
    try {
      return await agentService.steer(runId, text);
    } catch {
      return false;
    } finally {
      setState(previous => ({ ...previous, steering: false }));
    }
  }, []);

  return { state, start, abort, steer, reset };
}

/** Whether a run is still going, for disabling the composer. */
export const isBusy = (phase: RunPhase) => phase === 'starting' || phase === 'running';

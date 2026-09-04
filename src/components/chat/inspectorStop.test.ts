/**
 * The run inspector's Stop button has to stop the run.
 *
 * ## The defect
 *
 * The inspector rendered `RunView` with `onAbort={() => setInspectorRunId(null)}`.
 * `RunView` draws that callback under a button labelled **Stop**. So pressing
 * Stop closed the panel and did nothing else: the backend run carried on
 * holding a model server and burning its step budget, and the one piece of UI
 * that had been showing it was now gone. The gesture meaning "stop this work"
 * and the gesture meaning "hide this window" had become the same gesture, and
 * the more destructive reading was the one that did nothing.
 *
 * It is the worst shape a bug like this comes in, because the screen agrees
 * with the person: they pressed Stop, the run vanished from view, and every
 * visible signal said it had worked.
 *
 * ## Why these are source assertions
 *
 * `vitest.config.mjs` runs with `environment: 'node'` and this repository
 * deliberately vendors no DOM implementation, so a rendering test that clicked
 * the button is not available here. `src/contexts/activeRun.test.ts` already
 * pins a wiring invariant on this same file by reading it; that is the
 * precedent, and this follows it.
 *
 * These assertions are narrow on purpose. They do not check that the stop
 * *works* — only that the button is wired to something that sends an abort and
 * waits for the run to say it ended, rather than to a state setter. The
 * behaviour behind it is `agent_abort_run`, which is tested in the backend.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

/** The file with its comments removed, so prose cannot satisfy an assertion. */
function code(path: string): string {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter(line => {
      const trimmed = line.trim();
      return (
        !trimmed.startsWith('//') && !trimmed.startsWith('*') && !trimmed.startsWith('/*')
      );
    })
    .join('\n');
}

const CHAT_SURFACE = 'src/components/chat/ChatSurface.tsx';
const RUN_VIEW = 'src/components/run/RunView.tsx';

describe('the inspector Stop button stops the run', () => {
  it('does not hand RunView a callback that only closes the panel', () => {
    // The exact regression. `onNewTask` may still close the inspector — that
    // button says "New task", and closing is what it means.
    expect(code(CHAT_SURFACE)).not.toMatch(
      /onAbort=\{\(\)\s*=>\s*setInspectorRunId\(null\)\}/,
    );
  });

  it('sends an abort for the run being inspected', () => {
    const source = code(CHAT_SURFACE);
    expect(source).toContain('agentService.abort(runId)');
    // Scoped to the inspected run rather than whichever run is live: the
    // inspector can be opened on a run the composer is not driving.
    expect(source).toMatch(/const runId = inspectorRunId/);
  });

  it('waits for the run to report a terminal state', () => {
    // `agent_abort_run` resolving means the request was accepted, not that the
    // run ended. The acknowledgement is the run's own terminal state.
    expect(code(CHAT_SURFACE)).toContain('isTerminal');
  });

  it('does not treat a non-running phase as an acknowledgement', () => {
    // A run paused at a milestone gate is not running either. Waiting on
    // `phase` would report the work as over while it waits for a person.
    expect(code(CHAT_SURFACE)).not.toMatch(/stoppingRunId[\s\S]{0,200}view\.phase/);
  });

  it('reports a stop that is never acknowledged instead of waiting forever', () => {
    const source = code(CHAT_SURFACE);
    expect(source).toContain('STOP_ACKNOWLEDGEMENT_TIMEOUT_MS');
    expect(source).toContain('setStopProblem');
    // The message has to be shown, not merely held in state.
    expect(source).toMatch(/\{stopProblem &&/);
  });
});

describe('the button says which of the two moments it is in', () => {
  it('accepts a stopping flag and disables itself while it waits', () => {
    const source = code(RUN_VIEW);
    expect(source).toMatch(/stopping\?:\s*boolean/);
    expect(source).toContain('disabled={stopping}');
    expect(source).toMatch(/stopping \? 'Stopping/);
  });

  it('is told whether the run it is showing is the one being stopped', () => {
    // Keyed by id: switching the inspector to another run must not leave a
    // second run's button reading "Stopping".
    expect(code(CHAT_SURFACE)).toContain('stopping={stoppingRunId === inspectorRunId}');
  });
});

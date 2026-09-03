/**
 * One run, shared by every surface that shows it.
 *
 * ## Why this exists
 *
 * `useRun` is a hook, so every caller gets its own state: its own `runIdRef`,
 * its own event subscription, its own filter on the correlation id it issued.
 * Three pages called it independently — the demonstrator, the tasks list, and
 * the SIH dashboard.
 *
 * That is fine for the two that *start* runs. It is exactly wrong for the one
 * that only watches. `SIHDashboard` called `useRun()` and never called `start`,
 * so its instance had no run id and filtered every event away. Its routing,
 * plan, activity, verification and security panes showed the idle state for the
 * whole of a run somebody had launched from chat two panes over — and a panel
 * that renders "idle" honestly is indistinguishable from one wired to nothing
 * at all.
 *
 * So the instance is hoisted. One `useRun` lives here, at the provider, and
 * every surface reads it. A run started from the demonstrator, from a rerun, or
 * adopted from the chat is the same run in all of them, and the run id they
 * show is one id rather than three.
 *
 * ## How a chat-launched run gets in
 *
 * `ConversationContext` owns its own send path and issues its own run id — it
 * has to, because it reserves the assistant cell before the backend replies.
 * It publishes that id here through {@link ActiveRunValue.follow}, and this
 * provider adopts it: reads the run back off its durable record and then
 * follows its events like any other. The chat is the authority for what it
 * started; this is the authority for what everything else displays.
 */

import React, { createContext, useContext, useMemo } from 'react';
import { useRun, type RunViewState } from '../components/run/useRun';

export interface ActiveRunValue {
  /** The run every surface displays. Idle until something starts one. */
  state: RunViewState;
  /**
   * The run id every pane must agree on, or null when nothing is running.
   *
   * Read this rather than each pane keeping its own: two ids on one screen is
   * the symptom the whole provider exists to remove.
   */
  runId: string | null;
  /** Starts a run and makes it the active one. */
  start: ReturnType<typeof useRun>['start'];
  abort: ReturnType<typeof useRun>['abort'];
  steer: ReturnType<typeof useRun>['steer'];
  reset: ReturnType<typeof useRun>['reset'];
  /**
   * Follows a run this provider did not start.
   *
   * The chat surface issues its own run id before the backend has replied,
   * because it needs one to route streaming events. Handing it here is what
   * makes the dashboard show the run the person is actually watching.
   */
  follow: (runId: string) => Promise<void>;
}

const ActiveRunContext = createContext<ActiveRunValue | null>(null);

export function ActiveRunProvider({ children }: { children: React.ReactNode }) {
  const { state, start, abort, steer, reset, adopt } = useRun();

  const value = useMemo<ActiveRunValue>(
    () => ({
      state,
      runId: state.runId,
      start,
      abort,
      steer,
      reset,
      follow: async (runId: string) => {
        await adopt(runId);
      },
    }),
    [state, start, abort, steer, reset, adopt],
  );

  return <ActiveRunContext.Provider value={value}>{children}</ActiveRunContext.Provider>;
}

/**
 * What each pane of a run surface displays, derived from one state.
 *
 * The panes on the SIH dashboard — routing, plan, activity, verification,
 * security — used to be independent only in the sense that they read different
 * fields; the state they read them from was a `useRun` instance private to that
 * page, which never started a run and so never had one. Deriving them here, in
 * one function over one state, is what makes "all five agree" a property rather
 * than a coincidence.
 *
 * Every entry is the run id that pane is describing. They are all the same id
 * by construction, which is the point: a screen showing two run ids is showing
 * one pane's answer about another pane's run.
 */
export interface PaneView {
  /** The run this pane is describing, or null when nothing is running. */
  runId: string | null;
  /** Whether this pane has anything of that run's to show yet. */
  hasData: boolean;
}

export function paneViews(state: RunViewState): {
  routing: PaneView;
  plan: PaneView;
  activity: PaneView;
  verification: PaneView;
  security: PaneView;
} {
  // Read once, from the state handed in. A pane that reached for a source of
  // its own is how they came to describe different runs.
  const runId = state.runId;
  return {
    routing: { runId, hasData: state.summary !== null },
    plan: { runId, hasData: state.plan !== null },
    activity: { runId, hasData: state.activity.length > 0 },
    verification: { runId, hasData: Boolean(state.summary?.verification) },
    // The security pane reads the sovereignty broker rather than the run, but
    // it is captioned with the run it is shown beside — so it belongs to the
    // same run as everything else on the screen.
    security: { runId, hasData: true },
  };
}

/**
 * The one active run.
 *
 * Throws outside the provider rather than returning a private instance. A
 * silent fallback is what let the dashboard look wired while showing nothing,
 * and a component that renders run state outside the provider is a bug worth
 * failing loudly for.
 */
export function useActiveRun(): ActiveRunValue {
  const value = useContext(ActiveRunContext);
  if (!value) {
    throw new Error(
      'useActiveRun must be used inside an ActiveRunProvider. Every surface that shows a run ' +
        'reads the same one; a private instance would show a different run from the pane next ' +
        'to it.',
    );
  }
  return value;
}

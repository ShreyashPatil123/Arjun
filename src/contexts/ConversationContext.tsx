/**
 * Shared conversation state.
 *
 * The chat surface and the app shell both need to know what conversation is
 * active and which assistant cell is currently streaming. When those two
 * surfaces called `useConversation()` independently, each got its OWN
 * `useState` slice — different `conversation`, different `streamingContent`,
 * different `isStreaming` — so the header and the message list drifted out
 * of sync. The header would say "1 message" while the body rendered four.
 *
 * The fix is to host the state in a single `ConversationProvider` near the
 * root of the tree and have every caller read it through this context. The
 * reducer for a streaming run lives inside the provider as a per-run
 * closure so events for one run never touch another run's cell. See the
 * comment on `RunReducer` for the per-run isolation contract.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  agentService,
  listenAttachmentProgress,
  type AgentEvent,
  type AgentEventEnvelope,
  type ChatMessage,
  type Classification,
  type Conversation,
  type RunSummary,
  type ComposerAttachment,
} from '../services/agent.service';
import type { OcrDetent } from '../services/ocr.service';
import {
  applyProgress,
  type ProgressInput,
  type ProgressStep,
} from '../components/chat/runProgress';

const LAST_CONVERSATION_KEY = 'arjun.conversation.last';

function rememberConversation(id: string | null) {
  try {
    if (id) sessionStorage.setItem(LAST_CONVERSATION_KEY, id);
    else sessionStorage.removeItem(LAST_CONVERSATION_KEY);
  } catch {
    // A browser with storage disabled loses reattachment across a reload and
    // nothing else. Not worth failing a chat over.
  }
}

function lastConversation(): string | null {
  try {
    return sessionStorage.getItem(LAST_CONVERSATION_KEY);
  } catch {
    return null;
  }
}

function newMessageId(role: 'user' | 'assistant'): string {
  const prefix = role === 'user' ? 'u' : 'a';
  return `${prefix}-${crypto.randomUUID()}`;
}

const STREAMING_PERSIST_DEBOUNCE_MS = 400;
const STREAMING_MIRROR_DEBOUNCE_MS = 30;

export type MessageEvent = Extract<
  AgentEvent,
  { type: 'message_start' | 'message_update' | 'message_end' }
>;

export function isMessageEvent(event: AgentEvent): event is MessageEvent {
  return (
    event.type === 'message_start' ||
    event.type === 'message_update' ||
    event.type === 'message_end'
  );
}

/**
 * Events that describe progress rather than content.
 *
 * Both carry the assistant `messageId`, which is what lets a reducer accept
 * one before the server has issued its own run id without ever accepting
 * another run's.
 */
export type ProgressEvent = Extract<
  AgentEvent,
  { type: 'run_stage' | 'model_thinking' }
>;

export function isProgressEvent(event: AgentEvent): event is ProgressEvent {
  return event.type === 'run_stage' || event.type === 'model_thinking';
}

/** The `messageId` an event names, or null if it names none. */
export function eventMessageId(event: AgentEvent): string | null {
  if (isMessageEvent(event)) return event.messageId;
  if (event.type === 'model_thinking') return event.messageId;
  if (event.type === 'run_stage') {
    return typeof event.messageId === 'string' ? event.messageId : null;
  }
  return null;
}

/**
 * The streaming reducer for a single run.
 *
 * Each `send()` creates exactly one `RunReducer`. The reducer holds the
 * per-run state — messageId, conversationId, runId, live content, persist
 * timer — as plain fields on a normal object, not as React state and not
 * as module-level refs. Two runs in flight at once do not interfere
 * because each one only reads and writes its own fields.
 *
 * The reducer is wired to the event channel via a Tauri listener that
 * forwards envelopes through `envelope.runId`; the listener consults the
 * registry of `RunReducer` instances and routes each envelope to the
 * reducer whose `runId` or `messageId` it matches. Envelopes that match
 * no run are dropped.
 */
export class RunReducer {
  /** Whether the live content buffer has unsent deltas. */
  private dirty = false;
  /** Local content buffer, in arrival order, with repeats collapsed. */
  private content = '';
  /** Mirror debounce timer (pushes content to React state at a throttled rate). */
  private mirrorTimer: number | null = null;
  /** Persist debounce timer (writes the latest snapshot to disk). */
  private persistTimer: number | null = null;
  /** Whether `message_end` has been received. The reducer accepts at most
   * one of these for the lifetime of the run. */
  private receivedEnd = false;
  /** Server-issued runId, captured from the first `plan_ready` envelope. */
  private actualRunId: string | null = null;
  /** Timestamp the run started, used for elapsed-time metrics. */
  readonly startedAt: number;
  /** Set when `dispose()` runs. Late events are dropped on the floor. */
  private disposed = false;
  /**
   * The steps this turn has been through, newest last.
   *
   * Held on the reducer rather than in React state for the same reason the
   * content buffer is: two runs in flight each own their own list, and an
   * event that matched neither cannot append to either.
   */
  private progress: ProgressStep[] = [];

  constructor(
    private readonly registry: RunReducerRegistry,
    readonly runId: string,
    readonly conversationId: string,
    readonly messageId: string,
  ) {
    this.startedAt = Date.now();
    // Seeded before any event arrives, so the cell has a line to show during
    // the IPC round trip that reserves the turn. It names work that is
    // genuinely under way — the request has been handed over — rather than
    // filling the gap with a spinner that means nothing.
    this.pushProgress({ kind: 'submitted' });
  }

  /**
   * Folds in progress that arrived on a channel other than `agent://event`.
   *
   * The document reader reports its pages on `attachment:progress`, which is
   * an application-wide channel; the registry routes one of those to this
   * reducer only when it names this reducer's message. A read that names no
   * message reaches nobody, which is the safe direction to fail.
   */
  ingest(input: ProgressInput): void {
    if (this.disposed) return;
    this.pushProgress(input);
  }

  /** Folds one progress event in and publishes the new list. */
  private pushProgress(input: ProgressInput): void {
    const next = applyProgress(this.progress, input, Date.now());
    if (next === this.progress) return;
    this.progress = next;
    this.registry.publishProgress(this.messageId, next);
  }

  /** Push the latest content into React state on the next microtask. */
  private scheduleMirror(): void {
    if (this.mirrorTimer !== null) return;
    this.mirrorTimer = window.setTimeout(() => {
      this.mirrorTimer = null;
      this.registry.publishContent(this.messageId, this.content);
    }, STREAMING_MIRROR_DEBOUNCE_MS);
  }

  /** Persist the current content snapshot to the conversation store. */
  private schedulePersist(): void {
    this.dirty = true;
    if (this.persistTimer !== null) return;
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      if (!this.dirty) return;
      this.dirty = false;
      void agentService
        .updateStreamingContent(
          this.conversationId,
          this.messageId,
          this.content,
        )
        .then((next) => {
          if (next) this.registry.publishConversation(next);
        })
        .catch(() => undefined);
    }, STREAMING_PERSIST_DEBOUNCE_MS);
  }

  /** Force any pending persist to run now. Used on `message_end`. */
  private flushPersist(): void {
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    if (!this.dirty) return;
    this.dirty = false;
    void agentService
      .updateStreamingContent(
        this.conversationId,
        this.messageId,
        this.content,
      )
      .then((next) => {
        if (next) this.registry.publishConversation(next);
      })
      .catch(() => undefined);
  }

  /**
   * Apply one envelope to the reducer. Returns true if the envelope was
   * consumed (i.e. matched this run), false if it should be passed on to
   * any other reducers in the registry.
   */
  apply(envelope: AgentEventEnvelope): boolean {
    if (this.disposed) return false;
    const event = envelope.event;

    // `plan_ready` is the one envelope that teaches this reducer the
    // server's run id, by echoing back the correlation id the caller sent.
    if (event.type === 'plan_ready') {
      if (event.correlationId === this.runId) {
        this.actualRunId = envelope.runId;
        return true;
      }
      return this.actualRunId !== null && envelope.runId === this.actualRunId;
    }

    // Two ids can identify this run, and both are needed.
    //
    // The server's run id is authoritative once `plan_ready` has taught it
    // to us. Before that — which is most of a cold turn, because the stages
    // that report attachment reading, routing and model loading all happen
    // before the run has an id — the envelope carries the caller's own
    // correlation id, which is this reducer's `runId`. Matching only the
    // first would drop every stage emitted before generation started, which
    // is exactly the silence this work exists to remove.
    const namedMessage = eventMessageId(event);
    const matchesRun =
      (this.actualRunId !== null && envelope.runId === this.actualRunId) ||
      envelope.runId === this.runId;
    const matchesMessage =
      namedMessage !== null && namedMessage === this.messageId;
    if (!matchesRun && !matchesMessage) return false;

    // An event that names a message must name OURS, whatever run it claims.
    // Without this, a run holding two assistant cells would let the second
    // cell's tokens land in the first.
    if (namedMessage !== null && namedMessage !== this.messageId) return false;

    // A message event is stamped with the server's run id even before
    // `plan_ready` arrives, so a fast model whose first `message_start`
    // beats the plan still teaches us the id.
    if (
      this.actualRunId === null &&
      matchesMessage &&
      envelope.runId !== this.runId
    ) {
      this.actualRunId = envelope.runId;
    }

    if (isProgressEvent(event)) {
      if (event.type === 'model_thinking') {
        this.pushProgress({
          kind: 'thinking',
          state: event.state,
          characters: event.characters,
          elapsedMs: event.elapsedMs,
        });
      } else {
        const { type, stage, elapsedMs, ...detail } = event;
        void type;
        this.pushProgress({
          kind: 'stage',
          stage,
          elapsedMs,
          detail: detail as Record<string, unknown>,
        });
      }
      return true;
    }

    if (!isMessageEvent(event)) return true; // consume but no-op

    switch (event.type) {
      case 'message_start':
        this.content = '';
        this.registry.publishContent(this.messageId, this.content);
        return true;
      case 'message_update': {
        const next = this.content + event.delta;
        this.content = collapseRepeats(next, /[.!?]/.test(event.delta));
        // The first visible token is the only place composition can be
        // observed starting. No stage on the Rust side can know it, because
        // the Rust side is blocked on the loop by then.
        if (this.content.length > 0) this.pushProgress({ kind: 'text' });
        this.scheduleMirror();
        this.schedulePersist();
        return true;
      }
      case 'message_end': {
        if (this.receivedEnd) return true;
        this.receivedEnd = true;
        // Closes whatever was still open. A panel left showing an unfinished
        // "Writing the answer" under a finished answer is the same class of
        // lie as showing nothing at all.
        this.pushProgress({ kind: 'done' });
        if (this.mirrorTimer !== null) {
          window.clearTimeout(this.mirrorTimer);
          this.mirrorTimer = null;
        }
        this.registry.publishContent(this.messageId, this.content);
        this.flushPersist();
        const elapsed = Date.now() - this.startedAt;
        const runId = this.actualRunId ?? this.runId;
        const finalContent = this.content;
        void agentService
          .completeMessage({
            conversationId: this.conversationId,
            messageId: this.messageId,
            runId,
            finalContent,
            elapsedMs: elapsed,
            failed: false,
            tokensIn: event.tokensIn,
            tokensOut: event.tokensOut,
          })
          .then((next) => {
            if (next) this.registry.publishConversation(next);
            this.registry.onRunDone(this);
          })
          .catch(() => {
            this.registry.onRunDone(this);
          });
        return true;
      }
    }
  }

  /**
   * Tear the reducer down. Used when the run completes, fails, or is
   * cancelled — the reducer is removed from the registry so its memory
   * can be reclaimed and so the event listener stops trying to route
   * to it.
   */
  dispose(): void {
    this.disposed = true;
    if (this.mirrorTimer !== null) {
      window.clearTimeout(this.mirrorTimer);
      this.mirrorTimer = null;
    }
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
  }
}

/**
 * The registry of currently-active run reducers.
 *
 * Holds the `Map<runId, RunReducer>` and routes events to the right
 * reducer. Also publishes state changes (content updates, conversation
 * updates) up to the React context so the UI re-renders.
 */
export class RunReducerRegistry {
  private readonly reducers = new Map<string, RunReducer>();
  private readonly byMessageId = new Map<string, RunReducer>();
  private readonly unsubscribers = new Map<string, () => void>();
  private activeUnlisten: (() => void) | null = null;

  constructor(
    private readonly publish: {
      onContent: (messageId: string, content: string) => void;
      onProgress: (messageId: string, steps: ProgressStep[]) => void;
      onConversation: (next: Conversation) => void;
      onRunDone: (reducer: RunReducer) => void;
    },
  ) {}

  /**
   * Register a new run and subscribe to the event channel. The returned
   * function tears the registration down.
   */
  register(reducer: RunReducer): () => void {
    this.reducers.set(reducer.runId, reducer);
    this.byMessageId.set(reducer.messageId, reducer);
    if (this.reducers.size === 1) {
      this.activeUnlisten = this.subscribe();
    }
    let disposed = false;
    return () => {
      if (disposed) return;
      disposed = true;
      this.reducers.delete(reducer.runId);
      this.byMessageId.delete(reducer.messageId);
      const unlisten = this.unsubscribers.get(reducer.runId);
      if (unlisten) {
        this.unsubscribers.delete(reducer.runId);
        unlisten();
      }
      reducer.dispose();
      if (this.reducers.size === 0 && this.activeUnlisten) {
        this.activeUnlisten();
        this.activeUnlisten = null;
      }
    };
  }

  /** Per-reducer subscriber: each reducer gets its own listener so it
   * owns its filter logic in one place. */
  private subscribe(): () => void {
    let stopped = false;
    let unlisten: (() => void) | null = null;
    void agentService.subscribe((envelope) => {
      if (stopped) return;
      // Try the runId-keyed reducer first, then fall back to the
      // messageId-keyed one (for envelopes that arrive before the
      // server has stamped its own runId, like the first
      // `message_start` from a fast-streaming model).
      let reducer = this.reducers.get(envelope.runId);
      if (!reducer) {
        // The table is keyed by the caller's own run id, so every envelope
        // stamped with the *server's* id misses it. Both message events and
        // progress events name the assistant message, and that name is what
        // finds the reducer in that case.
        const named = eventMessageId(envelope.event);
        if (named !== null) reducer = this.byMessageId.get(named);
      }
      if (!reducer) return;
      reducer.apply(envelope);
    }).then((fn) => {
      if (stopped) fn();
      else unlisten = fn;
    });
    return () => {
      stopped = true;
      if (unlisten) unlisten();
    };
  }

  publishContent(messageId: string, content: string): void {
    this.publish.onContent(messageId, content);
  }

  publishProgress(messageId: string, steps: ProgressStep[]): void {
    this.publish.onProgress(messageId, steps);
  }

  /** Hands an off-channel progress event to the reducer that owns it. */
  ingestForMessage(messageId: string, input: ProgressInput): void {
    this.byMessageId.get(messageId)?.ingest(input);
  }

  publishConversation(next: Conversation): void {
    this.publish.onConversation(next);
  }

  onRunDone(reducer: RunReducer): void {
    this.publish.onRunDone(reducer);
    // The reducer already wrote the final state. The caller (the
    // reducer itself) is responsible for unregistering.
  }

  /** Iterate over the active reducers. Used for tests and cleanup. */
  values(): IterableIterator<RunReducer> {
    return this.reducers.values();
  }
}

/**
 * Collapse runs of repeated content into a single instance.
 *
 * Small models tend to fall into loops and emit the same sentence two or
 * three times in a row. The raw stream faithfully reflects the model,
 * but the duplication makes the answer look broken in the UI.
 *
 * Two passes:
 *  1. Sentence-level: if the last N characters end with the same
 *     sentence that already appears earlier, strip the earlier copy.
 *  2. Word-level: if the last 200 chars contain an immediate repeat
 *     of a chunk (8..200 chars), strip one copy.
 */
function collapseRepeats(s: string, deltaHadTerminator = true): string {
  if (s.length < 20) return s;

  // The sentence pass is the expensive half: it builds a `RegExp` and runs it
  // over the *whole* accumulated answer, so running it on every delta makes
  // the cost of streaming quadratic in the length of the answer. At a few
  // hundred characters that is invisible; at ten thousand it is the reducer
  // stuttering the stream it exists to smooth.
  //
  // It is also unnecessary. The pass can only find a new duplicate once a new
  // sentence has been *completed*, and that cannot happen in a delta with no
  // terminator in it. Skipping those is not an approximation — the result is
  // the same string, arrived at without re-scanning the answer a thousand
  // times. The word-level pass below is already bounded to the last 400
  // characters and runs every time.
  if (!deltaHadTerminator) return collapseTail(s);

  const sentenceRe = /[.!?]\s+([^.!?]{8,200})[.!?]/g;
  const tail1 = s.slice(-500);
  const tailSentences: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = sentenceRe.exec(tail1)) !== null) {
    tailSentences.push(m[1]);
  }
  if (tailSentences.length > 0) {
    const last = tailSentences[tailSentences.length - 1];
    const escaped = last.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const dupRe = new RegExp(
      `(?:^|[.!?]\\s+)${escaped}[.!?](?:\\s+${escaped}[.!?])+`,
      'g',
    );
    if (dupRe.test(s)) {
      return s.replace(dupRe, (match) => {
        const parts = match.split(
          new RegExp(`(?<=[.!?])\\s+(?=${escaped}[.!?])`),
        );
        return parts[parts.length - 1] ?? match;
      });
    }
  }

  return collapseTail(s);
}

/**
 * The cheap half: strip an immediate repeat of the last 8..200 characters.
 *
 * Bounded to the last 400 characters, so its cost does not grow with the
 * length of the answer.
 */
function collapseTail(s: string): string {
  const tail = s.slice(-400);
  for (let len = Math.min(200, Math.floor(tail.length / 2)); len >= 8; len -= 1) {
    const last = tail.slice(-len);
    const before = tail.slice(-len * 2, -len);
    if (last === before) {
      return s.slice(0, s.length - len);
    }
  }
  return s;
}

export interface UseConversation {
  conversation: Conversation | null;
  conversations: Conversation[];
  isStreaming: boolean;
  activeMessageId: string | null;
  activeRunId: string | null;
  streamingContents: Map<string, string>;
  /**
   * What each in-flight turn is doing, keyed by assistant `messageId`.
   *
   * Keyed by message rather than by run because that is what the cell that
   * renders it is keyed by, and a map from the thing rendering to the thing
   * rendered cannot put one turn's progress under another's answer.
   */
  progressByMessage: Map<string, ProgressStep[]>;
  send: (
    prompt: string,
    classification?: Classification,
    options?: {
      systemPrompt?: string;
      attachments?: ComposerAttachment[];
      ocrDetent?: OcrDetent;
    },
  ) => Promise<void>;
  open: (conversationId: string) => Promise<void>;
  newConversation: () => Promise<void>;
  refresh: () => Promise<void>;
  complete: (args: {
    finalContent: string;
    elapsedMs: number;
    modelName?: string;
    modelRole?: string;
    usedFallback?: boolean;
    error?: string;
    failed?: boolean;
  }) => Promise<void>;
  replay: (userMessage: ChatMessage) => Promise<void>;
}

const ConversationContext = createContext<UseConversation | null>(null);

export function ConversationProvider({ children }: { children: React.ReactNode }) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [streamingContents, setStreamingContents] = useState<Map<string, string>>(
    () => new Map(),
  );
  const [progressByMessage, setProgressByMessage] = useState<
    Map<string, ProgressStep[]>
  >(() => new Map());

  /**
   * A ref-based lock so two rapid `send()` invocations cannot both
   * proceed. `isStreaming` lives in React state and is one tick behind
   * the truth; a ref check inside `send` is the only synchronous way
   * to catch a double-press. The lock is released in `send`'s `finally`
   * regardless of how `start()` resolves.
   */
  const sendingRef = useRef(false);

  // The registry holds the per-run reducers and the shared event
  // subscription. The set of callbacks it calls is stable for the
  // lifetime of the provider.
  const registryRef = useRef<RunReducerRegistry | null>(null);
  if (registryRef.current === null) {
    registryRef.current = new RunReducerRegistry({
      onContent: (messageId, content) => {
        setStreamingContents((prev) => {
          const next = new Map(prev);
          next.set(messageId, content);
          return next;
        });
      },
      onProgress: (messageId, steps) => {
        setProgressByMessage((prev) => {
          const next = new Map(prev);
          next.set(messageId, steps);
          return next;
        });
      },
      onConversation: (next) => {
        setConversation(next);
      },
      onRunDone: () => {
        // The reducer handles its own teardown. We just mark the run
        // as no longer streaming if no other runs are active.
        if (registryRef.current && registryRef.current.values().next().done) {
          setIsStreaming(false);
          setActiveMessageId(null);
          setActiveRunId(null);
        }
      },
    });
  }

  // The document reader's own page counter, folded into the same step list
  // as everything else so a person sees one account of the turn rather than
  // two. Subscribed once for the life of the provider; the routing is by the
  // message id the reader stamps, never by which turn happens to be newest.
  useEffect(() => {
    const sub = listenAttachmentProgress((payload) => {
      const registry = registryRef.current;
      if (!registry || !payload.messageId) return;
      registry.ingestForMessage(payload.messageId, {
        kind: 'attachmentPage',
        name: payload.name,
        page: payload.page,
        pages: payload.pages,
        phase: payload.phase,
      });
    });
    return () => {
      void sub.then((un) => un());
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const list = await agentService.listConversations();
      setConversations(list);
    } catch {
      // A failing list call should not break the chat.
    }
  }, []);

  const reloadActive = useCallback(async (id: string | null) => {
    if (!id) {
      setConversation(null);
      return;
    }
    try {
      const next = await agentService.getConversation(id);
      setConversation(next);
    } catch {
      setConversation(null);
    }
  }, []);

  const open = useCallback(
    async (conversationId: string) => {
      rememberConversation(conversationId);
      await reloadActive(conversationId);
      await refresh();
    },
    [reloadActive, refresh],
  );

  const newConversation = useCallback(async () => {
    const created = await agentService.createConversation('New conversation');
    rememberConversation(created.id);
    await reloadActive(created.id);
    await refresh();
  }, [refresh, reloadActive]);

  const send = useCallback(
    async (
      prompt: string,
      classification?: Classification,
      options?: {
        systemPrompt?: string;
        attachments?: ComposerAttachment[];
        /** Where the accuracy-to-speed slider was left. Reading only. */
        ocrDetent?: OcrDetent;
      },
    ) => {
      // A turn with an attachment and no words is still a turn: "read this"
      // is implied by attaching it. Refusing here is how an attached document
      // silently goes nowhere.
      if (!prompt.trim() && !(options?.attachments ?? []).length) return;
      if (sendingRef.current) return;
      sendingRef.current = true;
      const registry = registryRef.current;
      if (!registry) {
        sendingRef.current = false;
        return;
      }

      let conv = conversation;
      if (!conv) {
        const title =
          prompt
            .split('\n')
            .map((line) => line.trim())
            .find((line) => line.length > 0) ?? 'New conversation';
        const created = await agentService.createConversation(title.slice(0, 80));
        conv = created;
        rememberConversation(created.id);
      }

      const runId = crypto.randomUUID();
      const messageId = newMessageId('assistant');

      // Reserve the user message and the assistant cell on the
      // conversation, and bind the run id so the runtime can route
      // streaming events to the right cell. Each call is its own
      // server-issued `messageId`, and the per-run reducer captures
      // the id so subsequent events only update THIS cell.
      const updated = await agentService.appendTurn(
        conv.id,
        runId,
        messageId,
        prompt,
      );
      if (updated) {
        setConversation(updated);
        // Clear any stale streaming content for this new cell.
        setStreamingContents((prev) => {
          if (!prev.has(messageId)) return prev;
          const next = new Map(prev);
          next.set(messageId, '');
          return next;
        });
      }
      await refresh();

      setIsStreaming(true);
      setActiveMessageId(messageId);
      setActiveRunId(runId);

      // Each run gets its OWN reducer. Two concurrent runs would each
      // have one, with no shared refs and no cross-contamination.
      const reducer = new RunReducer(registry, runId, conv.id, messageId);
      const unregister = registry.register(reducer);

      let failure: Error | null = null;
      try {
        const summary: RunSummary | null = await agentService.start({
          prompt,
          classification,
          systemPrompt: options?.systemPrompt,
          // Bound to THIS request. The backend keeps nothing between runs, so
          // a later turn cannot inherit this turn's document.
          attachments: options?.attachments,
          ocrDetent: options?.ocrDetent,
          conversationId: conv.id,
          messageId,
          correlationId: runId,
        });
        // `start` returning is the authoritative end-of-run signal.
        // The reducer's `message_end` handler will also have fired by
        // here in the normal path; resetting state unconditionally
        // covers runs whose summary lacks a `messageId` (refused,
        // tool-only, transient failure).
        if (summary?.messageId) {
          const next = await agentService
            .getConversation(conv.id)
            .catch(() => null);
          if (next) setConversation(next);
          await refresh();
        }
      } catch (error) {
        failure = error instanceof Error ? error : new Error(String(error));
      } finally {
        // The reducer may have already torn itself down on
        // `message_end`; if not (the run was cancelled or the
        // connection failed), tear it down now. Either way, the
        // unregister callback is idempotent.
        unregister();
        setIsStreaming(false);
        setActiveMessageId(null);
        setActiveRunId(null);
        sendingRef.current = false;
        if (failure) {
          const message = failure.message;
          await agentService
            .completeMessage({
              conversationId: conv.id,
              messageId,
              runId,
              finalContent: '',
              elapsedMs: 0,
              error: message,
              failed: true,
            })
            .catch(() => undefined);
          const next = await agentService
            .getConversation(conv.id)
            .catch(() => null);
          if (next) setConversation(next);
        }
      }
    },
    [conversation, refresh],
  );

  const complete = useCallback(
    async (args: {
      finalContent: string;
      elapsedMs: number;
      modelName?: string;
      modelRole?: string;
      usedFallback?: boolean;
      error?: string;
      failed?: boolean;
    }) => {
      if (!conversation || !activeMessageId || !activeRunId) return;
      const next = await agentService
        .completeMessage({
          conversationId: conversation.id,
          messageId: activeMessageId,
          runId: activeRunId,
          finalContent: args.finalContent,
          elapsedMs: args.elapsedMs,
          modelName: args.modelName,
          modelRole: args.modelRole,
          usedFallback: args.usedFallback,
          error: args.error,
          failed: args.failed ?? false,
        })
        .catch(() => null);
      if (next) setConversation(next);
      await refresh();
      setIsStreaming(false);
      setActiveMessageId(null);
      setActiveRunId(null);
    },
    [conversation, activeMessageId, activeRunId, refresh],
  );

  const replay = useCallback(
    async (userMessage: ChatMessage) => {
      if (userMessage.role !== 'user') return;
      if (!conversation) return;
      await send(userMessage.content);
    },
    [conversation, send],
  );

  // Restore the last conversation on mount. Falls back to a fresh one
  // if the remembered id is gone.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const remembered = lastConversation();
      if (remembered) {
        const conv = await agentService.getConversation(remembered);
        if (!cancelled) {
          if (conv) {
            setConversation(conv);
          } else {
            rememberConversation(null);
          }
        }
      }
      await refresh();
      if (cancelled) return;
      if (!conversation) {
        const created = await agentService.createConversation('New conversation');
        if (cancelled) return;
        rememberConversation(created.id);
        setConversation(created);
        await refresh();
      }
    })();

    const onTrigger = (event: Event) => {
      const detail = (event as CustomEvent<{
        prompt: string;
        title?: string;
        systemPrompt?: string;
        classification?: Classification;
      }>).detail;
      if (!detail || !detail.prompt) return;
      void (async () => {
        if (detail.title) {
          const created = await agentService.createConversation(detail.title);
          rememberConversation(created.id);
          await reloadActive(created.id);
          await refresh();
        }
        await send(detail.prompt, detail.classification, {
          systemPrompt: detail.systemPrompt,
        });
      })();
    };
    window.addEventListener('arjun:trigger-send', onTrigger);

    return () => {
      cancelled = true;
      window.removeEventListener('arjun:trigger-send', onTrigger);
    };
    // Mount-only effect; the actions are intentionally not in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drop streaming content entries for messages that no longer exist in
  // the conversation. When a conversation is reloaded (e.g. after a
  // remount) the persisted message set may have shrunk; keeping stale
  // streaming entries around would let the wrong cell render.
  useEffect(() => {
    if (!conversation) return;
    setStreamingContents((prev) => {
      if (prev.size === 0) return prev;
      const liveIds = new Set(
        conversation.messages
          .filter((m) => m.status === 'streaming')
          .map((m) => m.id),
      );
      let changed = false;
      const next = new Map<string, string>();
      for (const [id, content] of prev) {
        if (liveIds.has(id)) {
          next.set(id, content);
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    // Progress is pruned on a different rule from content: a step list stays
    // after its turn finishes, because "what did it spend forty seconds on"
    // is a question asked *after* the answer arrives, not during. What it
    // does not survive is leaving the conversation — the durable account of
    // a past run is the task record behind "View details", not a list this
    // session happened to still be holding.
    setProgressByMessage((prev) => {
      if (prev.size === 0) return prev;
      const known = new Set(conversation.messages.map((m) => m.id));
      let changed = false;
      const next = new Map<string, ProgressStep[]>();
      for (const [id, steps] of prev) {
        if (known.has(id)) next.set(id, steps);
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [conversation]);

  const value = useMemo<UseConversation>(
    () => ({
      conversation,
      conversations,
      isStreaming,
      activeMessageId,
      activeRunId,
      streamingContents,
      progressByMessage,
      send,
      open,
      newConversation,
      refresh,
      complete,
      replay,
    }),
    [
      conversation,
      conversations,
      isStreaming,
      activeMessageId,
      activeRunId,
      streamingContents,
      progressByMessage,
      send,
      open,
      newConversation,
      refresh,
      complete,
      replay,
    ],
  );

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

/**
 * Read the shared conversation state.
 *
 * Every component in the tree that needs to know the active conversation
 * or its streaming cell should call this hook. The hook is intentionally
 * context-backed so the header (rendered by the app shell) and the
 * message list (rendered by the chat surface) read the SAME state.
 */
export function useConversation(): UseConversation {
  const ctx = useContext(ConversationContext);
  if (!ctx) {
    throw new Error(
      'useConversation must be used within a ConversationProvider',
    );
  }
  return ctx;
}

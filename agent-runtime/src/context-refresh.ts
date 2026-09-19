/**
 * Asking Rust what this round is allowed to know, before making it.
 *
 * ## Why every round and not once at the start
 *
 * Run-start injection was what this product did, and the way it failed was
 * quiet. A turn that makes twelve tool calls used the context compiled before
 * the first one: an operator correction recorded at call three reached the
 * model at call four only if the model happened to re-read it, and a fact
 * another agent committed to the same task never arrived at all. Nothing said
 * the context was stale, so the loop looked like it was working with current
 * information.
 *
 * So the loop asks before each round — after a tool, after a compaction, on a
 * retry, on recovery — and Rust answers with a freshly authorised set at a
 * freshly read graph cursor.
 *
 * ## Why this side cannot decide any of it
 *
 * The three things that decide the answer all live on the Rust side. The
 * session is there, so authorisation is there. The graph revision is there, so
 * the cursor is there. And the admission rules — which claims count as
 * established and which are only proposals — are there. A loop that assembled
 * its own context would be assembling it without any of them.
 *
 * ## Why a failure here does not fail the round
 *
 * A refresh that could not be made costs this round the *newest* context; it
 * does not invalidate the transcript the round already has. Failing the turn
 * would turn a degraded answer into no answer. What is not acceptable is
 * failing silently, so every failure is reported — including the one that
 * matters most, a deployment with no graph at all.
 */

import { RpcError, type RpcPeer } from "./peer.js";

/** One block of context Rust authorised for this round. */
export interface ContextBlock {
  kind:
    | "objective"
    | "constraint"
    | "plan"
    | "receipt"
    | "pendingApproval"
    | "neighbour"
    | "evidence"
    | "artifact";
  itemId?: string;
  revision?: number;
  content: string;
  tokens: number;
  contentHash: string;
}

/** What Rust answered. */
export interface ContextRefresh {
  /** The changefeed cursor this was compiled against. */
  graphRevision: number | null;
  manifestHash: string;
  /**
   * True when the objective, constraints, plan, receipts and pending approvals
   * together need more than the window affords.
   *
   * Nothing was dropped from them. The round proceeds knowing it is over
   * budget, which is a decision somebody can see, rather than quietly losing a
   * correction.
   */
  mandatoryOverflowed: boolean;
  blocks: ContextBlock[];
  contentHashes: string[];
  omissions: Array<{ what: string; reason: string; detail: string }>;
}

export interface RefreshRequest {
  runId: string;
  taskId: string;
  agentId: string;
  definitionVersion: number;
  modelId: string;
  servedWindow: number;
  question: string;
  /** Content hashes this round already carries, so nothing is injected twice. */
  alreadyCarried: string[];
  projectId?: string;
  templateId?: string;
  reservedToolSchemas: number;
  reservedOutput: number;
  reservedFraming: number;
}

/**
 * Asks for the context this round may use.
 *
 * Returns `undefined` when the refresh could not be made — the round then runs
 * on what it already has, and the reason has been reported.
 */
export async function refreshContext(
  peer: RpcPeer,
  request: RefreshRequest,
  log: (line: string) => void = (line) => {
    process.stderr.write(`[agent-runtime:log] ${line}\n`);
  },
): Promise<ContextRefresh | undefined> {
  try {
    const refreshed = (await peer.request("context.refresh", request)) as ContextRefresh;

    if (refreshed?.mandatoryOverflowed) {
      log(
        `[context] run=${request.runId} the objective, constraints and receipts for this round ` +
          `need more than the window affords; nothing was dropped from them and this round is ` +
          `over budget`,
      );
    }
    for (const omission of refreshed?.omissions ?? []) {
      // Only the reasons a person can act on. A deduplication is the compiler
      // working and would be noise on every round.
      if (omission.reason === "budget" || omission.reason === "revoked") {
        log(`[context] run=${request.runId} ${omission.detail}`);
      }
    }
    return refreshed;
  } catch (cause) {
    const detail =
      cause instanceof RpcError
        ? `${cause.code}: ${cause.message}`
        : cause instanceof Error
          ? cause.message
          : String(cause);
    log(
      `[context] run=${request.runId} this round could not refresh its context (${detail}), so ` +
        `it runs on what it already had — a correction or a fact recorded since the last ` +
        `refresh will not have reached the model`,
    );
    return undefined;
  }
}

/**
 * Renders authorised blocks into the one message the round prepends.
 *
 * One message rather than several, because the transcript's shape is what the
 * compactor and the translator reason about, and a variable number of extra
 * messages per round would make both of them harder to predict than they
 * already are.
 *
 * Its content is data the model is told about, and every block carries its own
 * status label — an unverified proposal says so inside its own brackets — so a
 * stored sentence that tries to give instructions arrives as something with a
 * provenance rather than as a voice.
 */
export function renderContextBlocks(blocks: readonly ContextBlock[]): string | undefined {
  if (blocks.length === 0) return undefined;
  return (
    "--- TASK MEMORY (authorised for this turn; each line states its own status) ---\n" +
    blocks.map((block) => block.content).join("\n")
  );
}

/**
 * Wraps a stream function so every round it makes refreshes its context first.
 *
 * ## Why a wrapper rather than a call site
 *
 * Because there is no single call site. `Agent` calls `streamFn` for the first
 * round, for the round after each tool result, for the round after a
 * compaction, and again for any round the repair layer re-issues. A refresh
 * placed before the loop would have covered the first of those and none of the
 * rest — which is precisely the gap this closes.
 *
 * The blocks are prepended to the messages for that one call. They are not
 * written into the run's transcript: the transcript is what was *said*, and the
 * task memory is what Rust authorised for this round. Keeping them apart is
 * what lets the next round be given a different, freshly authorised set without
 * the previous one accumulating in the history.
 */
export function withContextRefresh<Fn extends (...args: never[]) => unknown>(
  inner: Fn,
  context: () => {
    peer: RpcPeer;
    request: RefreshRequest;
    remember: (hashes: readonly string[]) => void;
  },
): Fn {
  const wrapped = async (...args: unknown[]): Promise<unknown> => {
    const { peer, request, remember } = context();
    const refreshed = await refreshContext(peer, request);

    if (refreshed && refreshed.blocks.length > 0) {
      const rendered = renderContextBlocks(refreshed.blocks);
      // Found by shape rather than by position.
      //
      // `streamFn` has been called with different arities by different
      // versions of agent-core, and a wrapper that assumed argument zero would
      // silently stop injecting the moment that changed — which is the failure
      // mode this whole module exists to remove, reintroduced one layer up.
      const carrier = args.find(
        (arg): arg is { messages: unknown[] } =>
          typeof arg === "object" &&
          arg !== null &&
          Array.isArray((arg as { messages?: unknown }).messages),
      );
      if (rendered && carrier) {
        // Prepended, so it is read before the conversation rather than after
        // it. A constraint the model meets at the end of a long transcript is
        // a constraint it has already had several chances to break.
        carrier.messages.unshift({ role: "system", content: rendered });
      }
      remember(refreshed.contentHashes);
    }

    return (inner as unknown as (...rest: unknown[]) => unknown)(...args);
  };
  return wrapped as unknown as Fn;
}

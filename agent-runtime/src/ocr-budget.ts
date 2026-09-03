/**
 * How much of an OCR'd document actually goes into the prompt.
 *
 * ## The decision this makes, and why it has to be made somewhere
 *
 * `compose_prompt_with_attachments` pastes every attachment's text into the
 * turn, whole. For a one-page invoice that is exactly right. For a 40-page
 * scanned standard it is the failure `context-ledger.ts` was written to
 * diagnose — "Evidence dominates: whole documents are reaching the window
 * instead of references" — except that arriving at the diagnosis after the fact
 * does not help the run that already lost its history to it.
 *
 * So the size decision happens before the paste, and it is stated as a rule
 * rather than left to a threshold buried in a call site.
 *
 * ## The rule
 *
 * Let
 *
 * - `window` be the model's context window,
 * - `committed` be everything already spoken for this turn — the system prompt,
 *   the tool schemas, the notes, the transcript so far, and the reserve held
 *   back for the reply,
 * - `budget = window - committed`, what is genuinely free, and
 * - `docTokens` be the document's text, counted.
 *
 * Then:
 *
 * - `docTokens <= FULL_INCLUSION_SHARE * budget` → **the whole text goes in.**
 * - otherwise → **partial inclusion**: as much of the document as the
 *   allowance holds is injected, and the turn says plainly how much that was.
 *   (Ranking those passages by relevance to the question is the natural next
 *   step here and is deliberately not claimed yet — see the note below.)
 *
 * ## Why the share is a half and not all of it
 *
 * A document permitted to fill the whole free budget leaves nothing for the
 * answer to build on, or for a second turn to exist in. The person asks a
 * follow-up and the run compacts immediately, losing the very document it just
 * read. Half leaves room for the conversation the document was attached *for*.
 *
 * The number is a judgement, not a measurement, and it is written here once so
 * that it can be argued with in one place.
 *
 * ## Why not summarisation
 *
 * Summarising is the third option and it was not taken. It costs a second model
 * call on a machine that may have no GPU left after the first, and it replaces
 * the document's own words with a paraphrase — in a system whose evidence is
 * meant to be quotable back to the page it came from. Partial inclusion keeps
 * every word it does include verbatim.
 *
 * ## What this does not yet do
 *
 * The part included is the beginning of the document, not the part most
 * relevant to the question. {@link fill} is written against *ranked* chunks so
 * that relevance ordering can be dropped in without changing the budget rule —
 * but until something actually ranks them, no message here claims it does.
 */

/** The share of free budget a single document's text may occupy. */
export const FULL_INCLUSION_SHARE = 0.5;

/**
 * The floor under a document's allowance, in tokens.
 *
 * Without it, a turn that is already nearly full computes a budget of a few
 * hundred tokens and admits a handful of sentences — which reads to the model
 * as a document that says almost nothing, rather than as a document that could
 * not be shown. Below this floor the honest outcome is to say it does not fit,
 * which is what `plan` returns.
 */
export const MINIMUM_USEFUL_ALLOWANCE = 512;

/** What to do with one document's text. */
export type InjectionStrategy = "full" | "chunked" | "reference-only";

export interface InjectionPlan {
  strategy: InjectionStrategy;
  /** Tokens this document may spend. */
  allowance: number;
  /** The document's full size, as counted. */
  documentTokens: number;
  /** Free budget at the moment of the decision. */
  budget: number;
  /**
   * Why, in the words the person reads. Shown verbatim on the context row —
   * "about 30% of it" is the difference between a trustworthy answer and one
   * that silently rests on a third of the evidence.
   */
  explanation: string;
}

/**
 * Decides how one document enters the prompt.
 *
 * `window` of zero means the runtime was never told the model's window. That is
 * reported as `full` with an explanation saying so, rather than as a refusal:
 * an unknown window is not evidence that the document does not fit, and
 * refusing on it would break every run against a model whose window nobody
 * configured. The overflow, if it comes, is then the provider's own clear error
 * rather than a guess made here.
 */
export function plan(input: {
  documentTokens: number;
  committed: number;
  window: number;
}): InjectionPlan {
  const { documentTokens, committed, window } = input;

  if (window <= 0) {
    return {
      strategy: "full",
      allowance: documentTokens,
      documentTokens,
      budget: 0,
      explanation:
        "The model's context window is not known to this runtime, so the whole document was included. If it does not fit, the model server will say so.",
    };
  }

  const budget = Math.max(0, window - committed);
  const allowance = Math.floor(budget * FULL_INCLUSION_SHARE);

  if (documentTokens <= allowance) {
    return {
      strategy: "full",
      allowance,
      documentTokens,
      budget,
      explanation: `The whole document was included — ${documentTokens.toLocaleString()} tokens, within the ${allowance.toLocaleString()} available to it.`,
    };
  }

  if (allowance < MINIMUM_USEFUL_ALLOWANCE) {
    return {
      strategy: "reference-only",
      allowance: 0,
      documentTokens,
      budget,
      explanation: `There was no room for this document — ${documentTokens.toLocaleString()} tokens against ${allowance.toLocaleString()} available. It was read, but none of it is in this turn.`,
    };
  }

  const share = Math.round((allowance / documentTokens) * 100);
  return {
    strategy: "chunked",
    allowance,
    documentTokens,
    budget,
    explanation: `This document is ${documentTokens.toLocaleString()} tokens and ${allowance.toLocaleString()} were available, so roughly the first ${share}% of it was included. The rest is not in this turn.`,
  };
}

/**
 * Fills an allowance from ranked chunks, in rank order.
 *
 * Stops at the first chunk that does not fit rather than skipping it to fit a
 * later, smaller one. Packing greedily would reorder the document by size, and
 * a model handed passage 9 and passage 3 but not passage 4 reads a text with a
 * hole in it that nothing marks — which is how a confident answer gets built on
 * a paragraph that was never there.
 */
export function fill<T extends { tokens: number }>(
  ranked: readonly T[],
  allowance: number,
): { taken: T[]; used: number; omitted: number } {
  const taken: T[] = [];
  let used = 0;
  let omitted = 0;
  let stopped = false;
  for (const chunk of ranked) {
    if (!stopped && used + chunk.tokens <= allowance) {
      taken.push(chunk);
      used += chunk.tokens;
    } else {
      stopped = true;
      omitted += 1;
    }
  }
  return { taken, used, omitted };
}

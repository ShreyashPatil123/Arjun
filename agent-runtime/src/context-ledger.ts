/**
 * What is actually in the context window, section by section.
 *
 * ## The failure this prevents
 *
 * `RunCompactor` knows one number: how many tokens the whole projected context
 * came to. When that number is too large it summarises the oldest half. That
 * works, and it tells an operator nothing about *why* the window filled — which
 * is the question they actually have when a run compacts four times in twenty
 * turns. Was it the tool schemas? A skill body nobody uses? One 40-page tool
 * result pasted whole?
 *
 * Without an answer the only lever is "summarise sooner", which degrades the
 * run. With one, the lever is usually "stop pasting the document" — which does
 * not degrade anything, because the document is still retrievable by reference.
 *
 * ## Why the sections are these sections
 *
 * They are the things that can independently grow, and each has a different
 * remedy:
 *
 * - `system` — the base prompt. Fixed; growth here is a code change.
 * - `skill` — guidance loaded for this task. Progressive disclosure: a skill
 *   body belongs in context only while the step that needs it is live.
 * - `toolSchema` — the tool definitions. Grows with the *catalogue*, not the
 *   run, so a large number here means too many tools are offered at once.
 * - `evidence` — retrieved passages. Should be references plus the regions
 *   actually read, never whole documents.
 * - `notes` — the bounded working notes. If this is not small, the bound is
 *   wrong.
 * - `transcript` — the live message tail.
 * - `compaction` — the summary standing in for older history.
 * - `reserve` — held back for the model's own output and for the summarisation
 *   request. Not occupied; *committed*, which is the same thing for the purpose
 *   of deciding whether the next turn fits.
 *
 * ## What this is not
 *
 * Not a budget enforcer. It counts and reports; `RunCompactor` decides. Keeping
 * measurement apart from policy means a wrong count produces a wrong *number on
 * a screen* rather than a run that compacts when it should not.
 */

import { estimateContextTokens, type AgentMessage } from "@openclaw/agent-core";

/** The independently-growing parts of a context window. */
export const CONTEXT_SECTIONS = [
  "system",
  "skill",
  "toolSchema",
  "evidence",
  "notes",
  "transcript",
  "compaction",
  "reserve",
] as const;

export type ContextSection = (typeof CONTEXT_SECTIONS)[number];

/** Loose text as the estimator wants to see it. */
function asTextMessage(text: string): AgentMessage {
  return { role: "user", content: [{ type: "text", text }], timestamp: 0 } as AgentMessage;
}

/** One reading of the ledger. Plain data, so it crosses the wire unchanged. */
export interface ContextLedgerSnapshot {
  /** Token count per section. Every section is present, zero included, so a
   * reader never has to distinguish "absent" from "empty". */
  sections: Record<ContextSection, number>;
  /** Everything except `reserve` — what is really occupied right now. */
  occupied: number;
  /** Occupied plus reserve. What the next turn has to fit inside. */
  committed: number;
  /** The model's window, as the runtime was told it. Zero when unknown. */
  window: number;
  /** `window - committed`. Negative means the next turn does not fit. */
  headroom: number;
  /** Times this run has compacted so far. */
  compactions: number;
}

/**
 * Accumulates token counts for one run.
 *
 * Mutable and reused across turns rather than rebuilt: the fixed sections
 * (system, tool schemas) are measured once, and re-measuring them every turn
 * would spend real time counting characters that did not change.
 */
export class ContextLedger {
  readonly #sections: Record<ContextSection, number>;
  #window: number;
  #compactions = 0;

  constructor(window = 0) {
    this.#window = Math.max(0, Math.floor(window));
    this.#sections = Object.fromEntries(
      CONTEXT_SECTIONS.map((section) => [section, 0]),
    ) as Record<ContextSection, number>;
  }

  /** Replaces a section's count. Used for the parts measured once. */
  set(section: ContextSection, tokens: number): void {
    this.#sections[section] = Math.max(0, Math.floor(tokens));
  }

  /**
   * Measures text and records it as a section's whole content.
   *
   * Wrapped as a message rather than counted by character, so a section made of
   * loose text and one made of messages are measured by the same estimator. Two
   * estimators would make the ledger's own total disagree with the number
   * compaction decides on, and only one of them can be right.
   */
  setText(section: ContextSection, text: string): void {
    this.set(
      section,
      text ? estimateContextTokens([asTextMessage(text)]).tokens : 0,
    );
  }

  /**
   * Measures messages and records them as a section's whole content.
   *
   * Uses the harness estimator rather than a character count, because an image
   * block counts as a flat block of tokens there. ARJUN feeds rendered PDF
   * pages to vision models, and counting a page image by its characters reads
   * it as ~0 tokens — which is how a ledger ends up claiming plenty of headroom
   * on the turn that overflows.
   */
  setMessages(section: ContextSection, messages: AgentMessage[]): void {
    this.set(section, estimateContextTokens(messages).tokens);
  }

  /** Adds to a section. For things that arrive one at a time, like passages. */
  add(section: ContextSection, tokens: number): void {
    this.#sections[section] = Math.max(0, this.#sections[section] + Math.floor(tokens));
  }

  get(section: ContextSection): number {
    return this.#sections[section];
  }

  setWindow(window: number): void {
    this.#window = Math.max(0, Math.floor(window));
  }

  countCompaction(): void {
    this.#compactions += 1;
  }

  get compactions(): number {
    return this.#compactions;
  }

  snapshot(): ContextLedgerSnapshot {
    const sections = { ...this.#sections };
    const occupied = CONTEXT_SECTIONS.filter((section) => section !== "reserve").reduce(
      (total, section) => total + sections[section],
      0,
    );
    const committed = occupied + sections.reserve;
    return {
      sections,
      occupied,
      committed,
      window: this.#window,
      // An unknown window has no headroom to report. Zero would read as "full",
      // which is a claim this does not have the information to make; the caller
      // checks `window` before believing it.
      headroom: this.#window > 0 ? this.#window - committed : 0,
      compactions: this.#compactions,
    };
  }

  /**
   * Whether the next turn fits.
   *
   * False when the window is unknown is deliberate — an unknown window is not
   * evidence of room, and the caller that wants to proceed anyway can say so by
   * checking `window` itself.
   */
  fits(): boolean {
    const { window, committed } = this.snapshot();
    return window > 0 && committed <= window;
  }

  /**
   * The sections that grew most, largest first.
   *
   * The thing an operator reads. Naming the top two is usually the whole
   * diagnosis, and a table of eight numbers is not.
   *
   * `reserve` is excluded. It is set by policy from the window size rather than
   * grown by anything the run did, so ranking it here would regularly name it
   * as the cause on a small window — a diagnosis whose only remedy is to reserve
   * less, which is the one change guaranteed to make the run fail.
   */
  largest(count = 3): { section: ContextSection; tokens: number }[] {
    return CONTEXT_SECTIONS.filter((section) => section !== "reserve")
      .map((section) => ({ section, tokens: this.#sections[section] }))
      .filter((entry) => entry.tokens > 0)
      .sort((a, b) => b.tokens - a.tokens)
      .slice(0, count);
  }
}

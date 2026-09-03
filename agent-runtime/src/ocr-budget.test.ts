/**
 * Tests for the OCR injection threshold.
 *
 * The rule is stated in `ocr-budget.ts`; these pin the boundaries of it, and in
 * particular the two ways it could quietly do the wrong thing: admitting a
 * document that leaves no room for the conversation, and admitting so little of
 * one that the model reads it as a document that says nothing.
 */

import { describe, expect, it } from "vitest";

import { FULL_INCLUSION_SHARE, MINIMUM_USEFUL_ALLOWANCE, fill, plan } from "./ocr-budget.js";

describe("injection strategy", () => {
  it("includes a small document whole", () => {
    const decision = plan({ documentTokens: 1_000, committed: 2_000, window: 32_000 });
    expect(decision.strategy).toBe("full");
    expect(decision.explanation).toContain("whole document");
  });

  it("chunks a document larger than its share of the free budget", () => {
    // 40k of text against a 32k window is the case the rule exists for.
    const decision = plan({ documentTokens: 40_000, committed: 4_000, window: 32_000 });
    expect(decision.strategy).toBe("chunked");
    expect(decision.allowance).toBe(14_000); // (32000 - 4000) * 0.5
    // The wording must describe what actually happens — a prefix, not a
    // relevance-ranked selection. See the note in `ocr-budget.ts`.
    expect(decision.explanation).toMatch(/first \d+% of it/);
    expect(decision.explanation).not.toMatch(/relevant/);
  });

  it("leaves at least half the free budget for the conversation", () => {
    // The property behind the constant: whatever the document, what it is
    // allowed to take never exceeds half of what was free. A document that
    // consumed all of it would make the follow-up turn impossible, and the
    // follow-up is what the document was attached for.
    for (const documentTokens of [1_000, 10_000, 100_000, 1_000_000]) {
      const decision = plan({ documentTokens, committed: 8_000, window: 32_000 });
      const free = 32_000 - 8_000;
      expect(decision.allowance).toBeLessThanOrEqual(free * FULL_INCLUSION_SHARE);
    }
  });

  it("sits exactly on the boundary without chunking", () => {
    const free = 32_000 - 2_000;
    const exact = free * FULL_INCLUSION_SHARE;
    expect(plan({ documentTokens: exact, committed: 2_000, window: 32_000 }).strategy).toBe(
      "full",
    );
    expect(plan({ documentTokens: exact + 1, committed: 2_000, window: 32_000 }).strategy).toBe(
      "chunked",
    );
  });

  it("says a document does not fit rather than admitting a useless sliver", () => {
    // A nearly-full turn leaves a few hundred tokens. Injecting three sentences
    // and calling it the document is worse than saying it did not fit: the
    // model answers confidently from a fragment nobody knows is a fragment.
    const decision = plan({ documentTokens: 50_000, committed: 31_400, window: 32_000 });
    expect(decision.strategy).toBe("reference-only");
    expect(decision.allowance).toBe(0);
    expect(decision.explanation).toContain("no room");
    expect(MINIMUM_USEFUL_ALLOWANCE).toBeGreaterThan(0);
  });

  it("includes everything when the window is unknown, and says so", () => {
    // Not a refusal. An unconfigured window is not evidence that the document
    // does not fit, and the server's own error is a better message than a guess
    // made here.
    const decision = plan({ documentTokens: 90_000, committed: 0, window: 0 });
    expect(decision.strategy).toBe("full");
    expect(decision.explanation).toContain("not known");
  });

  it("treats an over-committed turn as having no budget, not negative budget", () => {
    const decision = plan({ documentTokens: 500, committed: 40_000, window: 32_000 });
    expect(decision.budget).toBe(0);
    expect(decision.allowance).toBe(0);
    expect(decision.strategy).toBe("reference-only");
  });
});

describe("filling an allowance", () => {
  const chunk = (tokens: number, id: string) => ({ tokens, id });

  it("takes chunks in rank order until the allowance is spent", () => {
    const { taken, used } = fill([chunk(300, "a"), chunk(300, "b"), chunk(300, "c")], 700);
    expect(taken.map((c) => c.id)).toEqual(["a", "b"]);
    expect(used).toBe(600);
  });

  it("stops rather than skipping ahead to something smaller", () => {
    // Greedy packing would take "a" then "c", producing a text with an unmarked
    // hole where "b" was. A model reading that answers from a document it
    // believes is continuous.
    const { taken, omitted } = fill([chunk(300, "a"), chunk(900, "b"), chunk(50, "c")], 400);
    expect(taken.map((c) => c.id)).toEqual(["a"]);
    expect(omitted).toBe(2);
  });

  it("takes nothing when the allowance is zero", () => {
    const { taken, used, omitted } = fill([chunk(10, "a")], 0);
    expect(taken).toEqual([]);
    expect(used).toBe(0);
    expect(omitted).toBe(1);
  });
});

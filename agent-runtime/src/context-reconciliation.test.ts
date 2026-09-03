/**
 * Tests for reconciling the ledger's estimate against what the model charged.
 *
 * The property under test is narrow and worth stating plainly: after a call
 * whose usage the provider reported, the ledger's total equals that report. Not
 * approximately, and not the estimate it started from. Everything else here
 * defends the cases where that correction must *not* happen — because a
 * correction applied to a number nobody measured is how a guess acquires the
 * authority of a measurement.
 */

import { describe, expect, it } from "vitest";

import { ContextLedger } from "./context-ledger.js";

/** A ledger with the fixed sections already measured, as a run has by turn 1. */
function ledger(window = 32_000) {
  const l = new ContextLedger(window);
  l.set("system", 1_000);
  l.set("toolSchema", 2_000);
  l.set("transcript", 3_000);
  l.set("reserve", 4_000);
  return l;
}

describe("reconciling one turn", () => {
  it("corrects the running total to what the provider charged", () => {
    const l = ledger();
    // Estimated input: system 1,000 + tools 2,000 + transcript 3,000 = 6,000.
    expect(l.snapshot().occupied).toBe(6_000);

    l.reconcile({ estimatedIn: 6_000, actualIn: 7_500, actualOut: 400 });
    l.applyMeasuredInput(7_500);

    // The whole point: occupied now *is* the measured figure, not a guess near
    // it. A meter that stays on 6,000 after the model said 7,500 is a meter
    // that will let the next turn overflow a window it reported as roomy.
    expect(l.snapshot().occupied).toBe(7_500);
  });

  it("keeps both halves so the estimator can be judged", () => {
    const l = ledger();
    l.reconcile({ estimatedIn: 6_000, actualIn: 7_500, actualOut: 400 });
    const record = l.snapshot().reconciliations[0]!;
    expect(record.estimatedIn).toBe(6_000);
    expect(record.actualIn).toBe(7_500);
    expect(record.driftRatio).toBeCloseTo(1.25);
  });

  it("leaves the estimate alone when the provider reported nothing", () => {
    // A local server that sends no usage block is common. Correcting toward
    // "null" — or, worse, toward zero — would replace a serviceable estimate
    // with a fiction, and the screen would show it with a measured figure's
    // confidence.
    const l = ledger();
    const before = l.snapshot().occupied;
    l.reconcile({ estimatedIn: 6_000, actualIn: null, actualOut: null });
    l.applyMeasuredInput(null);

    expect(l.snapshot().occupied).toBe(before);
    const record = l.snapshot().reconciliations[0]!;
    expect(record.actualIn).toBeNull();
    expect(record.driftRatio).toBeNull();
  });

  it("records one reconciliation per model call, numbered in order", () => {
    // "Recently performed" means every turn. Four calls, four records — a run
    // that reconciled once and coasted would be a guess for the rest of its
    // life.
    const l = ledger();
    for (let i = 0; i < 4; i++) {
      l.reconcile({ estimatedIn: 1_000, actualIn: 1_100, actualOut: 50 });
    }
    expect(l.snapshot().reconciliations.map((r) => r.turn)).toEqual([1, 2, 3, 4]);
  });

  it("books the correction on the transcript and leaves measured sections alone", () => {
    // The fixed sections were counted from text this process holds. The
    // transcript is the inferred part, so it is where an unexplained difference
    // belongs — see `applyMeasuredInput`.
    const l = ledger();
    l.applyMeasuredInput(7_500);
    const { sections } = l.snapshot();
    expect(sections.system).toBe(1_000);
    expect(sections.toolSchema).toBe(2_000);
    expect(sections.transcript).toBe(4_500);
  });

  it("floors the transcript at zero rather than going negative", () => {
    // A provider total below the fixed sections means those sections are
    // over-counted. A negative transcript would make the ledger sum to
    // something impossible; the discrepancy stays visible as drift instead.
    const l = ledger();
    l.applyMeasuredInput(500);
    expect(l.snapshot().sections.transcript).toBe(0);
    expect(l.snapshot().occupied).toBeGreaterThanOrEqual(0);
  });

  it("reports no drift ratio against a zero estimate", () => {
    const l = ledger();
    l.reconcile({ estimatedIn: 0, actualIn: 900, actualOut: 10 });
    expect(l.snapshot().reconciliations[0]!.driftRatio).toBeNull();
  });
});

describe("headroom after correction", () => {
  it("recomputes what the next turn has to fit inside", () => {
    const l = ledger(32_000);
    l.reconcile({ estimatedIn: 6_000, actualIn: 20_000, actualOut: 500 });
    l.applyMeasuredInput(20_000);
    const snapshot = l.snapshot();
    // committed = occupied 20,000 + reserve 4,000
    expect(snapshot.committed).toBe(24_000);
    expect(snapshot.headroom).toBe(8_000);
  });

  it("goes negative when the corrected total no longer fits", () => {
    // Negative headroom is the honest report and the signal the UI escalates
    // on. Clamping it to zero would make "does not fit" indistinguishable from
    // "fits exactly".
    const l = ledger(32_000);
    l.applyMeasuredInput(30_000);
    const snapshot = l.snapshot();
    expect(snapshot.headroom).toBeLessThan(0);
    expect(l.fits()).toBe(false);
  });
});

describe("itemisation under correction", () => {
  it("still sums to its sections after the total is corrected", () => {
    // The invariant has to survive reconciliation, or the rows stop explaining
    // the bar precisely when the bar becomes trustworthy.
    const l = ledger();
    l.upsertEntity({
      id: "sha-abc",
      section: "evidence",
      label: "invoice.pdf",
      tokens: 800,
      measurement: "exact",
      status: "active",
      pinned: false,
      sequence: 1,
    });
    l.set("evidence", 800);
    l.applyMeasuredInput(9_000);
    expect(l.snapshot().itemisationErrors).toEqual([]);
  });

  it("does not let a runtime update clear a pin a person set", () => {
    // The pin is a human instruction. The runtime re-registering the document
    // on the next turn must not quietly discard it, or the protection lasts
    // exactly one turn and the person never learns it stopped.
    const l = ledger();
    const doc = {
      id: "sha-abc",
      section: "evidence" as const,
      label: "invoice.pdf",
      tokens: 800,
      measurement: "exact" as const,
      status: "active" as const,
      pinned: false,
      sequence: 1,
    };
    l.upsertEntity(doc);
    l.setPinned("sha-abc", true);
    l.upsertEntity({ ...doc, tokens: 850 });

    const entity = l.snapshot().entities.find((e) => e.id === "sha-abc");
    expect(entity?.pinned).toBe(true);
    expect(entity?.tokens).toBe(850);
  });
});

/**
 * Tests for the itemised context layer.
 *
 * Three things are worth protecting here, and they are the three that would
 * make the meter lie rather than merely look wrong:
 *
 * 1. The entity rows sum to the section totals they claim to explain.
 * 2. The eviction order matches what the compactor actually does.
 * 3. A document still being read reports no size, rather than a guessed one.
 */

import { describe, expect, it } from "vitest";

import {
  type ContextEntity,
  evictionPlan,
  withRemainders,
  occupiedByEntities,
  pressure,
  reconcileSections,
  rollUp,
  shareOf,
} from "./context-entities.js";

function entity(over: Partial<ContextEntity> = {}): ContextEntity {
  return {
    id: "e1",
    section: "transcript",
    label: "Turn 1",
    tokens: 100,
    measurement: "estimated",
    status: "active",
    pinned: false,
    sequence: 0,
    ...over,
  };
}

describe("roll-up", () => {
  it("sums entities into their sections", () => {
    const totals = rollUp([
      entity({ id: "a", section: "transcript", tokens: 100 }),
      entity({ id: "b", section: "transcript", tokens: 50 }),
      entity({ id: "c", section: "evidence", tokens: 400 }),
    ]);
    expect(totals).toEqual({ transcript: 150, evidence: 400 });
  });

  it("counts a document still being read as nothing, not as a guess", () => {
    // The whole reason `pending` is a status rather than an absence. A row that
    // contributed an estimated size here would make the ledger disagree with
    // the projection it describes, and the disagreement would grow with the
    // document.
    const totals = rollUp([
      entity({ id: "reading", section: "evidence", status: "pending", tokens: 0 }),
      entity({ id: "read", section: "evidence", tokens: 300 }),
    ]);
    expect(totals.evidence).toBe(300);
  });

  it("excludes what has already been dropped", () => {
    const totals = rollUp([
      entity({ id: "gone", status: "dropped", tokens: 900 }),
      entity({ id: "here", tokens: 100 }),
    ]);
    expect(totals.transcript).toBe(100);
  });

  it("reports a section whose rows do not add up to it", () => {
    // This is the assertion that keeps every downstream number honest. If it
    // ever fires in production the breakdown is fiction, and it is better to
    // name the section than to draw a bar for it.
    const entities = [entity({ section: "evidence", tokens: 100 })];
    expect(reconcileSections(entities, { evidence: 100 })).toEqual([]);
    expect(reconcileSections(entities, { evidence: 250 })).toEqual([
      { section: "evidence", fromEntities: 100, fromSection: 250 },
    ]);
  });
});

describe("eviction order", () => {
  it("reclaims retrievable evidence before it summarises conversation", () => {
    // Mirrors `RunCompactor.transform`: `pruneStaleToolResults` runs first and
    // every turn, because a passage with a durable marker costs nothing to drop.
    const plan = evictionPlan(
      [
        entity({ id: "turn", section: "transcript", sequence: 5 }),
        entity({ id: "passage", section: "evidence", sequence: 9 }),
      ],
      100,
    );
    expect(plan.map((e) => e.id)).toEqual(["passage", "turn"]);
  });

  it("takes the oldest first within a section", () => {
    const plan = evictionPlan(
      [
        entity({ id: "new", sequence: 9 }),
        entity({ id: "old", sequence: 1 }),
        entity({ id: "mid", sequence: 4 }),
      ],
      100,
    );
    expect(plan.map((e) => e.id)).toEqual(["old", "mid", "new"]);
  });

  it("never offers the system prompt, tool schemas, notes or reserve", () => {
    const plan = evictionPlan(
      [
        entity({ id: "sys", section: "system" }),
        entity({ id: "tools", section: "toolSchema" }),
        entity({ id: "notes", section: "notes" }),
        entity({ id: "reserve", section: "reserve" }),
        entity({ id: "summary", section: "compaction" }),
        entity({ id: "turn", section: "transcript" }),
      ],
      100,
    );
    expect(plan.map((e) => e.id)).toEqual(["turn"]);
  });

  it("honours a pin", () => {
    const plan = evictionPlan(
      [
        entity({ id: "pinned-doc", section: "evidence", sequence: 0, pinned: true }),
        entity({ id: "turn", section: "transcript", sequence: 8 }),
      ],
      100,
    );
    expect(plan.map((e) => e.id)).toEqual(["turn"]);
    expect(plan.some((e) => e.pinned)).toBe(false);
  });

  it("protects the recent tail the compactor keeps", () => {
    // `keepRecentTokens` means the newest turns are not candidates. Promising
    // that one of them goes first would be a promise the compactor breaks.
    const plan = evictionPlan(
      [entity({ id: "old", sequence: 1 }), entity({ id: "recent", sequence: 20 })],
      10,
    );
    expect(plan.map((e) => e.id)).toEqual(["old"]);
  });

  it("promises nothing when the protected boundary is unknown", () => {
    // Infinity is the default, and it yields an empty plan rather than a
    // confident wrong one.
    expect(evictionPlan([entity({ id: "turn", sequence: 3 })])).toEqual([]);
  });
});

describe("pressure", () => {
  const entities = [entity({ id: "turn", sequence: 1 })];

  it("escalates with occupancy", () => {
    expect(pressure(100, 1000, entities, 99).level).toBe("ok");
    expect(pressure(700, 1000, entities, 99).level).toBe("tight");
    expect(pressure(950, 1000, entities, 99).level).toBe("critical");
    expect(pressure(1200, 1000, entities, 99).level).toBe("over");
  });

  it("stays quiet when the window is unknown rather than crying wolf", () => {
    // A model nobody configured a window for would otherwise show red on every
    // run, which teaches people that red means nothing.
    const unknown = pressure(5000, 0, entities, 99);
    expect(unknown.level).toBe("ok");
    expect(unknown.ratio).toBeNull();
  });

  it("names what goes first", () => {
    const p = pressure(
      950,
      1000,
      [
        entity({ id: "doc", section: "evidence", sequence: 2 }),
        entity({ id: "turn", section: "transcript", sequence: 1 }),
      ],
      99,
    );
    expect(p.firstToGo?.id).toBe("doc");
  });

  it("reports nothing to reclaim when everything is pinned or immovable", () => {
    // The case worth interrupting somebody for: the next turn fails outright
    // rather than degrading, and no amount of compaction changes that.
    const p = pressure(
      990,
      1000,
      [
        entity({ id: "sys", section: "system" }),
        entity({ id: "doc", section: "evidence", pinned: true, sequence: 1 }),
      ],
      99,
    );
    expect(p.level).toBe("critical");
    expect(p.firstToGo).toBeNull();
  });
});

describe("share", () => {
  it("is zero rather than NaN when nothing is committed", () => {
    // A NaN width renders as a full bar, which reads as "this filled the
    // window" — the exact opposite of the truth.
    expect(shareOf(entity({ tokens: 0 }), 0)).toBe(0);
  });

  it("totals to one across a full ledger", () => {
    const entities = [entity({ id: "a", tokens: 250 }), entity({ id: "b", tokens: 750 })];
    const committed = occupiedByEntities(entities);
    const total = entities.reduce((sum, e) => sum + shareOf(e, committed), 0);
    expect(total).toBeCloseTo(1);
  });
});

describe("remainders", () => {
  it("makes the rows add up to the section totals", () => {
    // The invariant the whole design rests on. After this, `reconcileSections`
    // must be empty — that is what makes a bar and the rows under it agree.
    const known = [entity({ id: "doc", section: "evidence", tokens: 400 })];
    const sections = { evidence: 400, transcript: 1_200 };
    const completed = withRemainders(known, sections);
    expect(reconcileSections(completed, sections)).toEqual([]);
  });

  it("does not add a row when the itemisation is already complete", () => {
    const known = [entity({ id: "doc", section: "evidence", tokens: 400 })];
    const completed = withRemainders(known, { evidence: 400 });
    expect(completed).toHaveLength(1);
  });

  it("shows an over-claiming itemisation instead of hiding it", () => {
    // Entities claiming more than the section holds is a defect. Clamping it to
    // zero would make the defect invisible, which is the one outcome that
    // guarantees nobody fixes it.
    const known = [entity({ id: "doc", section: "evidence", tokens: 900 })];
    const completed = withRemainders(known, { evidence: 400 });
    const remainder = completed.find((e) => e.id === "evidence:remainder");
    expect(remainder?.tokens).toBe(-500);
    expect(remainder?.status).toBe("dropped");
  });

  it("puts a remainder behind named rows in the eviction plan", () => {
    // A person can act on "invoice.pdf goes first". They cannot act on "the
    // rest of the conversation goes first", so the nameable thing is named.
    const completed = withRemainders(
      [entity({ id: "doc", section: "evidence", tokens: 100, sequence: 5 })],
      { evidence: 300 },
    );
    const plan = evictionPlan(completed, Number.POSITIVE_INFINITY);
    expect(plan[0]?.id).toBe("doc");
  });
});

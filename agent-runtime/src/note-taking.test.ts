/**
 * What the notes learn without the model's help.
 *
 * The failure guarded here is the one that costs real work: a run whose process
 * died after `create_docx` wrote the approval note, resuming and writing it
 * again. That only stays prevented while the effect is recorded by the code
 * that saw the tool succeed — a model asked to note its own side effects
 * records the interesting ones and forgets the completed ones.
 */

import { describe, expect, it } from "vitest";
import { markersIn, observeToolResult } from "./note-taking.js";
import { WorkingNotes } from "./working-notes.js";

describe("evidence markers", () => {
  it("takes every marker the model was shown, once each", () => {
    const text = "[E1] SOP, page 4\ntext\n\n[E2] SOP, page 5\nmore\n\n[E1] again";

    expect(markersIn(text)).toEqual(["E1", "E2"]);
  });

  it("finds none in a result that carries none", () => {
    expect(markersIn('No passages matched "unicorns".')).toEqual([]);
  });

  it("records a search's markers into the notes", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "search_documents",
      args: { query: "seal wear" },
      text: "[E1] SOP, page 4\ntext\n\n[E2] SOP, page 5\nmore",
    });

    expect(notes.state.evidenceIds).toEqual(["E1", "E2"]);
  });

  it("records markers from a loaded page range the same way", () => {
    // Both tools feed one numbered table, so both must feed one note list —
    // otherwise a marker pulled back by page would never become durable, and
    // its raw result would never be cleared.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "load_more_evidence",
      args: { documentSha256: "sop", fromPage: 11, toPage: 13 },
      text: "Read pages 11 to 13 of Maintenance SOP.\n\n[E3] SOP, page 11\ntext",
    });

    expect(notes.state.evidenceIds).toEqual(["E3"]);
  });

  it("does not read markers out of a tool that does not produce evidence", () => {
    // A drafted document quoting "[E1]" is the model citing, not the tool
    // retrieving. Treating it as evidence would make a marker durable that
    // nothing in the evidence table backs.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "read_scoped_file",
      args: { path: "draft.md" },
      text: "The seal is worn beyond the limit [E1].",
    });

    expect(notes.state.evidenceIds).toEqual([]);
  });
});

describe("side effects a resumption must not repeat", () => {
  it("records a produced document by name, not by path", () => {
    // The path contains the run's own workspace directory, which differs on
    // every attempt. A resumption comparing full paths would never recognise
    // its own earlier work and would write the document again.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "create_docx",
      args: { path: "C:\\data\\runs\\run-1\\approval-note.docx", template: "approval_note" },
      text: "Created approval-note.docx.",
    });

    expect(notes.hasDone("create_docx", "approval-note.docx")).toBe(true);
    expect(notes.state.artifactIds).toEqual(["approval-note.docx"]);
  });

  it("records a workspace write as an effect", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "write_scoped_file",
      args: { path: "draft.md", content: "…" },
      text: "Wrote draft.md.",
    });

    expect(notes.hasDone("write_scoped_file", "draft.md")).toBe(true);
  });

  it("records that code ran without claiming which execution it was", () => {
    // An execution has no stable name. Saying one happened warns a resumption;
    // naming it would let the resumption conclude a *different* execution was
    // already done and skip it.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "execute_code",
      args: { language: "python", source: "print(1)" },
      text: "1",
    });

    const completed = notes.state.completed;
    expect(completed).toHaveLength(1);
    expect(completed[0]?.tool).toBe("execute_code");
    // Not filed as an artifact: nothing was produced that can be re-opened.
    expect(notes.state.artifactIds).toEqual([]);
  });

  it("does not record a read as something that must not be repeated", () => {
    // Reading is free to redo. Recording it would fill the bounded list with
    // entries no resumption needs, pushing out the ones it does.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "read_scoped_file",
      args: { path: "draft.md" },
      text: "…",
    });

    expect(notes.state.completed).toEqual([]);
  });

  it("counts the same effect once however often it is observed", () => {
    const notes = new WorkingNotes();
    for (let i = 0; i < 5; i++) {
      observeToolResult(notes, {
        tool: "create_docx",
        args: { path: "approval-note.docx" },
        text: "Created approval-note.docx.",
      });
    }

    expect(notes.state.completed).toHaveLength(1);
  });
});

describe("calculations", () => {
  it("records a reference into the engine's record rather than the working", () => {
    // The steps are in the calculation record on the Rust side. Copying them
    // here would put the whole working in the context on every turn, which is
    // what the notes exist to avoid.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "run_calculation",
      args: { expression: "9.0 mm - 7.4 mm" },
      text: "1.6 mm",
    });
    observeToolResult(notes, {
      tool: "run_calculation",
      args: { expression: "1.6 mm / 9.0 mm" },
      text: "0.178",
    });

    expect(notes.state.calculationIds).toEqual(["C1", "C2"]);
    expect(notes.render()).not.toContain("9.0 mm - 7.4 mm");
  });
});

describe("what the notes then let compaction do", () => {
  it("makes a search's markers durable, which is what allows its raw text to be cleared", () => {
    // The chain requirement 12 depends on: the marker has to reach the notes
    // before the raw passage can be dropped, or the reference would point at
    // nothing a resumed run could resolve.
    const notes = new WorkingNotes();
    const text = `[E1] Maintenance SOP, page 4\n${"y".repeat(2_000)}`;
    observeToolResult(notes, { tool: "search_documents", args: { query: "seal" }, text });

    expect(notes.state.evidenceIds).toContain("E1");
    expect(markersIn(text).every((marker) => notes.state.evidenceIds.includes(marker))).toBe(true);
  });
});

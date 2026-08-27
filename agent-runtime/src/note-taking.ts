/**
 * Keeping the run's notes current from what its tools actually returned.
 *
 * ## Why the notes are not written by the model
 *
 * The obvious design is a `remember` tool the model calls when it learns
 * something. It fails in a specific and unrecoverable way: the model decides
 * what to record, and the entries that matter most are exactly the ones it is
 * least likely to write down. A model that has just produced a document does
 * not think to note that it produced it — it thinks about the next thing. Then
 * the process dies, the run resumes, and it produces the document again.
 *
 * So the entries that make a resumption *safe* are derived here, from what the
 * tools returned, without the model's participation. `create_docx` succeeded on
 * `approval-note.docx`, so that effect happened; a search returned `[E3]`, so
 * that marker resolves. Neither claim depends on the model having noticed.
 *
 * The model still contributes the judgement-shaped parts — the goal, the
 * decisions, the open questions — through `run.note` from the Rust side. Those
 * are the entries where a model's account is the only source there is, and also
 * the entries whose loss costs context rather than correctness.
 *
 * ## Why markers are parsed out of the text
 *
 * The evidence table lives in Rust and numbers each passage once for the life
 * of the run. The number the model is told to cite is in the rendered text it
 * receives. Parsing it back out of that text means the notes hold exactly the
 * markers the model saw — not the markers Rust believes it sent, which can
 * differ if a result was truncated on the way.
 *
 * That matters for pruning: `pruneStaleToolResults` clears a raw result only
 * when every marker in it is durable. Markers taken from the same text the
 * check runs against cannot disagree with it.
 */

import type { WorkingNotes } from "./working-notes.js";

/** Tools whose success is a side effect a resumed run must not repeat. */
const SIDE_EFFECTING = new Set([
  "create_docx",
  "create_xlsx",
  "write_scoped_file",
  "execute_code",
]);

/** Tools that return numbered evidence. */
const EVIDENCE_PRODUCING = new Set(["search_documents", "load_more_evidence"]);

/** Evidence markers appearing in a rendered tool result, de-duplicated. */
export function markersIn(text: string): string[] {
  return [...new Set([...text.matchAll(/\[E(\d+)\]/g)].map((match) => `E${match[1]}`))];
}

/** The argument that names what a side-effecting call acted on. */
function targetOf(tool: string, args: unknown): string | undefined {
  if (typeof args !== "object" || args === null) return undefined;
  const record = args as Record<string, unknown>;
  const path = record.path;
  if (typeof path === "string" && path.length > 0) {
    // The file name, not the path. The path includes the run's own workspace
    // directory, which is different on every attempt — so a resumed run
    // comparing full paths would never recognise its own earlier work.
    const name = path.split(/[\\/]/).pop();
    return name && name.length > 0 ? name : path;
  }
  if (tool === "execute_code") {
    const language = record.language;
    // Code has no name. The language is not a stable identity for one
    // execution, so this records that an execution happened without claiming
    // which — enough to warn a resumption, not enough to let it conclude that a
    // *different* execution was already done.
    return typeof language === "string" ? `${language} (an execution)` : "an execution";
  }
  return undefined;
}

/**
 * Folds one tool result into the notes.
 *
 * Pure with respect to everything except the notes it is given, so a caller can
 * test it without a peer, a model, or a running loop.
 */
export function observeToolResult(
  notes: WorkingNotes,
  observation: { tool: string; args: unknown; text: string },
): void {
  const { tool, args, text } = observation;

  if (EVIDENCE_PRODUCING.has(tool)) {
    for (const marker of markersIn(text)) notes.sawEvidence(marker);
  }

  if (tool === "run_calculation") {
    // Numbered by position, because the engine's own record is keyed that way
    // and the note is a reference into it rather than a copy of the working.
    notes.calculated(`C${notes.state.calculationIds.length + 1}`);
  }

  const target = targetOf(tool, args);
  if (target && SIDE_EFFECTING.has(tool)) {
    notes.didEffect(tool, target);
    if (tool !== "execute_code") {
      // A produced file is also an artifact the run can be asked about later.
      notes.produced(target);
    }
  }
}

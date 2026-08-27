/**
 * The run's own memory: what it is doing, what it has established, what is
 * still open — and nothing else.
 *
 * ## The failure this prevents
 *
 * The obvious fix for compaction losing things is to let the model keep notes.
 * The obvious way that fails is that the notes become a second transcript: the
 * model appends to them every turn, they are never pruned because nothing knows
 * what is stale, and within twenty turns the notes are larger than the history
 * they were meant to replace. Now compaction has two things to summarise and
 * the run is worse off than before.
 *
 * So the notes here are a **fixed-shape record with hard caps**, not a log.
 * Every field is either a single value that is replaced, or a list with a
 * ceiling. Writing past a ceiling drops the oldest entry and increments a
 * counter — the bound is enforced, and the fact that it bound something is
 * visible rather than silent.
 *
 * ## Why identifiers and not content
 *
 * `evidenceIds`, `calculationIds`, `artifactIds` hold markers — `E3`, `C1`,
 * `approval-note.docx` — never the passage, the working or the file. The full
 * thing lives on the Rust side, in the evidence table and the task record, and
 * is retrievable by that identifier. That is what makes the notes cheap enough
 * to keep in context permanently: a run that retrieved forty passages carries
 * forty short markers, not forty passages.
 *
 * It is also what makes recovery honest. A recovered run reads `completed`,
 * sees that `create_docx` already produced `approval-note.docx`, and does not
 * write it again — because the note says the effect happened, not that the
 * model believes it happened.
 */

/** Where the run has got to, in terms the plan uses. */
export interface NotesStage {
  /** The plan step now being worked, 1-based. Zero before the first. */
  ordinal: number;
  /** What that step is for, in the person's terms. */
  intent: string;
}

/** One decision the run made and is bound by. */
export interface NoteDecision {
  /** What was decided, in one line. */
  what: string;
  /** Why — the sentence a reviewer needs to judge it. */
  because: string;
  /** RFC 3339, UTC. */
  at: string;
}

/** A side effect that has already happened. Read on recovery. */
export interface CompletedEffect {
  tool: string;
  /** What it acted on — a file name, a path, an identifier. */
  target: string;
  /** RFC 3339, UTC. */
  at: string;
}

/** The notes as plain data. What crosses the wire and what is persisted. */
export interface WorkingNotesState {
  goal: string;
  stage: NotesStage;
  decisions: NoteDecision[];
  evidenceIds: string[];
  calculationIds: string[];
  artifactIds: string[];
  openQuestions: string[];
  nextAction: string;
  completed: CompletedEffect[];
  /** How many entries the caps have dropped, per list. Never reset. */
  dropped: Record<string, number>;
}

/**
 * The ceilings.
 *
 * Chosen so the rendered notes stay near a thousand tokens on a full run, which
 * is affordable against an 8k window and negligible against a 32k one. They are
 * not tuning knobs for the model: nothing the model writes can raise them.
 */
export const NOTE_LIMITS = {
  decisions: 12,
  evidenceIds: 64,
  calculationIds: 32,
  artifactIds: 16,
  openQuestions: 8,
  completed: 32,
  /** Longest single line kept, in characters. A note is a line, not a page. */
  lineChars: 240,
} as const;

function trim(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > NOTE_LIMITS.lineChars
    ? `${clean.slice(0, NOTE_LIMITS.lineChars - 1)}…`
    : clean;
}

/** Appends under a cap, dropping the oldest and counting the drop. */
function push<T>(
  list: T[],
  item: T,
  cap: number,
  dropped: Record<string, number>,
  key: string,
): void {
  list.push(item);
  while (list.length > cap) {
    list.shift();
    dropped[key] = (dropped[key] ?? 0) + 1;
  }
}

/**
 * One run's bounded notes.
 *
 * Held by the compactor, rendered into the context ahead of the transcript, and
 * persisted so a recovered run reads them instead of starting over.
 */
export class WorkingNotes {
  #goal = "";
  #stage: NotesStage = { ordinal: 0, intent: "" };
  #decisions: NoteDecision[] = [];
  #evidenceIds: string[] = [];
  #calculationIds: string[] = [];
  #artifactIds: string[] = [];
  #openQuestions: string[] = [];
  #nextAction = "";
  #completed: CompletedEffect[] = [];
  #dropped: Record<string, number> = {};

  static from(state: Partial<WorkingNotesState> | undefined): WorkingNotes {
    const notes = new WorkingNotes();
    if (!state) return notes;
    notes.setGoal(state.goal ?? "");
    if (state.stage) notes.atStage(state.stage.ordinal, state.stage.intent);
    notes.setNextAction(state.nextAction ?? "");
    // Carried forward so a restart does not reset the count and make the notes
    // look as though they had never bound anything.
    for (const [key, count] of Object.entries(state.dropped ?? {})) {
      notes.#dropped[key] = count;
    }
    // Re-applied through the setters so a persisted file written before a cap
    // was tightened is bounded on load rather than trusted.
    for (const decision of state.decisions ?? []) {
      notes.decided(decision.what, decision.because, decision.at);
    }
    for (const id of state.evidenceIds ?? []) notes.sawEvidence(id);
    for (const id of state.calculationIds ?? []) notes.calculated(id);
    for (const id of state.artifactIds ?? []) notes.produced(id);
    for (const question of state.openQuestions ?? []) notes.asked(question);
    for (const effect of state.completed ?? []) {
      notes.didEffect(effect.tool, effect.target, effect.at);
    }
    return notes;
  }

  setGoal(goal: string): void {
    this.#goal = trim(goal);
  }

  atStage(ordinal: number, intent: string): void {
    this.#stage = { ordinal: Math.max(0, Math.floor(ordinal)), intent: trim(intent) };
  }

  decided(what: string, because: string, at = new Date().toISOString()): void {
    push(
      this.#decisions,
      { what: trim(what), because: trim(because), at },
      NOTE_LIMITS.decisions,
      this.#dropped,
      "decisions",
    );
  }

  /** Records an evidence marker. Idempotent: `E3` seen twice is one marker. */
  sawEvidence(id: string): void {
    this.#addId(this.#evidenceIds, id, NOTE_LIMITS.evidenceIds, "evidenceIds");
  }

  calculated(id: string): void {
    this.#addId(this.#calculationIds, id, NOTE_LIMITS.calculationIds, "calculationIds");
  }

  produced(id: string): void {
    this.#addId(this.#artifactIds, id, NOTE_LIMITS.artifactIds, "artifactIds");
  }

  #addId(list: string[], id: string, cap: number, key: string): void {
    const clean = trim(id);
    if (!clean || list.includes(clean)) return;
    push(list, clean, cap, this.#dropped, key);
  }

  /** Adds an unresolved question. Idempotent on exact text. */
  asked(question: string): void {
    const clean = trim(question);
    if (!clean || this.#openQuestions.includes(clean)) return;
    push(this.#openQuestions, clean, NOTE_LIMITS.openQuestions, this.#dropped, "openQuestions");
  }

  /** Marks a question answered. Silently ignores one that was never open. */
  answered(question: string): void {
    const clean = trim(question);
    this.#openQuestions = this.#openQuestions.filter((open) => open !== clean);
  }

  setNextAction(action: string): void {
    this.#nextAction = trim(action);
  }

  /**
   * Records a side effect that has already taken place.
   *
   * The thing a recovered run reads before acting. Idempotent on
   * `tool + target`, so replaying the same completion does not grow the list.
   */
  didEffect(tool: string, target: string, at = new Date().toISOString()): void {
    if (this.hasDone(tool, target)) return;
    push(
      this.#completed,
      { tool, target: trim(target), at },
      NOTE_LIMITS.completed,
      this.#dropped,
      "completed",
    );
  }

  /** Whether this exact side effect is already known to have happened. */
  hasDone(tool: string, target: string): boolean {
    const clean = trim(target);
    return this.#completed.some((effect) => effect.tool === tool && effect.target === clean);
  }

  get state(): WorkingNotesState {
    return {
      goal: this.#goal,
      stage: { ...this.#stage },
      decisions: this.#decisions.map((decision) => ({ ...decision })),
      evidenceIds: [...this.#evidenceIds],
      calculationIds: [...this.#calculationIds],
      artifactIds: [...this.#artifactIds],
      openQuestions: [...this.#openQuestions],
      nextAction: this.#nextAction,
      completed: this.#completed.map((effect) => ({ ...effect })),
      dropped: { ...this.#dropped },
    };
  }

  /** True when nothing has been recorded. A run's first turn renders nothing. */
  get isEmpty(): boolean {
    return (
      !this.#goal &&
      !this.#nextAction &&
      this.#stage.ordinal === 0 &&
      this.#decisions.length === 0 &&
      this.#evidenceIds.length === 0 &&
      this.#calculationIds.length === 0 &&
      this.#artifactIds.length === 0 &&
      this.#openQuestions.length === 0 &&
      this.#completed.length === 0
    );
  }

  /**
   * The notes as the model reads them.
   *
   * Deliberately terse and in a fixed order. A stable rendering is what makes
   * this cacheable across turns and what stops the model from treating the
   * notes as prose to be continued.
   */
  render(): string {
    if (this.isEmpty) return "";
    const lines: string[] = ["## Working notes (carried across compaction)"];
    if (this.#goal) lines.push(`Goal: ${this.#goal}`);
    if (this.#stage.ordinal > 0) {
      lines.push(`Stage: step ${this.#stage.ordinal} — ${this.#stage.intent}`);
    }
    if (this.#decisions.length > 0) {
      lines.push("Decisions:");
      for (const decision of this.#decisions) {
        lines.push(`  - ${decision.what} (because ${decision.because})`);
      }
    }
    if (this.#evidenceIds.length > 0) {
      // Markers only. The passages are retrievable by marker, and pasting them
      // here would make the notes the largest thing in the window.
      lines.push(`Evidence held: ${this.#evidenceIds.join(", ")}`);
    }
    if (this.#calculationIds.length > 0) {
      lines.push(`Calculations: ${this.#calculationIds.join(", ")}`);
    }
    if (this.#artifactIds.length > 0) {
      lines.push(`Artifacts: ${this.#artifactIds.join(", ")}`);
    }
    if (this.#completed.length > 0) {
      lines.push("Already done — do not repeat:");
      for (const effect of this.#completed) {
        lines.push(`  - ${effect.tool} → ${effect.target}`);
      }
    }
    if (this.#openQuestions.length > 0) {
      lines.push("Unresolved:");
      for (const question of this.#openQuestions) lines.push(`  - ${question}`);
    }
    if (this.#nextAction) lines.push(`Next: ${this.#nextAction}`);

    const dropped = Object.entries(this.#dropped).filter(([, count]) => count > 0);
    if (dropped.length > 0) {
      // Said out loud. Notes that silently forgot something are worse than
      // notes that say they did, because only one of the two can be checked.
      lines.push(
        `(Older entries were dropped to keep these notes bounded: ${dropped
          .map(([key, count]) => `${count} ${key}`)
          .join(", ")}.)`,
      );
    }
    return lines.join("\n");
  }
}

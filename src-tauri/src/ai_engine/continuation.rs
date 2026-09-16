//! Carrying a task across more than one generation.
//!
//! [`crate::ai_engine::token_budget`] caps what a single inference call may
//! produce. That cap is correct and it is not the whole story: a task can
//! legitimately need more output than any one generation is allowed, and the
//! only honest answers to that are to finish it across several or to admit it
//! was not finished. Truncating and presenting the fragment is neither, which
//! is why [`crate::agent_runtime::outcome::RunOutcome::LengthLimited`] exists
//! as its own ending rather than as a completion.
//!
//! This module turns that ending into a decision:
//!
//! ```text
//! task -> gen 1 (<= cap) -> checkpoint -> gen 2 (<= cap) -> ... -> done
//!                              |
//!                              +-> not converging -> hand back, say so
//! ```
//!
//! ## Why the checkpoint is structured rather than a transcript
//!
//! The obvious continuation is "send the transcript back and say carry on". It
//! does not work on a long task, for the reason the task was long: the
//! transcript is already most of the window, and appending to it is how the
//! next generation has less room than the last. A checkpoint is a
//! *compression* — decisions reached, subtasks done and outstanding,
//! constraints that still bind, what comes next — and it is the shape this
//! codebase already keeps in `agent-runtime/src/working-notes.ts` and
//! [`crate::agent_runtime::resume`].
//!
//! ## Why there is a convergence guard
//!
//! Chained generation without one is an unbounded loop wearing a budget. A
//! model that restates its reasoning each round and advances nothing consumes
//! generations forever, and every round looks locally healthy: tokens are
//! produced, the stall guard is rearmed, the deadline is distant. The signal is
//! not tokens, it is *progress* — a checkpoint that does not differ from its
//! predecessor in completed work achieved nothing, whatever it emitted. Two of
//! those in a row ends the chain.
//!
//! This is the judgement `run.ts` makes with `MAX_TURNS` and its stall guard,
//! one level up: those bound a single run's turns, this bounds how many runs
//! one task may be spread across.

use serde::{Deserialize, Serialize};

/// The most generations one task may be spread across.
///
/// A backstop, not a policy — the policy is [`ContinuationChain::record`],
/// which stops as soon as progress does. This is the number at which something
/// has gone wrong that the progress check did not catch: eight generations at
/// the largest band is 65 536 tokens of output on one task, which is a report,
/// not an answer.
pub const MAX_GENERATIONS: u32 = 8;

/// Consecutive generations without progress before the chain is broken.
///
/// Two rather than one. A single round that finishes nothing is ordinary — a
/// model can spend a generation reading a long tool result and reasoning about
/// it before completing anything, and ending the task there would punish
/// exactly the deliberation the large bands exist to allow. Two in a row is a
/// pattern.
pub const STALLED_ROUNDS_BEFORE_BREAKING: u32 = 2;

/// What a task knows about itself between two generations.
///
/// Every field is what the *next* generation needs in order not to start over,
/// and nothing else. Deliberately not a transcript — see the module note.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Checkpoint {
    /// Which generation produced this, counting from 1.
    #[serde(default)]
    pub generation: u32,
    /// Decisions reached and not to be relitigated.
    #[serde(default)]
    pub conclusions: Vec<String>,
    /// Subtasks finished. The progress signal: a generation that adds nothing
    /// here achieved nothing, whatever it emitted.
    #[serde(default)]
    pub completed_subtasks: Vec<String>,
    /// Subtasks still outstanding.
    #[serde(default)]
    pub pending_subtasks: Vec<String>,
    /// Constraints and acceptance criteria that still bind.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// Tool calls already made and the artefacts they produced.
    ///
    /// Carried so the next generation does not repeat a side effect. The same
    /// hazard [`crate::agent_runtime::resume`] handles for a resumed run, and
    /// for the same reason: a document written twice is not a document written
    /// once.
    #[serde(default)]
    pub tool_outputs: Vec<String>,
    /// The one thing to do next. Empty when the task is finished.
    #[serde(default)]
    pub next_step: String,
}

impl Checkpoint {
    /// Whether this checkpoint says anything is left to do.
    ///
    /// Both halves are asked. A checkpoint with pending subtasks and no next
    /// step has lost its thread; one with a next step and no pending list is
    /// mid-subtask. Either is unfinished, and requiring both to be empty is
    /// what stops a task ending because the model forgot to restate its list.
    pub fn is_incomplete(&self) -> bool {
        !self.next_step.trim().is_empty() || !self.pending_subtasks.is_empty()
    }

    /// Whether this checkpoint represents real progress over `previous`.
    ///
    /// Completed work is the only measure. Conclusions and reasoning can grow
    /// without the task moving — that is precisely what a circular think loop
    /// looks like from outside — while a finished subtask or a consumed pending
    /// item cannot be produced by restating anything.
    pub fn advances_on(&self, previous: &Checkpoint) -> bool {
        if self.completed_subtasks.len() > previous.completed_subtasks.len() {
            return true;
        }
        if self.pending_subtasks.len() < previous.pending_subtasks.len() {
            return true;
        }
        // A tool that ran is a change to the world, which no amount of
        // re-reasoning produces.
        if self.tool_outputs.len() > previous.tool_outputs.len() {
            return true;
        }
        // Same counts, different work: one subtask swapped for another is
        // progress the lengths cannot see.
        self.completed_subtasks != previous.completed_subtasks
    }

    /// The checkpoint as a prompt preamble for the next generation.
    ///
    /// Phrased as a resumption rather than a briefing. A model handed a summary
    /// and a question treats the question as new and re-derives the summary;
    /// told plainly that this is its own prior work and that the reasoning
    /// behind it is settled, it continues from the next step instead. The
    /// difference is a whole generation's output.
    pub fn as_resumption_prompt(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "You are resuming your own unfinished work. The generation before this one \
             reached its output limit; it was not wrong and it was not rejected. Continue \
             from the next step below. Do not restate the reasoning that produced these \
             conclusions, and do not begin the task again.\n",
        );
        let section = |out: &mut String, title: &str, items: &[String]| {
            if items.is_empty() {
                return;
            }
            out.push('\n');
            out.push_str(title);
            out.push_str(":\n");
            for item in items {
                out.push_str("- ");
                out.push_str(item.trim());
                out.push('\n');
            }
        };
        section(&mut out, "Established, not to be revisited", &self.conclusions);
        section(&mut out, "Already done", &self.completed_subtasks);
        section(
            &mut out,
            "Already carried out, and must not be repeated",
            &self.tool_outputs,
        );
        section(&mut out, "Still outstanding", &self.pending_subtasks);
        section(&mut out, "Constraints that still apply", &self.constraints);
        if !self.next_step.trim().is_empty() {
            out.push_str("\nStart here: ");
            out.push_str(self.next_step.trim());
            out.push('\n');
        }
        out
    }
}

impl Checkpoint {
    /// A checkpoint from the working notes the run already keeps.
    ///
    /// Nothing here is new state. [`crate::agent_runtime::memory::RunMemory`]
    /// is what the loop records as it goes and what a resumed run already reads
    /// before it acts, and it carries every field a continuation needs. Asking
    /// the model to produce a separate summary would cost a generation and
    /// invite it to describe work it did not do; reading the ledger the run
    /// wrote costs nothing and cannot.
    ///
    /// `completed` is split across two fields on purpose. A side effect is both
    /// evidence of progress and a thing that must not happen twice, and the
    /// resumption prompt says each of those in its own words.
    pub fn from_run_memory(
        memory: &crate::agent_runtime::memory::RunMemory,
        generation: u32,
    ) -> Self {
        Self {
            generation,
            conclusions: memory
                .decisions
                .iter()
                .map(|decision| format!("{} — {}", decision.what, decision.because))
                .collect(),
            completed_subtasks: memory
                .completed
                .iter()
                .map(|effect| format!("{} {}", effect.tool, effect.target))
                .collect(),
            pending_subtasks: memory.open_questions.clone(),
            constraints: vec![memory.goal.clone()]
                .into_iter()
                .filter(|goal| !goal.trim().is_empty())
                .collect(),
            tool_outputs: memory
                .completed
                .iter()
                .map(|effect| format!("{} produced {}", effect.tool, effect.target))
                .collect(),
            next_step: memory.next_action.clone(),
        }
    }
}

/// What to do after a generation ended at its cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ContinuationDecision {
    /// Run another generation, seeded with this preamble.
    Continue {
        generation: u32,
        resumption_prompt: String,
    },
    /// The task is done. Nothing was truncated.
    Finished,
    /// Stop, and say why.
    ///
    /// The task is handed back unfinished rather than presented as complete —
    /// the distinction [`crate::agent_runtime::outcome::RunOutcome::LengthLimited`]
    /// exists to preserve, which this must not quietly undo.
    ///
    /// `escalate` marks the endings a caller should answer by delegating or
    /// recovering rather than merely reporting: the task had more to do and
    /// this model stopped making headway on it.
    Stop { detail: String, escalate: bool },
}

/// One task's progress across a chain of generations.
///
/// Holds only what the decision needs: how many generations have run, the last
/// checkpoint to compare against, and how many rounds in a row achieved
/// nothing. The checkpoints themselves are recorded in the event ledger by the
/// caller, which is where this product's durable history lives.
#[derive(Debug, Clone, Default)]
pub struct ContinuationChain {
    generations: u32,
    last: Option<Checkpoint>,
    stalled_rounds: u32,
}

impl ContinuationChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generations run so far.
    pub fn generations(&self) -> u32 {
        self.generations
    }

    /// Consecutive generations that advanced nothing.
    pub fn stalled_rounds(&self) -> u32 {
        self.stalled_rounds
    }

    /// The most recent checkpoint, for a caller recording or displaying it.
    pub fn last_checkpoint(&self) -> Option<&Checkpoint> {
        self.last.as_ref()
    }

    /// Records a generation's checkpoint and says what happens next.
    ///
    /// `hit_output_cap` is the question this turns on, and it is asked rather
    /// than inferred from the checkpoint. A generation that stopped of its own
    /// accord with work outstanding is a model that decided to hand back — that
    /// is an answer, however partial, and continuing it would be this code
    /// overruling the model rather than rescuing it. Only a generation the
    /// *deployment's cap* cut short is one this module may resume.
    pub fn record(&mut self, checkpoint: Checkpoint, hit_output_cap: bool) -> ContinuationDecision {
        self.generations = self.generations.saturating_add(1);

        let advanced = match &self.last {
            // The first generation has nothing to compare against and cannot
            // have stalled.
            None => true,
            Some(previous) => checkpoint.advances_on(previous),
        };
        self.stalled_rounds = if advanced {
            0
        } else {
            self.stalled_rounds.saturating_add(1)
        };

        let incomplete = checkpoint.is_incomplete();
        let stalled = self.stalled_rounds >= STALLED_ROUNDS_BEFORE_BREAKING;
        let generation = self.generations;
        self.last = Some(checkpoint);

        if !incomplete {
            return ContinuationDecision::Finished;
        }

        if !hit_output_cap {
            return ContinuationDecision::Stop {
                detail: "It stopped with work outstanding, without reaching the output limit. \
                         The remaining steps are listed rather than assumed done."
                    .to_string(),
                escalate: false,
            };
        }

        if stalled {
            return ContinuationDecision::Stop {
                detail: format!(
                    "It ran {generation} generations and the last {} finished nothing new, so \
                     it was stopped rather than left circling. What it did complete is kept.",
                    self.stalled_rounds
                ),
                escalate: true,
            };
        }

        if generation >= MAX_GENERATIONS {
            return ContinuationDecision::Stop {
                detail: format!(
                    "It reached the {MAX_GENERATIONS}-generation limit for one task with work \
                     still outstanding. What it completed is kept, and the rest is listed."
                ),
                escalate: true,
            };
        }

        let resumption_prompt = self
            .last
            .as_ref()
            .map(Checkpoint::as_resumption_prompt)
            .unwrap_or_default();
        ContinuationDecision::Continue {
            generation: generation.saturating_add(1),
            resumption_prompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(done: &[&str], pending: &[&str], next: &str) -> Checkpoint {
        Checkpoint {
            completed_subtasks: done.iter().map(|s| s.to_string()).collect(),
            pending_subtasks: pending.iter().map(|s| s.to_string()).collect(),
            next_step: next.to_string(),
            ..Checkpoint::default()
        }
    }

    /// The whole point: a task cut off at its cap is resumed, not abandoned and
    /// not presented as an answer.
    #[test]
    fn a_task_cut_off_with_work_left_is_continued() {
        let mut chain = ContinuationChain::new();
        let decision = chain.record(checkpoint(&["read the spec"], &["write it"], "write it"), true);

        match decision {
            ContinuationDecision::Continue {
                generation,
                resumption_prompt,
            } => {
                assert_eq!(generation, 2);
                assert!(resumption_prompt.contains("write it"));
                assert!(
                    resumption_prompt.contains("resuming"),
                    "the next generation must be told it is continuing its own work"
                );
            }
            other => panic!("expected a continuation, got {other:?}"),
        }
    }

    /// A finished task is finished, whether or not it used its whole budget.
    #[test]
    fn a_task_with_nothing_outstanding_is_finished() {
        let mut chain = ContinuationChain::new();
        let decision = chain.record(checkpoint(&["all of it"], &[], ""), true);
        assert_eq!(decision, ContinuationDecision::Finished);
    }

    /// A model that stopped on its own is not overruled.
    ///
    /// Continuing here would be this code deciding the model was wrong to hand
    /// back, which is not a judgement it is in a position to make.
    #[test]
    fn a_generation_that_stopped_on_its_own_is_not_resumed() {
        let mut chain = ContinuationChain::new();
        let decision = chain.record(checkpoint(&[], &["something"], "something"), false);
        match decision {
            ContinuationDecision::Stop { escalate, .. } => assert!(
                !escalate,
                "handing back deliberately is not a failure to escalate"
            ),
            other => panic!("expected a stop, got {other:?}"),
        }
    }

    /// Circling is caught by what the task completed, not by what it emitted.
    ///
    /// Each of these rounds produces a full generation of tokens and finishes
    /// nothing — which is what a repetitive think loop looks like from outside.
    #[test]
    fn a_chain_that_finishes_nothing_twice_running_is_broken() {
        let mut chain = ContinuationChain::new();
        let circling = || checkpoint(&["read the spec"], &["write it"], "write it");

        assert!(matches!(
            chain.record(circling(), true),
            ContinuationDecision::Continue { .. }
        ));
        // First unproductive round: tolerated, because deliberation is allowed.
        assert!(matches!(
            chain.record(circling(), true),
            ContinuationDecision::Continue { .. }
        ));
        assert_eq!(chain.stalled_rounds(), 1);

        match chain.record(circling(), true) {
            ContinuationDecision::Stop { escalate, detail } => {
                assert!(escalate, "a stalled task is the caller's to delegate");
                assert!(detail.contains("finished nothing new"), "{detail}");
            }
            other => panic!("expected the chain to break, got {other:?}"),
        }
    }

    /// Progress resets the guard, so a slow task is not punished for one round
    /// of reading.
    #[test]
    fn a_round_that_finishes_something_clears_the_stall_count() {
        let mut chain = ContinuationChain::new();
        chain.record(checkpoint(&["a"], &["b", "c"], "b"), true);
        chain.record(checkpoint(&["a"], &["b", "c"], "b"), true);
        assert_eq!(chain.stalled_rounds(), 1);

        chain.record(checkpoint(&["a", "b"], &["c"], "c"), true);
        assert_eq!(
            chain.stalled_rounds(),
            0,
            "a finished subtask is progress and clears the count"
        );
    }

    /// A task that keeps making progress still stops somewhere.
    #[test]
    fn a_productive_chain_is_still_bounded() {
        let mut chain = ContinuationChain::new();
        let mut decision = ContinuationDecision::Finished;
        for round in 0..MAX_GENERATIONS {
            // Every round finishes something, so the progress guard never bites
            // and only the generation ceiling can stop this.
            let done: Vec<String> = (0..=round).map(|n| format!("step {n}")).collect();
            decision = chain.record(
                Checkpoint {
                    completed_subtasks: done,
                    pending_subtasks: vec!["more".to_string()],
                    next_step: "more".to_string(),
                    ..Checkpoint::default()
                },
                true,
            );
        }
        match decision {
            ContinuationDecision::Stop { escalate, detail } => {
                assert!(escalate);
                assert!(detail.contains("generation limit"), "{detail}");
            }
            other => panic!("expected the generation ceiling to stop it, got {other:?}"),
        }
        assert_eq!(chain.generations(), MAX_GENERATIONS);
    }

    /// Work already carried out is named in the preamble, because a side effect
    /// repeated is a side effect that happened twice.
    #[test]
    fn the_preamble_names_the_tool_calls_that_must_not_be_repeated() {
        let checkpoint = Checkpoint {
            tool_outputs: vec!["create_docx wrote report.docx".to_string()],
            pending_subtasks: vec!["summarise it".to_string()],
            next_step: "summarise it".to_string(),
            ..Checkpoint::default()
        };
        let prompt = checkpoint.as_resumption_prompt();
        assert!(prompt.contains("create_docx wrote report.docx"));
        assert!(prompt.contains("must not be repeated"));
    }

    /// Swapping one finished subtask for another is progress the counts alone
    /// cannot see.
    #[test]
    fn different_work_at_the_same_count_still_counts_as_progress() {
        let before = checkpoint(&["a"], &["x"], "x");
        let after = checkpoint(&["b"], &["x"], "x");
        assert!(after.advances_on(&before));
    }
}

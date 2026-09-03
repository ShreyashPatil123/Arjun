//! Where a run is, and which events may legally move it.
//!
//! ## Why the states are explicit
//!
//! Before this, a run had four states — running, completed, failed, cancelled —
//! and everything interesting happened inside "running". That is enough to
//! colour a row on a list and not enough for anything else. A person looking at
//! a stalled run could not tell "waiting for you to approve something" from
//! "the model is thinking" from "a tool has been executing for four minutes",
//! and a process that came back after a crash could not tell which of those it
//! had interrupted.
//!
//! So the states are named, and the naming is the point: each one is a distinct
//! thing a person might need to do something about.
//!
//! ## Why the transitions are checked
//!
//! The state is folded from an event history that outlives the process which
//! wrote it. Events arrive from several places — the command that started the
//! run, the runtime supervisor, the approval bridge, recovery at the next
//! start — and nothing stops two of them from writing about one run at once.
//!
//! A fold that simply believed every event in the order it found them would let
//! a late `run_routed` drag a finished run back to *routed*, and the screen
//! would show a completed task as still in progress. So a transition that
//! cannot legally happen is **recorded and not applied**: the event stays in the
//! history, because it happened and hiding it would be worse, but it does not
//! move the state. [`Transition::Illegal`] carries the reason so the anomaly is
//! reportable rather than silent.
//!
//! ## Terminal states absorb
//!
//! Nothing follows an ending. A run that was cancelled a moment before its loop
//! reported completion has one true ending — the cancellation — and a history
//! carrying both would let a reader choose. The store refuses to append past a
//! terminal event; this refuses to fold past one, which is the same rule held
//! in the two places it has to hold.

use serde::{Deserialize, Serialize};

use super::model::TaskEventType;

/// Where a run is.
///
/// The non-terminal states are ordered by progress, which is what makes a
/// backwards transition detectable rather than merely surprising.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// Accepted and recorded. Nothing has been decided about it yet.
    Created,
    /// The sensitivity of the material is known, so the router can be narrowed.
    Classified,
    /// A model has been chosen, and the reasons are recorded.
    Routed,
    /// The plan and its budget are fixed. Nothing after this may widen them.
    Planned,
    /// The loop is working.
    Running,
    /// A person has been asked to allow something, and the run is waiting.
    AwaitingApproval,
    /// A tool has been authorised and is running.
    ExecutingTool,
    /// The tool's outcome is written down. Distinct from `Running` because it
    /// is the point at which a side effect is known to have settled.
    ToolResultRecorded,
    /// The answer is being checked against the evidence the run actually has.
    Verifying,

    // -- Terminal ---------------------------------------------------------
    /// Finished, with an answer.
    Completed,
    /// A person stopped it.
    Cancelled,
    /// It ended badly. Something decided this.
    Failed,
    /// It ran out of steps, time, or went in circles. The budget did its job;
    /// this is not a fault and is deliberately not `Failed`.
    StoppedByBudget,
    /// The model ran into the output cap for one turn, so the answer stops
    /// mid-way. Not a fault and deliberately not `Completed`: the text of a
    /// cut-off answer is indistinguishable from a short one, so the only place
    /// the difference can be recorded is here.
    StoppedByLength,
    /// The gateway or the plan refused something the run could not continue
    /// without. Also not a fault: the policy working is the system working.
    StoppedByPolicy,
    /// A person has to look before anything else happens.
    ///
    /// Reached when the process went away mid-run. Deliberately not `Failed`:
    /// nothing decided this outcome, so nobody should read a verdict into it.
    /// When the interruption caught a side-effecting tool call in flight, the
    /// effect is separately marked unknown and needs reconciling before any
    /// later run may reuse its key — see [`super::idempotency`].
    DegradedNeedsHuman,
}

impl RunState {
    pub const ALL: &'static [RunState] = &[
        RunState::Created,
        RunState::Classified,
        RunState::Routed,
        RunState::Planned,
        RunState::Running,
        RunState::AwaitingApproval,
        RunState::ExecutingTool,
        RunState::ToolResultRecorded,
        RunState::Verifying,
        RunState::Completed,
        RunState::Cancelled,
        RunState::Failed,
        RunState::StoppedByBudget,
        RunState::StoppedByLength,
        RunState::StoppedByPolicy,
        RunState::DegradedNeedsHuman,
    ];

    /// Stable database spelling. Written out rather than derived from the
    /// variant name so renaming a variant cannot rewrite history.
    pub const fn as_str(self) -> &'static str {
        match self {
            RunState::Created => "created",
            RunState::Classified => "classified",
            RunState::Routed => "routed",
            RunState::Planned => "planned",
            RunState::Running => "running",
            RunState::AwaitingApproval => "awaiting_approval",
            RunState::ExecutingTool => "executing_tool",
            RunState::ToolResultRecorded => "tool_result_recorded",
            RunState::Verifying => "verifying",
            RunState::Completed => "completed",
            RunState::Cancelled => "cancelled",
            RunState::Failed => "failed",
            RunState::StoppedByBudget => "stopped_by_budget",
            RunState::StoppedByLength => "stopped_by_length",
            RunState::StoppedByPolicy => "stopped_by_policy",
            RunState::DegradedNeedsHuman => "degraded_needs_human",
        }
    }

    pub fn from_str(raw: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|s| s.as_str() == raw).or_else(|| {
            // Spellings written by the first version of this store, before the
            // states were made explicit. Kept readable so a database from an
            // earlier build still lists its runs rather than dropping them.
            Some(match raw {
                "timed_out" => RunState::StoppedByBudget,
                "interrupted" => RunState::DegradedNeedsHuman,
                _ => return None,
            })
        })
    }

    /// Whether the run is over. Nothing may follow a terminal state.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            RunState::Completed
                | RunState::Cancelled
                | RunState::Failed
                | RunState::StoppedByBudget
                | RunState::StoppedByLength
                | RunState::StoppedByPolicy
                | RunState::DegradedNeedsHuman
        )
    }

    /// Whether the run finished the work it set out to do.
    ///
    /// Only `Completed`. A run stopped by its budget produced whatever it
    /// produced, and calling that success would make the budget decorative.
    pub const fn is_success(self) -> bool {
        matches!(self, RunState::Completed)
    }

    /// Whether a person needs to do something before this run can be closed.
    pub const fn needs_person(self) -> bool {
        matches!(self, RunState::AwaitingApproval | RunState::DegradedNeedsHuman)
    }

    /// How far through the run this state is, for detecting a transition that
    /// goes backwards. Terminal states share the top rank: they are not ordered
    /// against each other, only against everything before them.
    const fn rank(self) -> u8 {
        match self {
            RunState::Created => 0,
            RunState::Classified => 1,
            RunState::Routed => 2,
            RunState::Planned => 3,
            // The working states are one rank between them. A run moves among
            // them freely — authorise, execute, record, back to the loop — and
            // ordering them against each other would make ordinary progress
            // look like a fault.
            RunState::Running
            | RunState::AwaitingApproval
            | RunState::ExecutingTool
            | RunState::ToolResultRecorded => 4,
            RunState::Verifying => 5,
            _ => 6,
        }
    }

    /// What to show a person. A complete sentence: a status line that trails
    /// off reads as though the message itself was truncated.
    pub const fn describe(self) -> &'static str {
        match self {
            RunState::Created => "Accepted, and not yet started.",
            RunState::Classified => "Working out which model should take it.",
            RunState::Routed => "A model has been chosen.",
            RunState::Planned => "The plan is fixed. Starting.",
            RunState::Running => "Running.",
            RunState::AwaitingApproval => "Waiting for someone to approve an action.",
            RunState::ExecutingTool => "Running a tool.",
            RunState::ToolResultRecorded => "Running.",
            RunState::Verifying => "Checking the answer against the evidence.",
            RunState::Completed => "Finished.",
            RunState::Cancelled => "Stopped, because somebody stopped it.",
            RunState::Failed => "Stopped: it did not finish.",
            RunState::StoppedByBudget => {
                "Stopped: it reached the limit the plan set for it."
            }
            RunState::StoppedByLength => {
                "Stopped: the answer reached the output limit for one turn, so it is cut off."
            }
            RunState::StoppedByPolicy => {
                "Stopped: it needed to do something it is not permitted to do."
            }
            RunState::DegradedNeedsHuman => {
                "Interrupted. Somebody needs to look at this before it is relied on."
            }
        }
    }
}

/// What folding one event onto a state comes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transition {
    /// The event moves the run to this state.
    To(RunState),
    /// The event carries information but does not move the state — a step
    /// spent, a turn finished, a file produced.
    Stays,
    /// The event cannot legally follow this state.
    ///
    /// Recorded in the history and **not** applied to the state. The reason is
    /// kept so the anomaly can be reported rather than silently dropped.
    Illegal { reason: String },
}

/// Decides what one event does to a run's state.
///
/// Pure, so every combination can be driven in a test without a database.
pub fn advance(current: RunState, event: TaskEventType) -> Transition {
    use TaskEventType as E;

    // The one thing that may follow an ending: a person saying what actually
    // happened to a side effect the ending left unresolved.
    //
    // It is legal precisely because it does not compete with the ending. The
    // run still finished the way it finished; this says whether the half-written
    // document exists. Refusing it would mean the only way to answer that
    // question was to leave it unanswered forever.
    if event == E::ToolEffectReconciled {
        return Transition::Stays;
    }

    // Nothing else follows an ending. Checked before the rest, so a late event
    // about a finished run is reported as the anomaly it is rather than being
    // judged against whatever it would otherwise have meant.
    if current.is_terminal() {
        return Transition::Illegal {
            reason: format!(
                "the run already ended as {}, so a {} event cannot apply to it",
                current.as_str(),
                event.as_str()
            ),
        };
    }

    let next = match event {
        E::RunCreated => RunState::Created,
        E::RunClassified => RunState::Classified,
        E::RunRouted => RunState::Routed,
        E::PlanReady => RunState::Planned,
        E::RunStarted => RunState::Running,

        E::ApprovalRequested => RunState::AwaitingApproval,
        // Back to the loop either way. A rejection is a tool result the model
        // reads and acts on, not an ending — the run carries on and says what
        // it could not do.
        E::ApprovalDecided => RunState::Running,

        E::ToolAuthorized => RunState::ExecutingTool,
        E::ToolSucceeded | E::ToolFailed | E::ToolReplayed => RunState::ToolResultRecorded,
        // A refusal never reaches a tool, so it is not a tool *result*. The run
        // is simply still running, with one fewer option.
        E::ToolRefused => RunState::Running,

        E::VerificationStarted => RunState::Verifying,

        E::RunCompleted => RunState::Completed,
        E::RunCancelled => RunState::Cancelled,
        E::RunFailed => RunState::Failed,
        E::RunStoppedByBudget | E::RunTimedOut => RunState::StoppedByBudget,
        E::RunStoppedByLength => RunState::StoppedByLength,
        E::RunStoppedByPolicy => RunState::StoppedByPolicy,
        E::RunDegraded | E::RunInterrupted => RunState::DegradedNeedsHuman,

        // Information about a run that is under way, carrying no claim about
        // where it is. `plan_stopped` is deliberately here: the plan announcing
        // it will do no more is not itself the ending, and the ending event
        // that follows says which kind it was.
        E::PlanStep
        | E::PlanStopped
        | E::TurnEnded
        | E::ContextCompacted
        | E::ArtifactProduced
        | E::ToolEffectPending
        | E::ToolEffectUnknown
        | E::ToolEffectReconciled
        // Reading, refusing or promoting memory happens *within* whatever the
        // run is doing. None of it is an ending and none of it moves the run's
        // state; treating a recall as a transition would make a task that
        // consulted its notes look like a task that changed course.
        // A checkpoint is an observation about the run, not a move within it.
        // `run_resumed` is here for a subtler reason: the state a resumption
        // lands in is the state the checkpoint recorded, and that is restored
        // explicitly rather than implied by this event.
        | E::CheckpointTaken
        | E::CheckpointFailed
        | E::RunResumed
        // Loading a skill narrows what a run may do and adds instructions to
        // its context. Neither moves the run to a different state: it is still
        // running, with a smaller tool set.
        | E::SkillLoaded
        | E::SkillRefused
        | E::MemoryRecalled
        | E::MemoryRefused
        | E::MemoryPromoted
        | E::MemoryForgotten
        // A subagent starting and stopping happens *within* whatever the parent
        // is doing. It does not move the parent's own state, and treating it as
        // a transition would make a fan-out of four readers look like four
        // state changes.
        | E::SubagentStarted
        | E::SubagentStopped
        // A hook firing says something about a call the run was making, not
        // about where the run has got to. A refused call leaves the run
        // running with one fewer option, exactly as a gateway refusal does.
        | E::HookEvaluated => return Transition::Stays,
    };

    // A terminal event is always legal from any live state: a run can be
    // cancelled while it is merely `created`, and can fail before it routes.
    if next.is_terminal() || next.rank() >= current.rank() {
        return Transition::To(next);
    }

    Transition::Illegal {
        reason: format!(
            "a {} event would move the run from {} back to {}",
            event.as_str(),
            current.as_str(),
            next.as_str()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_walks_the_ordinary_path_end_to_end() {
        let path = [
            (TaskEventType::RunCreated, RunState::Created),
            (TaskEventType::RunClassified, RunState::Classified),
            (TaskEventType::RunRouted, RunState::Routed),
            (TaskEventType::PlanReady, RunState::Planned),
            (TaskEventType::RunStarted, RunState::Running),
            (TaskEventType::ToolAuthorized, RunState::ExecutingTool),
            (TaskEventType::ToolSucceeded, RunState::ToolResultRecorded),
            (TaskEventType::VerificationStarted, RunState::Verifying),
            (TaskEventType::RunCompleted, RunState::Completed),
        ];

        let mut state = RunState::Created;
        for (event, expected) in path {
            match advance(state, event) {
                Transition::To(next) => state = next,
                other => panic!("{event:?} from {state:?} was {other:?}"),
            }
            assert_eq!(state, expected);
        }
    }

    /// Every ending a run can have, and the state each one lands in.
    ///
    /// The point of the table is that no two of these collapse onto the same
    /// state. Before the typed outcome existed, an operator's stop, a provider
    /// error and a turn cut off at the output cap all arrived here as
    /// `RunCompleted`, and the state machine dutifully recorded three different
    /// things as `Completed`.
    #[test]
    fn each_ending_lands_in_its_own_state() {
        let endings = [
            (TaskEventType::RunCompleted, RunState::Completed),
            (TaskEventType::RunFailed, RunState::Failed),
            (TaskEventType::RunCancelled, RunState::Cancelled),
            (TaskEventType::RunStoppedByLength, RunState::StoppedByLength),
            (TaskEventType::RunStoppedByBudget, RunState::StoppedByBudget),
            (TaskEventType::RunStoppedByPolicy, RunState::StoppedByPolicy),
            (TaskEventType::RunDegraded, RunState::DegradedNeedsHuman),
        ];
        for (event, expected) in endings {
            assert_eq!(
                advance(RunState::Running, event),
                Transition::To(expected),
                "{event:?}"
            );
            assert!(expected.is_terminal(), "{expected:?} must end the run");
            assert!(event.is_terminal(), "{event:?} must end the run");
        }
        // And only one of them counts as having done the work.
        let successes = endings
            .iter()
            .filter(|(_, state)| state.is_success())
            .count();
        assert_eq!(successes, 1, "only a completion is a success");
    }

    #[test]
    fn a_run_cut_off_at_the_output_cap_is_not_recorded_as_finished() {
        // The distinction is invisible in the text: a cut-off answer reads
        // exactly like a short one, so the state is the only place it is said.
        let state = match advance(RunState::Running, TaskEventType::RunStoppedByLength) {
            Transition::To(next) => next,
            other => panic!("{other:?}"),
        };
        assert_ne!(state, RunState::Completed);
        assert!(!state.is_success());
        assert!(state.describe().contains("cut off"));
        // Nothing follows an ending, this one included.
        assert!(matches!(
            advance(state, TaskEventType::RunCompleted),
            Transition::Illegal { .. }
        ));
    }

    #[test]
    fn an_approval_suspends_the_run_and_returns_it() {
        let waiting = advance(RunState::Running, TaskEventType::ApprovalRequested);
        assert_eq!(waiting, Transition::To(RunState::AwaitingApproval));

        // Back to the loop whichever way the person decided: a rejection is a
        // tool result the model reads, not an ending.
        let resumed = advance(RunState::AwaitingApproval, TaskEventType::ApprovalDecided);
        assert_eq!(resumed, Transition::To(RunState::Running));
    }

    #[test]
    fn nothing_may_follow_an_ending() {
        for terminal in RunState::ALL.iter().filter(|s| s.is_terminal()) {
            let late = advance(*terminal, TaskEventType::RunCompleted);
            assert!(
                matches!(late, Transition::Illegal { .. }),
                "{terminal:?} accepted a late completion"
            );
        }
    }

    #[test]
    fn a_person_may_still_account_for_a_side_effect_after_the_run_has_ended() {
        // The single exception, and the reason for it: this does not compete
        // with the ending. The run finished the way it finished; this says
        // whether the half-written document exists. Refusing it would leave
        // that question permanently unanswerable.
        for terminal in RunState::ALL.iter().filter(|s| s.is_terminal()) {
            assert_eq!(
                advance(*terminal, TaskEventType::ToolEffectReconciled),
                Transition::Stays,
                "{terminal:?} refused a reconciliation"
            );
            // And it does not move the run out of the state it ended in.
            assert!(terminal.is_terminal());
        }
    }

    #[test]
    fn a_late_routing_event_cannot_drag_a_running_task_backwards() {
        // The race worth defending against: a slow write from the start path
        // landing after the loop is already working. Believed, it would show a
        // running task as still choosing a model.
        let backwards = advance(RunState::Running, TaskEventType::RunRouted);
        match backwards {
            Transition::Illegal { reason } => {
                assert!(reason.contains("back to routed"), "{reason}");
            }
            other => panic!("expected an illegal transition, got {other:?}"),
        }
    }

    #[test]
    fn a_run_may_be_cancelled_before_it_has_even_routed() {
        // Somebody presses stop while the model is still being chosen. A
        // terminal event is legal from every live state.
        for live in RunState::ALL.iter().filter(|s| !s.is_terminal()) {
            assert_eq!(
                advance(*live, TaskEventType::RunCancelled),
                Transition::To(RunState::Cancelled),
                "{live:?} refused a cancellation"
            );
        }
    }

    #[test]
    fn progress_events_carry_information_without_moving_the_state() {
        for event in [
            TaskEventType::PlanStep,
            TaskEventType::TurnEnded,
            TaskEventType::ContextCompacted,
            TaskEventType::ArtifactProduced,
            TaskEventType::PlanStopped,
        ] {
            assert_eq!(advance(RunState::Running, event), Transition::Stays);
        }
    }

    #[test]
    fn the_working_states_move_among_themselves_freely() {
        // Authorise, execute, record, and round again. Ordering these against
        // each other would make ordinary progress look like a fault.
        let mut state = RunState::ToolResultRecorded;
        for event in [
            TaskEventType::ToolAuthorized,
            TaskEventType::ToolSucceeded,
            TaskEventType::ToolAuthorized,
            TaskEventType::ToolFailed,
        ] {
            match advance(state, event) {
                Transition::To(next) => state = next,
                other => panic!("{event:?} from {state:?} was {other:?}"),
            }
        }
        assert_eq!(state, RunState::ToolResultRecorded);
    }

    #[test]
    fn a_budget_stop_is_not_a_failure_and_a_policy_stop_is_not_either() {
        // They read very differently to somebody scanning a list, and only one
        // of the three is a fault.
        assert_eq!(
            advance(RunState::Running, TaskEventType::RunStoppedByBudget),
            Transition::To(RunState::StoppedByBudget)
        );
        assert_eq!(
            advance(RunState::Running, TaskEventType::RunStoppedByPolicy),
            Transition::To(RunState::StoppedByPolicy)
        );
        assert!(!RunState::StoppedByBudget.is_success());
        assert_ne!(RunState::StoppedByBudget, RunState::Failed);
        assert_ne!(RunState::StoppedByPolicy, RunState::Failed);
    }

    #[test]
    fn the_states_written_by_the_earlier_store_still_read() {
        // A database from before the states were explicit must still list its
        // runs rather than dropping them.
        assert_eq!(RunState::from_str("timed_out"), Some(RunState::StoppedByBudget));
        assert_eq!(
            RunState::from_str("interrupted"),
            Some(RunState::DegradedNeedsHuman)
        );
        assert_eq!(RunState::from_str("nonsense"), None);
    }

    #[test]
    fn every_state_round_trips_through_its_database_spelling() {
        for state in RunState::ALL {
            assert_eq!(RunState::from_str(state.as_str()), Some(*state));
        }
    }

    #[test]
    fn every_ending_says_something_a_person_can_act_on() {
        for state in RunState::ALL {
            let sentence = state.describe();
            assert!(sentence.ends_with('.'), "{state:?}: {sentence}");
        }
    }
}

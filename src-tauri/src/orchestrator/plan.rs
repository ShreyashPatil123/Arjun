//! A plan the model cannot extend, and a run that knows when to stop.
//!
//! PS step 19: *"The plan includes a maximum number of steps, maximum execution
//! time, permitted tools, permitted files, model budget, and stop conditions.
//! The model is not allowed to extend the plan indefinitely."*
//!
//! And Part C, on failure behaviour: *"Agent loop repeats → Stop at the
//! step/time/tool budget and show the incomplete plan."*
//!
//! Those two sentences describe the difference between an agent and a runaway
//! process. An agent that cannot stop is not more capable than one that can —
//! it is a machine that will read the same document forty times while somebody
//! watches a spinner, and then produce nothing.
//!
//! ## The budget is set before the model sees anything
//!
//! Every limit here is fixed when the plan is created and is never adjusted by
//! anything the model emits. A model that could raise its own step budget has no
//! budget; a model that could add a tool to its permitted set has no tool
//! policy. Both are asked for by the problem statement precisely because both
//! are the obvious shortcut.
//!
//! ## Stopping is a result, not a failure
//!
//! A run that exhausts its budget returns what it managed, what it did not, and
//! why it stopped. Showing an incomplete plan honestly is far more useful than
//! either pretending it finished or discarding the work.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::tools::{ToolCall, ToolName};

/// Limits fixed when the plan is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Budget {
    /// Most steps this task may take.
    pub max_steps: u32,
    /// Wall-clock ceiling for the whole task.
    #[serde(with = "duration_seconds")]
    pub max_duration: Duration,
    /// Tools this task may use. A tool outside this set is refused even when the
    /// user holds the permission — the plan is narrower than the person.
    pub permitted_tools: Vec<ToolName>,
    /// How many times the same call may repeat before it is treated as a loop.
    pub repeat_limit: u32,
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(value: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(value.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

impl Budget {
    /// Sensible limits for an ordinary desk task.
    ///
    /// Twelve steps is enough for read → extract → search → compare → calculate
    /// → draft → validate with room to recover from a couple of mistakes, and
    /// few enough that a loop is caught before a person gives up waiting.
    pub fn standard(permitted_tools: Vec<ToolName>) -> Self {
        Self {
            max_steps: 12,
            max_duration: Duration::from_secs(10 * 60),
            permitted_tools,
            repeat_limit: 3,
        }
    }

    pub fn permits(&self, tool: ToolName) -> bool {
        self.permitted_tools.contains(&tool)
    }
}

/// One intended step, written when the plan is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanStep {
    pub ordinal: u32,
    /// What this step is for, in the user's terms.
    pub intent: String,
    pub done: bool,
}

/// Why a run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "reason")]
pub enum StopReason {
    /// Every step finished.
    Completed,
    /// The step budget ran out.
    StepsExhausted { taken: u32, allowed: u32 },
    /// The clock ran out.
    TimeExhausted { allowed_seconds: u64 },
    /// The same call kept coming back — the agent is going in circles.
    Looping { tool: String, repeats: u32 },
    /// Waiting on a person. Not a failure; the run resumes when they answer.
    AwaitingApproval { tool: String },
    /// A step failed and the plan cannot continue past it.
    Failed { detail: String },
}

impl StopReason {
    /// Whether the task got where it was going.
    pub fn is_success(&self) -> bool {
        matches!(self, StopReason::Completed)
    }

    /// What to tell the person, phrased so an incomplete run is legible rather
    /// than alarming.
    ///
    /// Always a complete sentence. Several variants embed a detail written
    /// elsewhere, which may or may not be punctuated, and a status line that
    /// sometimes trails off reads as though the message itself was truncated.
    pub fn explain(&self) -> String {
        let mut text = self.body();
        if !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
            text.push('.');
        }
        text
    }

    fn body(&self) -> String {
        match self {
            StopReason::Completed => "Finished.".to_string(),
            StopReason::StepsExhausted { taken, allowed } => format!(
                "Stopped after {taken} of {allowed} permitted steps. The work below is what was \
                 completed; the remaining steps were not attempted."
            ),
            StopReason::TimeExhausted { allowed_seconds } => format!(
                "Stopped after {} minutes, the time allowed for one task. The work below is what \
                 was completed.",
                allowed_seconds / 60
            ),
            StopReason::Looping { tool, repeats } => format!(
                "Stopped: the same {tool} call was attempted {repeats} times without progress, so \
                 the task was going in circles rather than getting closer to an answer."
            ),
            StopReason::AwaitingApproval { tool } => {
                format!("Waiting for you to approve the request to {tool}.")
            }
            StopReason::Failed { detail } => format!("Stopped: {detail}"),
        }
    }
}

/// A plan being carried out.
pub struct PlanRun {
    pub task_id: String,
    pub steps: Vec<PlanStep>,
    pub budget: Budget,
    started_at: Instant,
    steps_taken: u32,
    /// How many times each distinct call has been seen.
    seen: HashMap<String, u32>,
    stopped: Option<StopReason>,
}

/// Whether the run may take another step, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuation {
    Proceed,
    Stop(StopReason),
}

impl PlanRun {
    pub fn new(task_id: impl Into<String>, steps: Vec<String>, budget: Budget) -> Self {
        Self {
            task_id: task_id.into(),
            steps: steps
                .into_iter()
                .enumerate()
                .map(|(i, intent)| PlanStep {
                    ordinal: i as u32 + 1,
                    intent,
                    done: false,
                })
                .collect(),
            budget,
            started_at: Instant::now(),
            steps_taken: 0,
            seen: HashMap::new(),
            stopped: None,
        }
    }

    /// For tests and for resuming a run that waited on a person.
    pub fn started_at(&mut self, when: Instant) {
        self.started_at = when;
    }

    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    pub fn stopped(&self) -> Option<&StopReason> {
        self.stopped.as_ref()
    }

    /// Steps that were planned but never reached.
    ///
    /// Shown rather than hidden: a person who can see what was skipped can
    /// decide whether the partial answer is usable, and one who cannot has to
    /// assume the worst.
    pub fn unfinished(&self) -> Vec<&PlanStep> {
        self.steps.iter().filter(|s| !s.done).collect()
    }

    /// Checks whether the run may make this call.
    ///
    /// Budgets are checked before the tool gateway, because a task that is out
    /// of time should not be asking about permissions — and because the answer
    /// "you have run out of steps" is more useful than "that path is fine but
    /// nothing more will happen".
    pub fn may_call(&mut self, call: &ToolCall) -> Continuation {
        if let Some(reason) = &self.stopped {
            return Continuation::Stop(reason.clone());
        }

        // Time first: an overrunning task should stop even if it has steps left.
        if self.started_at.elapsed() >= self.budget.max_duration {
            return self.halt(StopReason::TimeExhausted {
                allowed_seconds: self.budget.max_duration.as_secs(),
            });
        }

        if self.steps_taken >= self.budget.max_steps {
            return self.halt(StopReason::StepsExhausted {
                taken: self.steps_taken,
                allowed: self.budget.max_steps,
            });
        }

        // A tool outside the plan is refused even when the person could use it
        // elsewhere. The plan is narrower than the permission, deliberately.
        let Some(tool) = ToolName::from_str(&call.tool) else {
            return self.halt(StopReason::Failed {
                detail: format!("the model asked for a tool that does not exist: {:?}", call.tool),
            });
        };

        if !self.budget.permits(tool) {
            return self.halt(StopReason::Failed {
                detail: format!(
                    "{} is not among the tools this task was allowed to use",
                    tool.as_str()
                ),
            });
        }

        // Loop detection. Keyed on the whole call, so re-reading a *different*
        // file is progress while re-reading the same one is not.
        let fingerprint = format!("{}::{}", call.tool, call.arguments);
        let repeats = self.seen.entry(fingerprint).or_insert(0);
        *repeats += 1;
        if *repeats > self.budget.repeat_limit {
            let repeats = *repeats;
            return self.halt(StopReason::Looping {
                tool: tool.as_str().to_string(),
                repeats,
            });
        }

        Continuation::Proceed
    }

    /// Records that a step ran.
    pub fn record_step(&mut self) {
        self.steps_taken += 1;
        if let Some(next) = self.steps.iter_mut().find(|s| !s.done) {
            next.done = true;
        }
    }

    /// Records that a tool call was spent, without claiming a step is finished.
    ///
    /// The distinction matters wherever one planned step takes more than one
    /// call. [`Self::record_step`] advances the checklist on every call, which
    /// is right when the caller drives the plan a step at a time. On the agent
    /// path a model may search four times to satisfy one step, and ticking four
    /// steps off would tell an operator the document had been produced and
    /// checked when nothing of the sort had happened.
    ///
    /// So the budget is spent either way, and only the claim differs.
    pub fn record_call(&mut self) {
        self.steps_taken += 1;
    }

    /// Ends the run because a person has to answer.
    ///
    /// Not a failure — the budget is preserved so the run continues from here
    /// once they do.
    pub fn await_approval(&mut self, tool: ToolName) -> StopReason {
        let reason = StopReason::AwaitingApproval {
            tool: tool.describe().to_string(),
        };
        self.stopped = Some(reason.clone());
        reason
    }

    /// Clears a pause so the run can carry on after an approval.
    pub fn resume(&mut self) {
        if matches!(self.stopped, Some(StopReason::AwaitingApproval { .. })) {
            self.stopped = None;
        }
    }

    pub fn complete(&mut self) -> StopReason {
        for step in &mut self.steps {
            step.done = true;
        }
        let reason = StopReason::Completed;
        self.stopped = Some(reason.clone());
        reason
    }

    fn halt(&mut self, reason: StopReason) -> Continuation {
        self.stopped = Some(reason.clone());
        Continuation::Stop(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolName> {
        vec![
            ToolName::SearchDocuments,
            ToolName::ReadScopedFile,
            ToolName::RunCalculation,
        ]
    }

    fn run() -> PlanRun {
        PlanRun::new(
            "task-1",
            vec![
                "Read the inspection report".into(),
                "Find the relevant SOP".into(),
                "Calculate the deviation".into(),
            ],
            Budget::standard(tools()),
        )
    }

    fn search(query: &str) -> ToolCall {
        ToolCall::new("search_documents", json!({ "query": query }))
    }

    #[test]
    fn a_run_within_its_budget_proceeds() {
        let mut run = run();
        assert_eq!(run.may_call(&search("wall thickness")), Continuation::Proceed);
    }

    #[test]
    fn the_step_budget_stops_the_run_and_says_how_far_it_got() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget {
                max_steps: 2,
                ..Budget::standard(tools())
            },
        );

        for i in 0..2 {
            assert_eq!(run.may_call(&search(&format!("q{i}"))), Continuation::Proceed);
            run.record_step();
        }

        match run.may_call(&search("q3")) {
            Continuation::Stop(StopReason::StepsExhausted { taken, allowed }) => {
                assert_eq!((taken, allowed), (2, 2));
            }
            other => panic!("expected the step budget to stop it, got {other:?}"),
        }
    }

    #[test]
    fn the_time_budget_stops_the_run_even_with_steps_left() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget {
                max_duration: Duration::from_secs(60),
                ..Budget::standard(tools())
            },
        );
        // Pretend the task started well over an hour ago.
        run.started_at(Instant::now() - Duration::from_secs(3700));

        assert!(matches!(
            run.may_call(&search("anything")),
            Continuation::Stop(StopReason::TimeExhausted { .. })
        ));
    }

    /// The plan is narrower than the person: a tool the user could use elsewhere
    /// is still refused if this task was not given it.
    #[test]
    fn a_tool_outside_the_plan_is_refused_even_when_the_user_holds_it() {
        let mut run = run();
        let call = ToolCall::new(
            "execute_code",
            json!({ "language": "python", "source": "print(1)" }),
        );

        match run.may_call(&call) {
            Continuation::Stop(StopReason::Failed { detail }) => {
                assert!(detail.contains("not among the tools"), "{detail}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_tool_that_does_not_exist_stops_the_run() {
        let mut run = run();
        let call = ToolCall::new("delete_everything", json!({}));
        assert!(matches!(
            run.may_call(&call),
            Continuation::Stop(StopReason::Failed { .. })
        ));
    }

    // ── Loops ────────────────────────────────────────────────────────────

    /// The failure this exists to catch: an agent reading the same document
    /// forty times while somebody watches a spinner.
    #[test]
    fn repeating_the_same_call_is_detected_as_a_loop() {
        let mut run = run();
        let call = search("wall thickness");

        for _ in 0..3 {
            assert_eq!(run.may_call(&call), Continuation::Proceed);
        }

        match run.may_call(&call) {
            Continuation::Stop(StopReason::Looping { tool, repeats }) => {
                assert_eq!(tool, "search_documents");
                assert_eq!(repeats, 4);
            }
            other => panic!("expected a loop to be caught, got {other:?}"),
        }
    }

    /// Re-reading a *different* file is progress; only the identical call is not.
    #[test]
    fn different_calls_to_the_same_tool_are_not_a_loop() {
        let mut run = run();
        for i in 0..8 {
            assert_eq!(
                run.may_call(&search(&format!("query {i}"))),
                Continuation::Proceed,
                "distinct queries should not look like a loop"
            );
        }
    }

    // ── Stopping honestly ────────────────────────────────────────────────

    #[test]
    fn an_incomplete_run_reports_the_steps_it_never_reached() {
        let mut run = run();
        run.record_step();

        let unfinished = run.unfinished();
        assert_eq!(unfinished.len(), 2);
        assert_eq!(unfinished[0].intent, "Find the relevant SOP");
    }

    #[test]
    fn a_completed_run_has_nothing_unfinished() {
        let mut run = run();
        assert!(run.complete().is_success());
        assert!(run.unfinished().is_empty());
    }

    #[test]
    fn every_stop_reason_explains_itself_in_plain_words() {
        let reasons = [
            StopReason::Completed,
            StopReason::StepsExhausted { taken: 12, allowed: 12 },
            StopReason::TimeExhausted { allowed_seconds: 600 },
            StopReason::Looping { tool: "search_documents".into(), repeats: 4 },
            StopReason::AwaitingApproval { tool: "write a file".into() },
            StopReason::Failed { detail: "the sandbox refused".into() },
        ];

        for reason in reasons {
            let text = reason.explain();
            assert!(!text.is_empty());
            assert!(text.ends_with('.'), "{text:?} should read as a sentence");
        }
    }

    #[test]
    fn the_step_exhausted_message_tells_a_person_what_they_have() {
        let text = StopReason::StepsExhausted { taken: 12, allowed: 12 }.explain();
        assert!(text.contains("what was completed"));
        assert!(text.contains("not attempted"));
    }

    // ── Approval pauses rather than fails ────────────────────────────────

    #[test]
    fn waiting_for_approval_is_not_a_failure_and_the_run_resumes() {
        let mut run = run();
        let reason = run.await_approval(ToolName::WriteScopedFile);

        assert!(!reason.is_success());
        assert!(matches!(
            run.may_call(&search("anything")),
            Continuation::Stop(StopReason::AwaitingApproval { .. })
        ));

        run.resume();
        assert_eq!(run.may_call(&search("anything")), Continuation::Proceed);
    }

    /// Resuming must not clear a stop that was not an approval pause.
    #[test]
    fn resuming_does_not_revive_a_run_that_ran_out_of_steps() {
        let mut run = PlanRun::new(
            "task-1",
            vec!["one".into()],
            Budget { max_steps: 0, ..Budget::standard(tools()) },
        );
        assert!(matches!(run.may_call(&search("q")), Continuation::Stop(_)));

        run.resume();
        assert!(
            matches!(run.may_call(&search("q")), Continuation::Stop(_)),
            "an exhausted run must stay stopped"
        );
    }

    /// Nothing the model emits may widen the budget.
    #[test]
    fn the_budget_is_fixed_when_the_plan_is_made() {
        let budget = Budget::standard(tools());
        let run = PlanRun::new("task-1", vec!["one".into()], budget.clone());

        // The only way to change limits is to build a different plan.
        assert_eq!(run.budget.max_steps, budget.max_steps);
        assert_eq!(run.budget.permitted_tools, budget.permitted_tools);
    }
}

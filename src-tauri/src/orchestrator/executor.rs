//! Running a task one step at a time, pausing when a person is needed.
//!
//! This is where the plan, the gateway and the tools meet. It deliberately runs
//! **one step per call** rather than looping internally, for three reasons:
//!
//! - The interface can show what is happening as it happens, instead of a
//!   spinner and then an answer.
//! - A person can interrupt between steps.
//! - Approval is a natural pause rather than a callback threaded through a loop.
//!
//! ## A refusal is a result, not a crash
//!
//! When the gateway refuses a call, the refusal is handed back to the model as
//! that call's *result*. An agent told "path must be inside the task workspace"
//! adjusts and carries on; one that gets an exception has nothing to work with.
//! This is the single most important behaviour here, because refusals are the
//! common case, not the exception — a probabilistic system proposes invalid
//! actions routinely and that is not a malfunction.
//!
//! ## Repeated failure is progress information
//!
//! Three refusals in a row means the model is not learning from them, and the
//! remaining budget will be spent producing the same refusal again. That is
//! caught here rather than left to exhaust the step budget, so the person gets
//! "it kept being refused for this reason" instead of "it ran out of steps".

use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::gateway::{GatewayVerdict, TaskContext, ToolGateway};
use super::plan::{Continuation, PlanRun, StopReason};
use super::tools::{ToolCall, ToolName};
use crate::audit::{AuditKind, AuditService};

/// Runs a tool that the gateway has already permitted.
///
/// A trait so the loop is testable without a filesystem, a sandbox or a model.
///
/// `resolved_path` is the path the gateway checked and approved, not the one the
/// model wrote. The runner must use it rather than re-deriving the path from the
/// call: two pieces of code resolving the same string separately will eventually
/// disagree, and the one that disagrees here is the one holding a file handle.
pub trait ToolRunner {
    fn run(
        &self,
        tool: ToolName,
        call: &ToolCall,
        resolved_path: Option<&std::path::Path>,
    ) -> Result<String, String>;
}

/// How many consecutive refusals mean the model is not learning from them.
const CONSECUTIVE_REFUSAL_LIMIT: u32 = 3;

/// What happened on one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepOutcome {
    pub tool: String,
    /// What to feed back to the model as this call's result — the tool's output
    /// on success, or the refusal text when it was refused.
    pub result: String,
    pub permitted: bool,
    pub took_ms: u64,
}

/// Where the task is now.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "state")]
pub enum TaskState {
    /// The step ran. Feed `outcome.result` back and ask for the next call.
    Stepped { outcome: StepOutcome },
    /// A person has to answer before this can proceed.
    AwaitingApproval {
        tool: String,
        /// Target, arguments and consequence, ready to show.
        prompt: String,
    },
    /// Nothing more will happen.
    Finished {
        reason: StopReason,
        /// Steps that were planned but never reached.
        unfinished: Vec<String>,
    },
}

impl TaskState {
    pub fn is_finished(&self) -> bool {
        matches!(self, TaskState::Finished { .. })
    }
}

pub struct Executor<'a> {
    pub runner: &'a dyn ToolRunner,
    pub audit: Option<&'a AuditService>,
    /// Counts refusals since the last call that actually ran.
    consecutive_refusals: u32,
}

impl<'a> Executor<'a> {
    pub fn new(runner: &'a dyn ToolRunner, audit: Option<&'a AuditService>) -> Self {
        Self {
            runner,
            audit,
            consecutive_refusals: 0,
        }
    }

    /// Takes one step: check the budget, check the gateway, then run or pause.
    pub fn step(
        &mut self,
        run: &mut PlanRun,
        context: &TaskContext<'_>,
        call: &ToolCall,
    ) -> TaskState {
        // 1. Does the plan allow another step at all? Checked first, so a task
        //    that is out of time is not told about permissions instead.
        if let Continuation::Stop(reason) = run.may_call(call) {
            return self.finish(run, reason, context);
        }

        // 2. Does the gateway allow this particular call?
        let started = Instant::now();
        let verdict = ToolGateway::decide(call, context);

        match verdict {
            GatewayVerdict::Refuse { reason } => {
                self.consecutive_refusals += 1;

                if self.consecutive_refusals >= CONSECUTIVE_REFUSAL_LIMIT {
                    // The model is not adjusting. Spending the rest of the
                    // budget producing this same refusal helps nobody.
                    let stop = StopReason::Failed {
                        detail: format!(
                            "the same kind of request was refused {} times in a row and the task \
                             was not adapting. The last reason was: {reason}",
                            self.consecutive_refusals
                        ),
                    };
                    return self.finish(run, stop, context);
                }

                self.record(context, call, false, &reason);

                // Handed back as the call's result. The step is counted, because
                // a refused attempt still consumed a turn of the model's budget.
                run.record_step();
                TaskState::Stepped {
                    outcome: StepOutcome {
                        tool: call.tool.clone(),
                        result: reason,
                        permitted: false,
                        took_ms: started.elapsed().as_millis() as u64,
                    },
                }
            }

            GatewayVerdict::NeedsApproval { tool, summary, .. } => {
                // Not a failure and not a step. The budget is untouched so the
                // wait costs the task nothing.
                run.await_approval(tool);
                self.record(context, call, false, "awaiting approval");
                TaskState::AwaitingApproval {
                    tool: tool.as_str().to_string(),
                    prompt: summary,
                }
            }

            GatewayVerdict::Allow {
                tool,
                ref resolved_path,
            } => {
                let result = match self.runner.run(tool, call, resolved_path.as_deref()) {
                    Ok(output) => {
                        self.consecutive_refusals = 0;
                        output
                    }
                    Err(failure) => {
                        // A tool that ran and failed is different from one that
                        // was refused: the model may well be able to recover, so
                        // the error goes back as the result.
                        format!("The tool ran but failed: {failure}")
                    }
                };

                self.record(context, call, true, &result);
                run.record_step();

                TaskState::Stepped {
                    outcome: StepOutcome {
                        tool: call.tool.clone(),
                        result,
                        permitted: true,
                        took_ms: started.elapsed().as_millis() as u64,
                    },
                }
            }
        }
    }

    /// Resumes a task after a person approved the pending action.
    pub fn approved(&mut self, run: &mut PlanRun) {
        run.resume();
        self.consecutive_refusals = 0;
    }

    /// Ends a task because a person rejected the pending action.
    ///
    /// A rejection is a decision, not an error — the task stops cleanly and says
    /// who stopped it.
    pub fn rejected(&mut self, run: &mut PlanRun, context: &TaskContext<'_>) -> TaskState {
        let reason = StopReason::Failed {
            detail: format!(
                "{} declined the action this task needed to continue",
                context.session.user.display_name
            ),
        };
        run.resume();
        self.finish(run, reason, context)
    }

    fn finish(
        &self,
        run: &mut PlanRun,
        reason: StopReason,
        context: &TaskContext<'_>,
    ) -> TaskState {
        let unfinished: Vec<String> = run
            .unfinished()
            .into_iter()
            .map(|step| step.intent.clone())
            .collect();

        if let Some(audit) = self.audit {
            let _ = audit.record(
                &context.session.user.id,
                AuditKind::Task,
                format!("Task {} ended: {}", run.task_id, reason.explain()),
                Some(serde_json::json!({
                    "taskId": run.task_id,
                    "stepsTaken": run.steps_taken(),
                    "unfinished": unfinished,
                    "reason": reason,
                })),
            );
        }

        TaskState::Finished { reason, unfinished }
    }

    fn record(&self, context: &TaskContext<'_>, call: &ToolCall, permitted: bool, detail: &str) {
        let Some(audit) = self.audit else { return };

        // The arguments are recorded, but never the tool's output: a document's
        // text is exactly what PS step 14 says must not be copied into a log
        // that more people can read than could read the document.
        let _ = audit.record(
            &context.session.user.id,
            AuditKind::PolicyDecision,
            format!(
                "{} {}",
                if permitted { "Ran" } else { "Refused" },
                call.tool
            ),
            Some(serde_json::json!({
                "tool": call.tool,
                "arguments": call.arguments,
                "permitted": permitted,
                "detail": if permitted { "ran" } else { detail },
            })),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, Session, User};
    use crate::orchestrator::plan::Budget;
    use crate::policy::ApprovalState;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct FakeRunner {
        calls: Mutex<Vec<String>>,
        fail: bool,
    }

    impl FakeRunner {
        fn working() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail: true,
            }
        }
        fn ran(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ToolRunner for FakeRunner {
        fn run(
            &self,
            tool: ToolName,
            _call: &ToolCall,
            _resolved_path: Option<&std::path::Path>,
        ) -> Result<String, String> {
            self.calls.lock().unwrap().push(tool.as_str().to_string());
            if self.fail {
                Err("the index was unavailable".into())
            } else {
                Ok("3 passages found".into())
            }
        }
    }

    fn session(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    fn tools() -> Vec<ToolName> {
        vec![
            ToolName::SearchDocuments,
            ToolName::ReadScopedFile,
            ToolName::WriteScopedFile,
        ]
    }

    fn run() -> PlanRun {
        PlanRun::new(
            "task-1",
            vec!["Search the SOPs".into(), "Draft the note".into()],
            Budget::standard(tools()),
        )
    }

    fn context<'a>(session: &'a Session, roots: &'a [PathBuf]) -> TaskContext<'a> {
        TaskContext {
            session,
            workspace_roots: roots,
            confidential_work_permitted: true,
            approval: ApprovalState::NotRequested,
        }
    }

    fn search(query: &str) -> ToolCall {
        ToolCall::new("search_documents", json!({ "query": query }))
    }

    #[test]
    fn a_permitted_call_runs_and_returns_its_output() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = executor.step(&mut run(), &context(&s, &roots), &search("wall thickness"));

        match state {
            TaskState::Stepped { outcome } => {
                assert!(outcome.permitted);
                assert_eq!(outcome.result, "3 passages found");
            }
            other => panic!("expected a step, got {other:?}"),
        }
        assert_eq!(runner.ran(), vec!["knowledge.search_authorized"]);
    }

    /// The most important behaviour here: a refusal comes back as the call's
    /// result so the model can adjust, rather than as an error it cannot use.
    #[test]
    fn a_refusal_is_handed_back_as_the_result_and_the_task_continues() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let state = executor.step(
            &mut plan,
            &context(&s, &roots),
            &ToolCall::new("read_scoped_file", json!({ "path": "C:/Windows/System32/config" })),
        );

        match state {
            TaskState::Stepped { outcome } => {
                assert!(!outcome.permitted);
                assert!(outcome.result.contains("outside this task's workspace"));
            }
            other => panic!("expected a refusal handed back as a step, got {other:?}"),
        }
        assert!(runner.ran().is_empty(), "a refused call must never reach the tool");
    }

    /// Spending the rest of the budget producing the same refusal helps nobody.
    #[test]
    fn repeated_refusals_stop_the_task_and_name_the_reason() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let bad = |n: u32| {
            ToolCall::new(
                "read_scoped_file",
                json!({ "path": format!("C:/Windows/attempt-{n}") }),
            )
        };

        assert!(!executor.step(&mut plan, &context(&s, &roots), &bad(1)).is_finished());
        assert!(!executor.step(&mut plan, &context(&s, &roots), &bad(2)).is_finished());

        match executor.step(&mut plan, &context(&s, &roots), &bad(3)) {
            TaskState::Finished { reason, .. } => {
                let text = reason.explain();
                assert!(text.contains("3 times in a row"), "{text}");
                assert!(text.contains("workspace"), "the last reason should be quoted: {text}");
            }
            other => panic!("expected the task to stop, got {other:?}"),
        }
    }

    /// A successful call clears the counter, so intermittent refusals during a
    /// task that is otherwise progressing do not accumulate into a stop.
    #[test]
    fn a_successful_step_resets_the_refusal_counter() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let bad = ToolCall::new("read_scoped_file", json!({ "path": "C:/Windows/x" }));

        executor.step(&mut plan, &context(&s, &roots), &bad);
        executor.step(&mut plan, &context(&s, &roots), &search("progress"));
        executor.step(&mut plan, &context(&s, &roots), &bad);

        assert!(
            !executor.step(&mut plan, &context(&s, &roots), &bad).is_finished(),
            "the counter should have been reset by the successful step"
        );
    }

    // ── Approval ─────────────────────────────────────────────────────────

    #[test]
    fn a_write_pauses_for_a_person_with_a_prompt_worth_reading() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        let state = executor.step(
            &mut run(),
            &context(&s, &roots),
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
            ),
        );

        match state {
            TaskState::AwaitingApproval { tool, prompt } => {
                assert_eq!(tool, "workspace.write_text");
                assert!(prompt.contains("note.txt"));
                assert!(prompt.contains("5 byte(s)"));
            }
            other => panic!("expected a pause, got {other:?}"),
        }
        assert!(runner.ran().is_empty(), "nothing runs before approval");
    }

    /// Waiting must cost the task nothing, or a slow reviewer would burn the
    /// budget the task needs to finish.
    #[test]
    fn waiting_for_approval_consumes_no_steps() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        executor.step(
            &mut plan,
            &context(&s, &roots),
            &ToolCall::new(
                "write_scoped_file",
                json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
            ),
        );
        assert_eq!(plan.steps_taken(), 0);
    }

    #[test]
    fn approving_lets_the_write_proceed() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        let call = ToolCall::new(
            "write_scoped_file",
            json!({ "path": "C:/arjun/tasks/1/note.txt", "content": "hello" }),
        );

        executor.step(&mut plan, &context(&s, &roots), &call);
        executor.approved(&mut plan);

        let mut granted = context(&s, &roots);
        granted.approval = ApprovalState::Granted;

        match executor.step(&mut plan, &granted, &call) {
            TaskState::Stepped { outcome } => assert!(outcome.permitted),
            other => panic!("expected the write to proceed, got {other:?}"),
        }
    }

    /// A rejection is a decision, not an error.
    #[test]
    fn rejecting_stops_the_task_cleanly_and_names_who_stopped_it() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = run();

        match executor.rejected(&mut plan, &context(&s, &roots)) {
            TaskState::Finished { reason, .. } => {
                assert!(reason.explain().contains("Kiran"));
                assert!(reason.explain().contains("declined"));
            }
            other => panic!("expected the task to stop, got {other:?}"),
        }
    }

    // ── Budgets and honesty ──────────────────────────────────────────────

    #[test]
    fn an_exhausted_budget_finishes_and_lists_what_was_never_reached() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut plan = PlanRun::new(
            "task-1",
            vec!["Search the SOPs".into(), "Draft the note".into()],
            Budget { max_steps: 1, ..Budget::standard(tools()) },
        );

        executor.step(&mut plan, &context(&s, &roots), &search("first"));

        match executor.step(&mut plan, &context(&s, &roots), &search("second")) {
            TaskState::Finished { unfinished, .. } => {
                assert_eq!(unfinished, vec!["Draft the note"]);
            }
            other => panic!("expected the task to finish, got {other:?}"),
        }
    }

    /// A tool that ran and failed is recoverable; the model may try another way.
    #[test]
    fn a_tool_failure_comes_back_as_a_result_rather_than_ending_the_task() {
        let runner = FakeRunner::failing();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];

        match executor.step(&mut run(), &context(&s, &roots), &search("anything")) {
            TaskState::Stepped { outcome } => {
                assert!(outcome.permitted, "it was permitted; it simply failed");
                assert!(outcome.result.contains("the index was unavailable"));
            }
            other => panic!("expected a step, got {other:?}"),
        }
    }

    #[test]
    fn nothing_runs_while_the_network_is_reachable() {
        let runner = FakeRunner::working();
        let mut executor = Executor::new(&runner, None);
        let s = session(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/1")];
        let mut ctx = context(&s, &roots);
        ctx.confidential_work_permitted = false;

        executor.step(&mut run(), &ctx, &search("anything"));
        assert!(runner.ran().is_empty());
    }
}

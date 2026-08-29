//! Deterministic checks at the points where a run changes what it is doing.
//!
//! ## What this is for, and what it is not
//!
//! ARJUN already refuses things: the tool gateway decides every call, the policy
//! gateway decides every entitlement, the broker decides every outbound packet.
//! This module does not replace any of them and must never be mistaken for a
//! second copy of one — two enforcement paths for the same rule drift, and the
//! weaker one silently becomes the real policy.
//!
//! What it adds is the ability to place a check *where the code is, rather than
//! where a rule happens to live*. A deployment needs to say "nothing may be
//! written outside the workspace" once, and have that hold whether the write
//! arrives through the artifact writer, the sandbox, or a subagent. Threading
//! that condition through three call sites is how one of them ends up missing
//! it. A hook at [`HookPoint::BeforeArtifactWrite`] is reached by all three.
//!
//! ## Hooks are not instructions
//!
//! The single most important property here: **a hook is code, and the model
//! cannot see it, address it, or argue with it.** Nothing in a prompt, a skill
//! body, a retrieved document or a tool result can register, disable or
//! influence a hook. This is why policy belongs here rather than in the system
//! prompt, and it is the difference between a rule and a request. A model told
//! "never write outside the workspace" complies until a document it retrieves
//! tells it otherwise; a hook does not read documents.
//!
//! ## Failing closed
//!
//! Points come in two kinds, and the distinction decides what a broken hook
//! costs:
//!
//! - A **gate** runs before something happens and may stop it. If a hook at a
//!   gate fails — panics, times out, returns nonsense — the gate **blocks**.
//!   Not because blocking is nice, but because the alternative is a deployment
//!   whose safety property can be removed by crashing the thing that enforces
//!   it. A hook that cannot answer has not said yes.
//! - An **observation** runs after the fact and cannot stop anything, because
//!   there is nothing left to stop. A failure there is recorded and the run
//!   continues. Letting an observation block would mean a broken logger could
//!   halt a task that had already succeeded.
//!
//! [`HookRegistry::dispatch`] enforces this: it catches panics rather than
//! trusting hooks not to have them, and converts a panic at a gate into a
//! refusal that names the hook.
//!
//! ## Output is evidence, and evidence is bounded
//!
//! Every dispatch produces a [`HookReport`] — typed, size-capped, and free of
//! document text by construction. It is persisted alongside the run's other
//! events so that "why did this run refuse to write that file?" has an answer
//! months later. The caps are not politeness: a hook that could return an
//! unbounded string would be a way to move a confidential document into an
//! event log that people without clearance read.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::sovereignty::OperatingMode;

pub mod policy;

#[cfg(test)]
mod tests;

/// The moments a run can be inspected at.
///
/// Ordered as a run encounters them, so reading the enum reads the lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HookPoint {
    /// A person signed in and a session opened.
    SessionStart,
    /// The sensitivity of the material is known.
    PromptClassified,
    /// A model was chosen for the run.
    ModelSelected,
    /// A skill's instructions were loaded into the run.
    SkillLoaded,
    /// About to send a request to the model.
    BeforeModelRequest,
    /// A model turn came back.
    AfterModelResponse,
    /// About to ask the gateway to authorise a tool call.
    BeforeToolAuthorize,
    /// A tool finished, either way.
    AfterToolExecute,
    /// About to put something to a person.
    ApprovalRequested,
    /// A person answered.
    ApprovalDecided,
    /// About to write a file a person will be handed.
    BeforeArtifactWrite,
    /// About to start the code sandbox.
    BeforeSandboxStart,
    /// About to replace older history with a summary.
    BeforeCompaction,
    /// History was replaced.
    AfterCompaction,
    /// A bounded worker is about to begin.
    SubagentStart,
    /// It ended.
    SubagentStop,
    /// The run finished, any way it finished.
    RunComplete,
}

impl HookPoint {
    pub const ALL: &'static [HookPoint] = &[
        HookPoint::SessionStart,
        HookPoint::PromptClassified,
        HookPoint::ModelSelected,
        HookPoint::SkillLoaded,
        HookPoint::BeforeModelRequest,
        HookPoint::AfterModelResponse,
        HookPoint::BeforeToolAuthorize,
        HookPoint::AfterToolExecute,
        HookPoint::ApprovalRequested,
        HookPoint::ApprovalDecided,
        HookPoint::BeforeArtifactWrite,
        HookPoint::BeforeSandboxStart,
        HookPoint::BeforeCompaction,
        HookPoint::AfterCompaction,
        HookPoint::SubagentStart,
        HookPoint::SubagentStop,
        HookPoint::RunComplete,
    ];

    /// Stable spelling for events and records.
    ///
    /// Written out rather than derived, so renaming a variant cannot rewrite
    /// history that already refers to it.
    pub const fn as_str(self) -> &'static str {
        match self {
            HookPoint::SessionStart => "session_start",
            HookPoint::PromptClassified => "prompt_classified",
            HookPoint::ModelSelected => "model_selected",
            HookPoint::SkillLoaded => "skill_loaded",
            HookPoint::BeforeModelRequest => "before_model_request",
            HookPoint::AfterModelResponse => "after_model_response",
            HookPoint::BeforeToolAuthorize => "before_tool_authorize",
            HookPoint::AfterToolExecute => "after_tool_execute",
            HookPoint::ApprovalRequested => "approval_requested",
            HookPoint::ApprovalDecided => "approval_decided",
            HookPoint::BeforeArtifactWrite => "before_artifact_write",
            HookPoint::BeforeSandboxStart => "before_sandbox_start",
            HookPoint::BeforeCompaction => "before_compaction",
            HookPoint::AfterCompaction => "after_compaction",
            HookPoint::SubagentStart => "subagent_start",
            HookPoint::SubagentStop => "subagent_stop",
            HookPoint::RunComplete => "run_complete",
        }
    }

    /// Whether a hook here can stop what is about to happen.
    ///
    /// A gate runs *before* an effect and has something left to prevent. An
    /// observation runs after, when refusing would be a claim about a thing
    /// that already happened.
    ///
    /// This is also what decides the cost of a broken hook — see the module
    /// note on failing closed — so it is a property of the point rather than
    /// something a hook chooses for itself. A hook that could declare its own
    /// point non-blocking could opt out of failing closed by crashing.
    pub const fn is_gate(self) -> bool {
        matches!(
            self,
            HookPoint::PromptClassified
                | HookPoint::ModelSelected
                | HookPoint::SkillLoaded
                | HookPoint::BeforeModelRequest
                | HookPoint::BeforeToolAuthorize
                | HookPoint::ApprovalRequested
                | HookPoint::BeforeArtifactWrite
                | HookPoint::BeforeSandboxStart
                | HookPoint::BeforeCompaction
                | HookPoint::SubagentStart
        )
    }
}

/// What a hook is told about the moment it was called at.
///
/// Deliberately narrow. A hook receives identifiers, classifications, paths and
/// counts — never a prompt, a passage, a document body or a model's output.
/// That is not a convenience: a hook's return value is persisted as evidence
/// read by people without clearance for the material, so material a hook never
/// receives is material it cannot leak. A hook needing the text of something to
/// decide is a hook asking the wrong question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookInput {
    /// [`HookPoint::SessionStart`]
    Session { user_id: String },
    /// [`HookPoint::PromptClassified`] and [`HookPoint::BeforeModelRequest`].
    Material {
        run_id: String,
        classification: Classification,
        mode: OperatingMode,
    },
    /// [`HookPoint::ModelSelected`]
    Model {
        run_id: String,
        model_id: String,
        /// True when the model runs on this machine.
        local: bool,
    },
    /// [`HookPoint::SkillLoaded`]
    Skill {
        run_id: String,
        skill: String,
        /// Digest of the instructions, so a changed skill is visible without
        /// the instructions themselves being copied into an event.
        sha256: String,
        /// Whether the skill declares that it needs the network.
        requires_network: bool,
        mode: OperatingMode,
    },
    /// [`HookPoint::BeforeToolAuthorize`] and [`HookPoint::AfterToolExecute`].
    Tool {
        run_id: String,
        tool: ToolName,
        /// Resolved target, when the call names one.
        path: Option<PathBuf>,
        mode: OperatingMode,
        /// Set only on [`HookPoint::AfterToolExecute`].
        succeeded: Option<bool>,
    },
    /// [`HookPoint::ApprovalRequested`] and [`HookPoint::ApprovalDecided`].
    Approval {
        run_id: String,
        approval_id: String,
        tool: ToolName,
        /// Set only on [`HookPoint::ApprovalDecided`].
        granted: Option<bool>,
    },
    /// [`HookPoint::BeforeArtifactWrite`]
    ArtifactWrite {
        run_id: String,
        path: PathBuf,
        /// Directories this run may write inside. Empty means none.
        roots: Vec<PathBuf>,
        classification: Classification,
    },
    /// [`HookPoint::BeforeSandboxStart`]
    Sandbox {
        run_id: String,
        language: String,
        /// Whether the sandbox this machine can offer actually isolates.
        isolated: bool,
        mode: OperatingMode,
    },
    /// [`HookPoint::BeforeCompaction`] and [`HookPoint::AfterCompaction`].
    Compaction {
        run_id: String,
        tokens_before: u32,
        /// Set only on [`HookPoint::AfterCompaction`].
        tokens_after: Option<u32>,
    },
    /// [`HookPoint::SubagentStart`] and [`HookPoint::SubagentStop`].
    Subagent {
        run_id: String,
        child_id: String,
        profile: String,
        /// Every tool the child is permitted. A child may never exceed this.
        allowed_tools: Vec<ToolName>,
        /// Set only on [`HookPoint::SubagentStop`].
        completed: Option<bool>,
    },
    /// [`HookPoint::RunComplete`]
    RunEnded { run_id: String, status: String },
}

impl HookInput {
    /// The run this is about, when it is about one.
    pub fn run_id(&self) -> Option<&str> {
        match self {
            HookInput::Session { .. } => None,
            HookInput::Material { run_id, .. }
            | HookInput::Model { run_id, .. }
            | HookInput::Skill { run_id, .. }
            | HookInput::Tool { run_id, .. }
            | HookInput::Approval { run_id, .. }
            | HookInput::ArtifactWrite { run_id, .. }
            | HookInput::Sandbox { run_id, .. }
            | HookInput::Compaction { run_id, .. }
            | HookInput::Subagent { run_id, .. }
            | HookInput::RunEnded { run_id, .. } => Some(run_id),
        }
    }
}

/// Longest refusal a hook may return.
///
/// Long enough to say what was refused and how to proceed; short enough that a
/// hook cannot use the field to move a document into the event log.
pub const MAX_REASON_CHARS: usize = 400;

/// Most annotations one hook may attach, and the longest each may be.
pub const MAX_ANNOTATIONS: usize = 8;
pub const MAX_ANNOTATION_CHARS: usize = 200;

/// What one hook decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookOutcome {
    /// Nothing to say. The overwhelmingly common answer.
    Proceed,
    /// Proceed, but record these. Facts for the trace, never instructions for
    /// the model: annotations are not put in front of it.
    Note(Vec<String>),
    /// Stop, for this reason. Only meaningful at a gate.
    Block { reason: String },
}

impl HookOutcome {
    /// A refusal, with the reason trimmed to the cap.
    pub fn block(reason: impl Into<String>) -> Self {
        HookOutcome::Block {
            reason: cap(reason.into(), MAX_REASON_CHARS),
        }
    }

    pub fn is_block(&self) -> bool {
        matches!(self, HookOutcome::Block { .. })
    }
}

/// Trims to `limit` characters, marking that it did.
///
/// Character-wise rather than byte-wise: slicing a UTF-8 string at an arbitrary
/// byte panics mid-character, and a panic inside the bounding code would be an
/// unusually silly way to take down a run.
fn cap(text: String, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text;
    }
    let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// One check.
///
/// `run` takes `&self` and returns a value: a hook holds no mutable state and
/// cannot accumulate anything across calls. That is what makes dispatch order
/// irrelevant and lets a gate's answer be reproduced from the input alone —
/// which is the property that makes a refusal reviewable.
pub trait Hook: Send + Sync {
    /// Stable identifier, recorded on every decision it makes.
    fn name(&self) -> &'static str;

    /// Where it runs.
    fn point(&self) -> HookPoint;

    fn run(&self, input: &HookInput) -> HookOutcome;
}

/// What a whole dispatch came to, ready to persist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookReport {
    pub point: String,
    pub run_id: Option<String>,
    /// True when something refused.
    pub blocked: bool,
    /// The refusing hook, when one refused.
    pub blocked_by: Option<String>,
    /// Why, already capped.
    pub reason: Option<String>,
    /// Hooks that ran without refusing.
    pub passed: Vec<String>,
    /// Hooks that failed to answer. At a gate these are also refusals.
    pub failed: Vec<String>,
    /// Bounded notes, by hook name.
    pub notes: BTreeMap<String, Vec<String>>,
}

impl HookReport {
    fn empty(point: HookPoint, run_id: Option<String>) -> Self {
        Self {
            point: point.as_str().to_string(),
            run_id,
            blocked: false,
            blocked_by: None,
            reason: None,
            passed: Vec::new(),
            failed: Vec::new(),
            notes: BTreeMap::new(),
        }
    }

    /// The sentence handed back to whatever was about to act.
    pub fn refusal(&self) -> Option<String> {
        self.reason.clone()
    }
}

/// The hooks a deployment has installed.
///
/// Built once at start-up from code, never from a file the model could reach
/// and never from anything a prompt can name. See the module note: a hook that
/// could be registered at runtime by something the model influences would be a
/// hook the model controls, which is the opposite of the point.
#[derive(Default)]
pub struct HookRegistry {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The registry a deployment actually runs with.
    ///
    /// The built-in policy hooks, and nothing else. A deployment adding its own
    /// does so here, in code that is compiled and reviewed.
    pub fn with_builtin_policy() -> Self {
        let mut registry = Self::new();
        for hook in policy::builtin() {
            registry.add_boxed(hook);
        }
        registry
    }

    pub fn add(&mut self, hook: impl Hook + 'static) -> &mut Self {
        self.hooks.push(Box::new(hook));
        self
    }

    pub fn add_boxed(&mut self, hook: Box<dyn Hook>) -> &mut Self {
        self.hooks.push(hook);
        self
    }

    /// How many hooks run at this point.
    pub fn count_at(&self, point: HookPoint) -> usize {
        self.hooks.iter().filter(|h| h.point() == point).count()
    }

    /// Runs every hook registered at `point`.
    ///
    /// ## Why every hook runs even after one refuses
    ///
    /// Stopping at the first refusal would make the report depend on
    /// registration order, and a reviewer asking "what else was wrong with this
    /// call?" would get one answer today and a different one after an unrelated
    /// hook was added. The first refusal is what the caller is told; the report
    /// holds all of them.
    ///
    /// ## Why panics are caught
    ///
    /// Not to be forgiving. A panicking hook at a gate is *counted as a
    /// refusal*, which is stricter than letting it propagate — an unwinding
    /// panic would abort the whole request, and a deployment that could disable
    /// a safety check by crashing it has no safety check. Catching converts a
    /// broken hook into a refusal that names it.
    pub fn dispatch(&self, point: HookPoint, input: &HookInput) -> HookReport {
        let mut report = HookReport::empty(point, input.run_id().map(str::to_string));

        for hook in self.hooks.iter().filter(|h| h.point() == point) {
            let name = hook.name();
            let outcome = catch_unwind(AssertUnwindSafe(|| hook.run(input)));

            match outcome {
                Ok(HookOutcome::Proceed) => report.passed.push(name.to_string()),
                Ok(HookOutcome::Note(notes)) => {
                    report.passed.push(name.to_string());
                    let bounded: Vec<String> = notes
                        .into_iter()
                        .take(MAX_ANNOTATIONS)
                        .map(|note| cap(note, MAX_ANNOTATION_CHARS))
                        .collect();
                    if !bounded.is_empty() {
                        report.notes.insert(name.to_string(), bounded);
                    }
                }
                Ok(HookOutcome::Block { reason }) => {
                    Self::record_block(&mut report, point, name, cap(reason, MAX_REASON_CHARS));
                }
                Err(_) => {
                    // The fail-closed path. The panic payload is deliberately
                    // not read: it is arbitrary text from a failing component,
                    // and this report is persisted where people without
                    // clearance read it.
                    report.failed.push(name.to_string());
                    Self::record_block(
                        &mut report,
                        point,
                        name,
                        format!(
                            "The {name} check could not complete, so this was not allowed to \
                             proceed. A check that cannot answer has not said yes."
                        ),
                    );
                }
            }
        }

        report
    }

    /// Books a refusal, keeping the first one as the reason given to the caller.
    ///
    /// A hook that refuses at an observation point is recorded and does *not*
    /// set `blocked`: there is nothing left to stop, and reporting a block that
    /// did not happen would tell a reviewer the run was prevented from doing
    /// something it in fact did.
    fn record_block(report: &mut HookReport, point: HookPoint, name: &str, reason: String) {
        if !point.is_gate() {
            report
                .notes
                .entry(name.to_string())
                .or_default()
                .push(cap(
                    format!("Objected after the fact, which cannot stop it: {reason}"),
                    MAX_ANNOTATION_CHARS,
                ));
            return;
        }
        if !report.blocked {
            report.blocked = true;
            report.blocked_by = Some(name.to_string());
            report.reason = Some(reason);
        } else {
            report
                .notes
                .entry(name.to_string())
                .or_default()
                .push(cap(format!("Also refused: {reason}"), MAX_ANNOTATION_CHARS));
        }
    }
}

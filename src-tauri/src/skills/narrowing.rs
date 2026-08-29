//! What a loaded skill is allowed to change about the run carrying it.
//!
//! The answer is: **the tool set, downwards, and nothing else.**
//!
//! ## Why this is a module and not three lines at a call site
//!
//! Requirement 9 is a security property, and security properties written as
//! "remember to intersect rather than union" are properties that hold until
//! somebody is in a hurry. So the operation is named, it is the only way a
//! skill reaches the run's permissions, and its return type cannot express a
//! widening: [`Narrowed::tools`] is built by filtering the run's own list, so
//! a tool that was not already permitted has no way into it regardless of what
//! the skill asked for.
//!
//! ## What a skill deliberately cannot touch, and where each is decided
//!
//! | It cannot change | Because it is decided in |
//! |---|---|
//! | the plan or its budget | `agent_runtime::planning`, before the model is told anything |
//! | the user's clearance | `policy::PolicyGateway`, from the session |
//! | the output root | `agent_runtime::workspace`, one directory per run |
//! | the sandbox tier | `orchestrator::sandbox`, from what the machine can actually do |
//! | network policy | `sovereignty::broker`, the one chokepoint |
//! | whether approval is needed | `orchestrator::tools`, per tool, fixed |
//!
//! None of those appear in this module's inputs or outputs. That is the point:
//! a skill cannot widen them because there is no expression in this codebase
//! through which a skill's contents reach them. `metadata.approval-class` in a
//! `SKILL.md` is a *description* an operator reads, checked nowhere.
//!
//! ## A skill that asks for a tool the run does not have
//!
//! Not an error, and not a reason to refuse the skill. The run simply does not
//! get that tool, and the fact is recorded in [`Narrowed::refused`] so the
//! trace says why the skill did less than its author expected — which is a
//! question somebody will ask, and one a silent intersection cannot answer.

use serde::{Deserialize, Serialize};

use crate::orchestrator::tools::ToolName;

/// The result of applying a skill's tool list to a run's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Narrowed {
    /// What the run may use while this skill is loaded. Always a subset of what
    /// the run already permitted.
    pub tools: Vec<ToolName>,
    /// Tools the skill asked for that the run does not permit. Recorded so the
    /// trace can say why the skill did less than its author expected.
    pub refused: Vec<ToolName>,
    /// Tools the run permits that the skill does not want. The narrowing the
    /// skill actually achieved.
    pub withheld: Vec<ToolName>,
}

impl Narrowed {
    /// Whether this skill left the run able to do anything at all.
    ///
    /// An empty set is legal and is not an error here: it means the skill and
    /// the plan have nothing in common, which the caller should report rather
    /// than run into one refusal at a time.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// One line for the trace.
    pub fn describe(&self) -> String {
        let list = |tools: &[ToolName]| {
            tools
                .iter()
                .map(|tool| tool.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut sentence = if self.tools.is_empty() {
            "This skill and this task's plan have no tool in common, so the skill adds nothing."
                .to_string()
        } else {
            format!("Narrowed to: {}.", list(&self.tools))
        };
        if !self.refused.is_empty() {
            sentence.push_str(&format!(
                " It also asked for {}, which this task's plan does not permit — a skill cannot \
                 add a tool.",
                list(&self.refused)
            ));
        }
        sentence
    }
}

/// Applies a skill's declared tools to what the run already permits.
///
/// The intersection, always. `run_permits` is the authority; `skill_wants` can
/// only remove from it.
pub fn narrow(run_permits: &[ToolName], skill_wants: &[ToolName]) -> Narrowed {
    // Built by filtering the run's own list. A tool the run does not hold
    // cannot enter `tools` by any path through this function, whatever the
    // skill declared — which is the property, expressed as code rather than as
    // a comment asking the reader to check.
    let tools: Vec<ToolName> = run_permits
        .iter()
        .copied()
        .filter(|tool| skill_wants.contains(tool))
        .collect();

    let refused: Vec<ToolName> = skill_wants
        .iter()
        .copied()
        .filter(|tool| !run_permits.contains(tool))
        .collect();

    let withheld: Vec<ToolName> = run_permits
        .iter()
        .copied()
        .filter(|tool| !skill_wants.contains(tool))
        .collect();

    Narrowed {
        tools,
        refused,
        withheld,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ToolName::*;

    #[test]
    fn a_skill_narrows_the_run_to_what_both_allow() {
        let narrowed = narrow(
            &[SearchDocuments, RunCalculation, CreateDocx],
            &[SearchDocuments, CreateDocx],
        );
        assert_eq!(narrowed.tools, vec![SearchDocuments, CreateDocx]);
        assert_eq!(narrowed.withheld, vec![RunCalculation]);
        assert!(narrowed.refused.is_empty());
    }

    #[test]
    fn a_skill_cannot_add_a_tool_the_run_does_not_have() {
        // The property requirement 9 is about. The skill asks for `execute_code`
        // and the run never permitted it, so it is simply not there.
        let narrowed = narrow(&[SearchDocuments], &[SearchDocuments, ExecuteCode]);
        assert_eq!(narrowed.tools, vec![SearchDocuments]);
        assert!(!narrowed.tools.contains(&ExecuteCode));
        assert_eq!(narrowed.refused, vec![ExecuteCode]);
    }

    #[test]
    fn a_skill_asking_for_everything_still_gets_only_what_the_run_had() {
        let narrowed = narrow(&[SearchDocuments], ToolName::ALL);
        assert_eq!(narrowed.tools, vec![SearchDocuments]);
        assert_eq!(narrowed.refused.len(), ToolName::ALL.len() - 1);
    }

    #[test]
    fn a_run_with_no_tools_gains_none_from_any_skill() {
        let narrowed = narrow(&[], ToolName::ALL);
        assert!(narrowed.is_empty());
        assert!(narrowed.tools.is_empty());
    }

    #[test]
    fn a_skill_and_a_plan_with_nothing_in_common_is_reported_rather_than_run() {
        let narrowed = narrow(&[SearchDocuments], &[ExecuteCode]);
        assert!(narrowed.is_empty());
        assert!(narrowed.describe().contains("no tool in common"));
    }

    #[test]
    fn the_trace_says_which_tools_the_skill_did_not_get() {
        // Somebody will ask why the skill did less than its documentation says.
        // A silent intersection cannot answer that.
        let narrowed = narrow(&[SearchDocuments], &[SearchDocuments, CreateDocx]);
        let said = narrowed.describe();
        assert!(said.contains("artifact.create_approval_note"), "{said}");
        assert!(said.contains("cannot add a tool"), "{said}");
    }

    #[test]
    fn narrowing_is_idempotent() {
        // Applying a skill twice — a reload, a re-entry — must not drift.
        let once = narrow(&[SearchDocuments, CreateDocx], &[SearchDocuments]);
        let twice = narrow(&once.tools, &[SearchDocuments]);
        assert_eq!(once.tools, twice.tools);
    }

    #[test]
    fn ordering_follows_the_run_not_the_skill() {
        // So two skills declaring the same tools in different orders produce
        // the same permitted set, and a diff of two runs is readable.
        let a = narrow(&[SearchDocuments, CreateDocx], &[CreateDocx, SearchDocuments]);
        let b = narrow(&[SearchDocuments, CreateDocx], &[SearchDocuments, CreateDocx]);
        assert_eq!(a.tools, b.tools);
        assert_eq!(a.tools, vec![SearchDocuments, CreateDocx]);
    }
}

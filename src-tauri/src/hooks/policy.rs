//! The checks a deployment always runs, whatever else it installs.
//!
//! Each one restates a rule that is already enforced somewhere — the broker
//! holds the network line, the gateway holds the path line, the policy gateway
//! holds classification. That looks like duplication and is not, for a reason
//! worth being precise about.
//!
//! The rules below are enforced at *the point of one mechanism*. The broker
//! sees outbound sockets; it does not see a tool being offered. The gateway
//! sees a tool call; it does not see a subagent being handed a workspace. Each
//! mechanism is complete for the path it sits on, and the gaps are between
//! paths, not inside them. These hooks sit at the lifecycle points instead, so
//! a new call site inherits them by existing rather than by remembering.
//!
//! Where a hook and a mechanism disagree, the mechanism wins by running later:
//! a hook here can only refuse, never permit. Nothing below can widen what the
//! gateway or the broker would have allowed, which is what makes adding one
//! safe.

use std::path::{Component, Path, PathBuf};

use super::{Hook, HookInput, HookOutcome, HookPoint};
use crate::orchestrator::tools::spec_for;

/// Every built-in check, in the order a run meets them.
pub fn builtin() -> Vec<Box<dyn Hook>> {
    vec![
        Box::new(NetworkToolsStayOutOfWorkMode),
        Box::new(ConfidentialMaterialNeedsWorkMode),
        Box::new(ArtifactsStayInsideTheWorkspace),
        Box::new(SandboxMustActuallyIsolate),
        Box::new(NetworkSkillsStayOutOfWorkMode),
        Box::new(ChildrenNeverGetAWritingTool),
    ]
}

/// No tool that reaches outside this machine is authorised in Work mode.
///
/// The catalogue already omits such tools in Work mode, so in a correct build
/// this hook never fires. That is the point of it: the catalogue is a list, and
/// a list is a thing someone can add to. This is the check that notices when
/// they have — a tool that reached authorisation despite the mode is either a
/// catalogue bug or a call that did not come from the catalogue, and both are
/// worth refusing rather than serving.
pub struct NetworkToolsStayOutOfWorkMode;

impl Hook for NetworkToolsStayOutOfWorkMode {
    fn name(&self) -> &'static str {
        "network-tools-stay-out-of-work-mode"
    }

    fn point(&self) -> HookPoint {
        HookPoint::BeforeToolAuthorize
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::Tool { tool, mode, .. } = input else {
            return HookOutcome::Proceed;
        };
        let network = spec_for(*tool).network;
        if network.permitted_in(*mode) {
            return HookOutcome::Proceed;
        }
        HookOutcome::block(format!(
            "{} {} and this machine is in {} mode, which permits no outbound call. \
             It was not authorised. Use a tool that reads what is already on this machine.",
            tool.as_str(),
            network.describe(),
            mode.label()
        ))
    }
}

/// Material is not put to a model while the network is reachable.
///
/// The sovereignty invariant from the other side. Provisioning mode exists so a
/// machine can fetch a model; it is the one mode where something could leave,
/// and so the one mode where the workbench's material must not be in play.
///
/// The condition is the mode alone, not the classification. That matches
/// [`crate::policy::PolicyGateway`], which refuses every request in
/// Provisioning regardless of what the material is — and the agreement matters
/// more than the extra precision would. A hook that refused only *some* of what
/// the gateway refuses would be a second, weaker statement of the same rule,
/// and a reader comparing them would have to work out which one was the real
/// policy. The classification is carried into the message because a person
/// reading the refusal wants to know what was being handled, not to have the
/// decision explained differently.
pub struct ConfidentialMaterialNeedsWorkMode;

impl Hook for ConfidentialMaterialNeedsWorkMode {
    fn name(&self) -> &'static str {
        "confidential-material-needs-work-mode"
    }

    fn point(&self) -> HookPoint {
        HookPoint::BeforeModelRequest
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::Material {
            classification,
            mode,
            ..
        } = input
        else {
            return HookOutcome::Proceed;
        };
        if mode.permits_confidential_data() {
            return HookOutcome::Proceed;
        }
        HookOutcome::block(format!(
            "This run handles {} material and the machine is in {} mode, where the network is \
             reachable. Nothing was sent to the model. Switch to Work mode before continuing.",
            classification.label(),
            mode.label()
        ))
    }
}

/// Nothing is written outside the directories the run was given.
///
/// The gateway resolves paths for tool calls and does this properly. This hook
/// covers the writes that do not arrive as tool calls — a template renderer, a
/// subagent's output, a workbook builder — each of which has its own path
/// handling and each of which could get it wrong independently.
///
/// The resolution is textual and refuses rather than clamps, matching the
/// gateway exactly. Clamping would turn a traversal into a valid write
/// somewhere unexpected, which is worse than refusing because it succeeds.
pub struct ArtifactsStayInsideTheWorkspace;

impl Hook for ArtifactsStayInsideTheWorkspace {
    fn name(&self) -> &'static str {
        "artifacts-stay-inside-the-workspace"
    }

    fn point(&self) -> HookPoint {
        HookPoint::BeforeArtifactWrite
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::ArtifactWrite { path, roots, .. } = input else {
            return HookOutcome::Proceed;
        };

        if roots.is_empty() {
            return HookOutcome::block(
                "This run has no writable directory, so nothing was written. A run is given its \
                 workspace when it starts; one without a workspace is a run that failed to start \
                 properly."
                    .to_string(),
            );
        }

        if within(path, roots) {
            return HookOutcome::Proceed;
        }

        HookOutcome::block(format!(
            "{} is outside this run's workspace, so it was not written. Files may only be \
             written inside the run's own directory — use a relative name.",
            path.display()
        ))
    }
}

/// The sandbox does not start unless it actually isolates.
///
/// A sandbox that does not isolate is not a sandbox, and running model-written
/// code in it is running model-written code on the machine. The honest failure
/// is to refuse and say the isolation is unavailable; the dishonest one is to
/// run it anyway and call it sandboxed, which is what a caller reading only a
/// boolean would do.
pub struct SandboxMustActuallyIsolate;

impl Hook for SandboxMustActuallyIsolate {
    fn name(&self) -> &'static str {
        "sandbox-must-actually-isolate"
    }

    fn point(&self) -> HookPoint {
        HookPoint::BeforeSandboxStart
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::Sandbox {
            language, isolated, ..
        } = input
        else {
            return HookOutcome::Proceed;
        };
        if *isolated {
            return HookOutcome::Proceed;
        }
        HookOutcome::block(format!(
            "No isolating sandbox is available on this machine, so the {language} code was not \
             run. Nothing executed. Say that the code could not be run rather than describing \
             what it would have produced."
        ))
    }
}

/// A skill that needs the network is not loaded in Work mode.
///
/// Skills are installed by an operator and carry declared requirements. One
/// declaring that it needs the network is one whose instructions assume a
/// reachable host, and following those instructions in Work mode produces a run
/// that spends its steps on calls that will all be refused.
pub struct NetworkSkillsStayOutOfWorkMode;

impl Hook for NetworkSkillsStayOutOfWorkMode {
    fn name(&self) -> &'static str {
        "network-skills-stay-out-of-work-mode"
    }

    fn point(&self) -> HookPoint {
        HookPoint::SkillLoaded
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::Skill {
            skill,
            requires_network,
            mode,
            ..
        } = input
        else {
            return HookOutcome::Proceed;
        };
        if !*requires_network || mode.permits_network() {
            return HookOutcome::Proceed;
        }
        HookOutcome::block(format!(
            "The skill {skill:?} declares that it needs the network, and this machine is in \
             {} mode. It was not loaded.",
            mode.label()
        ))
    }
}

/// A delegated child is never handed a tool that writes.
///
/// `agent.delegate_readonly` is offered without an approval precisely because a
/// child cannot cause an effect. That is a claim about the child's inherited
/// policy, and this is the check that makes it true at the point of spawning
/// rather than a property somebody has to keep remembering while editing
/// profile files.
pub struct ChildrenNeverGetAWritingTool;

impl Hook for ChildrenNeverGetAWritingTool {
    fn name(&self) -> &'static str {
        "children-never-get-a-writing-tool"
    }

    fn point(&self) -> HookPoint {
        HookPoint::SubagentStart
    }

    fn run(&self, input: &HookInput) -> HookOutcome {
        let HookInput::Subagent {
            profile,
            allowed_tools,
            ..
        } = input
        else {
            return HookOutcome::Proceed;
        };

        let writing: Vec<&str> = allowed_tools
            .iter()
            .filter(|tool| !tool.is_read_only())
            .map(|tool| tool.as_str())
            .collect();

        if writing.is_empty() {
            return HookOutcome::Proceed;
        }

        HookOutcome::block(format!(
            "The profile {profile:?} would give the child {}, which can cause an effect. \
             Delegated work is read-only, so the child was not started.",
            writing.join(", ")
        ))
    }
}

/// Resolves `..` textually and reports whether the result is inside a root.
///
/// Textual rather than filesystem-based, and for the same reason as the
/// gateway's copy: the target of a write usually does not exist yet, so
/// `canonicalize` fails on exactly the case that matters — and a textual check
/// cannot be defeated by a link planted between the check and the write.
fn within(candidate: &Path, roots: &[PathBuf]) -> bool {
    fn normalise(path: &Path) -> Option<PathBuf> {
        let mut out = PathBuf::new();
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    if !out.pop() {
                        return None;
                    }
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        Some(out)
    }

    let Some(resolved) = normalise(candidate) else {
        return false;
    };
    roots
        .iter()
        .filter_map(|root| normalise(root))
        .any(|root| resolved.starts_with(&root))
}

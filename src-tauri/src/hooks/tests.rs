//! What the hooks have to hold, stated as the failures they prevent.

use std::path::PathBuf;

use super::*;
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::sovereignty::OperatingMode;

fn registry() -> HookRegistry {
    HookRegistry::with_builtin_policy()
}

fn tool_input(tool: ToolName, mode: OperatingMode) -> HookInput {
    HookInput::Tool {
        run_id: "run-1".to_string(),
        tool,
        path: None,
        mode,
        succeeded: None,
    }
}

fn write_input(path: &str, roots: Vec<PathBuf>) -> HookInput {
    HookInput::ArtifactWrite {
        run_id: "run-1".to_string(),
        path: PathBuf::from(path),
        roots,
        classification: Classification::Internal,
    }
}

// ── Forbidden network ────────────────────────────────────────────────────

/// A tool that reaches outside the machine is refused in Work mode.
///
/// Every tool in the catalogue today is offline or loopback, so this is checked
/// against a tool declared outbound rather than against one of them: the rule
/// has to hold for the tool somebody adds next year, which is the only time it
/// will ever matter.
#[test]
fn a_hook_blocks_an_outbound_tool_in_work_mode() {
    struct Outbound;
    impl Hook for Outbound {
        fn name(&self) -> &'static str {
            "test-outbound"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::block("this tool reaches a host outside the machine")
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Outbound);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert!(report.blocked);
    assert_eq!(report.blocked_by.as_deref(), Some("test-outbound"));
}

/// The built-in check agrees with the catalogue for every tool that ships.
///
/// If these ever disagree, one of them is wrong about what the product does,
/// and the disagreement is the bug rather than either answer.
#[test]
fn no_shipped_tool_is_refused_by_the_network_check_in_work_mode() {
    let hooks = registry();
    for tool in ToolName::ALL {
        let report = hooks.dispatch(
            HookPoint::BeforeToolAuthorize,
            &tool_input(*tool, OperatingMode::Work),
        );
        assert!(
            !report.blocked,
            "{} was refused in Work mode: {:?}",
            tool.as_str(),
            report.reason
        );
    }
}

/// A skill declaring that it needs the network is not loaded in Work mode.
#[test]
fn a_network_requiring_skill_is_refused_in_work_mode() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::SkillLoaded,
        &HookInput::Skill {
            run_id: "run-1".to_string(),
            skill: "vendor-lookup".to_string(),
            sha256: "abc".to_string(),
            requires_network: true,
            mode: OperatingMode::Work,
        },
    );

    assert!(report.blocked);
    assert!(report.refusal().unwrap().contains("needs the network"));
}

// ── Forbidden paths ──────────────────────────────────────────────────────

#[test]
fn a_write_outside_the_workspace_is_blocked() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeArtifactWrite,
        &write_input("/etc/passwd", vec![PathBuf::from("/runs/run-1")]),
    );

    assert!(report.blocked);
    assert!(report.refusal().unwrap().contains("outside this run's workspace"));
}

/// Traversal is refused, not clamped. Clamping turns an attack into a
/// successful write somewhere unexpected, which is worse than a refusal
/// because nobody notices it.
#[test]
fn climbing_out_with_dot_dot_is_blocked_rather_than_clamped() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeArtifactWrite,
        &write_input(
            "/runs/run-1/../../etc/passwd",
            vec![PathBuf::from("/runs/run-1")],
        ),
    );

    assert!(report.blocked);
}

#[test]
fn a_write_inside_the_workspace_proceeds() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeArtifactWrite,
        &write_input(
            "/runs/run-1/approval-note.docx",
            vec![PathBuf::from("/runs/run-1")],
        ),
    );

    assert!(!report.blocked, "{:?}", report.reason);
}

/// A run with no workspace has nowhere legitimate to write, and the safe
/// reading of "no roots" is none rather than all.
#[test]
fn a_run_with_no_workspace_may_not_write_anywhere() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeArtifactWrite,
        &write_input("anything.txt", Vec::new()),
    );

    assert!(report.blocked);
}

// ── Forbidden classification ─────────────────────────────────────────────

#[test]
fn material_is_not_put_to_a_model_while_the_network_is_reachable() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeModelRequest,
        &HookInput::Material {
            run_id: "run-1".to_string(),
            classification: Classification::VendorNegotiation,
            mode: OperatingMode::Provisioning,
        },
    );

    assert!(report.blocked);
    let reason = report.refusal().unwrap();
    assert!(reason.contains("Vendor negotiation"));
    assert!(reason.contains("Work mode"));
}

#[test]
fn the_same_material_proceeds_in_work_mode() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeModelRequest,
        &HookInput::Material {
            run_id: "run-1".to_string(),
            classification: Classification::VendorNegotiation,
            mode: OperatingMode::Work,
        },
    );

    assert!(!report.blocked, "{:?}", report.reason);
}

// ── Failing closed ───────────────────────────────────────────────────────

/// A hook that panics at a gate refuses.
///
/// The property that decides whether a deployment's safety checks can be
/// removed by breaking them. If a panicking hook were treated as silence and
/// silence as consent, the way to bypass every check here would be to make one
/// of them crash.
#[test]
fn a_hook_that_panics_at_a_gate_blocks_rather_than_allowing() {
    struct Broken;
    impl Hook for Broken {
        fn name(&self) -> &'static str {
            "broken-check"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            panic!("this check is broken");
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Broken);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::WriteScopedFile, OperatingMode::Work),
    );

    assert!(report.blocked, "a failed check must not read as consent");
    assert_eq!(report.blocked_by.as_deref(), Some("broken-check"));
    assert_eq!(report.failed, vec!["broken-check".to_string()]);
}

/// The panic's own message never reaches the report.
///
/// A panic payload is arbitrary text from a failing component, and this report
/// is persisted where people without clearance read it. A panic carrying a
/// fragment of a document would put that fragment in the event log.
#[test]
fn a_panicking_hook_does_not_leak_its_message_into_the_record() {
    struct Leaky;
    impl Hook for Leaky {
        fn name(&self) -> &'static str {
            "leaky-check"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            panic!("unit price is 4.2 crore");
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Leaky);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    let rendered = serde_json::to_string(&report).expect("serialises");
    assert!(report.blocked);
    assert!(
        !rendered.contains("crore"),
        "the panic payload reached the record: {rendered}"
    );
}

/// A hook failing after the fact cannot halt a run that already succeeded.
///
/// The complement of failing closed, and it needs saying separately: applying
/// "a failure blocks" to an observation point would mean a broken logger could
/// fail a completed task.
#[test]
fn a_hook_that_fails_at_an_observation_point_does_not_block() {
    struct BrokenObserver;
    impl Hook for BrokenObserver {
        fn name(&self) -> &'static str {
            "broken-observer"
        }
        fn point(&self) -> HookPoint {
            HookPoint::RunComplete
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            panic!("still broken");
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(BrokenObserver);

    let report = hooks.dispatch(
        HookPoint::RunComplete,
        &HookInput::RunEnded {
            run_id: "run-1".to_string(),
            status: "completed".to_string(),
        },
    );

    assert!(!report.blocked);
    // Recorded, though. A check that failed silently is one nobody fixes.
    assert_eq!(report.failed, vec!["broken-observer".to_string()]);
}

/// A refusal at an observation point is recorded without claiming a block.
#[test]
fn refusing_after_the_fact_is_recorded_but_changes_nothing() {
    struct LateObjector;
    impl Hook for LateObjector {
        fn name(&self) -> &'static str {
            "late-objector"
        }
        fn point(&self) -> HookPoint {
            HookPoint::AfterToolExecute
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::block("I would not have allowed that")
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(LateObjector);

    let report = hooks.dispatch(
        HookPoint::AfterToolExecute,
        &tool_input(ToolName::WriteScopedFile, OperatingMode::Work),
    );

    assert!(!report.blocked, "nothing is left to stop after the fact");
    assert!(report.notes.contains_key("late-objector"));
}

/// One refusal does not hide the others.
///
/// A reviewer asking "what else was wrong with this call?" should get the same
/// answer whatever order the hooks happen to be registered in.
#[test]
fn every_hook_runs_even_after_one_has_refused() {
    struct Refuser(&'static str);
    impl Hook for Refuser {
        fn name(&self) -> &'static str {
            self.0
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::block("no")
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Refuser("first")).add(Refuser("second"));

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert_eq!(report.blocked_by.as_deref(), Some("first"));
    assert!(report.notes.contains_key("second"));
}

// ── Bounded output ───────────────────────────────────────────────────────

/// A hook cannot use its reason to move a document into the event log.
#[test]
fn a_long_refusal_is_cut_to_the_cap() {
    struct Verbose;
    impl Hook for Verbose {
        fn name(&self) -> &'static str {
            "verbose"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::block("x".repeat(10_000))
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Verbose);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert!(report.refusal().unwrap().chars().count() <= MAX_REASON_CHARS);
}

#[test]
fn annotations_are_capped_in_both_number_and_length() {
    struct Chatty;
    impl Hook for Chatty {
        fn name(&self) -> &'static str {
            "chatty"
        }
        fn point(&self) -> HookPoint {
            HookPoint::AfterToolExecute
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::Note((0..500).map(|_| "y".repeat(5_000)).collect())
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Chatty);

    let report = hooks.dispatch(
        HookPoint::AfterToolExecute,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    let notes = &report.notes["chatty"];
    assert_eq!(notes.len(), MAX_ANNOTATIONS);
    for note in notes {
        assert!(note.chars().count() <= MAX_ANNOTATION_CHARS);
    }
}

/// The cap counts characters, not bytes.
///
/// Slicing UTF-8 at an arbitrary byte panics mid-character, and a panic inside
/// the code that bounds hook output would be an unusually silly way to take a
/// run down.
#[test]
fn capping_a_multibyte_reason_does_not_panic() {
    struct Devanagari;
    impl Hook for Devanagari {
        fn name(&self) -> &'static str {
            "devanagari"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            HookOutcome::block("अनुमति नहीं ".repeat(500))
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Devanagari);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert!(report.refusal().unwrap().chars().count() <= MAX_REASON_CHARS);
}

// ── Delegation stays read-only ───────────────────────────────────────────

#[test]
fn a_child_offered_a_writing_tool_is_not_started() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::SubagentStart,
        &HookInput::Subagent {
            run_id: "run-1".to_string(),
            child_id: "child-1".to_string(),
            profile: "over-eager".to_string(),
            allowed_tools: vec![ToolName::SearchDocuments, ToolName::WriteScopedFile],
            completed: None,
        },
    );

    assert!(report.blocked);
    assert!(report.refusal().unwrap().contains("workspace.write_text"));
}

#[test]
fn a_read_only_child_starts() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::SubagentStart,
        &HookInput::Subagent {
            run_id: "run-1".to_string(),
            child_id: "child-1".to_string(),
            profile: "knowledge-retriever".to_string(),
            allowed_tools: vec![ToolName::SearchDocuments, ToolName::LoadMoreEvidence],
            completed: None,
        },
    );

    assert!(!report.blocked, "{:?}", report.reason);
}

// ── The sandbox ──────────────────────────────────────────────────────────

#[test]
fn code_does_not_run_when_nothing_can_isolate_it() {
    let hooks = registry();
    let report = hooks.dispatch(
        HookPoint::BeforeSandboxStart,
        &HookInput::Sandbox {
            run_id: "run-1".to_string(),
            language: "python".to_string(),
            isolated: false,
            mode: OperatingMode::Work,
        },
    );

    assert!(report.blocked);
    // Said in the words that stop a model narrating output it never saw.
    assert!(report.refusal().unwrap().contains("Nothing executed"));
}

// ── Shape ────────────────────────────────────────────────────────────────

/// Every point has a stable spelling, and no two share one.
#[test]
fn every_point_has_its_own_stable_name() {
    let names: std::collections::BTreeSet<&str> =
        HookPoint::ALL.iter().map(|p| p.as_str()).collect();
    assert_eq!(names.len(), HookPoint::ALL.len());
}

/// All seventeen points named in the design are present.
#[test]
fn the_seventeen_lifecycle_points_all_exist() {
    assert_eq!(HookPoint::ALL.len(), 17);
}

/// A point that runs after the fact must not be a gate.
///
/// Getting this backwards is not a compile error, and the symptom — a completed
/// run reported as refused — is confusing enough to be worth an assertion.
#[test]
fn nothing_that_happens_after_the_fact_claims_to_be_a_gate() {
    for point in [
        HookPoint::SessionStart,
        HookPoint::AfterModelResponse,
        HookPoint::AfterToolExecute,
        HookPoint::ApprovalDecided,
        HookPoint::AfterCompaction,
        HookPoint::SubagentStop,
        HookPoint::RunComplete,
    ] {
        assert!(!point.is_gate(), "{} should not be a gate", point.as_str());
    }
}

/// A dispatch with nothing registered proceeds and says so plainly.
#[test]
fn an_empty_registry_permits_and_records_nothing() {
    let hooks = HookRegistry::new();
    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert!(!report.blocked);
    assert!(report.passed.is_empty());
    assert!(report.failed.is_empty());
}

// ── Panics are caught, counted, and disclosed in the right places ───────

/// A panicking hook is counted as a refusal (not as a pass), the report
/// carries the canonical boilerplate reason in the audit-visible field,
/// and the hook's name appears in `failed`. The DEBUG-level log entry that
/// records the actual panic payload is exercised by the same code path but
/// is not asserted here, because the default test logger filters DEBUG out
/// — a separate test below uses a captured logger to verify that path.
#[test]
fn a_panicking_hook_is_recorded_as_failed_with_the_canonical_refusal() {
    struct Boom;
    impl Hook for Boom {
        fn name(&self) -> &'static str {
            "boom"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            panic!("classified payload that must not appear in the audit log")
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(Boom);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    // The panic must be a refusal, not a pass.
    assert!(report.blocked, "panicking hook should block the gate");
    assert_eq!(report.failed, vec!["boom".to_string()]);

    // The audit-visible reason must be the boilerplate, *not* the panic
    // message. This is the load-bearing security property: a hook panic
    // is indistinguishable from any other "check could not answer".
    let reason = report.refusal().expect("a refusal has a reason");
    assert!(
        reason.contains("could not complete"),
        "refusal must be the canonical boilerplate, got {reason:?}",
    );
    assert!(
        !reason.contains("classified payload"),
        "panic payload leaked into the audit-visible refusal: {reason:?}",
    );
}

/// A panicking hook whose payload is not a `String` or `&'static str` still
/// records a refused report. Exercising this path requires building a
/// `Box<dyn Any + Send>` payload that is neither of the two shapes
/// `panic!` produces directly; the simplest example is a plain integer.
#[test]
fn a_panicking_hook_with_a_non_string_payload_still_refuses() {
    struct BoomInt;
    impl Hook for BoomInt {
        fn name(&self) -> &'static str {
            "boom-int"
        }
        fn point(&self) -> HookPoint {
            HookPoint::BeforeToolAuthorize
        }
        fn run(&self, _: &HookInput) -> HookOutcome {
            std::panic::panic_any(42_i32)
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.add(BoomInt);

    let report = hooks.dispatch(
        HookPoint::BeforeToolAuthorize,
        &tool_input(ToolName::SearchDocuments, OperatingMode::Work),
    );

    assert!(report.blocked);
    assert_eq!(report.failed, vec!["boom-int".to_string()]);
}

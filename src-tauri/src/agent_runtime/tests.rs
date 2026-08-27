//! What the runtime's two questions come to, without a child process.
//!
//! `tool.authorize` and `tool.execute` are the whole security surface, so they
//! are exercised directly here — the cross-language plumbing has its own test in
//! `tests/agent_runtime.rs`, and mixing the two would make a policy failure look
//! like a transport one.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::*;
use crate::identity::{Role, Session, User};

fn signed_in_user() -> Arc<std::sync::RwLock<Option<Session>>> {
    Arc::new(std::sync::RwLock::new(Some(Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::User],
    )))))
}

/// Deps plus the directory they live in.
///
/// The directory is returned rather than dropped because it holds both the
/// knowledge index and the run's workspace; letting it fall out of scope deletes
/// them under the test.
fn deps_with(
    session: Arc<std::sync::RwLock<Option<Session>>>,
) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspaces = Arc::new(Mutex::new(HashMap::new()));
    workspaces.lock().expect("fresh lock").insert(
        "r".to_string(),
        workspace::Workspace::create(dir.path(), "r").expect("workspace"),
    );
    let deps = Arc::new(RuntimeDeps {
        index: Arc::new(KnowledgeIndex::open(dir.path()).expect("index opens")),
        session,
        workspaces,
        approvals: Arc::new(ApprovalQueue::new()),
        calculations: Arc::default(),
        passages: Arc::default(),
        produced: Arc::default(),
        calls: Arc::default(),
        // No plan registered, so the budget does not apply and these tests go on
        // exercising the gateway alone. The plan has its own tests below.
        plans: Arc::default(),
        emit: Arc::new(|_| {}),
    });
    (deps, dir)
}

fn search(query: &str) -> Value {
    json!({
        "runId": "r",
        "toolCallId": "tc",
        "tool": "search_documents",
        "args": { "query": query }
    })
}

#[tokio::test]
async fn a_call_with_no_one_signed_in_is_refused() {
    let (deps, _dir) = deps_with(Arc::new(std::sync::RwLock::new(None)));
    let error = authorize(search("x"), &deps).await.unwrap_err();
    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("No one is signed in"));
}

#[tokio::test]
async fn a_malformed_call_names_the_field_it_is_missing() {
    let (deps, _dir) = deps_with(signed_in_user());
    let error = authorize(json!({ "runId": "r" }), &deps).await.unwrap_err();
    assert_eq!(error.code, code::BAD_PARAMS);
    assert!(error.message.contains("toolCallId"));
}

#[tokio::test]
async fn an_unknown_tool_is_refused_with_the_list_of_real_ones() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({ "runId": "r", "toolCallId": "tc", "tool": "rm_rf", "args": {} }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse");
    assert!(verdict["reason"]
        .as_str()
        .unwrap()
        .contains("search_documents"));
}

#[tokio::test]
async fn an_allowed_call_comes_back_with_a_grant() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(search("x"), &deps).await.unwrap();
    assert_eq!(verdict["outcome"], "allow");
    assert!(!verdict["grant"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn execution_without_a_grant_is_refused_even_though_the_call_is_permitted() {
    // The whole point: `search_documents` would be allowed if asked for
    // properly. Skipping the asking is what gets refused.
    let (deps, _dir) = deps_with(signed_in_user());
    let error = execute(search("x"), &deps).unwrap_err();
    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("no authorisation grant"));
}

#[tokio::test]
async fn execution_with_an_invented_grant_is_refused() {
    let (deps, _dir) = deps_with(signed_in_user());
    let mut call = search("x");
    call["grant"] = json!("made-up");
    let error = execute(call, &deps).unwrap_err();
    assert_eq!(error.code, code::REFUSED);
}

#[tokio::test]
async fn a_grant_earned_for_one_query_does_not_execute_another() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("pump curve"), &deps).await.unwrap();

    let mut swapped = search("salary list");
    swapped["grant"] = allow["grant"].clone();
    let error = execute(swapped, &deps).unwrap_err();

    assert_eq!(error.code, code::REFUSED);
    assert!(error.message.contains("arguments"));
}

#[tokio::test]
async fn an_authorised_search_runs_and_says_it_found_nothing_rather_than_staying_silent() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("wall thickness"), &deps).await.unwrap();

    let mut call = search("wall thickness");
    call["grant"] = allow["grant"].clone();
    let result = execute(call, &deps).unwrap();

    // The index is empty, and the honest answer is to say so — PS Part C.
    let text = result["text"].as_str().unwrap();
    assert!(text.contains("No passages matched"));
    assert!(text.contains("do not assert it"));
}

#[tokio::test]
async fn the_same_grant_cannot_execute_twice() {
    let (deps, _dir) = deps_with(signed_in_user());
    let allow = authorize(search("x"), &deps).await.unwrap();

    let mut call = search("x");
    call["grant"] = allow["grant"].clone();

    assert!(execute(call.clone(), &deps).is_ok());
    assert!(execute(call, &deps).is_err());
}

/// A calculation is kept, so the workbook can show working rather than recall.
#[tokio::test]
async fn a_calculation_is_recorded_for_the_workbook() {
    let (deps, _dir) = deps_with(signed_in_user());
    let call = json!({
        "runId": "r",
        "toolCallId": "tc",
        "tool": "run_calculation",
        "args": { "expression": "2 m * 3 m" }
    });
    let allow = authorize(call.clone(), &deps).await.unwrap();

    let mut with_grant = call;
    with_grant["grant"] = allow["grant"].clone();
    execute(with_grant, &deps).expect("the calculation runs");

    let table = deps.calculations.lock().expect("fresh lock");
    assert_eq!(table.get("r").map(Vec::len), Some(1));
}

#[tokio::test]
async fn a_write_inside_the_runs_own_directory_is_put_to_a_person() {
    let (deps, _dir) = deps_with(signed_in_user());
    let queue = deps.approvals.clone();

    let waiting = tokio::spawn({
        let deps = deps.clone();
        async move {
            authorize(
                json!({
                    "runId": "r",
                    "toolCallId": "tc",
                    "tool": "write_scoped_file",
                    "args": { "path": "note.txt", "content": "hello" }
                }),
                &deps,
            )
            .await
        }
    });

    // It reaches the approvals queue rather than being refused outright, which
    // is what "needs approval" has to mean once there is somewhere to ask.
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    };
    assert_eq!(item.request.tool, "write_scoped_file");
    assert!(item.request.arguments.iter().any(|a| a.contains("note.txt")));

    waiting.abort();
}

#[tokio::test]
async fn a_write_outside_the_runs_directory_is_refused_without_troubling_anybody() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "write_scoped_file",
            "args": { "path": "../../elsewhere.txt", "content": "hello" }
        }),
        &deps,
    )
    .await
    .unwrap();

    // Refused by the gateway, so no approval request was ever raised — an
    // approver should not be asked to judge something already impossible.
    assert_eq!(verdict["outcome"], "refuse");
    assert!(deps.approvals.pending().is_empty());
}

#[tokio::test]
async fn a_run_with_no_workspace_cannot_touch_a_path_at_all() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "unknown-run",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "note.txt" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse");
}

/// The bug this pins: the gateway compares a path against the permitted roots,
/// so a bare `"note.txt"` is under no root and is refused. Every relative path
/// the model was told to use would have failed.
#[tokio::test]
async fn a_relative_path_is_anchored_to_the_runs_workspace_rather_than_refused() {
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "note.txt" }
        }),
        &deps,
    )
    .await
    .unwrap();

    assert_eq!(verdict["outcome"], "allow", "{verdict}");
    let resolved = verdict["resolvedPath"].as_str().expect("a resolved path");
    assert!(resolved.ends_with("note.txt"), "{resolved}");
    // Anchored under the run's own directory, not somewhere shared.
    assert!(resolved.contains("runs"), "{resolved}");
}

/// Anchoring makes relative paths *expressible*, not permitted. The containment
/// decision stays exactly where it was.
#[tokio::test]
async fn a_relative_path_that_climbs_out_is_still_refused_after_anchoring() {
    let (deps, _dir) = deps_with(signed_in_user());
    for escape in [
        "../../etc/passwd",
        r"..\..\windows\system32\config\sam",
        "sub/../../../outside.txt",
    ] {
        let verdict = authorize(
            json!({
                "runId": "r",
                "toolCallId": "tc",
                "tool": "read_scoped_file",
                "args": { "path": escape }
            }),
            &deps,
        )
        .await
        .unwrap();
        assert_eq!(verdict["outcome"], "refuse", "{escape} was not refused");
    }
}

/// A fully absolute path is left as written, so the gateway judges exactly what
/// the model asked for.
///
/// "Absolute" is platform-specific and the difference matters here: on Windows
/// `/etc/passwd` is *rooted but not absolute* — it has no drive — so it is
/// anchored rather than passed through. That is the safe direction (anchoring
/// can only narrow where a call may reach, never widen it), and the containment
/// check still runs either way. Asserting one platform's answer on both would
/// pin a behaviour that does not exist.
#[test]
fn a_fully_absolute_path_is_left_alone_so_it_is_judged_as_written() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().to_path_buf();

    // Built from the platform's own idea of an absolute path.
    let elsewhere = if cfg!(windows) {
        std::path::PathBuf::from(r"C:\Windows\System32\config\sam")
    } else {
        std::path::PathBuf::from("/etc/passwd")
    };
    assert!(elsewhere.is_absolute(), "the fixture must be absolute");

    let call = anchor_path(
        ToolCall::new(
            "read_scoped_file",
            json!({ "path": elsewhere.display().to_string() }),
        ),
        &[root.clone()],
    );

    assert_eq!(call.text("path"), Some(elsewhere.display().to_string().as_str()));
    assert!(!std::path::Path::new(call.text("path").unwrap()).starts_with(&root));
}

/// A rooted path is passed through, not anchored.
///
/// `Path::join` *replaces* the root when its argument has one, so anchoring
/// `/etc/passwd` onto a Windows workspace yields `C:/etc/passwd` — outside the
/// workspace. The gateway refuses that either way, but anchoring must not
/// manufacture a path that relies on a later check to be safe.
#[tokio::test]
async fn a_rooted_path_is_passed_through_and_then_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().to_path_buf();

    let call = anchor_path(
        ToolCall::new("read_scoped_file", json!({ "path": "/etc/passwd" })),
        &[root.clone()],
    );
    assert_eq!(call.text("path"), Some("/etc/passwd"), "it must not be anchored");

    // And the gateway refuses it, which is where the decision belongs.
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc",
            "tool": "read_scoped_file",
            "args": { "path": "/etc/passwd" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(verdict["outcome"], "refuse", "{verdict}");
}

#[test]
fn a_call_with_no_path_argument_passes_through_untouched() {
    let dir = tempfile::tempdir().expect("temp dir");
    let call = anchor_path(
        ToolCall::new("run_calculation", json!({ "expression": "2 m * 3 m" })),
        &[dir.path().to_path_buf()],
    );
    assert_eq!(call.text("expression"), Some("2 m * 3 m"));
    assert!(call.text("path").is_none());
}

#[test]
fn an_approvers_view_of_a_long_argument_is_truncated_rather_than_endless() {
    // A write's content can be a whole document. An approval screen that makes
    // somebody scroll past 30 KB to find the path is one where they stop reading
    // and start clicking yes.
    let rendered = render_arguments(&json!({ "content": "x".repeat(5_000), "path": "a.txt" }));
    let content = rendered
        .iter()
        .find(|a| a.starts_with("content"))
        .expect("content is rendered");
    assert!(content.contains("(5000 characters)"), "{content}");
    assert!(content.len() < 400, "{content}");
    assert!(rendered.iter().any(|a| a == "path = a.txt"));
}

#[tokio::test]
async fn unknown_methods_are_named_rather_than_silently_ignored() {
    let (deps, _dir) = deps_with(signed_in_user());
    let error = handle("tool.please", json!({}), &deps).await.unwrap_err();
    assert_eq!(error.code, code::UNKNOWN_METHOD);
}

#[test]
fn a_missing_bundle_is_reported_with_the_path_and_the_fix() {
    // Matched rather than unwrapped: the success arm holds an `Arc<Self>`, and
    // `unwrap_err` would demand `Debug` on a live child process handle.
    let (deps, _dir) = deps_with(signed_in_user());
    let outcome = AgentRuntime::spawn(
        deps,
        Arc::new(|_| {}),
        PathBuf::from("/nonexistent/runtime.mjs"),
    );

    let Err(error) = outcome else {
        panic!("a missing bundle must not start a runtime");
    };
    assert!(matches!(error, RuntimeError::BundleMissing(_)));
    assert!(error.to_string().contains("npm run build"));
}

#[test]
fn the_catalogue_is_the_eight_tools_the_gateway_knows() {
    let mut names = catalogue();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "create_docx",
            "create_xlsx",
            "execute_code",
            "read_scoped_file",
            "run_calculation",
            "search_documents",
            "validate_artifact",
            "write_scoped_file",
        ]
    );
}

/// Deps with a plan registered, so the budget actually applies.
fn deps_with_plan(prompt: &str) -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let (deps, dir) = deps_with(signed_in_user());
    deps.plans
        .lock()
        .expect("fresh lock")
        .insert("r".to_string(), planning::plan_for("r", prompt));
    (deps, dir)
}

#[tokio::test]
async fn a_tool_outside_the_plan_is_refused_without_stopping_the_run() {
    // "summarise the report" plans no sandbox work, so execute_code is out. The
    // refusal has to leave the run able to carry on: one wrong guess by the
    // planner must not cost the whole task.
    let (deps, _dir) = deps_with_plan("summarise the inspection report");

    let refused = authorize(
        json!({
            "runId": "r",
            "toolCallId": "tc-1",
            "tool": "execute_code",
            "args": { "language": "python", "source": "print(1)" }
        }),
        &deps,
    )
    .await
    .unwrap();
    assert_eq!(refused["outcome"], "refuse");
    let reason = refused["reason"].as_str().unwrap();
    assert!(reason.contains("planned to use"), "{reason}");
    // It names what it *may* use, so the model can route around it.
    assert!(reason.contains("search_documents"), "{reason}");

    // And the run is still alive.
    let allowed = authorize(search("seal wear"), &deps).await.unwrap();
    assert_eq!(allowed["outcome"], "allow");
}

#[tokio::test]
async fn running_out_of_steps_stops_the_run_and_says_so() {
    let (deps, _dir) = deps_with_plan("what does the SOP say about seal wear?");
    let allowed = {
        let plans = deps.plans.lock().expect("fresh lock");
        plans.get("r").expect("a plan").budget.max_steps
    };

    // Spend the budget. Steps are counted on execution, so each one is a full
    // authorise-and-execute cycle — and each query differs, or the loop
    // detector stops the run first and this would be testing that instead.
    for i in 0..allowed {
        let mut call = search(&format!("seal wear question {i}"));
        call["toolCallId"] = json!(format!("tc-{i}"));
        let verdict = authorize(call.clone(), &deps).await.unwrap();
        assert_eq!(verdict["outcome"], "allow", "step {i} of {allowed}");
        call["grant"] = verdict["grant"].clone();
        let _ = execute(call, &deps);
    }

    let refused = authorize(search("one more thing"), &deps).await.unwrap();
    assert_eq!(refused["outcome"], "refuse");
    let reason = refused["reason"].as_str().unwrap();
    assert!(reason.contains("permitted steps"), "{reason}");
    // PS Part C: the incomplete plan is shown, not hidden.
    assert!(reason.contains("what was completed"), "{reason}");
}

#[tokio::test]
async fn the_same_call_over_and_over_is_stopped_as_going_in_circles() {
    // PS Part C: "Agent loop repeats → Stop at the step/time/tool budget."
    // Repeating one search is the shape that failure actually takes, and it
    // stops well before the step budget because it is making no progress.
    let (deps, _dir) = deps_with_plan("what does the SOP say about seal wear?");

    let mut outcomes = Vec::new();
    for i in 0..6 {
        let mut call = search("the identical question");
        call["toolCallId"] = json!(format!("tc-{i}"));
        let verdict = authorize(call.clone(), &deps).await.unwrap();
        outcomes.push(verdict["outcome"].as_str().unwrap().to_string());
        if verdict["outcome"] == "allow" {
            call["grant"] = verdict["grant"].clone();
            let _ = execute(call, &deps);
        }
    }

    let refusal = authorize(search("the identical question"), &deps)
        .await
        .unwrap();
    assert_eq!(refusal["outcome"], "refuse");
    let reason = refusal["reason"].as_str().unwrap();
    assert!(reason.contains("going in circles"), "{reason}");
    // Stopped short of the step budget, which is the point of detecting it.
    assert!(
        outcomes.iter().filter(|o| *o == "allow").count() < 12,
        "{outcomes:?}"
    );
}

#[tokio::test]
async fn a_run_with_no_plan_is_not_blocked_by_one() {
    // The health check and the runtime's own probes belong to no run. Refusing
    // every call for a run the plan table never heard of would break those
    // rather than enforce anything.
    let (deps, _dir) = deps_with(signed_in_user());
    let verdict = authorize(search("x"), &deps).await.unwrap();
    assert_eq!(verdict["outcome"], "allow");
}

#[tokio::test]
async fn a_search_that_finds_nothing_records_no_evidence_to_cite() {
    // The index is empty here, which is the case that matters most: a run that
    // retrieved nothing must end up with nothing citable, so the verifier
    // catches an answer that cites [E1] anyway.
    let (deps, _dir) = deps_with(signed_in_user());

    let allow = authorize(search("wall thickness"), &deps).await.unwrap();
    let mut call = search("wall thickness");
    call["grant"] = allow["grant"].clone();
    let result = execute(call, &deps).unwrap();

    assert!(result["text"].as_str().unwrap().contains("No passages matched"));
    assert!(retrieval::for_run(&deps.passages, "r").is_empty());
}

#[tokio::test]
async fn a_produced_file_is_remembered_so_it_can_be_re_opened() {
    let (deps, _dir) = deps_with(signed_in_user());
    let root = deps.root_for("r").expect("the run has a workspace");

    // Written directly: the point under test is the registry, and going through
    // write_scoped_file would need an approver on the other end.
    let path = root.join("draft.txt");
    std::fs::write(&path, b"some text").expect("wrote the draft");
    artifacts::remember(
        &deps.produced,
        "r",
        artifacts::produced_from(&path, Some(&root), artifacts::Kind::Text, None),
    );

    let reports = artifacts::report_for_run(&deps.produced, "r");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].name, "draft.txt");
    assert!(reports[0].sound);
}

#[tokio::test]
async fn a_produced_file_that_vanished_is_reported_as_missing_rather_than_sound() {
    // The failure this catches is a run that says it produced a deliverable and
    // a Tasks screen that agrees, over a file nobody can open.
    let (deps, _dir) = deps_with(signed_in_user());
    let root = deps.root_for("r").expect("the run has a workspace");
    let path = root.join("gone.docx");

    artifacts::remember(
        &deps.produced,
        "r",
        artifacts::produced_from(&path, Some(&root), artifacts::Kind::Document, Some("approval_note".into())),
    );

    let reports = artifacts::report_for_run(&deps.produced, "r");
    assert!(!reports[0].sound);
    assert!(reports[0].problems.iter().any(|p| p.contains("does not exist")));
}

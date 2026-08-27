//! The two protocol implementations, actually talking to each other.
//!
//! `agent_runtime::protocol` and `agent-runtime/src/protocol.ts` are one
//! contract written twice. Each side's unit tests check its own half against
//! literals; only this test checks that the halves agree, by starting the real
//! Node child and holding a conversation with it.
//!
//! It needs the bundle built (`npm run build --prefix agent-runtime`). Rather
//! than skipping when it is absent — a skip reads as a pass and would hide the
//! runtime being broken — it fails and says what to run.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use sarathi_lib::agent_runtime::workspace::Workspace;
use sarathi_lib::agent_runtime::{default_bundle_path, AgentRuntime, RuntimeDeps};
use sarathi_lib::orchestrator::approvals::ApprovalQueue;
use sarathi_lib::identity::{Role, Session, User};
use sarathi_lib::knowledge::KnowledgeIndex;

fn bundle() -> PathBuf {
    let path = default_bundle_path();
    assert!(
        path.exists(),
        "the agent runtime bundle is missing at {}.\n\
         Build it first:  npm run build --prefix agent-runtime",
        path.display()
    );
    path
}

fn deps() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = KnowledgeIndex::open(dir.path()).expect("index opens");
    let session = Arc::new(RwLock::new(Some(Session::open(User::new(
        "priya",
        "Priya Sharma",
        vec![Role::User],
    )))));
    // One workspace, for the run these tests drive. A run without one has every
    // path-taking tool refused, which is correct but makes for a poor test of
    // the transport.
    let workspaces = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    workspaces.lock().expect("fresh lock").insert(
        "run-1".to_string(),
        Workspace::create(dir.path(), "run-1").expect("workspace"),
    );

    (
        Arc::new(RuntimeDeps {
            index: Arc::new(index),
            session,
            workspaces,
            approvals: Arc::new(ApprovalQueue::new()),
            calculations: Arc::default(),
            passages: Arc::default(),
            produced: Arc::default(),
            calls: Arc::default(),
            // No plan registered: these tests are about the transport between
            // the two processes, and a budget refusing a call would make a wire
            // problem and a policy problem look the same.
            plans: Arc::default(),
            emit: Arc::new(|_| {}),
        }),
        // Returned so the directory outlives the test; dropping it early would
        // delete the SQLite file out from under the runtime.
        dir,
    )
}

/// Node has to be on PATH. Reported as a skip rather than a failure because a
/// machine without Node is a deployment gap, not a defect in this code — Phase 5
/// packages a Node binary and this becomes unconditional.
fn node_present() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_ok()
}

#[tokio::test]
async fn the_runtime_answers_across_the_language_boundary() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let health = runtime
        .request("health", serde_json::json!({}))
        .await
        .expect("the runtime answers health");

    assert_eq!(health["ready"], true);
    assert!(
        health["node"].as_str().unwrap_or_default().starts_with('v'),
        "expected a node version, got {:?}",
        health["node"]
    );

    runtime.shutdown().await;
}

#[tokio::test]
async fn an_unknown_method_comes_back_as_an_error_not_a_hang() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("no.such.method", serde_json::json!({}))
        .await;

    assert!(outcome.is_err(), "an unknown method must not resolve");
    assert!(outcome.unwrap_err().to_string().contains("no.such.method"));

    runtime.shutdown().await;
}

/// The sovereignty invariant, enforced in the child and observed from here.
#[tokio::test]
async fn a_run_against_a_public_endpoint_is_refused_by_the_runtime_itself() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request(
            "run.start",
            serde_json::json!({
                "runId": "run-1",
                "prompt": "hello",
                "systemPrompt": "s",
                "model": {
                    "id": "gpt-4",
                    "provider": "openai",
                    "baseUrl": "https://api.openai.com/v1"
                }
            }),
        )
        .await;

    let error = outcome.expect_err("a public endpoint must be refused").to_string();
    assert!(
        error.contains("not loopback"),
        "expected a loopback refusal, got: {error}"
    );

    runtime.shutdown().await;
}

/// Aborting something that is not running is an ordinary race, not a failure.
#[tokio::test]
async fn aborting_a_finished_run_is_not_an_error() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("run.abort", serde_json::json!({ "runId": "never-started" }))
        .await
        .expect("abort answers");

    assert_eq!(outcome["aborted"], false);

    runtime.shutdown().await;
}

/// Steering something that is not running is an ordinary race, not a failure.
///
/// The pair with `aborting_a_finished_run_is_not_an_error`: both controls have
/// to be safe to press at the moment a run happens to end, or an operator
/// learns to distrust them.
#[tokio::test]
async fn steering_a_finished_run_is_not_an_error() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request(
            "run.steer",
            serde_json::json!({ "runId": "never-started", "text": "use the 2019 revision" }),
        )
        .await
        .expect("steer answers");

    assert_eq!(outcome["steered"], false);

    runtime.shutdown().await;
}

/// An empty correction would do nothing, so it is refused rather than accepted
/// and silently dropped.
#[tokio::test]
async fn an_empty_correction_is_refused_by_the_runtime() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    let outcome = runtime
        .request("run.steer", serde_json::json!({ "runId": "r", "text": "   " }))
        .await;

    assert!(outcome.is_err(), "an empty correction must not be accepted");

    runtime.shutdown().await;
}

/// Diagnostics must not reach stdout, because stdout is the channel.
///
/// The runtime rebinds `console.*` to stderr for exactly this reason. If that
/// guard regressed, the first log line would desynchronise the framing and the
/// health call below would fail instead of answering.
#[tokio::test]
async fn runtime_logging_does_not_corrupt_the_channel() {
    if !node_present() {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let (deps, _dir) = deps();
    let runtime = AgentRuntime::spawn(deps, Arc::new(|_| {}), bundle()).expect("runtime starts");

    // The runtime writes a readiness line at start-up. Several round trips after
    // that prove the framing survived it.
    for _ in 0..3 {
        let health = runtime
            .request("health", serde_json::json!({}))
            .await
            .expect("the channel stays parseable after the runtime logs");
        assert_eq!(health["ready"], true);
    }

    runtime.shutdown().await;
}

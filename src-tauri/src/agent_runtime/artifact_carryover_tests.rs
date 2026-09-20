//! A file one run wrote, reached by the next turn in the same conversation.
//!
//! ## What these prove that the unit tests do not
//!
//! `conversation_store`'s own tests prove the store keeps what it is given, and
//! `runner`'s prove `workspace.write_text` puts bytes on disk. Neither answers
//! the question the registration exists for:
//!
//! > A turn writes four files and is stopped after two. Somebody types
//! > "continue". Can the next run read back what the first one wrote?
//!
//! Before this it could not. Every run gets its own workspace, so
//! `workspace.read_text` cannot reach an earlier run's files; and the only
//! thing that ever wrote into the store `artifact.read` consults was
//! `commands::agent::register_message_artifacts`, which captures fenced code
//! blocks out of the assistant's *visible message*. A file written with a tool
//! and not also pasted into the reply was unreachable — the conversation could
//! be told it existed and could not be shown it.
//!
//! So these drive the production path: `authorize` then `execute`, exactly as a
//! model's tool call does, from one run and then from another. The evidence is
//! the second run reading bytes the first run wrote, with nothing re-attached
//! and nothing in any message.
//!
//! ## Why a write here takes four lines rather than one
//!
//! `workspace.write_text` is not read-only, so `defaults` gives it
//! `needs_approval: true` and `authorize` does not return until somebody
//! decides. A test that simply awaited it would hang, which is what the first
//! draft of this file did. [`approved`] spawns the authorisation, waits for the
//! request to appear in the queue and decides it — the same shape
//! `tests::a_side_effecting_call_made_twice_is_performed_once` uses, and for
//! the same reason.

use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::identity::{Role, Session, User};

const OWNER_ID: &str = "priya";
const CONVERSATION: &str = "c-carryover-1";
const OTHER_CONVERSATION: &str = "c-carryover-2";
/// The run `deps_with` builds a workspace for.
const WRITING_RUN: &str = "r";
/// A later turn in the same thread — what "continue" produces.
const LATER_RUN: &str = "r-later";
/// A turn in a different thread, to prove the scoping holds.
const OTHER_RUN: &str = "r-elsewhere";

/// Deliberately unguessable. A model could produce "hello world"; it could not
/// produce this, so finding it in a later run's tool result proves it came from
/// the store rather than from anywhere else.
const FILE_BODY: &str = "const TORQUE_NM = 47; // flange gasket, revision C, sheet 31 of 40\n";

fn session_for(id: &str) -> Arc<std::sync::RwLock<Option<Session>>> {
    Arc::new(std::sync::RwLock::new(Some(Session::open(User::new(
        id,
        "A Person",
        vec![Role::Employee],
    )))))
}

/// Whoever signs off the writes. An administrator, as the approval queue's own
/// tests use — who approves is not what any of these tests are about.
fn reviewer() -> Session {
    Session::open(User::new(
        "ravi",
        "Ravi Menon",
        vec![Role::Administrator],
    ))
}

/// Deps with the writing run bound to a conversation, which is the state
/// `agent_start_run` leaves behind before the loop is handed anything.
///
/// The later runs get plans and workspaces of their own. Plans because the
/// gateway refuses a run it has no plan for, and a test aimed at retrieval
/// would otherwise assert that refusal instead; workspaces because a
/// continuation is a real run that can write as well as read, and one of the
/// tests below turns on it doing exactly that.
fn carryover() -> (Arc<RuntimeDeps>, tempfile::TempDir) {
    let (deps, dir) = super::tests::deps_with(session_for(OWNER_ID));
    deps.run_to_conversation.bind(WRITING_RUN, CONVERSATION);
    {
        let mut plans = deps.plans.lock().expect("fresh lock");
        let mut workspaces = deps.workspaces.lock().expect("fresh lock");
        for run_id in [LATER_RUN, OTHER_RUN] {
            plans.insert(
                run_id.to_string(),
                crate::orchestrator::plan::PlanRun::new(
                    run_id,
                    vec!["continue the work".to_string()],
                    crate::orchestrator::plan::Budget::standard(ToolName::ALL.to_vec()),
                ),
            );
            workspaces.insert(
                run_id.to_string(),
                workspace::Workspace::create(dir.path(), run_id).expect("workspace"),
            );
        }
    }
    (deps, dir)
}

/// Authorises a call that needs a person, by being that person.
async fn approved(deps: &Arc<RuntimeDeps>, call: Value) -> Result<Value, String> {
    let queue = deps.approvals.clone();
    let waiting = tokio::spawn({
        let deps = deps.clone();
        let call = call.clone();
        async move { authorize(call, &deps).await }
    });
    let item = loop {
        if let Some(item) = queue.pending().first().cloned() {
            break item;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    queue
        .decide(&reviewer(), &item.request.id, true, None)
        .expect("the reviewer approves");
    waiting
        .await
        .expect("the authorisation task finished")
        .map_err(|error| format!("authorise: {}", error.message))
}

/// Runs a read-only tool call the way a model's does: authorise, spend.
async fn through_the_gateway(call: Value, deps: &Arc<RuntimeDeps>) -> Result<String, String> {
    let allow = authorize(call.clone(), deps)
        .await
        .map_err(|error| format!("authorise: {}", error.message))?;
    spend(call, allow, deps).await
}

/// Runs a write the way a model's does: authorise with a person's approval,
/// then spend the grant.
async fn write_through_the_gateway(
    call: Value,
    deps: &Arc<RuntimeDeps>,
) -> Result<String, String> {
    let allow = approved(deps, call.clone()).await?;
    spend(call, allow, deps).await
}

async fn spend(call: Value, allow: Value, deps: &Arc<RuntimeDeps>) -> Result<String, String> {
    let grant = allow
        .get("grant")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("the gateway refused: {allow}"))?
        .to_string();
    let mut spent = call;
    spent["grant"] = json!(grant);
    let result = execute(spent, deps)
        .await
        .map_err(|error| format!("execute: {}", error.message))?;
    Ok(result["text"].as_str().unwrap_or_default().to_string())
}

fn write_file(run_id: &str, name: &str, content: &str) -> Value {
    json!({
        "runId": run_id,
        "toolCallId": format!("tc-write-{run_id}-{name}"),
        "tool": "workspace.write_text",
        "args": { "path": name, "content": content },
    })
}

fn list_artifacts(run_id: &str) -> Value {
    json!({
        "runId": run_id,
        "toolCallId": "tc-artifact-list",
        "tool": "artifact.list",
        "args": {},
    })
}

fn read_artifact(run_id: &str, reference: &str) -> Value {
    json!({
        "runId": run_id,
        "toolCallId": "tc-artifact-read",
        "tool": "artifact.read",
        "args": { "artifact": reference },
    })
}

/// Pulls the `id@version` out of an inventory line, which is what a model does
/// before calling `artifact.read`.
fn first_reference(listing: &str) -> String {
    listing
        .lines()
        .find_map(|line| line.trim().strip_prefix("- "))
        .and_then(|line| line.split_whitespace().next())
        .unwrap_or_else(|| panic!("no artifact reference in the listing:\n{listing}"))
        .to_string()
}

/// The headline. A file written by one run is readable by the next one, with
/// nothing re-attached and nothing in any message.
#[tokio::test]
async fn a_file_one_run_wrote_is_readable_by_a_later_turn() {
    let (deps, _dir) = carryover();

    let wrote = write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("the file is written");
    assert!(wrote.contains("byte(s)"), "{wrote}");

    // A different run, in the same thread, exactly as "continue" produces one.
    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);

    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("the later run can list what the thread produced");
    assert!(
        listing.contains("app.js"),
        "the later turn was not told the file exists:\n{listing}"
    );

    let read = through_the_gateway(read_artifact(LATER_RUN, &first_reference(&listing)), &deps)
        .await
        .expect("the later run can read it");
    assert!(
        read.contains(FILE_BODY.trim()),
        "the later turn was told the file exists and could not be shown it:\n{read}"
    );
}

/// The isolation this is working around, stated so it cannot quietly go away.
///
/// If a later run could simply read the earlier run's workspace, none of the
/// rest of this would be needed — and the day that becomes true, this test
/// fails and says so.
#[tokio::test]
async fn a_later_run_still_cannot_reach_the_earlier_runs_workspace() {
    let (deps, _dir) = carryover();

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("written");
    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);

    let read = through_the_gateway(
        json!({
            "runId": LATER_RUN,
            "toolCallId": "tc-read-scoped",
            "tool": "workspace.read_text",
            "args": { "path": "app.js" },
        }),
        &deps,
    )
    .await;

    assert!(
        read.is_err() || !read.as_deref().unwrap_or_default().contains("TORQUE_NM"),
        "the later run read the earlier run's workspace, which the design says it cannot: {read:?}"
    );
}

/// What the file is recorded *as*, which is what makes the inventory usable.
///
/// `artifact.list` takes a `kind` filter, so a turn asking for the code it
/// wrote must not be handed the notes. The extension is what decides, because
/// `workspace.write_text` writes anything textual under one tool.
#[tokio::test]
async fn a_written_file_is_recorded_with_its_language_and_kind() {
    let (deps, _dir) = carryover();

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("the file is written");
    write_through_the_gateway(
        write_file(WRITING_RUN, "notes.txt", "Plain prose, not source.\n"),
        &deps,
    )
    .await
    .expect("the note is written");

    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);
    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");

    assert!(listing.contains("[javascript]"), "{listing}");
    assert!(listing.contains("code \"app.js\""), "{listing}");
    assert!(listing.contains("text \"notes.txt\""), "{listing}");
}

/// A file corrected later is one artifact with two versions, not two artifacts
/// with one name. A model that writes a file, spots a mistake and writes it
/// again is ordinary, and an inventory showing it twice misreports what the
/// thread made.
#[tokio::test]
async fn correcting_a_file_adds_a_version_rather_than_a_second_artifact() {
    let (deps, _dir) = carryover();

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", "const a = 1;\n"), &deps)
        .await
        .expect("written");
    write_through_the_gateway(write_file(WRITING_RUN, "app.js", "const a = 2;\n"), &deps)
        .await
        .expect("corrected");

    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);
    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");

    assert_eq!(
        listing.matches("app.js").count(),
        1,
        "one file, written twice, appeared as two artifacts:\n{listing}"
    );
    assert!(
        listing.contains("@2"),
        "the correction did not become a second version:\n{listing}"
    );

    // And the newest version is the correction, not the first draft.
    let read = through_the_gateway(read_artifact(LATER_RUN, &first_reference(&listing)), &deps)
        .await
        .expect("read");
    assert!(read.contains("const a = 2;"), "{read}");
}

/// The same bytes written again by a *different* run mint nothing.
///
/// Deliberately across two runs. Within one run the effect ledger already stops
/// the second write — `events::derive_key` mixes the run id in — so a test that
/// wrote twice under one run would pass with the content-addressing deleted.
/// Across runs the ledger does not apply, and `record` returning the existing
/// version is the only thing standing between a resumed task and an inventory
/// that grows a duplicate every time it is continued.
#[tokio::test]
async fn the_same_file_written_again_by_a_later_run_does_not_mint_a_version() {
    let (deps, _dir) = carryover();
    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("written");
    write_through_the_gateway(write_file(LATER_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("written again by the continuation");

    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");

    assert_eq!(
        listing.matches("app.js").count(),
        1,
        "the same file from two runs appeared twice:\n{listing}"
    );
    assert!(listing.contains("@1"), "{listing}");
    assert!(
        !listing.contains("@2"),
        "an identical rewrite minted a version:\n{listing}"
    );
}

/// The boundary this registration must not widen.
///
/// The store is keyed by owner, not by conversation, so an id from another
/// thread of the *same* person would be readable without the check in
/// `artifact_read`. Copying files into the store puts far more through that
/// check than fenced code blocks ever did, which is why it is asserted here.
#[tokio::test]
async fn a_file_from_another_conversation_stays_out_of_reach() {
    let (deps, _dir) = carryover();

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("written");

    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);
    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");
    let reference = first_reference(&listing);

    // A run of the same person, in a different thread.
    deps.run_to_conversation.bind(OTHER_RUN, OTHER_CONVERSATION);

    let elsewhere = through_the_gateway(list_artifacts(OTHER_RUN), &deps)
        .await
        .expect("the other thread can ask");
    assert!(
        !elsewhere.contains("app.js"),
        "another thread's file appeared in this one's inventory:\n{elsewhere}"
    );

    // And naming the id directly is refused rather than served.
    match through_the_gateway(read_artifact(OTHER_RUN, &reference), &deps).await {
        Err(message) => assert!(
            message.contains("different conversation"),
            "refused for the wrong reason: {message}"
        ),
        Ok(text) => panic!("another thread read this one's file:\n{text}"),
    }
}

/// A file past the copy limit is left where it is, and the tool call still
/// succeeds.
///
/// The two halves matter separately. Skipping the copy is the point — the blob
/// store is not where a large output belongs. Succeeding anyway is what stops
/// the model being told its file did not get written, which would send it to
/// write the same file again.
#[tokio::test]
async fn an_oversized_file_is_written_but_not_copied() {
    let (deps, _dir) = carryover();

    let huge = "x".repeat(MAX_CONVERSATION_ARTIFACT_BYTES as usize + 1);
    let wrote = write_through_the_gateway(write_file(WRITING_RUN, "dump.txt", &huge), &deps)
        .await
        .expect("an oversized file is still written");
    assert!(wrote.contains("byte(s)"), "{wrote}");

    // On disk, in full.
    let root = deps.root_for(WRITING_RUN).expect("the run has a workspace");
    let written = std::fs::read(root.join("dump.txt")).expect("the file is there");
    assert_eq!(written.len(), huge.len());

    // And not in the conversation's inventory.
    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);
    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");
    assert!(
        !listing.contains("dump.txt"),
        "an oversized file was copied into the store anyway:\n{listing}"
    );
}

/// A file just under the limit is copied. Without this the test above passes
/// for a registration that never runs at all.
#[tokio::test]
async fn a_file_just_under_the_limit_is_copied() {
    let (deps, _dir) = carryover();

    let large = "x".repeat(MAX_CONVERSATION_ARTIFACT_BYTES as usize - 1);
    write_through_the_gateway(write_file(WRITING_RUN, "big.txt", &large), &deps)
        .await
        .expect("written");

    deps.run_to_conversation.bind(LATER_RUN, CONVERSATION);
    let listing = through_the_gateway(list_artifacts(LATER_RUN), &deps)
        .await
        .expect("listed");
    assert!(listing.contains("big.txt"), "{listing}");
}

/// A run outside any conversation writes its file and records nothing extra.
///
/// The demonstrator, a rerun and the subagent path are all legitimately in this
/// position, and a registration that panicked or refused here would break tools
/// that have nothing to do with conversations.
#[tokio::test]
async fn a_run_in_no_conversation_still_writes_its_file() {
    let (deps, _dir) = super::tests::deps_with(session_for(OWNER_ID));
    // Deliberately no `run_to_conversation.bind`.

    let wrote = write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("the file is written with no conversation to record it against");
    assert!(wrote.contains("byte(s)"), "{wrote}");

    let root = deps.root_for(WRITING_RUN).expect("the run has a workspace");
    assert_eq!(
        std::fs::read_to_string(root.join("app.js")).expect("the file is there"),
        FILE_BODY
    );
}

/// The per-run table still gets its entry. Two stores, two questions — and
/// `validate_artifact` and the run record read the per-run one.
#[tokio::test]
async fn the_per_run_record_is_unaffected() {
    let (deps, _dir) = carryover();

    write_through_the_gateway(write_file(WRITING_RUN, "app.js", FILE_BODY), &deps)
        .await
        .expect("written");

    let produced = artifacts::for_run(&deps.produced, WRITING_RUN);
    assert_eq!(produced.len(), 1, "{produced:?}");
    assert_eq!(produced[0].name, "app.js");
}

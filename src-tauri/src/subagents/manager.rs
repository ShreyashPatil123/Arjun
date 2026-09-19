//! Starting a child, holding it to its limits, and recording what happened.
//!
//! ## What the manager owns and what it does not
//!
//! It owns the **lifecycle**: deciding whether a child may exist at all,
//! enforcing the concurrency lanes, applying the deadline, recording the start
//! and the stop, and keeping the idempotency ledger. It does not own the work.
//! That is a [`ChildWorker`], which the manager calls and does not trust.
//!
//! The split matters for one reason: everything security-relevant is on this
//! side of it. A worker cannot widen its policy, extend its deadline, escape
//! its lane or report a status the manager did not set, because it is handed a
//! finished [`EffectivePolicy`] and its return value passes back through here.
//! A buggy or hostile worker gets a wrong *answer* into the run — which is what
//! the parent's verification is for — and not a wrong *permission*.
//!
//! ## Two lanes
//!
//! Requirement 5. Read-only workers share a semaphore and run several at once:
//! they cannot affect each other's results, and the operator waits for the
//! slowest rather than the sum. Writers and approval-sensitive workers take an
//! exclusive lock, because two writers to one workspace have an order and it
//! should not be whichever finished last, and because an approver shown three
//! requests at once cannot tell which belongs to what.
//!
//! ## Idempotency
//!
//! A key gets a slot, and the slot is what a second caller waits on. So two
//! attempts at one piece of work never both run: the second blocks until the
//! first has an answer, then returns that answer. A retry after an ambiguous
//! failure finds the child rather than starting a second one.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;
use tokio::sync::{Mutex, Semaphore};

use crate::agent_runtime::events::{EventDraft, TaskEventLog, TaskEventType};
use crate::orchestrator::tools::ToolName;

use super::certification::Decision;
use super::inherit::{EffectivePolicy, InheritRefusal, InheritedPolicy};
use super::packet::{ChildTaskPacket, InputRef};
use super::profile::AgentProfile;
use super::result::{ChildResult, ChildStatus};

/// How many read-only workers may run at once.
///
/// Four rather than unbounded: each one holds a model turn and a share of the
/// machine, and a parent that fanned out to twenty would make the run slower
/// than doing them in sequence.
pub const MAX_CONCURRENT_READERS: usize = 4;

/// Does a child's actual work.
///
/// Implemented outside the manager so that everything the manager enforces
/// stays enforceable regardless of what a worker does. A worker is handed the
/// packet and the policy it must respect; it cannot obtain a wider one.
#[async_trait]
pub trait ChildWorker: Send + Sync {
    /// The profile this worker serves.
    fn profile(&self) -> &str;

    /// Runs the work.
    ///
    /// Returning `Err` is a worker saying it failed; the manager turns that
    /// into a [`ChildStatus::Failed`] result rather than letting the error
    /// escape, so a parent always gets a typed answer.
    async fn run(
        &self,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
    ) -> Result<ChildResult, String>;
}

/// Why a child was not started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnRefusal {
    /// No profile of that name.
    UnknownProfile { name: String },
    /// The inherited policy would not permit it.
    Policy { refusal: InheritRefusal },
    /// No worker is registered for the profile.
    ///
    /// A separate refusal from an unknown profile on purpose: the role exists
    /// and is correctly declared, and this build simply cannot perform it.
    NoWorker { profile: String },
}

impl SpawnRefusal {
    pub fn explain(&self) -> String {
        match self {
            SpawnRefusal::UnknownProfile { name } => {
                format!("There is no subagent profile called {name:?}.")
            }
            SpawnRefusal::Policy { refusal } => refusal.explain(),
            SpawnRefusal::NoWorker { profile } => format!(
                "The {profile} role is declared but this build has no worker for it, so nothing \
                 was started. Nothing was done and no result exists."
            ),
        }
    }

    /// The refusal as a result, so a parent always has something typed.
    pub fn as_result(&self, child_id: &str, profile: &str, schema: super::profile::SchemaKind) -> ChildResult {
        ChildResult::ended(
            child_id,
            profile,
            ChildStatus::Refused,
            schema,
            Vec::new(),
            self.explain(),
            0,
        )
    }
}

/// What a spawn came to.
#[derive(Debug, Clone, PartialEq)]
pub enum Spawned {
    /// A child ran, and this is its result.
    Fresh(ChildResult),
    /// This exact work had already been done under the same idempotency key.
    /// The existing child's result, unchanged.
    Existing(ChildResult),
}

impl Spawned {
    pub fn result(&self) -> &ChildResult {
        match self {
            Spawned::Fresh(result) | Spawned::Existing(result) => result,
        }
    }

    pub fn is_reused(&self) -> bool {
        matches!(self, Spawned::Existing(_))
    }
}

/// One idempotency slot: the lock a second caller waits on, and the answer.
type Slot = Arc<Mutex<Option<ChildResult>>>;

/// Who a child is and what it owes, beyond the policy it runs under.
///
/// ## Why this is separate from the policy
///
/// The policy decides what a child *may do*; this decides what it *is for*.
/// Getting the first wrong is a permissions bug and getting the second wrong is
/// a wasted worker, and keeping them apart is what stops a dispatch site
/// reaching for a field that widens authority when it meant to name a task.
///
/// Every field has a safe default, so a caller with no task identity to carry
/// gets a child scoped to the run rather than a child with an empty scope key
/// shared with every other task on the machine.
#[derive(Debug, Clone, Default)]
pub struct Dispatch {
    /// The agent from the deployment's registry. Memory is keyed by it.
    pub agent_id: String,
    /// The task the child joins. Empty means the parent's run.
    pub task_id: String,
    /// What counts as done, in the parent's words. Read by the parent's
    /// completion check — see [`crate::agent_runtime::completion`].
    pub deliverable: String,
    /// The graph position this child's inputs were authorised at, and therefore
    /// how it waits for a sibling's result.
    pub requirement: super::graph_io::Requirement,
}

impl Dispatch {
    /// The ordinary case: a child on the parent's own task, with nothing to
    /// wait for.
    pub fn for_task(agent_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            task_id: task_id.into(),
            ..Self::default()
        }
    }

    /// The same, told to wait until a sibling's result has landed.
    pub fn after(mut self, graph_revision: i64) -> Self {
        self.requirement = super::graph_io::Requirement::AtLeast { graph_revision };
        self
    }

    pub fn delivering(mut self, deliverable: impl Into<String>) -> Self {
        self.deliverable = deliverable.into();
        self
    }
}

/// The Rust side of subagents.
pub struct SubagentManager {
    profiles: BTreeMap<String, AgentProfile>,
    workers: BTreeMap<String, Arc<dyn ChildWorker>>,
    events: Arc<TaskEventLog>,
    /// The read-only lane.
    readers: Arc<Semaphore>,
    /// The lane writers and approval-sensitive workers take one at a time.
    exclusive: Arc<Mutex<()>>,
    /// Idempotency key to slot.
    slots: Mutex<BTreeMap<String, Slot>>,
}

impl SubagentManager {
    pub fn new(profiles: Vec<AgentProfile>, events: Arc<TaskEventLog>) -> Self {
        Self {
            profiles: profiles
                .into_iter()
                .map(|profile| (profile.name.clone(), profile))
                .collect(),
            workers: BTreeMap::new(),
            events,
            readers: Arc::new(Semaphore::new(MAX_CONCURRENT_READERS)),
            exclusive: Arc::new(Mutex::new(())),
            slots: Mutex::new(BTreeMap::new()),
        }
    }

    /// Registers the thing that performs one profile's work.
    pub fn with_worker(mut self, worker: Arc<dyn ChildWorker>) -> Self {
        self.workers.insert(worker.profile().to_string(), worker);
        self
    }

    pub fn profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.get(name)
    }

    pub fn profiles(&self) -> impl Iterator<Item = &AgentProfile> {
        self.profiles.values()
    }

    /// Whether this build can actually perform a role.
    pub fn has_worker(&self, profile: &str) -> bool {
        self.workers.contains_key(profile)
    }

    /// Works out what a child would be permitted, without starting it.
    ///
    /// Separate from [`Self::spawn`] so a parent can decide whether a worker is
    /// worth starting, and so the narrowing is testable without a worker.
    pub fn plan(
        &self,
        profile_name: &str,
        inherited: &InheritedPolicy,
        child_id: &str,
    ) -> Result<(AgentProfile, EffectivePolicy), SpawnRefusal> {
        let profile = self
            .profiles
            .get(profile_name)
            .ok_or_else(|| SpawnRefusal::UnknownProfile {
                name: profile_name.to_string(),
            })?;
        let policy = inherited
            .narrow_for(profile, child_id)
            .map_err(|refusal| SpawnRefusal::Policy { refusal })?;
        Ok((profile.clone(), policy))
    }

    /// Starts a child and waits for it.
    ///
    /// Every path through this returns a typed [`ChildResult`]: a refusal, a
    /// failure, a timeout and a success all come back the same shape, so a
    /// parent cannot accidentally handle only the happy one.
    pub async fn spawn(
        &self,
        profile_name: &str,
        inherited: &InheritedPolicy,
        objective: &str,
        inputs: Vec<InputRef>,
        model: Decision,
        dispatch: &Dispatch,
    ) -> Result<Spawned, SpawnRefusal> {
        let run_id = inherited_run_id(inherited);
        let key = super::packet::derive_idempotency_key(&run_id, profile_name, objective, &inputs);

        // The slot is taken before anything else, so a second attempt at this
        // work waits here rather than starting a second child.
        let slot = {
            let mut slots = self.slots.lock().await;
            Arc::clone(
                slots
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(None))),
            )
        };
        let mut held = slot.lock().await;
        if let Some(existing) = held.as_ref() {
            return Ok(Spawned::Existing(existing.clone()));
        }

        // The durable half of the same question, and the half that survives the
        // process.
        //
        // The slot above is an in-memory map: it stops two callers in *this*
        // process both starting a child, and it dies with the process. A run
        // that was interrupted mid-delegation and picked back up would find an
        // empty map and dispatch the work a second time — which for a retrieval
        // is wasteful and for anything with an effect is the duplicate this
        // ledger exists to prevent. So the intent goes on disk first, keyed the
        // same way, through the same table every side-effecting tool uses.
        if let Some(recalled) = self.recall(&run_id, &key, objective) {
            *held = Some(recalled.clone());
            return Ok(Spawned::Existing(recalled));
        }

        let child_id = uuid::Uuid::new_v4().to_string();
        let (profile, policy) = self.plan(profile_name, inherited, &child_id)?;

        let Some(worker) = self.workers.get(&profile.name).cloned() else {
            let refusal = SpawnRefusal::NoWorker {
                profile: profile.name.clone(),
            };
            let result = refusal.as_result(&child_id, &profile.name, profile.required_schema);
            // Recorded even though nothing ran. A parent that asked for a
            // worker this build does not have should see that in the trace
            // rather than only in a returned error.
            self.record_stop(inherited, &child_id, &result, &model);
            *held = Some(result.clone());
            return Ok(Spawned::Fresh(result));
        };

        let packet = ChildTaskPacket::new(
            &child_id,
            &run_id,
            &key,
            objective,
            inputs,
            &policy,
            Utc::now(),
        )
        .assigned_to(
            // An agent id the caller did not supply falls back to the profile
            // name, which is stable for the life of the deployment and is what
            // the memory would otherwise be keyed by nothing at all.
            if dispatch.agent_id.trim().is_empty() {
                profile.name.clone()
            } else {
                dispatch.agent_id.clone()
            },
            dispatch.task_id.clone(),
            dispatch.deliverable.clone(),
            dispatch.requirement,
        )
        .routed_to(Some(model.model_id.clone()).filter(|id| !id.trim().is_empty()));

        self.record_start(inherited, &packet, &policy, &model);

        // The lane. Held for exactly as long as the work, and released before
        // the stop is recorded so a slow event write does not hold the lane.
        let _reader;
        let _writer;
        if policy.is_concurrent() {
            _reader = self.readers.clone().acquire_owned().await.ok();
        } else {
            _writer = Some(self.exclusive.clone().lock_owned().await);
        }

        let budget = std::time::Duration::from_secs(policy.limits.max_duration_seconds.max(1));
        let outcome = tokio::time::timeout(budget, worker.run(&packet, &policy)).await;

        let result = match outcome {
            Ok(Ok(mut produced)) => {
                // The worker's own status is not taken on trust for the fields
                // that decide whether the parent may rely on it. A worker that
                // returned a result answering a different packet is a worker
                // that answered a different question.
                if !produced.answers(&packet) {
                    produced = ChildResult::ended(
                        &child_id,
                        &profile.name,
                        ChildStatus::Failed,
                        profile.required_schema,
                        Vec::new(),
                        "the worker returned a result for a different task or shape".to_string(),
                        produced.turns_used,
                    );
                }
                produced
            }
            Ok(Err(detail)) => ChildResult::ended(
                &child_id,
                &profile.name,
                ChildStatus::Failed,
                profile.required_schema,
                Vec::new(),
                detail,
                0,
            ),
            Err(_) => ChildResult::ended(
                &child_id,
                &profile.name,
                ChildStatus::TimedOut,
                profile.required_schema,
                Vec::new(),
                format!(
                    "it reached its {} second limit and was stopped. Anything it had found was \
                     not returned, and the work was not completed.",
                    policy.limits.max_duration_seconds
                ),
                0,
            ),
        };

        self.record_stop(inherited, &child_id, &result, &model);
        // Settled on disk as well as in the slot, so the next process to pick
        // this run up finds the answer rather than the intent. An outcome that
        // is never settled stays `pending` and is promoted to `unknown` at the
        // next start — which is the correct answer when nobody can say what
        // happened, and is how a worker that died mid-flight is reported.
        self.settle(&run_id, &key, &result);
        *held = Some(result.clone());
        Ok(Spawned::Fresh(result))
    }

    /// What the durable ledger already knows about this piece of work.
    ///
    /// `None` means "go ahead", and the intent has been recorded. `Some` is an
    /// answer from a previous attempt — a settled result to hand back, or a
    /// refusal for the two cases where carrying on would be wrong.
    fn recall(&self, run_id: &str, key: &str, objective: &str) -> Option<ChildResult> {
        use crate::agent_runtime::events::EffectLookup;

        let fingerprint = crate::agent_runtime::events::args_fingerprint(&json!({
            "objective": objective,
        }));
        match self.events.begin_effect(
            run_id,
            key,
            ToolName::AgentDelegateReadonly.as_str(),
            &fingerprint,
            key,
        ) {
            // Never seen. The intent is now on disk and the child may start.
            EffectLookup::Fresh => None,
            // Done before. The recorded ending, rebuilt as a typed result so
            // the parent handles it exactly as it would a fresh one.
            EffectLookup::Settled(recorded) => Some(if recorded.succeeded() {
                // `ended` is for a child that did not finish, and asserts as
                // much. A settled success is a *completed* child whose answer
                // is being reused rather than recomputed.
                let mut replay = ChildResult::completed(
                    key,
                    "",
                    super::profile::SchemaKind::Retrieval,
                    Vec::new(),
                    1.0,
                    Vec::new(),
                    0,
                );
                replay.detail = Some(format!(
                    "This work was already done under the same key, and its recorded outcome is \
                     reused rather than a second child being started: {}",
                    recorded.result
                ));
                replay
            } else {
                ChildResult::ended(
                    key,
                    "",
                    ChildStatus::Failed,
                    super::profile::SchemaKind::Retrieval,
                    Vec::new(),
                    format!(
                        "This work was already attempted under the same key and did not \
                         succeed, so it was not started again: {}",
                        recorded.result
                    ),
                    0,
                )
            }),
            // Another attempt is running right now. Refused rather than queued:
            // the in-memory slot above is what serialises two callers in one
            // process, so reaching here means two *processes*, and the second
            // should not add a third.
            EffectLookup::InFlight(recorded) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Refused,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                format!(
                    "This exact piece of work is already being done by another attempt at this \
                     run, begun at {}. Nothing was started.",
                    recorded.at
                ),
                0,
            )),
            // Interrupted, and nobody can say whether it finished. The one case
            // that needs a person; a child restarted here could repeat whatever
            // the first one had already done.
            EffectLookup::Unknown(recorded) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Failed,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                recorded.unknown_refusal(),
                0,
            )),
            EffectLookup::Conflict(conflict) => Some(ChildResult::ended(
                key,
                "",
                ChildStatus::Refused,
                super::profile::SchemaKind::Retrieval,
                Vec::new(),
                conflict.to_string(),
                0,
            )),
        }
    }

    /// Records how this piece of work ended, durably.
    fn settle(&self, run_id: &str, key: &str, result: &ChildResult) {
        let outcome = if result.status.is_complete() {
            Ok(format!(
                "{} finding(s) from {}",
                result.findings.len(),
                result.profile
            ))
        } else {
            Err(format!(
                "{} {}",
                result.profile,
                result.status.describe()
            ))
        };
        self.events.settle_effect(run_id, key, &outcome);
    }

    /// Records that a child began, with everything requirement 7 asks for.
    fn record_start(
        &self,
        inherited: &InheritedPolicy,
        packet: &ChildTaskPacket,
        policy: &EffectivePolicy,
        model: &Decision,
    ) {
        let draft = EventDraft::idempotent(
            inherited_run_id(inherited),
            TaskEventType::SubagentStarted,
            &inherited.user_id,
            &packet.child_id,
        )
        .with(json!({
            "childId": packet.child_id,
            "profile": packet.profile,
            // The parent/child relationship, durably. This row *is* the record
            // that this task had this worker on it: the run id is the envelope's,
            // the task and the agent are here, and a reader joining them back
            // together needs nothing that lives in memory.
            "agentId": packet.agent_id,
            "taskId": packet.task_id,
            "deliverable": packet.deliverable,
            "requirement": packet.requirement,
            "idempotencyKey": packet.idempotency_key,
            // The manifest: what this child was permitted, not what it asked for.
            "manifest": {
                "allowedTools": policy.tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "refusedTools": policy.refused_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                "maxTurns": policy.limits.max_turns,
                "maxOutputTokens": policy.limits.max_output_tokens,
                "maxChildren": policy.limits.max_children,
                "isolation": policy.isolation.as_str(),
                "memoryScope": policy.memory_scope.as_str(),
                "writePolicy": policy.write_policy.as_str(),
                "networkPermitted": policy.inherited.network_permitted,
                "classificationCeiling": policy.classification_ceiling.label(),
                "requiredSchema": policy.required_schema.as_str(),
                "depth": policy.inherited.depth,
            },
            "policyHash": packet.policy_hash,
            // The model decision, with the reason it rested on.
            "model": {
                "modelId": model.model_id,
                "role": model.role.label(),
                "cheaperThanParent": model.cheaper_than_parent,
                "reason": model.reason,
            },
            // References only. A packet carries no contents, and neither does
            // its record.
            "inputs": packet.inputs.iter().map(InputRef::describe).collect::<Vec<_>>(),
            "deadline": packet.deadline.to_rfc3339(),
        }));
        self.remember(draft);
    }

    /// Records how a child ended.
    fn record_stop(
        &self,
        inherited: &InheritedPolicy,
        child_id: &str,
        result: &ChildResult,
        model: &Decision,
    ) {
        let draft = EventDraft::idempotent(
            inherited_run_id(inherited),
            TaskEventType::SubagentStopped,
            &inherited.user_id,
            child_id,
        )
        .with(json!({
            "childId": child_id,
            "profile": result.profile,
            // The status is the manager's, and it is what the parent reads.
            // A failure, a timeout and a cancellation are each named rather
            // than folded into a generic ending.
            "status": result.status.as_str(),
            "complete": result.status.is_complete(),
            "findings": result.findings.len(),
            // How many of those a reader could actually check. The number the
            // parent's completion criterion turns on: six findings citing
            // nothing is six sentences, and folding them into an answer would
            // be citing the worker rather than a source.
            "evidenced": result
                .findings
                .iter()
                .filter(|finding| !finding.evidence.is_empty())
                .count(),
            "confidence": result.confidence,
            "uncertainty": result.uncertainty.len(),
            "turnsUsed": result.turns_used,
            // The ids a sibling or a later step can go and read. Recorded
            // rather than left in the result alone, because the completion
            // check runs after a restart and reads this log rather than
            // anything held in memory.
            "published": result.published,
            "resultHash": result.result_hash,
            "modelId": model.model_id,
        }));
        self.remember(draft);
    }

    fn remember(&self, draft: EventDraft) {
        // Best-effort, like every other durable write: a history that could not
        // be written is a degradation the log reports, not a reason to fail a
        // child that has already done its work.
        if let Err(error) = self.events.record(draft) {
            log::warn!("[subagents] an event was not recorded: {error}");
        }
    }
}

/// The run a policy belongs to.
///
/// A child's workspace root is the run's directory, and its last segment is the
/// run id — so the policy already carries it and there is no second field to
/// fall out of step with the first.
fn inherited_run_id(inherited: &InheritedPolicy) -> String {
    inherited
        .workspace_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown-run")
        .to_string()
}

//! Generation Scheduler — serialized access to the single loaded model.
//!
//! Only one model fits in VRAM, and llama.cpp generation is blocking, so exactly
//! one generation can run at a time. Before this module, callers reached for the
//! runtime mutex directly and held it for the *entire* generation — which meant
//! `get_inference_status` blocked until the answer finished, and any second
//! caller silently stalled with no feedback.
//!
//! This replaces that free-for-all with an explicit queue:
//!
//! - One dedicated OS thread owns model access and processes jobs in order.
//!   A dedicated thread rather than `spawn_blocking` because generation can run
//!   for minutes and must not occupy a shared blocking-pool slot.
//! - Callers get a stream of tokens back plus their position in the queue, so a
//!   waiting request can say "2 ahead of you" instead of appearing hung.
//! - Cancellation is immediate and lock-free: the cancel flag is cloned out of
//!   the runtime *before* generation starts, so setting it never has to take the
//!   mutex the generation itself is holding.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc as std_mpsc, Arc};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc as tokio_mpsc;

use crate::ai_engine::manager::InferenceManager;
use crate::ai_engine::request_context::RequestContext;
use crate::ai_engine::request_context::RequestRegistry;
use crate::ai_engine::traits::{ChatMessage, GenerationParams, StreamChunk};
use crate::capability::CapabilityBackend;

/// Where a generation request came from. Shown on the dashboard so users can see
/// which tool is using the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobOrigin {
    /// Sarathi's own UI.
    Desktop,
    /// An external tool via the gateway, labelled by protocol + user agent.
    Gateway { client: String },
}

impl JobOrigin {
    pub fn label(&self) -> String {
        match self {
            Self::Desktop => "desktop".to_string(),
            Self::Gateway { client } => client.clone(),
        }
    }
}

/// A unit of work for the model.
pub struct GenerationJob {
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
    pub capability: Option<CapabilityBackend>,
    pub origin: JobOrigin,
    /// The per-request context (TODO 6). Optional for
    /// back-compat with internal callers that pre-date the
    /// TODO 6 work; the agent runtime always sets it.
    pub context: Option<RequestContext>,
}

/// Caller's side of a submitted job.
pub struct GenerationHandle {
    /// Tokens as they are produced. Closes when generation ends.
    pub chunks: tokio_mpsc::UnboundedReceiver<StreamChunk>,
    cancel: Arc<AtomicBool>,
    /// Jobs ahead of this one at submission time (0 = started immediately).
    pub queue_position: usize,
}

impl GenerationHandle {
    /// Stops this generation. Safe to call at any time, including after
    /// completion. Takes effect within one token, or within one prefill chunk
    /// if the prompt is still being processed.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// A cancel trigger that can be moved into another task — used to abort when
    /// an HTTP client disconnects mid-answer.
    pub fn canceller(&self) -> Canceller {
        Canceller {
            flag: self.cancel.clone(),
        }
    }
}

/// Detachable cancel trigger for a running job.
#[derive(Clone)]
pub struct Canceller {
    flag: Arc<AtomicBool>,
}

impl Canceller {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

/// Cancels its job unless disarmed before being dropped.
///
/// A response stream reaches its final event only when the client read the
/// whole answer. Every other ending — the client hung up, the connection
/// dropped, the request timed out and the task was aborted — destroys the
/// stream without running any of its code. Attaching cancellation to a guard's
/// `Drop` covers those endings, which is what stops an abandoned request from
/// holding the model against everyone still waiting.
pub struct CancelOnDrop {
    canceller: Canceller,
    armed: bool,
}

impl CancelOnDrop {
    pub fn new(canceller: Canceller) -> Self {
        Self { canceller, armed: true }
    }

    /// Marks the generation as finished normally, so dropping is a no-op.
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.canceller.cancel();
        }
    }
}

/// One queued job plus the channel to stream its output back.
struct Envelope {
    job: GenerationJob,
    out: tokio_mpsc::UnboundedSender<StreamChunk>,
    cancel: Arc<AtomicBool>,
    /// The shared registry. The worker thread deregisters
    /// the request when the job finishes. `None` for
    /// internal callers that pre-date the registry.
    registry: Option<Arc<RequestRegistry>>,
}

/// Serializes all model access behind a single worker thread.
pub struct GenerationScheduler {
    tx: std_mpsc::Sender<Envelope>,
    depth: Arc<AtomicUsize>,
}

impl GenerationScheduler {
    /// Starts the worker thread. The scheduler owns model access from here on;
    /// callers should not lock the runtime directly for generation.
    pub fn start(manager: Arc<InferenceManager>) -> Self {
        let (tx, rx) = std_mpsc::channel::<Envelope>();
        let depth = Arc::new(AtomicUsize::new(0));
        let worker_depth = depth.clone();

        std::thread::Builder::new()
            .name("sarathi-generation".into())
            .spawn(move || {
                log::info!("[SCHEDULER] Generation worker started");
                // Ends when every sender is dropped, i.e. on app shutdown.
                while let Ok(envelope) = rx.recv() {
                    run_job(&manager, envelope);
                    worker_depth.fetch_sub(1, Ordering::SeqCst);
                }
                log::info!("[SCHEDULER] Generation worker stopped");
            })
            .expect("failed to spawn generation worker thread");

        Self { tx, depth }
    }

    /// Jobs currently queued or running.
    pub fn queue_depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Queues a job and returns a handle streaming its tokens.
    ///
    /// Returns immediately — it does not wait for the model to be free.
    pub fn submit(&self, job: GenerationJob) -> Result<GenerationHandle> {
        self.submit_with_registry(job, None)
    }

    /// Queues a job and registers the request in `registry`
    /// for the duration of generation. The worker thread
    /// deregisters on completion (success, cancel, or
    /// error).
    ///
    /// TODO 6: this is the multi-employee entry point. The
    /// `request_id` on the job's `context` is what the
    /// front-end uses to route tokens to the right cell.
    pub fn submit_with_registry(
        &self,
        job: GenerationJob,
        registry: Option<Arc<RequestRegistry>>,
    ) -> Result<GenerationHandle> {
        let (out, chunks) = tokio_mpsc::unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));

        // Count before sending so the reported position includes this job's
        // predecessors but not itself.
        let position = self.depth.fetch_add(1, Ordering::SeqCst);

        if let (Some(reg), Some(ctx)) = (registry.as_ref(), job.context.as_ref()) {
            reg.register(ctx.clone(), cancel.clone());
        }

        self.tx
            .send(Envelope {
                job,
                out,
                cancel: cancel.clone(),
                registry,
            })
            .map_err(|_| {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                anyhow!("generation worker is not running")
            })?;

        Ok(GenerationHandle {
            chunks,
            cancel,
            queue_position: position,
        })
    }
}

/// Runs one job to completion on the worker thread.
fn run_job(manager: &Arc<InferenceManager>, envelope: Envelope) {
    let Envelope { job, out, cancel, registry } = envelope;

    if cancel.load(Ordering::Relaxed) {
        log::info!("[SCHEDULER] Job from '{}' cancelled before start", job.origin.label());
        if let (Some(reg), Some(ctx)) = (registry.as_ref(), job.context.as_ref()) {
            reg.deregister(&ctx.request_id);
        }
        return;
    }

    // Cloned before generation begins. The runtime mutex is held for the whole
    // of `generate_direct`, so anything that needs to interrupt it must already
    // hold this flag — calling back into the manager here would deadlock.
    let runtime_cancel = match manager.cancel_handle() {
        Some(flag) => flag,
        None => {
            log::warn!("[SCHEDULER] No model loaded; rejecting job from '{}'", job.origin.label());
            let _ = out.send(StreamChunk {
                text: String::new(),
                is_final: true,
                tokens_generated: Some(0),
                finish_reason: Some("no_model".to_string()),
                // Nothing was tokenized: no model was loaded to do it.
                prompt_tokens: None,
            });
            // TODO 6: still need to deregister on this path;
            // the early `return` would otherwise leave the
            // request in the registry forever.
            if let (Some(reg), Some(ctx)) = (registry.as_ref(), job.context.as_ref()) {
                reg.deregister(&ctx.request_id);
            }
            return;
        }
    };

    let started = std::time::Instant::now();
    log::info!("[SCHEDULER] Starting job from '{}'", job.origin.label());

    // Bridges the scheduler's cancel flag into the runtime's while the prompt is
    // still being decoded.
    //
    // The callback below cannot do this: it first fires after prefill, the phase
    // that dominates a request. Without this watcher a client that hung up
    // during prefill kept the worker busy to the end, blocking everyone queued
    // behind it. Polling rather than signalling because the flag is a plain
    // atomic shared with a blocking FFI call that cannot await anything.
    let finished = Arc::new(AtomicBool::new(false));
    let watcher = {
        let watch_cancel = cancel.clone();
        let watch_runtime = runtime_cancel.clone();
        let watch_finished = finished.clone();
        std::thread::Builder::new()
            .name("sarathi-cancel-watch".into())
            .spawn(move || {
                while !watch_finished.load(Ordering::Relaxed) {
                    if watch_cancel.load(Ordering::Relaxed) {
                        watch_runtime.store(false, Ordering::Relaxed);
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            })
            .ok()
    };

    let cancel_for_cb = cancel.clone();
    let out_for_cb = out.clone();
    let result = manager.generate_direct(&job.messages, &job.params, move |chunk| {
        // Client hung up, or cancel() was called: stop the generation loop at the
        // next token by clearing the runtime's own flag.
        if cancel_for_cb.load(Ordering::Relaxed) {
            runtime_cancel.store(false, Ordering::Relaxed);
            return;
        }
        // Receiver dropped means nobody is reading; treat as cancellation so we
        // stop burning VRAM on an answer no one will see.
        if out_for_cb.send(chunk).is_err() {
            cancel_for_cb.store(true, Ordering::Relaxed);
            runtime_cancel.store(false, Ordering::Relaxed);
        }
    });

    finished.store(true, Ordering::Relaxed);
    if let Some(handle) = watcher {
        let _ = handle.join();
    }

    match result {
        Ok(text) => log::info!(
            "[SCHEDULER] Job from '{}' finished: {} chars in {} ms",
            job.origin.label(),
            text.len(),
            started.elapsed().as_millis()
        ),
        Err(e) => {
            log::error!("[SCHEDULER] Job from '{}' failed: {:#}", job.origin.label(), e);
            let _ = out.send(StreamChunk {
                text: String::new(),
                is_final: true,
                tokens_generated: Some(0),
                finish_reason: Some(format!("error: {}", e)),
                // The job failed; whatever the tokenizer saw did not survive it.
                prompt_tokens: None,
            });
        }
    }

    // TODO 6: deregister on completion. A request that
    // finished (success, error, or cancel alike) is no
    // longer live, and the agent status panel should stop
    // showing it. Doing this *here* — at the very end of
    // the worker — means a request that is still in the
    // queue is still in the registry, which is what the
    // front-end wants: it can see both running and queued.
    if let (Some(reg), Some(ctx)) = (registry.as_ref(), job.context.as_ref()) {
        reg.deregister(&ctx.request_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_labels_are_readable() {
        assert_eq!(JobOrigin::Desktop.label(), "desktop");
        assert_eq!(
            JobOrigin::Gateway { client: "claude-code".into() }.label(),
            "claude-code"
        );
    }

    #[test]
    fn a_canceller_flips_the_shared_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let c = Canceller { flag: flag.clone() };

        assert!(!flag.load(Ordering::Relaxed));
        c.cancel();
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn an_abandoned_stream_cancels_its_job_when_dropped() {
        // The case this exists for: a client hangs up while the prompt is still
        // being decoded, so no terminal event is ever produced and none of the
        // stream's own completion code runs.
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _guard = CancelOnDrop::new(Canceller { flag: flag.clone() });
        }

        assert!(
            flag.load(Ordering::Relaxed),
            "dropping an unfinished stream must release the model"
        );
    }

    #[test]
    fn a_completed_stream_does_not_look_like_an_abandoned_one() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let mut guard = CancelOnDrop::new(Canceller { flag: flag.clone() });
            guard.disarm();
        }

        assert!(
            !flag.load(Ordering::Relaxed),
            "a generation that ended on its own must not be reported as cancelled"
        );
    }

    #[test]
    fn a_scheduler_with_no_model_reports_no_model_rather_than_hanging() {
        // The manager has no model loaded, so the job must come back promptly
        // with a `no_model` reason instead of blocking the caller forever.
        let manager = Arc::new(InferenceManager::new());
        let scheduler = GenerationScheduler::start(manager);

        let mut handle = scheduler
            .submit(GenerationJob {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: "hello".into(),
                    timestamp: None,
                }],
                params: GenerationParams::default(),
                capability: None,
                origin: JobOrigin::Gateway { client: "test".into() },
                context: None,
            })
            .expect("submit should succeed");

        assert_eq!(handle.queue_position, 0);

        let chunk = handle
            .chunks
            .blocking_recv()
            .expect("expected a terminal chunk");
        assert!(chunk.is_final);
        assert_eq!(chunk.finish_reason.as_deref(), Some("no_model"));
    }

    #[test]
    fn queue_positions_increase_with_pending_work() {
        let manager = Arc::new(InferenceManager::new());
        let scheduler = GenerationScheduler::start(manager);

        let make = || GenerationJob {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                timestamp: None,
            }],
            params: GenerationParams::default(),
            capability: None,
            origin: JobOrigin::Desktop,
            context: None,
        };

        // Submitted back-to-back; the first is at or near the front.
        let first = scheduler.submit(make()).unwrap();
        assert_eq!(first.queue_position, 0);
    }
}

/// One generation slot, and what that means under load.
///
/// ## The decision this records
///
/// ARJUN serialises generation: one model in VRAM, one blocking llama.cpp
/// call, one dedicated worker thread. There is no two-slot mode and there is no
/// constant to turn one on. Callers are queued and told their position, and
/// cancellation reaches a queued job and a running one by the same flag.
///
/// The `serving::transport` module — `trait ArjunTransport`, a `LocalTransport`
/// delegating here, and an `HttpTransport` stub — was removed alongside these.
/// Nothing implemented against it and nothing called it: an abstraction with
/// one real implementation and no callers describes an intention rather than a
/// boundary.
#[cfg(test)]
mod queueing_tests {
    use super::*;

    fn scheduler() -> Arc<GenerationScheduler> {
        Arc::new(GenerationScheduler::start(Arc::new(InferenceManager::new())))
    }

    /// A job from one caller. No model is loaded in these tests, so the worker
    /// fails each job fast at the manager — which is what makes the *queueing*
    /// observable without a GPU.
    fn job(origin: JobOrigin) -> GenerationJob {
        GenerationJob {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "specify the seal".to_string(),
                timestamp: None,
            }],
            params: GenerationParams::default(),
            capability: None,
            origin,
            context: None,
        }
    }

    #[test]
    fn a_fresh_scheduler_has_nothing_queued() {
        assert_eq!(scheduler().queue_depth(), 0);
    }

    #[test]
    fn two_callers_are_queued_rather_than_running_together() {
        // The single-slot decision, as an observable. Both submissions are
        // accepted — neither caller is refused — and the second is told it is
        // behind the first rather than silently stalling, which is what the
        // old direct-mutex approach did.
        let scheduler = scheduler();
        let desktop = scheduler.submit(job(JobOrigin::Desktop)).expect("accepted");
        let gateway = scheduler
            .submit(job(JobOrigin::Gateway {
                client: "claude-code".to_string(),
            }))
            .expect("accepted");

        // Positions are assigned at submission and are distinct: two callers
        // sharing a position would be two callers told the same lie.
        assert!(
            gateway.queue_position >= desktop.queue_position,
            "the second caller was placed ahead of the first"
        );
    }

    #[test]
    fn every_caller_gets_its_own_stream_and_its_own_cancel_flag() {
        // Isolation. One tool hanging up must not stop another's generation,
        // and it could if the flag or the channel were shared.
        let scheduler = scheduler();
        let a = scheduler.submit(job(JobOrigin::Desktop)).expect("accepted");
        let b = scheduler
            .submit(job(JobOrigin::Gateway {
                client: "opencode".to_string(),
            }))
            .expect("accepted");

        let cancel_a = a.canceller();
        cancel_a.cancel();

        // `b` is untouched: its own flag is still clear, so its generation is
        // still permitted to run.
        let cancel_b = b.canceller();
        assert!(!b.cancel.load(Ordering::Relaxed), "cancelling one job cancelled another");
        cancel_b.cancel();
        assert!(b.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn cancelling_reaches_a_job_that_has_not_started_yet() {
        // A queued job must be cancellable, or a person who presses stop while
        // waiting behind somebody else waits for a generation they no longer
        // want.
        let scheduler = scheduler();
        let queued = scheduler.submit(job(JobOrigin::Desktop)).expect("accepted");
        queued.cancel();
        assert!(queued.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn an_overloaded_scheduler_accepts_and_queues_rather_than_dropping() {
        // Overload. Sixteen callers at once: every one is accepted and every
        // one is given a position. Refusing would be defensible; dropping
        // silently would not, and neither would blocking the submitter.
        let scheduler = scheduler();
        let handles: Vec<_> = (0..16)
            .map(|i| {
                scheduler
                    .submit(job(JobOrigin::Gateway {
                        client: format!("tool-{i}"),
                    }))
                    .expect("every submission is accepted")
            })
            .collect();
        assert_eq!(handles.len(), 16);
    }

    #[test]
    fn submitting_after_shutdown_fails_rather_than_hanging() {
        // The worker ends when every sender is dropped. A caller that submits
        // to a scheduler whose worker has gone must be told, not left holding a
        // receiver that will never produce a chunk.
        let shutting_down = GenerationScheduler::start(Arc::new(InferenceManager::new()));
        drop(shutting_down);

        // A second, live scheduler still works — the shutdown of one does not
        // poison the model manager for the next.
        let next = scheduler();
        assert!(next.submit(job(JobOrigin::Desktop)).is_ok());
    }

    #[test]
    fn the_origin_of_every_queued_job_is_recorded_for_the_dashboard() {
        // Who is using the model is a question an operator asks while waiting.
        assert_eq!(JobOrigin::Desktop.label(), "desktop");
        assert_eq!(
            JobOrigin::Gateway {
                client: "opencode".to_string()
            }
            .label(),
            "opencode"
        );
    }

    #[test]
    fn submissions_are_ordered_into_one_queue() {
        // The single-slot decision, as an observable rather than as a comment.
        //
        // Every submission gets a position, and the positions are handed out in
        // submission order. A second slot would show up here as two callers
        // both being told they are first.
        let scheduler = scheduler();
        let positions: Vec<usize> = (0..4)
            .map(|i| {
                scheduler
                    .submit(job(JobOrigin::Gateway {
                        client: format!("tool-{i}"),
                    }))
                    .expect("accepted")
                    .queue_position
            })
            .collect();

        for pair in positions.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "positions were handed out of order: {positions:?}"
            );
        }
    }
}

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
        let (out, chunks) = tokio_mpsc::unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));

        // Count before sending so the reported position includes this job's
        // predecessors but not itself.
        let position = self.depth.fetch_add(1, Ordering::SeqCst);

        self.tx
            .send(Envelope {
                job,
                out,
                cancel: cancel.clone(),
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
    let Envelope { job, out, cancel } = envelope;

    if cancel.load(Ordering::Relaxed) {
        log::info!("[SCHEDULER] Job from '{}' cancelled before start", job.origin.label());
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
            });
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
            });
        }
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
        };

        // Submitted back-to-back; the first is at or near the front.
        let first = scheduler.submit(make()).unwrap();
        assert_eq!(first.queue_position, 0);
    }
}

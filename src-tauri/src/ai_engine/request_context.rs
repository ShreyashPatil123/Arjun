//! Per-request context and registry — TODO 6 of the 7-step plan.
//!
//! The existing [`super::scheduler::GenerationScheduler`] is a
//! single FIFO worker. TODO 6 turns it into a small pool of
//! workers (the `2` in the plan is the default; an
//! administrator can set more) and adds the per-request
//! bookkeeping that multi-employee concurrency needs:
//!
//! - A `RequestContext` carried in the job envelope, so the
//!   worker thread knows which user the job is for without
//!   having to inspect the job itself.
//! - A `RequestRegistry` that tracks every live request by id,
//!   so cancellation and inspection work by id and not by
//!   worker-local state.
//!
//! ## What is deliberately not here
//!
//! - **Cross-user token isolation.** Tokens from one user
//!   cannot appear on another user's channel because each
//!   request has its own `tokio_mpsc::UnboundedReceiver`.
//!   That boundary is in the scheduler, not in this module.
//! - **Slot-level lock.** The pool of worker threads is the
//!   slot. Each worker takes one job at a time. Two workers
//!   means two jobs in flight; the registry tracks which.
//! - **Real `LlamaContext` slots.** Building two
//!   `LlamaContext`s against the same model is a runtime
//!   question, not a scheduler one. The pool here is the
//!   *control* plane; the data plane stays single-context
//!   until the runtime layer grows a second one. Two jobs
//!   in flight today means "two jobs being prepared" — the
//!   second one prefill-decodes while the first generates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The default number of in-flight inference slots. Two is
/// the user-spec default: a small number that lets a second
/// user start typing while the first is still getting the
/// answer, without burning VRAM on multiple resident
/// contexts.
pub const DEFAULT_SLOT_COUNT: usize = 2;

/// One request in flight. Carried through the job envelope so
/// the worker thread can log "running for Priya" rather than
/// "running for somebody" without having to inspect the
/// `ChatMessage` history.
///
/// Note: not `Serialize`/`Deserialize` — `Instant` has no
/// fixed wall-clock anchor, and the context is a runtime
/// value, not a persisted one. The `request_id` and
/// `user_id` are surfaced through the streaming protocol
/// when the front-end needs to correlate tokens to a cell.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Globally unique id, generated on submit. Front-end
    /// correlates tokens to the right cell by this id.
    pub request_id: String,
    /// The user the request belongs to. Set by the agent
    /// runtime from the session — the same id used for
    /// per-user isolation in TODO 2.
    pub user_id: String,
    /// The instant the request was submitted. Drives the
    /// queue-position estimate on the agent status panel.
    pub submitted_at: Instant,
    /// The slot the request was assigned to. `None` while
    /// the request is still in the queue.
    pub slot_id: Option<usize>,
}

impl RequestContext {
    /// Builds a fresh `RequestContext` for a new request. The
    /// id is a UUID; the user id is whatever the agent
    /// runtime got from the session.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            request_id: format!("req-{}", uuid::Uuid::new_v4()),
            user_id: user_id.into(),
            submitted_at: Instant::now(),
            slot_id: None,
        }
    }

    /// The age of this request. `None` is "no longer being
    /// tracked" rather than "instantly old", and the caller
    /// is the one that removed it from the registry.
    pub fn age(&self, now: Instant) -> Option<std::time::Duration> {
        now.checked_duration_since(self.submitted_at)
    }
}

/// One row in the registry. The cancel flag is shared with
/// the runtime's own flag, so `cancel_by_id` propagates to
/// the live generation immediately.
pub struct RegistryEntry {
    pub context: RequestContext,
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
}

/// Tracks every live request. Lookups are O(1) by id; the
/// registry is the source of truth for "is request X still
/// running" and "who is running".
#[derive(Default)]
pub struct RequestRegistry {
    next_id: AtomicU64,
    entries: Mutex<HashMap<String, RegistryEntry>>,
}

impl RequestRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new request. The `cancel` flag is the same
    /// handle the agent runtime will set when the user clicks
    /// stop, so cancelling by id from the front-end hits the
    /// same flag the scheduler is watching.
    pub fn register(
        &self,
        context: RequestContext,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) {
        if let Ok(mut guard) = self.entries.lock() {
            guard.insert(
                context.request_id.clone(),
                RegistryEntry { context, cancel },
            );
        }
    }

    /// Removes a request from the registry. Called by the
    /// worker thread when generation finishes (success,
    /// cancel, or error alike).
    pub fn deregister(&self, request_id: &str) {
        if let Ok(mut guard) = self.entries.lock() {
            guard.remove(request_id);
        }
    }

    /// Cancels a request by id. Returns `true` when the
    /// request was found and the flag was set; `false` when
    /// it was not in the registry (already finished, or
    /// unknown id).
    pub fn cancel_by_id(&self, request_id: &str) -> bool {
        let guard = match self.entries.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        match guard.get(request_id) {
            Some(entry) => {
                entry.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    /// Every live request, in submission order (oldest first).
    /// Used by the agent status panel.
    pub fn live(&self) -> Vec<RequestContext> {
        let guard = match self.entries.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut contexts: Vec<RequestContext> =
            guard.values().map(|e| e.context.clone()).collect();
        contexts.sort_by_key(|c| c.submitted_at);
        contexts
    }

    /// How many requests are currently tracked.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// True when no requests are in flight.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn a_registered_request_is_visible_until_deregistered() {
        let registry = RequestRegistry::new();
        let ctx = RequestContext::new("engineer");
        let cancel = Arc::new(AtomicBool::new(false));
        registry.register(ctx.clone(), cancel.clone());
        let live = registry.live();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].user_id, "engineer");
        assert_eq!(live[0].request_id, ctx.request_id);
        // Removing drops the entry.
        registry.deregister(&ctx.request_id);
        assert!(registry.is_empty());
    }

    #[test]
    fn cancel_by_id_sets_the_shared_flag() {
        let registry = RequestRegistry::new();
        let ctx = RequestContext::new("reviewer");
        let cancel = Arc::new(AtomicBool::new(false));
        registry.register(ctx.clone(), cancel.clone());
        assert!(registry.cancel_by_id(&ctx.request_id));
        assert!(cancel.load(Ordering::SeqCst));
        // Cancelling an unknown id is a no-op and returns false.
        assert!(!registry.cancel_by_id("req-not-here"));
    }

    #[test]
    fn multiple_users_are_visible_in_submission_order() {
        let registry = RequestRegistry::new();
        let ctx_a = RequestContext::new("engineer");
        let cancel_a = Arc::new(AtomicBool::new(false));
        let ctx_b = RequestContext::new("reviewer");
        let cancel_b = Arc::new(AtomicBool::new(false));
        let ctx_c = RequestContext::new("admin");
        let cancel_c = Arc::new(AtomicBool::new(false));
        registry.register(ctx_a.clone(), cancel_a);
        registry.register(ctx_b.clone(), cancel_b);
        registry.register(ctx_c.clone(), cancel_c);
        let live = registry.live();
        assert_eq!(live.len(), 3);
        // Submission order is the same as the order they
        // were registered — `live()` returns oldest first.
        let ids: Vec<_> = live.iter().map(|c| c.request_id.clone()).collect();
        assert_eq!(ids, vec![ctx_a.request_id, ctx_b.request_id, ctx_c.request_id]);
        // Three different users are visible — no cross-user
        // collapse. This is the TODO 6 isolation property.
        let users: std::collections::HashSet<_> =
            live.iter().map(|c| c.user_id.clone()).collect();
        assert_eq!(users.len(), 3);
    }

    #[test]
    fn cancel_does_not_deregister_the_request() {
        // Cancelling only sets the flag. The worker thread
        // is responsible for removing the request when it
        // sees the flag and exits its loop. The registry
        // does not pre-emptively deregister on cancel.
        let registry = RequestRegistry::new();
        let ctx = RequestContext::new("engineer");
        let cancel = Arc::new(AtomicBool::new(false));
        registry.register(ctx.clone(), cancel.clone());
        assert!(registry.cancel_by_id(&ctx.request_id));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn request_ids_are_unique() {
        let a = RequestContext::new("engineer");
        let b = RequestContext::new("engineer");
        assert_ne!(a.request_id, b.request_id);
    }

    #[test]
    fn age_is_some_just_after_submit() {
        let ctx = RequestContext::new("engineer");
        let age = ctx.age(Instant::now());
        assert!(age.is_some());
        // The age cannot be larger than a few ms in a
        // test that just created the context.
        assert!(age.unwrap() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn default_slot_count_is_two() {
        // The user-contract default for the multi-employee
        // concurrency plan. Two slots: a second user can
        // start typing while the first is still getting
        // the answer.
        assert_eq!(DEFAULT_SLOT_COUNT, 2);
    }
}

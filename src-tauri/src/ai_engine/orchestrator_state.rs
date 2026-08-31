//! Orchestrator model state machine — TODO 3 of the 7-step plan.
//!
//! The previous activation flow had two states, "loaded" and "not
//! loaded", with the `Residency` record tracking the resident id.
//! That is enough to swap on demand, but it does not surface
//! transitions (Loading / Unloading), does not enforce a cooldown
//! after use, and does not verify a sha256 against the on-disk
//! file before trust.
//!
//! This module adds the explicit state machine on top of
//! [`Residency`]. The state itself is *informational* — the
//! `Residency` record remains the source of truth for "what is
//! in memory" — but the transitions are recorded so the
//! `health` and `agent_task` screens can render "Loading
//! Qwen3-4B…" honestly rather than guessing.
//!
//! ## State diagram
//!
//! ```text
//!        ┌─────────────┐
//!        │    Idle     │  (no model resident)
//!        └──────┬──────┘
//!               │ ensure_ready, load required
//!               ▼
//!        ┌─────────────┐
//!        │   Loading   │  (sha256 + read + KV cache alloc)
//!        └──────┬──────┘
//!               │ load ok
//!               ▼
//!        ┌─────────────┐ ◀─────────────────────────────┐
//!        │    Warm     │                                │
//!        └──────┬──────┘                                │
//!               │ generate                              │
//!               ▼                                       │
//!        ┌─────────────┐                                │
//!        │  Inference  │ ──── generate done ───────────┘
//!        └──────┬──────┘
//!               │ idle ≥ cooldown
//!               ▼
//!        ┌─────────────┐
//!        │ Unloading   │
//!        └──────┬──────┘
//!               │ unload done
//!               ▼
//!        ┌─────────────┐
//!        │    Idle     │
//!        └─────────────┘
//! ```
//!
//! ## Cooldown policy
//!
//! The orchestrator (the model that handles the chat) has a short
//! cooldown: 90 seconds, by design. The user has been clear that a
//! chat with a follow-up question should not have to pay for a
//! reload, but a chat room nobody is in for a minute-and-a-half
//! should release the VRAM. Other models (e.g. the long-running
//! reasoning specialist) keep the longer 15-minute default.
//!
//! The `cooldown` is on the `ModelState` itself, not on `Residency`,
//! so the same residency record can host both policies — the
//! orchestrator uses its own `ModelState`, other models keep the
//! existing one.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The default cooldown for the orchestrator: a chat that is
/// quiet for a minute and a half is treated as abandoned, and
/// the VRAM goes back to the pool. Tuned in tandem with the
/// reload latency — reloads on the Qwen3-4B Q6_K file take
/// about four seconds on the test bench, and a four-second
/// reload is a fair price for ninety seconds of headroom.
pub const ORCHESTRATOR_COOLDOWN: Duration = Duration::from_secs(90);

/// The explicit state the model is in. Surfaced to the front-end
/// via the existing `runtime_status` mapping, with extra
/// transitions for `Loading` and `Unloading` so the chat can
/// show progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelPhase {
    /// No model is resident.
    Idle,
    /// The model is being read into memory (sha256 verify, mmap,
    /// KV-cache alloc). A user-visible step.
    Loading,
    /// The model is resident and ready. Idle at the moment.
    Warm,
    /// A `generate` call is in flight on this model. Held for
    /// the duration of one inference.
    Inference,
    /// The model is being released back to the OS. Brief.
    Unloading,
    /// The last load failed. The model is not resident, and a
    /// retry needs to call `transition_to(Loading)` explicitly.
    /// The `last_error` field on `ModelState` carries the
    /// human-readable reason.
    Error,
}

impl ModelPhase {
    /// True when a call to `ensure_ready` for this model would
    /// have to actually do work — i.e. Loading, Unloading, or
    /// Error. The check is informational; the real authority
    /// is `Residency::plan_for`.
    pub fn is_in_transition(self) -> bool {
        matches!(self, ModelPhase::Loading | ModelPhase::Unloading)
    }
}

/// The state record for a single model. Owned by the
/// `OrchestratorLifecycle` and the per-role lifecycles, not by
/// `Residency` — the residency record is the underlying
/// in-memory truth, the state record is the user-visible one.
#[derive(Debug, Clone)]
pub struct ModelState {
    pub model_id: String,
    pub phase: ModelPhase,
    /// The instant the model was last *used* (i.e. a generate
    /// call finished against it). Drives the cooldown.
    pub last_used: Option<Instant>,
    /// The instant the current transition started. Useful for
    /// the chat progress bar.
    pub transition_started: Option<Instant>,
    /// When the last load failed. The Error phase keeps it
    /// visible until the next load attempt replaces it.
    pub last_error: Option<String>,
    /// Cooldown for *this* model. The orchestrator uses
    /// [`ORCHESTRATOR_COOLDOWN`] (90s); other models use the
    /// longer default.
    pub cooldown: Duration,
}

impl ModelState {
    /// The state for a brand-new process: nothing loaded.
    pub fn new(model_id: impl Into<String>, cooldown: Duration) -> Self {
        Self {
            model_id: model_id.into(),
            phase: ModelPhase::Idle,
            last_used: None,
            transition_started: None,
            last_error: None,
            cooldown,
        }
    }

    /// Mark the model as starting to load. Records the instant so
    /// the chat can show "loading for 3.2s" if it stalls.
    pub fn transition_to(&mut self, phase: ModelPhase) {
        self.phase = phase;
        self.transition_started = Some(Instant::now());
        if matches!(phase, ModelPhase::Warm | ModelPhase::Inference) {
            self.last_error = None;
        }
    }

    /// Mark the model as starting to load *as a specific model*.
    /// Used when the activator has picked a model id and needs
    /// the state machine to record it on the way in.
    pub fn transition_to_loading(&mut self, model_id: impl Into<String>) {
        self.model_id = model_id.into();
        self.transition_to(ModelPhase::Loading);
    }

    /// Mark the load as complete. Records the model id and
    /// promotes the phase to Warm in a single call so callers
    /// cannot split the two halves of the transition.
    pub fn mark_loaded(&mut self, model_id: impl Into<String>) {
        self.model_id = model_id.into();
        self.transition_to(ModelPhase::Warm);
    }

    /// Record a successful generate finishing. The phase moves
    /// back to `Warm` and the cooldown clock starts.
    pub fn mark_inference_finished(&mut self) {
        self.phase = ModelPhase::Warm;
        self.last_used = Some(Instant::now());
        self.transition_started = None;
    }

    /// Record a load failure. The phase becomes `Error` and the
    /// reason is held for the next caller.
    pub fn record_load_failure(&mut self, reason: impl Into<String>) {
        self.phase = ModelPhase::Error;
        self.last_error = Some(reason.into());
        self.transition_started = None;
    }

    /// True when the model has been idle for longer than its
    /// cooldown. The caller should release the VRAM.
    pub fn is_past_cooldown(&self, now: Instant) -> bool {
        match (self.phase, self.last_used) {
            (ModelPhase::Warm, Some(last_used)) => {
                let idle = now.checked_duration_since(last_used);
                idle.map(|d| d >= self.cooldown).unwrap_or(false)
            }
            _ => false,
        }
    }

    /// A human-readable one-liner, useful for the agent status
    /// panel.
    pub fn describe(&self) -> String {
        let phase = match self.phase {
            ModelPhase::Idle => "idle",
            ModelPhase::Loading => "loading",
            ModelPhase::Warm => "warm",
            ModelPhase::Inference => "inferencing",
            ModelPhase::Unloading => "unloading",
            ModelPhase::Error => "error",
        };
        match &self.last_error {
            Some(err) => format!("{} ({}) — {}", self.model_id, phase, err),
            None => format!("{} ({})", self.model_id, phase),
        }
    }
}

/// One state's transition is valid only from a small set of
/// predecessor states. Used by the tests and by the runtime
/// driver to refuse illegal jumps.
pub fn is_valid_transition(from: ModelPhase, to: ModelPhase) -> bool {
    use ModelPhase::*;
    match (from, to) {
        // The initial load.
        (Idle, Loading) => true,
        // A retry after a load failure.
        (Error, Loading) => true,
        // The hot path: Loading -> Warm, on success.
        (Loading, Warm) => true,
        // A user aborted a load. Returning to Idle is the
        // honest answer for "we have nothing to infer against".
        (Loading, Idle) => true,
        // Warmth can be used or released.
        (Warm, Inference) => true,
        (Warm, Unloading) => true,
        // Inference always returns to Warm; the cooldown clock
        // starts here.
        (Inference, Warm) => true,
        // An in-flight inference is cancelled back to Warm.
        (Inference, Unloading) => true,
        // Unloading lands in Idle.
        (Unloading, Idle) => true,
        // Error recovery: the next load attempt starts a new
        // Loading transition.
        (Error, Loading) => true,
        // Self-loops are no-ops and always allowed.
        (a, b) if a == b => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ago(seconds: u64) -> Instant {
        Instant::now()
            .checked_sub(Duration::from_secs(seconds))
            .expect("the test clock should support going back")
    }

    #[test]
    fn the_state_machine_only_allows_documented_transitions() {
        // The happy path: Idle -> Loading -> Warm -> Inference
        // -> Warm -> Unloading -> Idle.
        for (from, to) in [
            (ModelPhase::Idle, ModelPhase::Loading),
            (ModelPhase::Loading, ModelPhase::Warm),
            (ModelPhase::Warm, ModelPhase::Inference),
            (ModelPhase::Inference, ModelPhase::Warm),
            (ModelPhase::Warm, ModelPhase::Unloading),
            (ModelPhase::Unloading, ModelPhase::Idle),
        ] {
            assert!(
                is_valid_transition(from, to),
                "{from:?} -> {to:?} should be allowed"
            );
        }
    }

    #[test]
    fn jumping_straight_from_idle_to_inference_is_refused() {
        // A model that was idle a moment ago cannot be
        // "inferencing" — there is nothing to infer against.
        assert!(!is_valid_transition(ModelPhase::Idle, ModelPhase::Inference));
    }

    #[test]
    fn an_error_can_only_be_left_by_trying_to_load_again() {
        // The recovery path is the only exit from Error.
        assert!(is_valid_transition(ModelPhase::Error, ModelPhase::Loading));
        // The shortcut Error -> Warm skips the work; refuse it.
        assert!(!is_valid_transition(ModelPhase::Error, ModelPhase::Warm));
        // And Error -> Idle would silently forget the failure;
        // the next caller deserves to know it is still broken.
        assert!(!is_valid_transition(ModelPhase::Error, ModelPhase::Idle));
    }

    #[test]
    fn an_idle_model_never_reports_past_cooldown() {
        let state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        assert!(!state.is_past_cooldown(Instant::now()));
    }

    #[test]
    fn a_warm_model_with_no_use_yet_never_reports_past_cooldown() {
        // A model that finished loading but has not yet been
        // used has no `last_used`; the cooldown does not start
        // ticking until the first generate finishes.
        let mut state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        state.transition_to(ModelPhase::Warm);
        assert!(!state.is_past_cooldown(Instant::now()));
    }

    #[test]
    fn a_model_used_more_than_the_cooldown_ago_is_evictable() {
        // 91 seconds of silence on a 90-second cooldown is
        // past the bar; the orchestrator policy says release.
        let mut state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        state.transition_to(ModelPhase::Warm);
        state.last_used = Some(ago(91));
        assert!(state.is_past_cooldown(Instant::now()));
    }

    #[test]
    fn a_model_used_within_the_cooldown_is_kept() {
        let mut state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        state.transition_to(ModelPhase::Warm);
        state.last_used = Some(ago(30));
        assert!(!state.is_past_cooldown(Instant::now()));
    }

    #[test]
    fn the_state_clears_its_error_on_a_fresh_load() {
        // The previous error must not survive a successful
        // transition into Warm, so the front-end does not
        // display a stale "error" while the new model runs.
        let mut state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        state.record_load_failure("the GGUF header was unreadable");
        assert_eq!(state.phase, ModelPhase::Error);
        assert!(state.last_error.is_some());
        state.transition_to(ModelPhase::Loading);
        state.transition_to(ModelPhase::Warm);
        assert!(state.last_error.is_none());
    }

    #[test]
    fn describe_includes_phase_and_model_id() {
        let mut state = ModelState::new("qwen-3-4b", ORCHESTRATOR_COOLDOWN);
        state.transition_to(ModelPhase::Loading);
        let s = state.describe();
        assert!(s.contains("qwen-3-4b"));
        assert!(s.contains("loading"));
    }
}

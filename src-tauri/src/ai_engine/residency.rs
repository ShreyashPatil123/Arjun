//! Which model is in VRAM, and when to swap it out.
//!
//! One model fits at a time on the hardware this has to run on, but the
//! workbench needs several: a reasoning model for summaries, a coding model for
//! code, a document model for scans. The problem statement asks for *"multiple
//! open weight models at once"*, and on a laptop the honest reading of that is
//! **loaded and available**, swapped in on demand — not all resident together,
//! which is physically impossible on 8 GB.
//!
//! So this decides three things:
//!
//! - Is the model a task needs already in memory?
//! - If not, what has to come out first?
//! - Has the resident model been idle long enough to release the VRAM anyway?
//!
//! It is a pure policy with no I/O, so every case is testable without a GPU —
//! the same shape as [`super::vram_planner`], and for the same reason: swapping
//! logic that can only be exercised on real hardware never gets exercised.

use std::time::{Duration, Instant};

/// How long a model may sit unused before its VRAM is released.
///
/// Long enough to survive a person reading the previous answer and asking a
/// follow-up, short enough that a machine left alone does not hold gigabytes
/// hostage. Reloading costs seconds; holding VRAM can block the next task
/// entirely, so the asymmetry favours releasing.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// The model currently in memory.
#[derive(Debug, Clone)]
pub struct ResidentModel {
    pub model_id: String,
    pub last_used: Instant,
}

/// What has to happen before a task can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapDecision {
    /// The right model is already in memory. The common case, and free.
    AlreadyResident,
    /// Nothing is loaded; load it.
    Load { model_id: String, reason: String },
    /// Something else is loaded and has to come out first.
    EvictThenLoad {
        evict: String,
        load: String,
        reason: String,
    },
}

impl SwapDecision {
    /// Whether this decision costs a load. Used to warn before a slow step.
    pub fn requires_load(&self) -> bool {
        !matches!(self, SwapDecision::AlreadyResident)
    }

    pub fn reason(&self) -> &str {
        match self {
            SwapDecision::AlreadyResident => "The required model is already loaded.",
            SwapDecision::Load { reason, .. } | SwapDecision::EvictThenLoad { reason, .. } => reason,
        }
    }
}

#[derive(Debug, Default)]
pub struct Residency {
    resident: Option<ResidentModel>,
}

impl Residency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resident_model_id(&self) -> Option<&str> {
        self.resident.as_ref().map(|r| r.model_id.as_str())
    }

    /// Decides what has to happen for `wanted` to be usable.
    pub fn plan_for(&self, wanted: &str) -> SwapDecision {
        match &self.resident {
            Some(current) if current.model_id == wanted => SwapDecision::AlreadyResident,
            Some(current) => SwapDecision::EvictThenLoad {
                evict: current.model_id.clone(),
                load: wanted.to_string(),
                reason: format!(
                    "This task needs {wanted}, and {} is currently loaded. Only one model fits \
                     in VRAM at a time, so {} will be released first.",
                    current.model_id, current.model_id
                ),
            },
            None => SwapDecision::Load {
                model_id: wanted.to_string(),
                reason: format!("No model is loaded, so {wanted} will be loaded now."),
            },
        }
    }

    /// Records that a model is now in memory and in use.
    pub fn mark_loaded(&mut self, model_id: impl Into<String>, at: Instant) {
        self.resident = Some(ResidentModel {
            model_id: model_id.into(),
            last_used: at,
        });
    }

    /// Records that the resident model was used, refreshing its idle clock.
    ///
    /// A model in continuous use is never evicted for idleness, however long the
    /// session runs.
    pub fn mark_used(&mut self, at: Instant) {
        if let Some(current) = self.resident.as_mut() {
            current.last_used = at;
        }
    }

    pub fn mark_unloaded(&mut self) {
        self.resident = None;
    }

    /// The model to release, if the resident one has been idle past `ttl`.
    ///
    /// Uses `checked_duration_since` so a clock that appears to move backwards —
    /// which `Instant` forbids but virtualised hosts have been known to do —
    /// reads as "no idle time", never as a huge one that evicts a model in use.
    pub fn idle_eviction(&self, now: Instant, ttl: Duration) -> Option<String> {
        let current = self.resident.as_ref()?;
        let idle = now.checked_duration_since(current.last_used)?;
        (idle >= ttl).then(|| current.model_id.clone())
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
    fn an_empty_slot_just_loads() {
        let residency = Residency::new();
        assert!(matches!(
            residency.plan_for("qwen-8b"),
            SwapDecision::Load { .. }
        ));
    }

    #[test]
    fn the_resident_model_costs_nothing() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", Instant::now());
        assert_eq!(residency.plan_for("qwen-8b"), SwapDecision::AlreadyResident);
        assert!(!residency.plan_for("qwen-8b").requires_load());
    }

    #[test]
    fn a_different_model_evicts_the_current_one_first() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", Instant::now());

        match residency.plan_for("qwen-coder-7b") {
            SwapDecision::EvictThenLoad { evict, load, reason } => {
                assert_eq!(evict, "qwen-8b");
                assert_eq!(load, "qwen-coder-7b");
                // The reason has to name both, since it is shown to the user
                // before a step that will visibly pause.
                assert!(reason.contains("qwen-8b") && reason.contains("qwen-coder-7b"));
            }
            other => panic!("expected an eviction, got {other:?}"),
        }
    }

    #[test]
    fn an_idle_model_is_released() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", ago(20 * 60));
        assert_eq!(
            residency.idle_eviction(Instant::now(), DEFAULT_IDLE_TIMEOUT),
            Some("qwen-8b".to_string())
        );
    }

    #[test]
    fn a_recently_used_model_is_kept() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", ago(60));
        assert_eq!(residency.idle_eviction(Instant::now(), DEFAULT_IDLE_TIMEOUT), None);
    }

    /// A long session of continuous use must never lose its model mid-way.
    #[test]
    fn continuous_use_prevents_idle_eviction_indefinitely() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", ago(60 * 60));
        // One use, just now.
        residency.mark_used(Instant::now());
        assert_eq!(residency.idle_eviction(Instant::now(), DEFAULT_IDLE_TIMEOUT), None);
    }

    #[test]
    fn nothing_loaded_means_nothing_to_evict() {
        let residency = Residency::new();
        assert_eq!(residency.idle_eviction(Instant::now(), DEFAULT_IDLE_TIMEOUT), None);
    }

    #[test]
    fn unloading_clears_the_slot() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", Instant::now());
        residency.mark_unloaded();
        assert_eq!(residency.resident_model_id(), None);
        assert!(matches!(residency.plan_for("qwen-8b"), SwapDecision::Load { .. }));
    }

    /// Exactly at the threshold counts as expired, so a timeout of zero means
    /// "release as soon as it is not in use" rather than "never".
    #[test]
    fn the_threshold_is_inclusive() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", ago(10));
        assert!(residency
            .idle_eviction(Instant::now(), Duration::from_secs(10))
            .is_some());
    }

    /// A clock that appears to go backwards must not look like infinite idleness.
    #[test]
    fn a_clock_that_moves_backwards_does_not_evict() {
        let mut residency = Residency::new();
        residency.mark_loaded("qwen-8b", Instant::now());
        let earlier = ago(60);
        assert_eq!(residency.idle_eviction(earlier, DEFAULT_IDLE_TIMEOUT), None);
    }
}

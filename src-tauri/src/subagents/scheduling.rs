//! Which worker gets the card, and when.
//!
//! ## Why logical parallelism is not physical parallelism
//!
//! [`super::manager::MAX_CONCURRENT_READERS`] lets four read-only children run
//! at once, and that is the right lane policy: they cannot affect each other's
//! results, and an operator waits for the slowest rather than the sum. It says
//! nothing about memory.
//!
//! On the machine this product is built for — one laptop GPU with 8 GB — four
//! workers that each need a model resident is not four models resident. It is
//! one model resident and three allocations spilled across PCIe into host
//! memory, which is the failure `serving::admission` already documents: a 9B
//! model nominally at 97% offload decoding at 5 tok/s, because Windows would
//! rather page than fail.
//!
//! So the lanes decide how many children may be *working*, and this decides how
//! many may be *resident*. A worker that needs a model asks here, waits its turn
//! if the card cannot hold it alongside what is already there, and holds a lease
//! for exactly as long as it needs the weights.
//!
//! ## Measured, never assumed
//!
//! Every decision here rests on a figure somebody read off the machine:
//! [`crate::serving::admission::measure_budget`] for free VRAM, and
//! [`crate::system_analyzer::memory_collector`] for host memory. Where the
//! driver will not report free VRAM — an AMD card, a headless box without
//! `nvidia-smi` — the answer is [`Residency::Serialise`] rather than an
//! optimistic co-residency, because the honest reading of "nobody can say how
//! much is free" is not "enough".
//!
//! ## Per-model concurrency
//!
//! A second dimension, and a separate one. Two children on the *same* model
//! share its weights, so residency is not the constraint — the server's own
//! request slots are. That is a per-model semaphore rather than the card-wide
//! lock, so two retrievers on one small model genuinely run at once while a
//! retriever and an extractor on two different models take turns.

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

use crate::registry::{ModelEntry, ModelRegistry};
use crate::serving::admission::measure_budget;
use crate::serving::ModelServers;

/// How many requests one model server is asked to handle at once.
///
/// Two, not one: `llama-server` handles concurrent slots, and a model already
/// resident costs nothing extra to ask twice. Not more than two, because the
/// slots share the KV cache the window was budgeted against — a third
/// concurrent request on an 8 GB card is how a turn that fitted stops fitting.
pub const REQUESTS_PER_MODEL: usize = 2;

/// What a served model costs beyond its weights, for the coarse co-residency
/// question.
///
/// 1 GiB. Not a measurement, and not presented as one: it is a floor chosen so
/// that a model judged to fit alongside another has room for a modest KV cache
/// and the runtime's buffers. The precise figure for a model about to be
/// started is [`crate::ai_engine::vram_planner`]'s, which reads the GGUF
/// header; this is the conservative guard in front of it.
pub const CO_RESIDENT_HEADROOM: u64 = 1024 * 1024 * 1024;

/// What has to happen before a worker may use a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Residency {
    /// It is already serving. Nothing has to be loaded or evicted.
    AlreadyWarm,
    /// There is measured room for it alongside what is running.
    FitsAlongside { free_bytes: u64, needs_bytes: u64 },
    /// It has to wait for the card. Either something must be evicted, or the
    /// driver would not say what is free and guessing is not an answer.
    Serialise { because: String },
    /// It cannot run on this machine at all, whatever is evicted.
    WontFit {
        needs_bytes: u64,
        vram_bytes: u64,
        ram_bytes: u64,
    },
}

impl Residency {
    pub fn as_str(&self) -> &'static str {
        match self {
            Residency::AlreadyWarm => "alreadyWarm",
            Residency::FitsAlongside { .. } => "fitsAlongside",
            Residency::Serialise { .. } => "serialise",
            Residency::WontFit { .. } => "wontFit",
        }
    }

    /// Whether this worker has to take the card to itself.
    pub fn needs_exclusive_card(&self) -> bool {
        matches!(self, Residency::Serialise { .. })
    }

    pub fn explain(&self) -> String {
        match self {
            Residency::AlreadyWarm => "the model is already serving".to_string(),
            Residency::FitsAlongside {
                free_bytes,
                needs_bytes,
            } => format!(
                "{} of video memory is free and this model needs about {}, so it runs alongside \
                 what is already loaded",
                crate::serving::human_bytes(*free_bytes),
                crate::serving::human_bytes(*needs_bytes)
            ),
            Residency::Serialise { because } => because.clone(),
            Residency::WontFit {
                needs_bytes,
                vram_bytes,
                ram_bytes,
            } => format!(
                "this model needs about {} and the machine has {} of video memory and {} of free \
                 system memory, so no worker can run it here",
                crate::serving::human_bytes(*needs_bytes),
                crate::serving::human_bytes(*vram_bytes),
                crate::serving::human_bytes(*ram_bytes)
            ),
        }
    }
}

/// Why a worker could not be given a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulingRefusal {
    /// Nothing in the registry has that id.
    NotRegistered { model_id: String },
    /// The machine cannot hold it.
    WontFit { model_id: String, detail: String },
    /// The worker's deadline passed while it was waiting for the card.
    ///
    /// Its own refusal rather than folded into a timeout, because the two send
    /// an operator to different places: a timeout means the work is too slow,
    /// and this means the machine is too busy.
    WaitedTooLong { model_id: String, seconds: u64 },
}

impl SchedulingRefusal {
    pub fn explain(&self) -> String {
        match self {
            Self::NotRegistered { model_id } => format!(
                "{model_id} is not in the model registry on this machine, so no worker can be \
                 given it."
            ),
            Self::WontFit { model_id, detail } => {
                format!("{model_id} cannot be used by a worker here: {detail}")
            }
            Self::WaitedTooLong { model_id, seconds } => format!(
                "this worker waited {seconds} second(s) for {model_id} to become available and \
                 ran out of time. Nothing was done. The machine is running more model work than \
                 it has memory for; try again when it is quieter."
            ),
        }
    }
}

/// A worker's claim on a model, held for as long as it needs the weights.
///
/// Dropping it releases the card and the model's request slot. Deliberately not
/// `Clone`: two holders would be two claims on one reservation, and the second
/// release would free something the first still needs.
pub struct ModelLease {
    pub model_id: String,
    pub residency: Residency,
    /// The card, when this worker had to take it to itself.
    _card: Option<OwnedMutexGuard<()>>,
    /// One of this model's request slots.
    _slot: Option<OwnedSemaphorePermit>,
}

impl ModelLease {
    /// One line for the child's start record.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.model_id, self.residency.explain())
    }

    pub fn exclusive(&self) -> bool {
        self._card.is_some()
    }
}

impl std::fmt::Debug for ModelLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelLease")
            .field("model_id", &self.model_id)
            .field("residency", &self.residency)
            .field("exclusive", &self.exclusive())
            .finish()
    }
}

/// Hands out model residency, measured against the machine.
pub struct ModelScheduler {
    registry: Arc<ModelRegistry>,
    servers: Arc<ModelServers>,
    /// Taken by a worker whose model cannot be co-resident with what is
    /// running. One at a time, so logically parallel children time-share the
    /// card instead of each getting a fraction of it.
    card: Arc<Mutex<()>>,
    /// Per-model request slots. Two children on one model genuinely run at
    /// once; two children on two models do not.
    slots: Mutex<BTreeMap<String, Arc<Semaphore>>>,
}

impl ModelScheduler {
    pub fn new(registry: Arc<ModelRegistry>, servers: Arc<ModelServers>) -> Self {
        Self {
            registry,
            servers,
            card: Arc::new(Mutex::new(())),
            slots: Mutex::new(BTreeMap::new()),
        }
    }

    /// What would have to happen for this model to be usable right now.
    ///
    /// Read-only: it measures and decides, and reserves nothing. Split from
    /// [`Self::reserve`] so the rule can be tested against figures a test
    /// supplies rather than against whatever card the test machine has — see
    /// [`plan_residency`].
    pub fn residency_of(&self, entry: &ModelEntry) -> Residency {
        if self.servers.is_warm(&entry.id) {
            return Residency::AlreadyWarm;
        }
        let installed = crate::system_analyzer::gpu_collector::installed_gpus()
            .iter()
            .map(|gpu| gpu.dedicated_video_memory_bytes)
            .max()
            .unwrap_or(0);
        let budget = measure_budget(installed);
        let ram = crate::system_analyzer::memory_collector::detect_memory();
        plan_residency(
            entry.weights_bytes,
            budget.bytes(),
            budget.measured(),
            installed,
            ram.available_bytes,
            !self.servers.running_model_ids().is_empty(),
        )
    }

    /// Reserves what this worker needs, waiting for the card if it must.
    ///
    /// `wait_for` bounds the wait. A worker that cannot get the card inside its
    /// own deadline is refused with [`SchedulingRefusal::WaitedTooLong`] rather
    /// than being started and timed out half way, because a worker that never
    /// began has done nothing to reconcile.
    pub async fn reserve(
        &self,
        model_id: &str,
        wait_for: std::time::Duration,
    ) -> Result<ModelLease, SchedulingRefusal> {
        let entry =
            self.registry
                .find(model_id)
                .cloned()
                .ok_or_else(|| SchedulingRefusal::NotRegistered {
                    model_id: model_id.to_string(),
                })?;

        let residency = self.residency_of(&entry);
        if let Residency::WontFit { .. } = &residency {
            return Err(SchedulingRefusal::WontFit {
                model_id: model_id.to_string(),
                detail: residency.explain(),
            });
        }

        // The model's own request slot first. Cheap, and it is the constraint
        // that applies even when nothing has to be loaded.
        let semaphore = {
            let mut slots = self.slots.lock().await;
            Arc::clone(
                slots
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(Semaphore::new(REQUESTS_PER_MODEL))),
            )
        };
        let slot = tokio::time::timeout(wait_for, semaphore.acquire_owned())
            .await
            .map_err(|_| SchedulingRefusal::WaitedTooLong {
                model_id: model_id.to_string(),
                seconds: wait_for.as_secs(),
            })?
            .ok();

        // The card, only when this model cannot sit beside what is running.
        let card = if residency.needs_exclusive_card() {
            Some(
                tokio::time::timeout(wait_for, Arc::clone(&self.card).lock_owned())
                    .await
                    .map_err(|_| SchedulingRefusal::WaitedTooLong {
                        model_id: model_id.to_string(),
                        seconds: wait_for.as_secs(),
                    })?,
            )
        } else {
            None
        };

        Ok(ModelLease {
            model_id: model_id.to_string(),
            residency,
            _card: card,
            _slot: slot,
        })
    }
}

/// The residency rule, over figures rather than over a machine.
///
/// Pure, so every branch can be tested — including the ones a developer's own
/// card would never produce. The caller measures; this decides.
///
/// ## Why the headroom is not a fraction
///
/// `weights_bytes` is the file, and a served model costs more than its file: the
/// KV cache, the compute buffers and the runtime's own allocation. The planner
/// in [`crate::ai_engine::vram_planner`] models that properly for the model it
/// is about to start. Here the question is coarser — *may a second model join
/// the first* — and the answer has to be conservative, because being wrong in
/// the optimistic direction is the spill this module exists to prevent. So a
/// model only fits alongside when the measured free memory holds its weights
/// **and** [`CO_RESIDENT_HEADROOM`] on top.
pub fn plan_residency(
    weights_bytes: u64,
    free_vram_bytes: u64,
    vram_measured: bool,
    installed_vram_bytes: u64,
    available_ram_bytes: u64,
    something_already_running: bool,
) -> Residency {
    // Nothing on this machine can hold it, whatever is evicted first.
    if weights_bytes > installed_vram_bytes.saturating_add(available_ram_bytes) {
        return Residency::WontFit {
            needs_bytes: weights_bytes,
            vram_bytes: installed_vram_bytes,
            ram_bytes: available_ram_bytes,
        };
    }

    // Nothing else is loaded, so there is nothing to be co-resident *with* and
    // the ordinary admission path will place it.
    if !something_already_running {
        return Residency::FitsAlongside {
            free_bytes: free_vram_bytes,
            needs_bytes: weights_bytes,
        };
    }

    // The driver would not say. "Nobody can say how much is free" does not read
    // as "enough" — see `VramBudget::InstalledOnly`, which exists because
    // routing once planned against a number that was not a measurement.
    if !vram_measured {
        return Residency::Serialise {
            because: "this machine's driver does not report free video memory, so whether a \
                      second model fits alongside the first cannot be measured. The workers \
                      take turns rather than risk both being spilled into system memory."
                .to_string(),
        };
    }

    let needs = weights_bytes.saturating_add(CO_RESIDENT_HEADROOM);
    if free_vram_bytes >= needs {
        Residency::FitsAlongside {
            free_bytes: free_vram_bytes,
            needs_bytes: needs,
        }
    } else {
        Residency::Serialise {
            because: format!(
                "{} of video memory is free and a second model needs about {}, so this worker \
                 waits for the card rather than being spilled into system memory alongside the \
                 one already loaded",
                crate::serving::human_bytes(free_vram_bytes),
                crate::serving::human_bytes(needs)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The machine this product is built for: an 8 GB card with a model already
    /// on it. A second 4 GB model does not join it.
    #[test]
    fn two_models_that_do_not_both_fit_take_turns() {
        let residency = plan_residency(4 * GIB, 3 * GIB, true, 8 * GIB, 16 * GIB, true);
        assert!(residency.needs_exclusive_card(), "{residency:?}");
        assert!(
            residency.explain().contains("waits for the card"),
            "{}",
            residency.explain()
        );
    }

    /// A small model on a card with room genuinely runs alongside.
    #[test]
    fn a_small_model_with_room_runs_alongside() {
        let residency = plan_residency(2 * GIB, 6 * GIB, true, 24 * GIB, 32 * GIB, true);
        assert!(!residency.needs_exclusive_card(), "{residency:?}");
        assert!(matches!(residency, Residency::FitsAlongside { .. }));
    }

    /// The headroom is the point: weights that *just* fit do not, because a
    /// served model is more than its file.
    #[test]
    fn weights_that_only_just_fit_are_not_treated_as_fitting() {
        // 3 GiB free, a 2.5 GiB model. The file fits; the file plus a KV cache
        // does not.
        let residency = plan_residency(2 * GIB + GIB / 2, 3 * GIB, true, 8 * GIB, 16 * GIB, true);
        assert!(residency.needs_exclusive_card(), "{residency:?}");
    }

    /// A driver that will not report free memory is not an optimistic answer.
    #[test]
    fn an_unmeasurable_card_serialises_rather_than_guessing() {
        let residency = plan_residency(2 * GIB, 24 * GIB, false, 24 * GIB, 32 * GIB, true);
        assert!(residency.needs_exclusive_card(), "{residency:?}");
        assert!(
            residency.explain().contains("cannot be measured"),
            "{}",
            residency.explain()
        );
    }

    /// With nothing else loaded there is nothing to be co-resident with, so the
    /// ordinary admission path places it.
    #[test]
    fn the_first_worker_does_not_wait_for_a_card_nobody_is_using() {
        let residency = plan_residency(6 * GIB, GIB, true, 8 * GIB, 16 * GIB, false);
        assert!(!residency.needs_exclusive_card(), "{residency:?}");
    }

    /// A model larger than the whole machine is refused rather than queued
    /// behind a card it will never fit on.
    #[test]
    fn a_model_bigger_than_the_machine_is_refused_not_queued() {
        let residency = plan_residency(64 * GIB, GIB, true, 8 * GIB, 16 * GIB, true);
        match &residency {
            Residency::WontFit { .. } => {}
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(!residency.needs_exclusive_card());
        assert!(residency.explain().contains("no worker can run it here"));
    }

    /// A model already serving needs nothing decided: the weights are there.
    #[test]
    fn a_warm_model_is_already_resident() {
        assert_eq!(Residency::AlreadyWarm.as_str(), "alreadyWarm");
        assert!(!Residency::AlreadyWarm.needs_exclusive_card());
    }
}

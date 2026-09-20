//! Deciding whether a model can be served, and what to release so it can.
//!
//! ## The failure this exists for
//!
//! Routing planned the GPU offload against `dedicated_video_memory_bytes` —
//! the VRAM the card *has*. Nothing read the VRAM that was *left*. Measured on
//! the development machine: an 8151 MiB card with 6742 MiB already held by
//! another llama-server and the desktop, so 1158 MiB free, while the planner
//! budgeted 8151 − 900 = 7251 MiB and asked llama.cpp to place a model that
//! could not possibly fit. The consequences were the two symptoms reported:
//!
//! - A 9B model nominally at 97% offload running at 5 tok/s, because Windows
//!   spilled the allocation across PCIe into host memory rather than failing.
//! - A 12B model that never came up at all, where the surface sat on
//!   "Thinking" for the full 180-second readiness timeout before it could even
//!   report a failure.
//!
//! ## The three things this module does
//!
//! 1. **Plans against free VRAM.** Asking the driver is the only budget that
//!    accounts for consumers ARJUN does not control — another llama-server, an
//!    Ollama daemon, the compositor, a second copy of the app.
//! 2. **Reclaims only when reclaiming is needed.** A model that fits alongside
//!    what is already running does not evict it. This matters for documents:
//!    the OCR model and the chat model coexist happily on a large card, and a
//!    blanket eviction would reload one of them on every page.
//! 3. **Refuses what cannot run.** A model larger than the machine's memory is
//!    reported as that, immediately, instead of being started and waited for.
//!
//! ## Generic by construction
//!
//! Nothing here names a model, a family, or a quantisation. Layer count and
//! context length come from the GGUF header; size comes from the file; the
//! budget comes from the driver. A model nobody has heard of is planned the
//! same way as one that ships with the product.

use std::path::Path;

use crate::ai_engine::gguf_meta;
use crate::ai_engine::gguf_meta::KvCost;
use crate::ai_engine::vram_planner::{plan_gpu_offload_at, ContextChoice, GpuOffloadPlan};
use crate::registry::{ModelEntry, ModelRegistry};
use crate::serving::{ModelServers, ServingError};
use crate::system_analyzer::{gpu_collector, memory_collector};

/// Which VRAM figure a plan was made against.
///
/// Reported rather than folded away, because "planned against 1.1 GB free" and
/// "planned against 8 GB installed because the driver would not say" are very
/// different confidences in the same number, and an operator reading a slow
/// answer deserves to know which one they got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramBudget {
    /// Measured free VRAM.
    Free(u64),
    /// The driver reported no free figure, so the installed total was used and
    /// the plan is only as good as the card being otherwise idle.
    InstalledOnly(u64),
}

impl VramBudget {
    pub const fn bytes(self) -> u64 {
        match self {
            VramBudget::Free(bytes) | VramBudget::InstalledOnly(bytes) => bytes,
        }
    }

    pub const fn measured(self) -> bool {
        matches!(self, VramBudget::Free(_))
    }
}

/// What admitting one model settled.
#[derive(Debug, Clone)]
pub struct Admission {
    pub plan: GpuOffloadPlan,
    /// Servers stopped to make room, newest first. Empty when none were.
    pub released: Vec<String>,
    /// True when the in-process model was unloaded to make room.
    pub released_in_process: bool,
    pub budget: VramBudget,
    /// Layers read from the GGUF header, or `None` when it could not be read
    /// and the planner had to assume.
    pub layers: Option<u32>,
    /// Whether this model's own chat template exposes a reasoning switch.
    ///
    /// Carried out of here because the header has already been read for the
    /// layer count, and opening a multi-gigabyte file twice per turn to answer
    /// a second question about it would be wasteful.
    pub supports_reasoning: bool,
}

/// Plans the offload for one model, reclaiming VRAM only if it has to.
///
/// The caller passes the result straight to [`ModelServers::endpoint_for`].
/// Splitting the decision from the spawning keeps this testable without a
/// llama-server binary, which is where the interesting mistakes live.
pub async fn admit(
    servers: &ModelServers,
    entry: &ModelEntry,
    models_dir: &Path,
) -> Result<Admission, ServingError> {
    let weights = models_dir.join(&entry.path);

    // The layer count the planner otherwise assumes is 32. That happens to be
    // right for some models and wrong for most — Gemma 3 12B has 48 — and the
    // assumption scales the offload fraction, so a wrong count silently leaves
    // layers on the CPU that the plan believed were on the GPU.
    let header = gguf_meta::capabilities(&weights);
    let layers = header.layers;
    let supports_reasoning = header.supports_toggled_reasoning;
    // The exact KV geometry, from the header this line already read.
    //
    // It used to be dropped here, so every plan on this path was costed by the
    // size band — a proxy for `layers x kv_heads x head_dim` that is right
    // within a factor for a dense model and wrong by four for a hybrid one.
    // Spark-X2.5-4B has 27 of its 36 blocks on a 512-token window: the band
    // charged 80 KB a token where the model costs 18 KB, and the planner
    // answered by serving 16 384 tokens on a card measured to hold 65 536.
    let kv_cost = header.kv_cost;

    let installed = gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);

    // A model that does not fit in memory at all cannot be rescued by any
    // offload split, so it is refused here rather than started and waited for.
    let ram = memory_collector::detect_memory();
    if entry.weights_bytes > 0 && entry.weights_bytes > ram.available_bytes.saturating_add(installed)
    {
        return Err(ServingError::WontFit {
            model: entry.name.clone(),
            model_bytes: entry.weights_bytes,
            vram_bytes: installed,
            ram_bytes: ram.available_bytes,
        });
    }

    let mut budget = measure_budget(installed);
    let mut plan = plan_for(budget.bytes(), entry, layers, kv_cost);

    // Already comfortable, or the card is not the constraint. Nothing is
    // disturbed — an OCR server mid-document keeps its memory.
    if plan.full_offload || !budget.measured() {
        return Ok(Admission {
            plan,
            released: Vec::new(),
            released_in_process: false,
            budget,
            layers,
            supports_reasoning,
        });
    }

    // The in-process model goes first, and it is usually the whole problem.
    //
    // ARJUN has two ways to run a model and they do not know about each other:
    // `ai_engine::manager` loads one inside this process for the gateway, and
    // `ModelServers` starts `llama-server` children for chat and documents.
    // On the reported machine the startup restore loaded Qwen in-process,
    // taking 4.3 GB, and the first chat message then started a second copy of
    // the same model in a child — two loads of one model on an 8 GB card. The
    // card was exhausted by ARJUN talking to itself.
    //
    // Released before any server because it is one contiguous reclaim of a
    // whole model, where the servers may each be small.
    let mut released_in_process = false;
    if let Some(inference) = crate::ai_engine::manager::global() {
        if inference.resident_model_id().is_some() {
            match inference.unload_active_model_direct() {
                Ok(()) => {
                    log::info!(
                        "[serving] unloaded the in-process model to make room for {}",
                        entry.name
                    );
                    released_in_process = true;
                    gpu_collector::invalidate_free_vram_cache();
                    budget = measure_budget(installed);
                    plan = plan_for(budget.bytes(), entry, layers, kv_cost);
                }
                Err(error) => log::warn!(
                    "[serving] the in-process model could not be unloaded, so {} is planned                      against what is left: {error:#}",
                    entry.name
                ),
            }
        }
    }

    // Least recently used first.
    //
    // This was `running_model_ids()`, which returns `HashMap` keys — so the
    // server reclaimed first was whichever the hasher happened to yield. On the
    // two-model turn this product runs constantly (read the attachment with the
    // OCR model, answer with the chat model) that is a coin toss between the
    // model just used and the model about to be used, and losing the toss costs
    // a cold start and a destroyed prompt cache. Twice per message, on a
    // constrained card.
    let others: Vec<String> = servers
        .eviction_order()
        .into_iter()
        .filter(|id| id != &entry.id)
        .collect();
    if plan.full_offload || others.is_empty() {
        return Ok(Admission {
            plan,
            released: Vec::new(),
            released_in_process,
            budget,
            layers,
            supports_reasoning,
        });
    }

    // Reclaim one server at a time and re-measure between each, so the
    // smallest number of them is disturbed. Stopping every server to fit a
    // model that only needed one released is how a document read loses its
    // OCR model to a chat message.
    let mut released = Vec::new();
    for id in others {
        servers.stop(&id).await;
        gpu_collector::invalidate_free_vram_cache();
        released.push(id);

        budget = measure_budget(installed);
        plan = plan_for(budget.bytes(), entry, layers, kv_cost);
        if plan.full_offload {
            break;
        }
    }

    Ok(Admission {
        plan,
        released,
        released_in_process,
        budget,
        layers,
        supports_reasoning,
    })
}

/// One offload plan for this entry against a measured budget.
///
/// A named helper rather than four copies of the same five arguments: `admit`
/// re-plans after every reclaim, and the four calls disagreeing about what the
/// KV cache costs is precisely the class of drift this wraps up.
///
/// The window is `Planned`, which is what this path has always passed — the
/// registry entry states the model's trained window and the planner is free to
/// walk down it to buy layers. An operator's fixed window is set in Settings
/// and handled on the in-process path, not here.
fn plan_for(
    budget_bytes: u64,
    entry: &ModelEntry,
    layers: Option<u32>,
    kv_cost: Option<KvCost>,
) -> GpuOffloadPlan {
    plan_gpu_offload_at(
        budget_bytes,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        layers,
        kv_cost,
        // Asked of the binary that will actually serve this, rather than
        // assumed. See `KvPrecision`: assuming `q8_0` on a build that does not
        // accept `-ctk` sizes the window against half the memory the server
        // goes on to allocate.
        super::llama_server_kv_precision(),
    )
}

/// Free VRAM where the driver will say, the installed total where it will not.
///
/// Public because routing has to ask the same question. The router used to plan
/// against `installed_gpus().max()` while admission planned against this, so on
/// a card already holding a server the router would pick the largest model that
/// fits a budget that does not exist, and admission would then partially
/// offload it — the exact failure this module's header describes.
///
/// Falling back to the installed figure rather than to zero is deliberate: a
/// machine whose driver reports no free figure — an AMD card, a headless box
/// without `nvidia-smi` — still has VRAM, and refusing to use it would be a
/// worse answer than the over-optimistic plan that was there before.
pub fn measure_budget(installed: u64) -> VramBudget {
    match gpu_collector::free_vram_bytes() {
        Some(free) => VramBudget::Free(free),
        None => VramBudget::InstalledOnly(installed),
    }
}

/// VRAM that ARJUN is holding and [`admit`] would release to make room.
///
/// ## Why routing needs this and free VRAM alone is not enough
///
/// Routing and loading have to answer to the same number, and an earlier fix
/// got them halfway there: routing used to plan against the *installed* total
/// while admission measured what was free, so the router picked the largest
/// model that "fits in 8 GB" and admission then partially offloaded it. Passing
/// free VRAM fixed that direction.
///
/// It opened the opposite one. Admission does not merely *measure* free VRAM —
/// it reclaims, unloading the in-process model and then servers until the model
/// fits. Free VRAM therefore understates what admission can actually offer, by
/// exactly the amount ARJUN is already using.
///
/// The visible failure is a feedback loop. Measured on an 8 GB laptop card:
/// with nothing loaded, 7.4 GB is free and a 4.07 GB model plans as a full
/// offload. Once it is resident, 2.3 GB is free — so the *next* turn plans the
/// same model, already sitting on the GPU, as not fitting, marks the decision
/// `used_fallback`, and the panel reports "partly on CPU" for a model that is
/// entirely on the card. Worse than a wrong label: the clamp in
/// `ai_engine::manager` then lowers the layer count to match the shrunken
/// budget, so the next load genuinely is partly on the CPU. The mislabel makes
/// itself true.
///
/// So the budget a router plans against is what ARJUN can *make* available:
/// what is free, plus what it would give back.
///
/// Counted from the registry's recorded `weights_bytes` rather than by asking
/// the driver, because the driver reports a total for the process and cannot
/// say which model an allocation belongs to. A model with no recorded size
/// contributes nothing, which errs toward the smaller budget.
pub fn reclaimable_bytes(registry: &ModelRegistry, servers: &ModelServers) -> u64 {
    let weights_of = |model_id: &str| -> u64 {
        registry
            .all()
            .iter()
            .find(|entry| entry.id == model_id)
            .map(|entry| entry.weights_bytes)
            .unwrap_or(0)
    };

    let in_process = crate::ai_engine::manager::global()
        .and_then(|inference| inference.resident_model_id())
        .map(|id| weights_of(&id))
        .unwrap_or(0);

    let in_servers: u64 = servers
        .running_model_ids()
        .iter()
        .map(|id| weights_of(id))
        .sum();

    in_process.saturating_add(in_servers)
}

/// The budget a routing decision should be planned against.
///
/// [`measure_budget`] plus [`reclaimable_bytes`], capped at the card — ARJUN
/// giving memory back cannot produce more VRAM than the GPU has. When the
/// driver reports no free figure the installed total is used unchanged, which
/// is what [`VramBudget::InstalledOnly`] already means.
pub fn routable_budget(
    installed: u64,
    registry: &ModelRegistry,
    servers: &ModelServers,
) -> VramBudget {
    match measure_budget(installed) {
        VramBudget::Free(free) => {
            VramBudget::Free(widen(free, reclaimable_bytes(registry, servers), installed))
        }
        installed_only => installed_only,
    }
}

/// Free plus reclaimable, bounded by the card.
///
/// Separated from [`routable_budget`] because this is the part with edge cases
/// and the rest needs a GPU and a running server to exercise.
///
/// The ceiling is `installed.max(free)` rather than `installed`: a driver that
/// reports more free than this build recorded as installed is describing a
/// machine this function should not silently contradict, and clamping to a
/// stale smaller total would throw away real memory. Whichever is larger is the
/// honest ceiling.
fn widen(free: u64, reclaimable: u64, installed: u64) -> u64 {
    free.saturating_add(reclaimable).min(installed.max(free))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    /// The feedback loop, in numbers taken from the machine that showed it.
    ///
    /// 8 GB card, a 4.07 GB model resident. Free VRAM alone says 2.3 GB and the
    /// model that is *already on the card* does not fit in it. Adding back what
    /// ARJUN would release restores the budget it actually has.
    #[test]
    fn a_resident_model_is_added_back_to_the_budget() {
        let installed = 8 * GB;
        let resident = 4 * GB + GB / 10;
        let free = 2 * GB + GB / 3;

        assert!(
            free < resident,
            "the premise: free VRAM alone cannot hold the model already loaded"
        );
        assert!(
            widen(free, resident, installed) >= resident,
            "the budget must cover a model ARJUN could reload after reclaiming"
        );
    }

    /// Giving memory back cannot produce more than the card holds.
    #[test]
    fn the_budget_never_exceeds_the_card() {
        let installed = 8 * GB;
        assert_eq!(widen(7 * GB, 6 * GB, installed), installed);
        assert_eq!(widen(installed, 4 * GB, installed), installed);
    }

    /// Nothing resident means nothing to add: the old behaviour, unchanged.
    #[test]
    fn with_nothing_loaded_the_budget_is_what_is_free() {
        assert_eq!(widen(7 * GB, 0, 8 * GB), 7 * GB);
    }

    /// A driver reporting more free than the recorded total is not clamped
    /// down to the stale figure.
    #[test]
    fn a_free_figure_above_the_recorded_total_is_not_thrown_away() {
        assert_eq!(widen(9 * GB, 0, 8 * GB), 9 * GB);
    }

    /// Absurd inputs saturate rather than wrap.
    #[test]
    fn the_arithmetic_saturates() {
        assert_eq!(widen(u64::MAX, u64::MAX, u64::MAX), u64::MAX);
    }
}

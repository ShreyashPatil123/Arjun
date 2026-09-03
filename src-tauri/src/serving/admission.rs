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
use crate::ai_engine::vram_planner::{plan_gpu_offload, GpuOffloadPlan};
use crate::registry::ModelEntry;
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
    let header = gguf_meta::read_gguf_metadata(&weights).ok();
    let layers = header
        .as_ref()
        .map(|meta| meta.block_count)
        .filter(|count| *count > 0);
    let supports_reasoning = header
        .as_ref()
        .map(|meta| meta.supports_toggled_reasoning)
        .unwrap_or(false);

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

    let budget = measure_budget(installed);
    let plan = plan_gpu_offload(budget.bytes(), entry.weights_bytes, entry.context_length, layers);

    // Already comfortable, or the card is not the constraint. Nothing is
    // disturbed — an OCR server mid-document keeps its memory.
    if plan.full_offload || !budget.measured() {
        return Ok(Admission {
            plan,
            released: Vec::new(),
            budget,
            layers,
            supports_reasoning,
        });
    }

    let others: Vec<String> = servers
        .running_model_ids()
        .into_iter()
        .filter(|id| id != &entry.id)
        .collect();
    if others.is_empty() {
        return Ok(Admission {
            plan,
            released: Vec::new(),
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
    let mut budget = budget;
    let mut plan = plan;
    for id in others {
        servers.stop(&id).await;
        gpu_collector::invalidate_free_vram_cache();
        released.push(id);

        budget = measure_budget(installed);
        plan = plan_gpu_offload(budget.bytes(), entry.weights_bytes, entry.context_length, layers);
        if plan.full_offload {
            break;
        }
    }

    Ok(Admission {
        plan,
        released,
        budget,
        layers,
        supports_reasoning,
    })
}

/// Free VRAM where the driver will say, the installed total where it will not.
///
/// Falling back to the installed figure rather than to zero is deliberate: a
/// machine whose driver reports no free figure — an AMD card, a headless box
/// without `nvidia-smi` — still has VRAM, and refusing to use it would be a
/// worse answer than the over-optimistic plan that was there before.
fn measure_budget(installed: u64) -> VramBudget {
    match gpu_collector::free_vram_bytes() {
        Some(free) => VramBudget::Free(free),
        None => VramBudget::InstalledOnly(installed),
    }
}

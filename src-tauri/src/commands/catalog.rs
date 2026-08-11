//! IPC for browsing models by category.
//!
//! Returns presentation-ready cards — publisher, licence, popularity, age,
//! categories, and every quantization sized against this machine's memory — so
//! the browser can render and filter without re-deriving anything.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::model_providers::huggingface::card::{build_card, ModelCard, ModelCategory};
use crate::model_providers::huggingface::discovery::GgufRepo;
use crate::model_providers::huggingface::live_catalog;

/// How long a browse sweep stays fresh.
///
/// Each sweep costs one search plus a detail request per repository — around a
/// hundred calls. Without this, every visit to the model browser repeated the
/// whole thing, and React's StrictMode doubles it in development: four full
/// sweeps were observed on a single startup, which walks straight into
/// HuggingFace's anonymous rate limit.
const BROWSE_CACHE_TTL: Duration = Duration::from_secs(600);

/// Cached sweep, with the authentication state that produced it.
///
/// The token decides how much of the library a sweep reaches — one search page
/// anonymously against five with a token — so results gathered under one state
/// do not describe the other. Without the flag a cached authenticated sweep
/// could be served to an anonymous caller, which showed a full 182-model
/// listing underneath a notice offering to unlock the full library.
type BrowseCache = Mutex<Option<(Instant, bool, Vec<GgufRepo>)>>;

fn browse_cache() -> &'static BrowseCache {
    static CACHE: OnceLock<BrowseCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Serialises sweeps so simultaneous callers share one fetch instead of each
/// starting their own — the cache is only written once a sweep completes.
fn browse_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn cached_repos(authenticated: bool) -> Option<Vec<GgufRepo>> {
    let guard = browse_cache().lock().ok()?;
    let (fetched_at, was_authenticated, repos) = guard.as_ref()?;
    (*was_authenticated == authenticated && fetched_at.elapsed() < BROWSE_CACHE_TTL)
        .then(|| repos.clone())
}

fn store_repos(authenticated: bool, repos: &[GgufRepo]) {
    if let Ok(mut guard) = browse_cache().lock() {
        *guard = Some((Instant::now(), authenticated, repos.to_vec()));
    }
}

/// Drops the cached sweep so the next browse re-fetches.
///
/// Called when the HuggingFace token changes: the cached results were gathered
/// under the old credentials, and an anonymous sweep reaches far less of the
/// library than an authenticated one. Without this, adding a token appeared to
/// do nothing until the ten-minute TTL expired.
pub fn invalidate_browse_cache() {
    if let Ok(mut guard) = browse_cache().lock() {
        *guard = None;
    }
}

/// Cards plus the facts needed to explain an empty or partial result.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogPage {
    pub cards: Vec<ModelCard>,
    /// Every category present in these results, so the sidebar only offers
    /// filters that would actually match something.
    pub categories: Vec<CategoryCount>,
    /// Memory available for model weights, used to mark which quantizations fit.
    /// Zero when hardware could not be read — nothing is then marked as fitting.
    pub weight_budget_bytes: u64,
    /// Set when the sweep was cut short, e.g. by rate limiting. The results are
    /// still usable; the message explains why there are fewer than expected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryCount {
    pub category: ModelCategory,
    pub label: String,
    pub count: usize,
}

/// Memory this machine can give to model weights.
///
/// Budgets against the same card the runtime would actually load onto, so the
/// "runs here" column and what happens on Load cannot disagree. Returns 0 when
/// no usable GPU is detected, so callers mark nothing as fitting rather than
/// guessing.
///
/// Two things this deliberately does not do. It does not take the largest
/// reported VRAM: an integrated GPU advertises a slice of system RAM as its
/// own, and on this machine the Radeon 780M's 13 GB outranked an RTX 5060's
/// real 8 GB — which marked 9 GB models as fitting on a card that cannot hold
/// them. And it budgets from *total* rather than free VRAM, because a browse
/// listing that shifts as other processes take memory would be unreadable.
fn weight_budget() -> u64 {
    let analyzer = crate::system_analyzer::get_system_analyzer_manager();
    let Some(profile) = analyzer.get_profile() else {
        return 0;
    };
    weight_budget_for(&profile.gpus.current())
}

/// The budget arithmetic, separated from hardware detection so it can be tested.
///
/// Charges the same three costs the VRAM planner does, in the same order: the
/// OS reserve, then the KV cache, then compute buffers. Weights get the rest.
///
/// Omitting the KV cache made this noticeably more generous than the planner —
/// 6.46 GB against its 5.46 GB on an 8 GB card — so a quantization the browser
/// marked "fits" could still fail to fully offload on Load. The KV cost depends
/// on context, and browsing has no session to ask, so it is priced at a typical
/// working context rather than a model's advertised maximum.
fn weight_budget_for(gpus: &[crate::system_analyzer::traits::GpuInfo]) -> u64 {
    /// Context the browse listing prices the KV cache at.
    const BROWSE_CONTEXT: u64 = 8192;
    const OS_RESERVE: u64 = 900 * 1024 * 1024;

    let Some(gpu) = crate::ai_engine::manager::select_inference_gpu(gpus) else {
        return 0;
    };

    let usable = gpu.vram_total_bytes.saturating_sub(OS_RESERVE);

    // Sized for a mid-range model: the exact per-token figure depends on the
    // model, which differs per card in the listing, so this uses the band the
    // planner applies to everything from ~3–8 GB of weights.
    let kv_bytes = crate::ai_engine::vram_planner::estimate_kv_bytes_per_token(4 * 1024 * 1024 * 1024)
        * BROWSE_CONTEXT;

    let after_kv = usable.saturating_sub(kv_bytes);
    after_kv.saturating_sub((after_kv as f64 * 0.12) as u64)
}

fn tally_categories(cards: &[ModelCard]) -> Vec<CategoryCount> {
    ModelCategory::all()
        .iter()
        .filter_map(|cat| {
            let count = cards.iter().filter(|c| c.categories.contains(cat)).count();
            // Offering a filter that matches nothing is a dead end.
            (count > 0).then(|| CategoryCount {
                category: *cat,
                label: cat.label().to_string(),
                count,
            })
        })
        .collect()
}

/// Browses the model catalog, optionally filtered by a search term.
///
/// A search queries HuggingFace directly rather than filtering the cached
/// listing: the cache holds only the popular sweep, and someone searching for a
/// specific fine-tune would otherwise be told it does not exist.
#[tauri::command]
pub async fn browse_model_cards(query: Option<String>) -> Result<CatalogPage, String> {
    let token = crate::config::hf_token::get();
    let search = query.as_deref().map(str::trim).filter(|q| !q.is_empty());

    let repos = match search {
        // A search is specific and user-initiated, so it always goes to the Hub
        // — serving it from the popular-sweep cache would report that a model
        // the user named does not exist.
        Some(term) => live_catalog::discover_repos(Some(term), 1, token.as_deref())
            .await
            // Rate limiting is the common, fixable case and its message says
            // what to do, so it is passed through unchanged.
            .map_err(|e| e.to_string())?,

        None => {
            let authenticated = token.is_some();

            if let Some(cached) = cached_repos(authenticated) {
                log::debug!("[CATALOG] Serving {} repositories from cache", cached.len());
                cached
            } else {
                let _guard = browse_lock().lock().await;

                // Re-check: another caller may have finished the same sweep
                // while this one waited for the lock.
                if let Some(cached) = cached_repos(authenticated) {
                    cached
                } else {
                    let pages = live_catalog::pages_for(token.as_deref());
                    let fetched = live_catalog::discover_repos(None, pages, token.as_deref())
                        .await
                        .map_err(|e| e.to_string())?;
                    store_repos(authenticated, &fetched);
                    fetched
                }
            }
        }
    };

    let budget = weight_budget();
    let now = chrono::Utc::now();

    // Adapters are not models and must not be browsed as though they were.
    //
    // A LoRA is a patch that cannot load on its own; listing it beside runnable
    // models offered a Download button for something that would never start.
    // They stay discoverable in the place they make sense — the adapter list on
    // the base model they were trained against.
    let cards: Vec<ModelCard> = repos
        .iter()
        .filter(|r| !r.is_lora_adapter)
        .map(|r| build_card(r, budget, now))
        .collect();

    let notice = (token.is_none() && search.is_none()).then(|| {
        "Showing the most popular models. Add a free HuggingFace token in Settings to browse \
         the full library."
            .to_string()
    });

    Ok(CatalogPage {
        categories: tally_categories(&cards),
        cards,
        weight_budget_bytes: budget,
        notice,
    })
}

/// Adapters published for a model, plus how usable they are here.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterPage {
    pub adapters: Vec<live_catalog::AdapterListing>,
    /// How many can be loaded as they are. The rest are PEFT safetensors, which
    /// Sarathi converts to GGUF during installation.
    pub ready_count: usize,
    /// Shown when none are directly usable, explaining what that means.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

/// Lists LoRA adapters published for a base model.
///
/// Looks up HuggingFace's `base_model:adapter:` tag, which adapter authors set
/// to declare their parent — a real relationship, not a name-similarity guess.
#[tauri::command]
pub async fn find_model_adapters(base_model_id: String) -> Result<AdapterPage, String> {
    let token = crate::config::hf_token::get();

    let adapters = live_catalog::find_adapters(&base_model_id, 20, token.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    let ready_count = adapters.iter().filter(|a| a.gguf_ready).count();

    let notice = if adapters.is_empty() {
        Some("No LoRA adapters published for this model yet.".to_string())
    } else if ready_count == 0 {
        Some(format!(
            "{} adapter(s) found. None ship GGUF, so Sarathi converts them during install — \
             which needs this base model installed and its family supported.",
            adapters.len()
        ))
    } else {
        None
    };

    Ok(AdapterPage { adapters, ready_count, notice })
}

/// Every category the app knows about, for a sidebar that stays stable while
/// results load.
#[tauri::command]
pub fn list_model_categories() -> Vec<CategoryCount> {
    ModelCategory::all()
        .iter()
        .map(|c| CategoryCount { category: *c, label: c.label().to_string(), count: 0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_analyzer::traits::GpuInfo;

    fn gpu(model: &str, dedicated: bool, total: u64, cuda: bool) -> GpuInfo {
        GpuInfo {
            vendor: String::new(),
            model: model.to_string(),
            gpu_type: String::new(),
            is_dedicated: dedicated,
            dedicated_video_memory_bytes: if dedicated { total } else { 0 },
            dedicated_system_memory_bytes: 0,
            shared_system_memory_bytes: if dedicated { 0 } else { total },
            total_available_graphics_memory_bytes: total,
            vram_total_bytes: total,
            vram_free_bytes: total,
            driver_version: None,
            vendor_id: None,
            device_id: None,
            compute_capability: None,
            cuda_supported: cuda,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: true,
            opencl_supported: true,
            detection_source: "test".into(),
            confidence: "High".into(),
        }
    }

    #[test]
    fn an_integrated_gpus_shared_memory_never_sets_the_budget() {
        // This machine exactly: a Radeon 780M reporting 13 GB of shared system
        // memory next to an RTX 5060 with 8.28 GB of its own.
        //
        // Taking the larger figure produced a ~9.5 GB budget, which marked a
        // 9.1 GB Q6_K as fitting on a card that holds 8.28 GB.
        let gpus = vec![
            gpu("AMD Radeon 780M", false, 13_050_000_000, false),
            gpu("NVIDIA GeForce RTX 5060 Laptop GPU", true, 8_280_000_000, true),
        ];

        let budget = weight_budget_for(&gpus);

        assert!(budget < 8_280_000_000, "budget cannot exceed the card's own VRAM");
        assert!(
            budget < 9_100_000_000,
            "a 9.1 GB Q6_K must not be marked as fitting; budget was {budget}"
        );
        // Roughly 5.5 GB once the KV cache and compute buffers are charged.
        assert!((5_000_000_000..6_200_000_000).contains(&budget), "unexpected budget {budget}");
    }

    #[test]
    fn the_browse_budget_is_not_more_generous_than_the_loader() {
        // These two answer the same question and used to disagree: the listing
        // marked a 6.09 GB Q3_K_M as fitting, then the loader refused to fully
        // offload it because its budget also carried the KV cache.
        let gpus = vec![gpu("NVIDIA GeForce RTX 5060 Laptop GPU", true, 8_280_000_000, true)];
        let browse = weight_budget_for(&gpus);

        let plan_budget = {
            const OS: u64 = 900 * 1024 * 1024;
            let usable = 8_280_000_000u64.saturating_sub(OS);
            let kv = crate::ai_engine::vram_planner::estimate_kv_bytes_per_token(4 * 1024 * 1024 * 1024)
                * 8192;
            let after_kv = usable.saturating_sub(kv);
            after_kv.saturating_sub((after_kv as f64 * 0.12) as u64)
        };

        assert!(
            browse <= plan_budget,
            "browse budget {browse} must not exceed the loader's {plan_budget}"
        );
    }

    #[test]
    fn no_usable_gpu_means_nothing_is_marked_as_fitting() {
        // Zero is the signal for "unknown", and `to_option` marks nothing as
        // fitting rather than guessing that everything does.
        assert_eq!(weight_budget_for(&[]), 0);
    }

    #[test]
    fn an_integrated_gpu_still_sets_the_budget_when_it_is_all_there_is() {
        let gpus = vec![gpu("AMD Radeon 780M", false, 13_050_000_000, false)];
        assert!(weight_budget_for(&gpus) > 0);
    }
}

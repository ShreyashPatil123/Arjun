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

type BrowseCache = Mutex<Option<(Instant, Vec<GgufRepo>)>>;

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

fn cached_repos() -> Option<Vec<GgufRepo>> {
    let guard = browse_cache().lock().ok()?;
    let (fetched_at, repos) = guard.as_ref()?;
    (fetched_at.elapsed() < BROWSE_CACHE_TTL).then(|| repos.clone())
}

fn store_repos(repos: &[GgufRepo]) {
    if let Ok(mut guard) = browse_cache().lock() {
        *guard = Some((Instant::now(), repos.to_vec()));
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
/// Uses the largest accelerated GPU's VRAM, less the reserve the planner keeps
/// for the OS and compute buffers. Returns 0 when no usable GPU is detected, so
/// callers mark nothing as fitting rather than guessing.
fn weight_budget() -> u64 {
    let analyzer = crate::system_analyzer::get_system_analyzer_manager();
    let Some(profile) = analyzer.get_profile() else {
        return 0;
    };

    profile
        .gpus
        .current()
        .iter()
        .filter(|g| g.cuda_supported || g.vulkan_supported)
        .map(|g| g.vram_total_bytes)
        .max()
        .map(|vram| {
            // Same shape as the VRAM planner: hold back the OS reserve, then a
            // slice for compute buffers. Weights get what is left.
            const OS_RESERVE: u64 = 900 * 1024 * 1024;
            let usable = vram.saturating_sub(OS_RESERVE);
            usable.saturating_sub((usable as f64 * 0.12) as u64)
        })
        .unwrap_or(0)
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
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty());
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
            if let Some(cached) = cached_repos() {
                log::debug!("[CATALOG] Serving {} repositories from cache", cached.len());
                cached
            } else {
                let _guard = browse_lock().lock().await;

                // Re-check: another caller may have finished the same sweep
                // while this one waited for the lock.
                if let Some(cached) = cached_repos() {
                    cached
                } else {
                    let pages = live_catalog::pages_for(token.as_deref());
                    let fetched = live_catalog::discover_repos(None, pages, token.as_deref())
                        .await
                        .map_err(|e| e.to_string())?;
                    store_repos(&fetched);
                    fetched
                }
            }
        }
    };

    let budget = weight_budget();
    let now = chrono::Utc::now();
    let cards: Vec<ModelCard> = repos.iter().map(|r| build_card(r, budget, now)).collect();

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
    /// How many can be loaded as they are. The rest are PEFT safetensors and
    /// need converting to GGUF first, which Sarathi cannot yet do — saying so
    /// is better than offering a download that will not load.
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
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty());

    let adapters = live_catalog::find_adapters(&base_model_id, 20, token.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    let ready_count = adapters.iter().filter(|a| a.gguf_ready).count();

    let notice = if adapters.is_empty() {
        Some("No LoRA adapters published for this model yet.".to_string())
    } else if ready_count == 0 {
        Some(format!(
            "{} adapter(s) found, but none ship GGUF files. They are PEFT safetensors and need \
             converting before llama.cpp can load them.",
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

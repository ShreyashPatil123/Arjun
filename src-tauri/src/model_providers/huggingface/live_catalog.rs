//! Live catalog fetching against the HuggingFace Hub.
//!
//! Turns [`discovery`] parsing into actual network calls. Two phases, because
//! bulk search omits the `gguf` metadata object:
//!
//! 1. One search request returns candidate repositories.
//! 2. Per-repo detail requests (run concurrently, in bounded batches) return
//!    parameter counts, architecture, context length, and exact file sizes.
//!
//! ## Authentication
//!
//! No token is required. Searching, reading model metadata, and reading the
//! metadata of *gated* repositories all work anonymously — verified against the
//! live API. A token is needed only to **download** files from a gated repo such
//! as `meta-llama/*` or `google/gemma-*`, and to raise anonymous rate limits.
//!
//! So a token is optional here and requested only when a download needs it,
//! rather than being demanded during setup.

use std::time::Duration;

use anyhow::{anyhow, Result};

use crate::model_providers::huggingface::discovery::{
    detail_url, estimate_architecture, family_name, search_url, GgufRepo, RawModelInfo,
};
use crate::model_recommendation::traits::{ModelArchitecture, ModelMetadata};

/// Concurrent detail requests. Kept modest so anonymous rate limits are not hit
/// and a large sweep does not saturate the user's connection.
const CONCURRENCY: usize = 8;

/// Search pages to sweep without a token.
///
/// HuggingFace rate-limits anonymous callers by IP, and every repository costs
/// one detail request on top of the search. A 5-page sweep is ~500 requests and
/// reliably trips the limit, which returns 429 and empties the catalog. One page
/// keeps anonymous use inside the allowance.
pub const ANONYMOUS_PAGES: u32 = 1;

/// Search pages to sweep with a token. Authenticated limits are far higher.
pub const AUTHENTICATED_PAGES: u32 = 5;

/// Pages to sweep given whether a token is available.
pub fn pages_for(token: Option<&str>) -> u32 {
    if token.map(str::trim).is_some_and(|t| !t.is_empty()) {
        AUTHENTICATED_PAGES
    } else {
        ANONYMOUS_PAGES
    }
}

/// Raised when HuggingFace rejects a request for rate-limiting reasons.
///
/// Distinguished from other failures so the UI can tell the user the specific,
/// fixable cause — add a free token — rather than a generic network error.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimited {
    pub had_token: bool,
}

impl std::fmt::Display for RateLimited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.had_token {
            write!(
                f,
                "HuggingFace rate limit reached even with a token. Wait a few minutes and try again."
            )
        } else {
            write!(
                f,
                "HuggingFace rate-limited this connection. Add a free HuggingFace token in \
                 Settings to browse the full model library."
            )
        }
    }
}

impl std::error::Error for RateLimited {}

/// Per-request timeout. Detail calls are small JSON documents.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Repos with fewer downloads than this are skipped — the long tail is mostly
/// abandoned experiments and personal scratch uploads.
const MIN_DOWNLOADS: u64 = 50;

fn client(token: Option<&str>) -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}"))
            .map_err(|_| anyhow!("HuggingFace token contains invalid characters"))?;
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .timeout(REQUEST_TIMEOUT)
        .default_headers(headers)
        .build()
        .map_err(Into::into)
}

/// Searches the Hub for GGUF repositories.
///
/// `query` is an optional free-text filter, so the UI can search the full Hub
/// rather than only what the default listing returns.
pub async fn search_repos(
    query: Option<&str>,
    limit: u32,
    page: u32,
    token: Option<&str>,
) -> Result<Vec<String>> {
    let client = client(token)?;
    let url = search_url(query, limit, page);

    let resp = client.get(&url).send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !status.is_success() {
        return Err(anyhow!("HuggingFace search returned {status}"));
    }

    let raw: Vec<RawModelInfo> = resp.json().await?;
    Ok(raw
        .into_iter()
        .filter(|r| r.downloads >= MIN_DOWNLOADS)
        .map(|r| r.id)
        .collect())
}

/// Fetches full details for one repository.
pub async fn fetch_repo(repo_id: &str, token: Option<&str>) -> Result<GgufRepo> {
    let client = client(token)?;
    let resp = client.get(detail_url(repo_id)).send().await?;

    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !status.is_success() {
        return Err(anyhow!("HuggingFace returned {status} for {repo_id}"));
    }

    let raw: RawModelInfo = resp.json().await?;
    raw.into_repo()
        .ok_or_else(|| anyhow!("{repo_id} has no usable GGUF files"))
}

/// Fetches many repositories concurrently, in bounded batches.
///
/// Individual failures are logged and skipped rather than failing the sweep —
/// one unreachable repo must not empty the catalog.
pub async fn fetch_repos(repo_ids: &[String], token: Option<&str>) -> Vec<GgufRepo> {
    let mut out = Vec::with_capacity(repo_ids.len());

    for chunk in repo_ids.chunks(CONCURRENCY) {
        let futures = chunk.iter().map(|id| async move {
            match fetch_repo(id, token).await {
                Ok(repo) => (Some(repo), false),
                Err(e) => {
                    let limited = e.downcast_ref::<RateLimited>().is_some();
                    if !limited {
                        log::debug!("[HF_CATALOG] Skipping {id}: {e}");
                    }
                    (None, limited)
                }
            }
        });

        let results = futures_util::future::join_all(futures).await;

        // Once the limit is hit, every remaining request will fail the same way.
        // Stop and keep what we have rather than spending hundreds of doomed
        // requests and digging the rate limit deeper.
        let limited = results.iter().any(|(_, l)| *l);
        out.extend(results.into_iter().filter_map(|(r, _)| r));

        if limited {
            log::warn!(
                "[HF_CATALOG] Rate limited after {} repositories — keeping partial results. {}",
                out.len(),
                RateLimited { had_token: token.is_some() }
            );
            break;
        }
    }

    out
}

/// A LoRA adapter published for some base model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterListing {
    pub repo_id: String,
    pub name: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    /// True when the repo ships `.gguf` files, which llama.cpp can load directly.
    ///
    /// Most published adapters are PEFT safetensors and are **not** loadable as
    /// they stand — they need converting to GGUF first. Saying so on the card is
    /// the difference between "download this and it works" and a file that
    /// silently fails to load.
    pub gguf_ready: bool,
    /// Short description derived from the repo name, e.g. `text to sql`.
    pub focus: String,
}

/// Finds LoRA adapters published for a given base model.
///
/// Uses HuggingFace's `base_model:adapter:` tag, which adapter authors set to
/// declare their parent — so this is a real lookup, not a name-similarity guess.
pub async fn find_adapters(
    base_model_id: &str,
    limit: u32,
    token: Option<&str>,
) -> Result<Vec<AdapterListing>> {
    let base = base_model_id.trim();
    if base.is_empty() || !base.contains('/') {
        // Without an org/name id there is nothing to match against.
        return Ok(Vec::new());
    }

    let client = client(token)?;
    let url = format!(
        "https://huggingface.co/api/models?filter=base_model:adapter:{}&sort=downloads&direction=-1&limit={}&full=true",
        base,
        limit.clamp(1, 50)
    );

    let resp = client.get(&url).send().await?;
    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !resp.status().is_success() {
        return Err(anyhow!("HuggingFace returned {} searching adapters", resp.status()));
    }

    let raw: Vec<RawModelInfo> = resp.json().await?;
    Ok(raw.into_iter().map(to_adapter_listing).collect())
}

fn to_adapter_listing(raw: RawModelInfo) -> AdapterListing {
    let author = raw
        .author
        .clone()
        .unwrap_or_else(|| raw.id.split('/').next().unwrap_or("").to_string());

    // Asking "does it contain a .gguf?" is not enough. Some repositories carry
    // the `base_model:adapter:` tag while shipping fully merged weights, and a
    // model GGUF passes that test — the Get button would then start a
    // multi-gigabyte download that can never be bound as an adapter.
    //
    // The installability rule lives in one place so the button and the download
    // cannot disagree about what is loadable.
    let filenames: Vec<String> = raw.siblings.iter().map(|s| s.rfilename.clone()).collect();
    let gguf_ready = crate::adapter_manager::store::check_installable(&filenames).is_ok();

    let short = raw.id.split('/').next_back().unwrap_or(&raw.id);
    let name = short.replace(['-', '_'], " ");

    // Trim only the boilerplate suffixes, keeping the rest of the name intact.
    //
    // An earlier version filtered out family names, digits, and anything ending
    // in "b", trying to distil a "focus". It mangled real names into single
    // meaningless words — "Persim", "Iraqi", "Merged" — because it stripped
    // everything that looked like model metadata and kept whatever was left.
    // The author's own name is more informative than a guess at its meaning.
    let focus = {
        let lowered = name.to_lowercase();
        let mut trimmed = lowered.as_str();
        for suffix in [" lora", " peft", " adapter", " finetune", " ft"] {
            trimmed = trimmed.strip_suffix(suffix).unwrap_or(trimmed);
        }
        let cleaned = trimmed.trim();
        if cleaned.is_empty() { name.clone() } else { cleaned.to_string() }
    };

    AdapterListing {
        repo_id: raw.id,
        name,
        author,
        downloads: raw.downloads,
        likes: raw.likes,
        gguf_ready,
        focus: if focus.trim().is_empty() { "general".to_string() } else { focus },
    }
}

/// Converts a discovered repository into the engine's model record.
///
/// Returns `None` when the repo lacks the GGUF metadata needed to reason about
/// it — better to omit a model than to recommend one on invented numbers.
pub fn to_model_metadata(repo: &GgufRepo) -> Option<ModelMetadata> {
    let gguf = repo.gguf.as_ref()?;
    if gguf.total_parameters == 0 {
        return None;
    }

    let arch = estimate_architecture(gguf.total_parameters);

    // `ModelArchitecture::MixtureOfExperts` requires expert counts, and the Hub's
    // GGUF metadata does not expose them. Rather than invent numbers, discovered
    // models are recorded as Dense.
    //
    // This is safe for the calculation that matters: `gguf.total` already counts
    // every expert's weights, because llama.cpp must load all of them, so memory
    // budgeting is correct either way. Expert counts feed only speed estimation,
    // which is not yet implemented.
    let architecture = ModelArchitecture::Dense;

    let mut use_cases = vec!["chat".to_string(), "general".to_string()];
    let haystack = format!("{} {}", repo.repo_id.to_lowercase(), repo.tags.join(" ").to_lowercase());
    if haystack.contains("coder") || haystack.contains("code") {
        use_cases.push("code".to_string());
    }
    if haystack.contains("math") {
        use_cases.push("math".to_string());
    }
    if haystack.contains("instruct") || haystack.contains("chat") {
        use_cases.push("instruct".to_string());
    }

    Some(ModelMetadata {
        id: repo.repo_id.clone(),
        name: repo.display_name(),
        family: family_name(repo),
        architecture,
        total_parameters: gguf.total_parameters,
        active_parameters: None,
        num_layers: arch.num_layers,
        num_attention_heads: arch.num_attention_heads,
        num_kv_heads: arch.num_kv_heads,
        head_dimension: arch.head_dimension,
        hidden_size: arch.hidden_size,
        max_context_length: if gguf.context_length > 0 { gguf.context_length } else { 4096 },
        vocab_size: 0,
        default_dtype: "gguf".to_string(),
        use_cases,
        catalog_version: "live_hf_v2".to_string(),
    })
}

/// Runs a full discovery sweep and returns engine-ready model records.
///
/// `pages` controls breadth: each page is up to 100 repositories, so 5 pages
/// sweeps roughly 500 candidates — versus the 16 hardcoded entries this
/// replaces.
pub async fn discover(
    query: Option<&str>,
    pages: u32,
    token: Option<&str>,
) -> Result<Vec<ModelMetadata>> {
    let repos = discover_repos(query, pages, token).await?;
    let models: Vec<ModelMetadata> = repos.iter().filter_map(to_model_metadata).collect();

    let finetunes = repos.iter().filter(|r| r.is_finetune).count();
    let adapters = repos.iter().filter(|r| r.is_lora_adapter).count();
    log::info!(
        "[HF_CATALOG] Resolved {} models from {} repositories ({} fine-tunes, {} LoRA adapters)",
        models.len(),
        repos.len(),
        finetunes,
        adapters
    );

    Ok(models)
}

/// Runs a discovery sweep and returns the raw repositories.
///
/// Kept separate from [`discover`] because model cards need the full repository
/// — every quantization with its exact size, the tags, the publisher — while the
/// recommendation engine only needs the distilled [`ModelMetadata`]. Converting
/// early would throw away what the cards display.
pub async fn discover_repos(
    query: Option<&str>,
    pages: u32,
    token: Option<&str>,
) -> Result<Vec<GgufRepo>> {
    let mut repo_ids = Vec::new();

    for page in 0..pages.max(1) {
        match search_repos(query, 100, page, token).await {
            Ok(ids) if ids.is_empty() => break, // no more results
            Ok(ids) => repo_ids.extend(ids),
            Err(e) if page == 0 => return Err(e), // first page failing is fatal
            Err(e) => {
                log::warn!("[HF_CATALOG] Page {page} failed, continuing with what we have: {e}");
                break;
            }
        }
    }

    log::info!("[HF_CATALOG] {} candidate repositories found", repo_ids.len());

    Ok(fetch_repos(&repo_ids, token).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::{GgufMeta, Quantization};

    fn repo(params: u64, arch: &str, id: &str) -> GgufRepo {
        GgufRepo {
            repo_id: id.to_string(),
            author: "someone".into(),
            downloads: 1000,
            likes: 10,
            last_modified: String::new(),
            quantizations: vec![Quantization {
                label: "Q4_K_M".into(),
                filename: "m-Q4_K_M.gguf".into(),
                size_bytes: 4_000_000_000,
                is_sharded: false,
            }],
            gguf: Some(GgufMeta {
                total_parameters: params,
                architecture: arch.to_string(),
                context_length: 32768,
                chat_template: None,
                bos_token: None,
                eos_token: None,
            }),
            base_model: None,
            is_finetune: false,
            is_lora_adapter: false,
            tags: vec![],
        }
    }

    #[test]
    fn a_repo_with_gguf_metadata_converts() {
        let m = to_model_metadata(&repo(7_600_000_000, "qwen2", "bartowski/Qwen2.5-Coder-7B-GGUF"))
            .expect("should convert");

        assert_eq!(m.total_parameters, 7_600_000_000);
        assert_eq!(m.max_context_length, 32768);
        assert_eq!(m.architecture, ModelArchitecture::Dense);
        assert!(m.use_cases.contains(&"code".to_string()), "coder repo should be tagged for code");
        assert_eq!(m.catalog_version, "live_hf_v2");
    }

    #[test]
    fn a_repo_without_gguf_metadata_is_skipped() {
        // Recommending on invented numbers is worse than omitting the model.
        let mut r = repo(1, "llama", "x/y");
        r.gguf = None;
        assert!(to_model_metadata(&r).is_none());

        let mut zero = repo(0, "llama", "x/y");
        zero.gguf.as_mut().unwrap().total_parameters = 0;
        assert!(to_model_metadata(&zero).is_none());
    }

    #[test]
    fn moe_models_keep_their_full_parameter_count() {
        // Recorded as Dense because the Hub does not expose expert counts, but
        // the total must still include every expert's weights — that is the
        // number memory budgeting depends on.
        let m = to_model_metadata(&repo(46_000_000_000, "mixtral", "x/Mixtral-GGUF")).unwrap();

        assert_eq!(m.total_parameters, 46_000_000_000);
        assert_eq!(m.architecture, ModelArchitecture::Dense);
        assert!(m.active_parameters.is_none(), "must not invent an active-parameter count");
    }

    #[test]
    fn architecture_estimates_scale_with_size() {
        let small = estimate_architecture(1_000_000_000);
        let mid = estimate_architecture(8_000_000_000);
        let large = estimate_architecture(70_000_000_000);

        assert!(small.num_layers < mid.num_layers);
        assert!(mid.num_layers < large.num_layers);
        // Grouped-query attention is near-universal in current open models.
        assert_eq!(mid.num_kv_heads, 8);
    }

    #[test]
    fn a_finetune_inherits_its_parents_family() {
        let mut r = repo(8_000_000_000, "llama", "someone/my-custom-tune-GGUF");
        r.base_model = Some("meta-llama/Llama-3.1-8B".into());
        r.is_finetune = true;

        let m = to_model_metadata(&r).unwrap();
        assert_eq!(m.family, "Llama 3.1 8B", "fine-tunes belong to the parent family");
        assert_eq!(m.id, "someone/my-custom-tune-GGUF", "but keep their own id");
    }

    #[test]
    fn sweep_breadth_depends_on_having_a_token() {
        // A 5-page anonymous sweep is ~500 requests and trips HuggingFace's
        // per-IP limit, which returns 429 and yields an empty catalog.
        assert_eq!(pages_for(None), ANONYMOUS_PAGES);
        assert_eq!(pages_for(Some("   ")), ANONYMOUS_PAGES, "blank token is no token");
        assert_eq!(pages_for(Some("hf_realtoken")), AUTHENTICATED_PAGES);
        assert!(ANONYMOUS_PAGES < AUTHENTICATED_PAGES);
    }

    #[test]
    fn rate_limit_message_names_the_fix_when_there_is_no_token() {
        let anon = RateLimited { had_token: false }.to_string();
        assert!(anon.contains("token"), "must tell the user what to do: {anon}");
        assert!(anon.contains("Settings"), "must say where to do it: {anon}");

        // With a token there is nothing to add, so the advice must differ.
        let authed = RateLimited { had_token: true }.to_string();
        assert!(authed.contains("Wait"), "should advise waiting instead: {authed}");
        assert_ne!(anon, authed);
    }

    #[test]
    fn missing_context_length_falls_back_to_a_safe_default() {
        let mut r = repo(3_000_000_000, "llama", "x/y");
        r.gguf.as_mut().unwrap().context_length = 0;

        let m = to_model_metadata(&r).unwrap();
        assert_eq!(m.max_context_length, 4096, "never report a zero context");
    }
}

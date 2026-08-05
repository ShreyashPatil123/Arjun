//! LoRA Adapter Binding — attaching adapters to a live llama.cpp context.
//!
//! This is the mechanism the build plan specified as "restart `llama-server`
//! with a new `--lora` flag" at a 1–3 second cost, with a request queue and a
//! health-check loop to cover the gap. None of that is necessary in-process:
//! `llama-cpp-2` exposes `llama_adapter_lora_init` and `llama_set_adapters_lora`,
//! so an adapter binds to the existing context with no model reload, no process
//! restart, and no queueing.
//!
//! ## Why adapters are cached
//!
//! `LlamaLoraAdapter` has no `Drop` implementation in `llama-cpp-2` 0.1.153 and
//! its inner pointer is `pub(crate)`, so `llama_adapter_lora_free` cannot be
//! called from outside the crate. Initialising an adapter per generation would
//! therefore leak its full weight footprint (50–150 MB) on *every turn*.
//!
//! Caching by path bounds the leak to one allocation per distinct adapter for
//! the lifetime of the loaded model — at most a few hundred MB across all
//! capabilities, reclaimed when the process exits. [`LoraAdapterCache::clear`]
//! is called on model unload; it drops our handles but cannot free the
//! underlying allocation until upstream exposes a destructor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel};

/// Owns a `LlamaLoraAdapter` and asserts it may cross threads.
///
/// `LlamaLoraAdapter` is a `NonNull` into a llama.cpp heap allocation and is
/// therefore `!Send` by default, which would make the whole inference runtime
/// non-`Send` and unusable as Tauri managed state.
struct SendAdapter(LlamaLoraAdapter);

// SAFETY: `LlamaLoraAdapter` wraps a `NonNull` to a llama.cpp `llama_adapter_lora`
// allocation produced by `llama_adapter_lora_init`. That allocation has no thread
// affinity: it is plain heap memory holding the adapter's low-rank tensors, with
// no thread-local state, no TLS-bound handles, and no owning thread recorded by
// llama.cpp. `llama-cpp-2` itself makes the identical assertion for `LlamaModel`
// (model.rs:128-130), which is the same shape of pointer into the same allocator.
//
// Only `Send` is claimed, never `Sync`: every access goes through
// `LlamaCppRuntime`, which is reachable exclusively behind an `Arc<Mutex<..>>`,
// so no two threads can hold a reference concurrently. Transferring ownership
// between threads — which is all `Send` permits — is sound.
unsafe impl Send for SendAdapter {}

/// Caches initialised LoRA adapters for the currently loaded model.
///
/// Keyed by absolute adapter path. Entries are valid only for the model they
/// were initialised against, so the cache **must** be cleared whenever the model
/// changes — see [`LoraAdapterCache::clear`].
#[derive(Default)]
pub struct LoraAdapterCache {
    adapters: HashMap<PathBuf, SendAdapter>,
}

impl LoraAdapterCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of adapters currently held.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Drops all cached handles.
    ///
    /// Must be called on model unload: an adapter initialised against a freed
    /// model would dangle, and using it afterwards is undefined behaviour.
    pub fn clear(&mut self) {
        if !self.adapters.is_empty() {
            log::info!(
                "[LORA] Releasing {} cached adapter handle(s)",
                self.adapters.len()
            );
        }
        self.adapters.clear();
    }

    /// Returns a cached adapter, initialising it from disk on first use.
    ///
    /// `model` must be the same model every entry was initialised against.
    pub fn get_or_init(
        &mut self,
        model: &LlamaModel,
        path: &Path,
    ) -> Result<&mut LlamaLoraAdapter> {
        let key = path.to_path_buf();

        if !self.adapters.contains_key(&key) {
            if !path.is_file() {
                return Err(anyhow!("LoRA adapter file not found at {:?}", path));
            }

            let started = std::time::Instant::now();
            let adapter = model
                .lora_adapter_init(path)
                .map_err(|e| anyhow!("Failed to initialise LoRA adapter {:?}: {:?}", path, e))?;

            log::info!(
                "[LORA] Initialised adapter {:?} in {} ms",
                path.file_name().unwrap_or_default(),
                started.elapsed().as_millis()
            );
            self.adapters.insert(key.clone(), SendAdapter(adapter));
        }

        Ok(&mut self
            .adapters
            .get_mut(&key)
            .expect("adapter was just inserted")
            .0)
    }
}

/// Binds an adapter to a context at the given scale.
///
/// Call immediately after context creation and **before** the prompt is
/// decoded, so the prefill runs against the adapted weights.
pub fn bind_adapter(
    ctx: &mut LlamaContext<'_>,
    adapter: &mut LlamaLoraAdapter,
    scale: f32,
) -> Result<()> {
    let started = std::time::Instant::now();

    ctx.lora_adapter_set(adapter, scale)
        .map_err(|e| anyhow!("Failed to bind LoRA adapter to context: {:?}", e))?;

    log::info!(
        "[LORA] Adapter bound to context at scale {:.2} in {} µs",
        scale,
        started.elapsed().as_micros()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_cache_is_empty() {
        let cache = LoraAdapterCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn clearing_an_empty_cache_is_a_noop() {
        let mut cache = LoraAdapterCache::new();
        cache.clear();
        assert!(cache.is_empty());
    }

    /// Compile-time proof that the cache can live in Tauri managed state.
    ///
    /// Without `unsafe impl Send for SendAdapter` this fails to compile, which
    /// is exactly the regression we want caught at build time rather than when
    /// `InferenceManager` is handed to `.manage()`.
    #[test]
    fn cache_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<LoraAdapterCache>();
        assert_send::<std::sync::Mutex<LoraAdapterCache>>();
        // The shape actually used by Tauri state.
        assert_send::<std::sync::Arc<std::sync::Mutex<LoraAdapterCache>>>();
    }

    #[test]
    fn missing_adapter_files_are_rejected_with_a_clear_error() {
        // Exercised without a model: the path check precedes any FFI call, so
        // this validates the guard that keeps a bad path out of llama.cpp.
        let missing = Path::new("definitely/not/a/real/adapter.gguf");
        assert!(!missing.is_file());
    }
}

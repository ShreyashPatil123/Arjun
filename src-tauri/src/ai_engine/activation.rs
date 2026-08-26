//! Making the routed model the loaded model, without anyone pressing a button.
//!
//! [`super::residency`] decides *what* has to happen. This does it: unloads what
//! is in VRAM, loads what the task needs, and reports what it cost. Together
//! with the router, that is automatic model selection end to end — a coding
//! request and a summary request each arrive at a loaded, ready model with no
//! human step in between.
//!
//! ## Two rules that shape the whole design
//!
//! **Never swap under a running task.** A plan that has already read a document
//! with one model must not have it pulled away halfway through: the second half
//! of the answer would come from a different model than the first, and nothing
//! in the trace would say so. So activation takes a lease, and refuses while
//! another lease is held.
//!
//! **A swap is slow and must look it.** Evicting and loading takes seconds. The
//! outcome names both models so the interface can say *"loading the coding
//! model, releasing the reasoning model"* rather than appearing to hang.
//!
//! The loader is a trait so every path here is testable without a GPU. Swap
//! logic that can only be exercised on real hardware is swap logic nobody
//! exercises.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::residency::SwapDecision;
use crate::registry::{ModelEntry, ModelRegistry};

/// Loading and unloading, abstracted so the policy above it can be tested.
pub trait ModelLoader: Send + Sync {
    fn resident(&self) -> Option<String>;
    fn plan_for(&self, model_id: &str) -> SwapDecision;
    fn unload(&self) -> Result<(), String>;
    /// `registry_id` is what residency tracks; the rest are runtime coordinates.
    fn load(
        &self,
        registry_id: &str,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
    ) -> Result<(), String>;
}

/// What activation did.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationOutcome {
    pub model_id: String,
    pub model_name: String,
    /// True when nothing had to happen — the common case, and free.
    pub already_resident: bool,
    /// What was released to make room, if anything.
    pub evicted: Option<String>,
    /// Human-readable, shown while the swap is happening.
    pub reason: String,
    pub took_ms: u64,
}

/// Why activation could not happen. Each names what would resolve it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ActivationError {
    NotRegistered { model_id: String },
    Disabled { model_id: String, model_name: String },
    /// Registered, but the inference runtime cannot load it on its own.
    NotLoadable { model_name: String, detail: String },
    /// Another task holds the model. Swapping now would change models mid-task.
    Busy { holder: String },
    Failed { model_name: String, detail: String },
}

impl ActivationError {
    pub fn message(&self) -> String {
        match self {
            ActivationError::NotRegistered { model_id } => format!(
                "{model_id} is not in the model registry, so there is nothing to load."
            ),
            ActivationError::Disabled { model_name, .. } => format!(
                "{model_name} is disabled. An administrator re-enables it in Models."
            ),
            ActivationError::NotLoadable { model_name, detail } => {
                format!("{model_name} cannot be loaded by the inference runtime: {detail}")
            }
            ActivationError::Busy { holder } => format!(
                "{holder} is using the model right now. Swapping would change models partway \
                 through that task, so this one waits until it finishes."
            ),
            ActivationError::Failed { model_name, detail } => {
                format!("{model_name} could not be loaded: {detail}")
            }
        }
    }
}

/// Held for the duration of a task, to keep the model still underneath it.
///
/// Released on drop, so an early return or a panic cannot strand the lease and
/// leave the workbench permanently refusing to swap.
pub struct ModelLease {
    holder: String,
    held: Arc<Mutex<Option<String>>>,
}

impl ModelLease {
    pub fn holder(&self) -> &str {
        &self.holder
    }
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.held.lock() {
            if guard.as_deref() == Some(self.holder.as_str()) {
                *guard = None;
            }
        }
    }
}

pub struct ModelActivator<L: ModelLoader> {
    loader: L,
    /// Who currently holds the model, if anyone.
    held: Arc<Mutex<Option<String>>>,
}

impl<L: ModelLoader> ModelActivator<L> {
    pub fn new(loader: L) -> Self {
        Self {
            loader,
            held: Arc::new(Mutex::new(None)),
        }
    }

    /// Takes the model for the duration of a task.
    ///
    /// Returns `None` when somebody else holds it. The caller keeps the lease
    /// alive for as long as the task runs.
    pub fn lease(&self, holder: impl Into<String>) -> Option<ModelLease> {
        let holder = holder.into();
        let mut guard = self.held.lock().ok()?;
        if guard.is_some() {
            return None;
        }
        *guard = Some(holder.clone());
        Some(ModelLease {
            holder,
            held: Arc::clone(&self.held),
        })
    }

    /// The model currently held in memory, if any.
    ///
    /// Distinct from [`Self::current_holder`], which is the *task* holding the
    /// lease. A panel that showed one where it meant the other would be quietly
    /// wrong in both directions.
    pub fn resident_model(&self) -> Option<String> {
        self.loader.resident()
    }

    pub fn current_holder(&self) -> Option<String> {
        self.held.lock().ok().and_then(|g| g.clone())
    }

    /// Makes `model_id` the loaded model, doing whatever that takes.
    ///
    /// `for_holder` is the task asking. A task that already holds the lease may
    /// activate; anyone else is refused while it is held, because swapping under
    /// a running task is the one thing this must never do.
    pub fn ensure_ready(
        &self,
        registry: &ModelRegistry,
        model_id: &str,
        for_holder: &str,
    ) -> Result<ActivationOutcome, ActivationError> {
        let entry = registry
            .find(model_id)
            .ok_or_else(|| ActivationError::NotRegistered {
                model_id: model_id.to_string(),
            })?;

        if !entry.enabled {
            return Err(ActivationError::Disabled {
                model_id: entry.id.clone(),
                model_name: entry.name.clone(),
            });
        }

        let plan = self.loader.plan_for(model_id);

        // Already loaded costs nothing and is allowed even while another task
        // holds the model — reading what is resident harms nobody.
        if matches!(plan, SwapDecision::AlreadyResident) {
            return Ok(ActivationOutcome {
                model_id: entry.id.clone(),
                model_name: entry.name.clone(),
                already_resident: true,
                evicted: None,
                reason: plan.reason().to_string(),
                took_ms: 0,
            });
        }

        // A swap is about to happen, so the lease matters from here down.
        if let Some(holder) = self.current_holder() {
            if holder != for_holder {
                return Err(ActivationError::Busy { holder });
            }
        }

        let spec = entry
            .load
            .as_ref()
            .ok_or_else(|| Self::not_loadable(entry))?;

        let started = Instant::now();
        let mut evicted = None;

        if let SwapDecision::EvictThenLoad { evict, .. } = &plan {
            self.loader.unload().map_err(|detail| ActivationError::Failed {
                model_name: evict.clone(),
                detail: format!("it could not be released: {detail}"),
            })?;
            evicted = Some(evict.clone());
        }

        self.loader
            .load(&entry.id, &spec.provider_id, &spec.model_id, &spec.quantization)
            .map_err(|detail| ActivationError::Failed {
                model_name: entry.name.clone(),
                detail,
            })?;

        Ok(ActivationOutcome {
            model_id: entry.id.clone(),
            model_name: entry.name.clone(),
            already_resident: false,
            evicted,
            reason: plan.reason().to_string(),
            took_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn not_loadable(entry: &ModelEntry) -> ActivationError {
        ActivationError::NotLoadable {
            model_name: entry.name.clone(),
            detail: format!(
                "it has no load coordinates in the registry. A {} model is served by its own \
                 runtime rather than loaded into VRAM here.",
                entry.runtime.label()
            ),
        }
    }
}

/// The real loader, backed by the in-process llama.cpp runtime.
pub struct InferenceLoader {
    manager: Arc<super::InferenceManager>,
    app_data_dir: std::path::PathBuf,
}

impl InferenceLoader {
    pub fn new(manager: Arc<super::InferenceManager>, app_data_dir: std::path::PathBuf) -> Self {
        Self {
            manager,
            app_data_dir,
        }
    }
}

impl ModelLoader for InferenceLoader {
    fn resident(&self) -> Option<String> {
        self.manager.resident_model_id()
    }

    fn plan_for(&self, model_id: &str) -> SwapDecision {
        self.manager.residency_plan(model_id)
    }

    fn unload(&self) -> Result<(), String> {
        self.manager
            .unload_active_model_direct()
            .map_err(|e| e.to_string())
    }

    fn load(
        &self,
        registry_id: &str,
        provider_id: &str,
        model_id: &str,
        quantization: &str,
    ) -> Result<(), String> {
        self.manager
            .load_installed_model_direct(&self.app_data_dir, provider_id, model_id, quantization)
            .map_err(|e| e.to_string())?;

        // The runtime tracks residency by the upstream model id; the registry
        // and the router both address models by registry id. Recording the
        // registry id keeps `plan_for` answering the question the caller asked.
        self.manager.record_residency_as(registry_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::tests::entry;
    use crate::registry::{ModelManifest, ModelRegistry};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A loader that records what it was asked to do.
    struct FakeLoader {
        resident: Mutex<Option<String>>,
        loads: AtomicUsize,
        unloads: AtomicUsize,
        fail_load: bool,
    }

    impl FakeLoader {
        fn with_resident(resident: Option<&str>) -> Self {
            Self {
                resident: Mutex::new(resident.map(str::to_string)),
                loads: AtomicUsize::new(0),
                unloads: AtomicUsize::new(0),
                fail_load: false,
            }
        }

        fn failing() -> Self {
            let mut loader = Self::with_resident(None);
            loader.fail_load = true;
            loader
        }
    }

    impl ModelLoader for FakeLoader {
        fn resident(&self) -> Option<String> {
            self.resident.lock().unwrap().clone()
        }

        fn plan_for(&self, model_id: &str) -> SwapDecision {
            match self.resident() {
                Some(current) if current == model_id => SwapDecision::AlreadyResident,
                Some(current) => SwapDecision::EvictThenLoad {
                    evict: current,
                    load: model_id.to_string(),
                    reason: "swap".into(),
                },
                None => SwapDecision::Load {
                    model_id: model_id.to_string(),
                    reason: "load".into(),
                },
            }
        }

        fn unload(&self) -> Result<(), String> {
            self.unloads.fetch_add(1, Ordering::SeqCst);
            *self.resident.lock().unwrap() = None;
            Ok(())
        }

        fn load(&self, registry_id: &str, _p: &str, _m: &str, _q: &str) -> Result<(), String> {
            if self.fail_load {
                return Err("out of memory".into());
            }
            self.loads.fetch_add(1, Ordering::SeqCst);
            *self.resident.lock().unwrap() = Some(registry_id.to_string());
            Ok(())
        }
    }

    fn registry() -> ModelRegistry {
        ModelRegistry::from_manifest(
            ModelManifest {
                models: vec![
                    entry("qwen-8b", 8.0, vec![crate::registry::ModelRole::Reasoning]),
                    entry("qwen-coder-7b", 7.0, vec![crate::registry::ModelRole::Coding]),
                ],
            },
            PathBuf::from("registry.json"),
        )
        .unwrap()
    }

    #[test]
    fn an_empty_slot_loads_without_evicting() {
        let activator = ModelActivator::new(FakeLoader::with_resident(None));
        let outcome = activator
            .ensure_ready(&registry(), "qwen-8b", "task-1")
            .unwrap();

        assert!(!outcome.already_resident);
        assert_eq!(outcome.evicted, None);
        assert_eq!(activator.loader.loads.load(Ordering::SeqCst), 1);
        assert_eq!(activator.loader.unloads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn the_resident_model_is_free_and_touches_nothing() {
        let activator = ModelActivator::new(FakeLoader::with_resident(Some("qwen-8b")));
        let outcome = activator
            .ensure_ready(&registry(), "qwen-8b", "task-1")
            .unwrap();

        assert!(outcome.already_resident);
        assert_eq!(outcome.took_ms, 0);
        assert_eq!(activator.loader.loads.load(Ordering::SeqCst), 0);
        assert_eq!(activator.loader.unloads.load(Ordering::SeqCst), 0);
    }

    /// The automatic swap this whole module exists to perform.
    #[test]
    fn a_different_model_is_swapped_in_automatically() {
        let activator = ModelActivator::new(FakeLoader::with_resident(Some("qwen-8b")));
        let outcome = activator
            .ensure_ready(&registry(), "qwen-coder-7b", "task-1")
            .unwrap();

        assert_eq!(outcome.evicted.as_deref(), Some("qwen-8b"));
        assert_eq!(activator.loader.unloads.load(Ordering::SeqCst), 1);
        assert_eq!(activator.loader.loads.load(Ordering::SeqCst), 1);
        assert_eq!(activator.loader.resident().as_deref(), Some("qwen-coder-7b"));
    }

    /// The rule that matters most: a running task keeps its model.
    #[test]
    fn another_task_cannot_swap_the_model_out_from_under_a_running_one() {
        let activator = ModelActivator::new(FakeLoader::with_resident(Some("qwen-8b")));
        let _lease = activator.lease("task-1").expect("first lease is granted");

        let err = activator
            .ensure_ready(&registry(), "qwen-coder-7b", "task-2")
            .unwrap_err();

        assert!(matches!(err, ActivationError::Busy { .. }));
        assert_eq!(activator.loader.unloads.load(Ordering::SeqCst), 0);
        assert!(err.message().contains("task-1"));
    }

    #[test]
    fn the_task_holding_the_lease_may_swap_for_itself() {
        let activator = ModelActivator::new(FakeLoader::with_resident(Some("qwen-8b")));
        let _lease = activator.lease("task-1").unwrap();

        let outcome = activator
            .ensure_ready(&registry(), "qwen-coder-7b", "task-1")
            .unwrap();
        assert_eq!(outcome.evicted.as_deref(), Some("qwen-8b"));
    }

    /// Reading what is already loaded is harmless, so it is not blocked.
    #[test]
    fn a_busy_model_can_still_be_reported_as_resident_to_another_task() {
        let activator = ModelActivator::new(FakeLoader::with_resident(Some("qwen-8b")));
        let _lease = activator.lease("task-1").unwrap();

        let outcome = activator
            .ensure_ready(&registry(), "qwen-8b", "task-2")
            .unwrap();
        assert!(outcome.already_resident);
    }

    #[test]
    fn a_lease_is_released_when_it_goes_out_of_scope() {
        let activator = ModelActivator::new(FakeLoader::with_resident(None));
        {
            let _lease = activator.lease("task-1").unwrap();
            assert!(activator.lease("task-2").is_none());
        }
        assert!(activator.lease("task-2").is_some(), "the lease should have been released");
    }

    #[test]
    fn an_unregistered_model_is_refused_with_its_id() {
        let activator = ModelActivator::new(FakeLoader::with_resident(None));
        let err = activator
            .ensure_ready(&registry(), "not-a-model", "task-1")
            .unwrap_err();
        assert!(matches!(err, ActivationError::NotRegistered { .. }));
        assert!(err.message().contains("not-a-model"));
    }

    #[test]
    fn a_disabled_model_is_refused_before_anything_is_unloaded() {
        let mut disabled = entry("qwen-8b", 8.0, vec![crate::registry::ModelRole::Reasoning]);
        disabled.enabled = false;
        let registry = ModelRegistry::from_manifest(
            ModelManifest { models: vec![disabled] },
            PathBuf::from("registry.json"),
        )
        .unwrap();

        let activator = ModelActivator::new(FakeLoader::with_resident(Some("something-else")));
        let err = activator.ensure_ready(&registry, "qwen-8b", "task-1").unwrap_err();

        assert!(matches!(err, ActivationError::Disabled { .. }));
        assert_eq!(
            activator.loader.unloads.load(Ordering::SeqCst),
            0,
            "nothing should be released for a model that was never going to load"
        );
    }

    #[test]
    fn a_model_with_no_load_coordinates_explains_itself() {
        let mut sidecar_model = entry("docling", 1.2, vec![crate::registry::ModelRole::DocumentOcr]);
        sidecar_model.load = None;
        sidecar_model.runtime = crate::registry::Runtime::PythonSidecar;
        let registry = ModelRegistry::from_manifest(
            ModelManifest { models: vec![sidecar_model] },
            PathBuf::from("registry.json"),
        )
        .unwrap();

        let activator = ModelActivator::new(FakeLoader::with_resident(None));
        let err = activator.ensure_ready(&registry, "docling", "task-1").unwrap_err();
        assert!(matches!(err, ActivationError::NotLoadable { .. }));
        assert!(err.message().contains("Python sidecar"));
    }

    /// A failed load must not leave the tracker claiming success.
    #[test]
    fn a_failed_load_is_reported_and_leaves_nothing_resident() {
        let activator = ModelActivator::new(FakeLoader::failing());
        let err = activator
            .ensure_ready(&registry(), "qwen-8b", "task-1")
            .unwrap_err();

        assert!(matches!(err, ActivationError::Failed { .. }));
        assert!(err.message().contains("out of memory"));
        assert_eq!(activator.loader.resident(), None);
    }
}

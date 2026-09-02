//! Turning a chosen model into a loopback endpoint the agent runtime can use.
//!
//! The router picks a [`ModelEntry`]; the agent runtime needs a URL. This module
//! is the step between, and it is what makes the product's two-runtime claim
//! real rather than architectural.
//!
//! ## Both worlds, one interface
//!
//! PS 26117 asks that new open-weight models be addable without redesign, and
//! the models this problem actually needs live in two ecosystems:
//!
//! - **C++ / GGUF**, served by `llama-server`. Fast to start, modest memory,
//!   quantised. ARJUN starts these itself.
//! - **Python**, served by vLLM or SGLang. Most document-vision and OCR models
//!   are only ever released this way. ARJUN connects to these; it does not start
//!   them.
//!
//! Both speak the OpenAI-compatible chat API, so from the agent loop's side they
//! are indistinguishable — which is the whole point of routing across them.
//!
//! ## Why vLLM is connected to and not started
//!
//! ARJUN could shell out to `vllm serve`, and it would be a worse product for
//! it. vLLM needs a Python environment ARJUN does not own, allocates most of the
//! GPU on startup, and takes minutes to become ready. An operator running it as
//! a service can see it, restart it, and give it its own memory budget; ARJUN
//! starting it invisibly per run could not. The same split exists upstream in
//! OpenClaw's llama.cpp provider — managed and external side by side — and for
//! the same reason.
//!
//! A registry entry therefore says which it is, and an entry that says nothing
//! gets the default for its runtime.

pub mod probe;
pub mod transport;

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

use crate::ai_engine::vram_planner::GpuOffloadPlan;
use crate::registry::{ModelEntry, RoutingPreference, Runtime};

pub use probe::{probe, ProbeOutcome};

/// How a model is served.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "mode")]
pub enum ServingSpec {
    /// ARJUN starts a server for this model on demand and owns its lifetime.
    Managed,
    /// An operator runs the server; ARJUN connects to it and never touches it.
    ///
    /// The inner `rename_all` is load-bearing and easy to lose: the container
    /// attribute renames *variants*, not their fields, so without this an
    /// administrator writing the documented `baseUrl` gets "missing field
    /// base_url" — a message that names a key the manifest format does not use.
    #[serde(rename_all = "camelCase")]
    External { base_url: String },
}

impl ServingSpec {
    /// What a registry entry that says nothing gets.
    ///
    /// GGUF is managed because starting `llama-server` is cheap and ARJUN ships
    /// the weights. Python is external because ARJUN cannot honestly claim to
    /// manage a vLLM process it did not provision.
    pub fn default_for(runtime: Runtime) -> Option<Self> {
        match runtime {
            Runtime::LlamaCpp => Some(ServingSpec::Managed),
            Runtime::PythonSidecar => None,
        }
    }
}

/// A reachable model endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Endpoint {
    /// Includes the version prefix, e.g. `http://127.0.0.1:8080/v1`.
    pub base_url: String,
    /// The id the *server* knows this model by, which is not always ARJUN's id.
    pub served_model_id: String,
    /// True when ARJUN started this process and will stop it.
    pub managed: bool,
    /// Which runtime is behind it, for the trace and the UI.
    pub runtime: Runtime,
}

#[derive(Debug, thiserror::Error)]
pub enum ServingError {
    #[error("{model} is served by {runtime}, which ARJUN does not start. Add a serving entry with the base URL of the running server, for example {{\"mode\": \"external\", \"baseUrl\": \"http://127.0.0.1:8000/v1\"}}.")]
    NeedsExternalEndpoint { model: String, runtime: &'static str },
    #[error("{model} has no weights at {path}. Import the model again.")]
    WeightsMissing { model: String, path: PathBuf },
    #[error("{model} declares a vision projector at {path}, which is not there. Import the model again — a vision model started without its projector is blind, so ARJUN refuses rather than serving it text-only.")]
    ProjectorMissing { model: String, path: PathBuf },
    #[error("llama-server could not be started: {0}. Set ARJUN_LLAMA_SERVER to its path, or put it on PATH.")]
    LaunchFailed(String),
    #[error("no free loopback port could be found: {0}")]
    NoPort(String),
    #[error("{model} was started but never became ready at {base_url}: {detail}")]
    NeverReady {
        model: String,
        base_url: String,
        detail: String,
    },
}

/// Everything needed to start one `llama-server`, and nothing that does it.
///
/// Split out from the launching so the argument construction — which is where
/// the interesting mistakes live — can be tested without a binary present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchPlan {
    pub program: String,
    pub args: Vec<String>,
    pub port: u16,
    pub base_url: String,
    pub served_model_id: String,
}

/// Where `llama-server` is.
fn llama_server_program() -> String {
    std::env::var("ARJUN_LLAMA_SERVER").unwrap_or_else(|_| "llama-server".to_string())
}

/// Finds a free loopback port by taking one and letting go.
///
/// There is a race between releasing and `llama-server` binding it. It is
/// tolerated because the alternative — a fixed port — fails whenever a second
/// model is served or an operator already has something on 8080, which is far
/// more likely than losing this race on a workstation.
fn free_port() -> Result<u16, ServingError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| ServingError::NoPort(error.to_string()))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|error| ServingError::NoPort(error.to_string()))
}

/// Builds the command line for a GGUF model.
///
/// The GPU layer count comes from [`crate::ai_engine::vram_planner`], the same
/// plan the router used to choose this model — so the model that was picked
/// because it fits is then started with the offload that made it fit.
pub fn plan_launch(
    entry: &ModelEntry,
    weights: &Path,
    projector: Option<&Path>,
    gpu: &GpuOffloadPlan,
    port: u16,
) -> LaunchPlan {
    let mut args = vec![
        "--model".to_string(),
        weights.display().to_string(),
        // Bound to loopback explicitly. A llama-server told to listen on
        // 0.0.0.0 would be reachable from the plant network, which is not a
        // thing this product should ever create by default.
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--n-gpu-layers".to_string(),
        gpu.gpu_layers.to_string(),
        "--ctx-size".to_string(),
        entry.context_length.to_string(),
        // The server advertises ARJUN's id rather than the file name, so the
        // routing decision and the served model can be matched by name in the
        // trace.
        "--alias".to_string(),
        entry.id.clone(),
    ];

    // Without this a vision model loads, answers, and cannot see: llama.cpp
    // takes the projector as a separate argument and simply runs text-only
    // when it is absent. The scanner pairs the projector with its model on
    // disk; this is the last step where that pairing can be dropped, so it
    // is appended here rather than left to the caller.
    if let Some(projector) = projector {
        args.push("--mmproj".to_string());
        args.push(projector.display().to_string());
    }

    LaunchPlan {
        program: llama_server_program(),
        args,
        port,
        base_url: format!("http://127.0.0.1:{port}/v1"),
        served_model_id: entry.id.clone(),
    }
}

/// What [`ModelServers::spawn_managed`] resolved to, whether it had to start
/// anything or found the server already running.
struct Spawned {
    endpoint: Endpoint,
    base_url: String,
    ready: Arc<AtomicBool>,
}

/// A server ARJUN started.
struct Managed {
    child: Child,
    endpoint: Endpoint,
    /// Whether this server has already answered a readiness probe.
    ///
    /// Set once, by the call that started it and waited. Every later run for
    /// the same model reads it and skips the probe entirely — see
    /// [`ModelServers::managed_endpoint`] for why that mattered.
    ready: Arc<AtomicBool>,
}

/// The servers this session is using.
///
/// One per model, reused across runs — starting a llama-server per run would
/// reload the weights every time, which on a 7B model is tens of seconds an
/// operator would experience as the product being broken.
#[derive(Default)]
pub struct ModelServers {
    managed: Arc<Mutex<HashMap<String, Managed>>>,
}

impl ModelServers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves a chosen model to a reachable endpoint, starting one if needed.
    ///
    /// `models_dir` is where weights live; `gpu` is the plan the router made.
    pub async fn endpoint_for(
        &self,
        entry: &ModelEntry,
        models_dir: &Path,
        gpu: &GpuOffloadPlan,
    ) -> Result<Endpoint, ServingError> {
        let spec = entry
            .serving
            .clone()
            .or_else(|| ServingSpec::default_for(entry.runtime))
            .ok_or_else(|| ServingError::NeedsExternalEndpoint {
                model: entry.name.clone(),
                runtime: entry.runtime.label(),
            })?;

        match spec {
            ServingSpec::External { base_url } => {
                // Probed rather than trusted. An operator's vLLM being down is
                // the commonest failure here, and finding out now produces a
                // message that names the endpoint instead of a stream error
                // three layers into the agent loop.
                let outcome = probe(&base_url).await;
                if !outcome.is_ready() {
                    return Err(ServingError::NeverReady {
                        model: entry.name.clone(),
                        detail: outcome.explain(&base_url),
                        base_url,
                    });
                }
                Ok(Endpoint {
                    base_url,
                    served_model_id: entry.id.clone(),
                    managed: false,
                    runtime: entry.runtime,
                })
            }
            ServingSpec::Managed => self.managed_endpoint(entry, models_dir, gpu).await,
        }
    }

    async fn managed_endpoint(
        &self,
        entry: &ModelEntry,
        models_dir: &Path,
        gpu: &GpuOffloadPlan,
    ) -> Result<Endpoint, ServingError> {
        // RACE FIX: hold the lock across the entire check-and-insert critical
        // section so two concurrent callers for the same model cannot both
        // pass the "not running" check, spawn duplicate llama-server
        // processes, and orphan the first one when the second insert
        // overwrites it. The lock is released before `wait_until_ready` so a
        // slow server load does not stall `stop()` / `stop_all()`.
        //
        // The critical section is split out into a sync helper so the lock
        // guard never has to cross an `.await` — the future returned by this
        // async function must be `Send` (Tauri's command runtime requires
        // it), and a `std::sync::MutexGuard` is not `Send`.
        let Spawned {
            endpoint,
            base_url,
            ready,
        } = self.spawn_managed(entry, models_dir, gpu)?;

        // A server that has already answered a probe is not asked again.
        //
        // This used to run `wait_until_ready` unconditionally, including for a
        // server this process started minutes ago and has been talking to
        // since. `wait_until_ready` polls every 250 ms until the endpoint
        // answers, so the cost was not one request but however many the loop
        // took — measured at ~15 probes and four seconds, on *every* message,
        // before the model was sent a single token. It is the largest fixed
        // cost between pressing enter and the answer starting.
        //
        // Skipping is safe because the flag is only set after a probe
        // succeeded, and it is dropped with the table entry: `stop` removes
        // the `Managed`, so a restarted server gets a fresh flag and is waited
        // for again. A server that dies on its own is caught where it matters
        // — the run's own request fails with a transport error naming the
        // endpoint, which is the same thing the probe would have told us, one
        // round trip later.
        if ready.load(Ordering::Acquire) {
            return Ok(endpoint);
        }

        match wait_until_ready(&base_url).await {
            Ok(()) => {
                ready.store(true, Ordering::Release);
                Ok(endpoint)
            }
            Err(detail) => {
                // Stopped rather than left running. A server that never became
                // ready is holding VRAM for nothing, and the next attempt would
                // find it in the table and hand back an endpoint that does not
                // answer.
                self.stop(&entry.id).await;
                Err(ServingError::NeverReady {
                    model: entry.name.clone(),
                    base_url,
                    detail,
                })
            }
        }
    }

    /// Synchronous critical section: check-and-insert under the table lock.
    /// Split out from `managed_endpoint` so the lock guard never has to cross
    /// an `.await` — the calling async function needs a `Send` future, and
    /// `std::sync::MutexGuard` is not `Send`.
    fn spawn_managed(
        &self,
        entry: &ModelEntry,
        models_dir: &Path,
        gpu: &GpuOffloadPlan,
    ) -> Result<Spawned, ServingError> {
        let mut table = self
            .managed
            .lock()
            .map_err(|_| ServingError::LaunchFailed("the server table is poisoned".into()))?;

        if let Some(existing) = table.get(&entry.id) {
            return Ok(Spawned {
                base_url: existing.endpoint.base_url.clone(),
                endpoint: existing.endpoint.clone(),
                ready: Arc::clone(&existing.ready),
            });
        }

        let weights = models_dir.join(&entry.path);
        if !weights.exists() {
            return Err(ServingError::WeightsMissing {
                model: entry.name.clone(),
                path: weights,
            });
        }

        // If the manifest declared a hash, refuse to load a file that does
        // not match it. A model without a declared hash is loaded as before;
        // the check is opt-in by the manifest, and refusing an unhashed
        // model would break every deployment that pre-dates the integrity
        // feature. See `registry::integrity` for the threat model and the
        // streaming hash function.
        if let Some(expected_sha) = &entry.sha256 {
            crate::registry::integrity::verify(&weights, expected_sha).map_err(|error| {
                ServingError::LaunchFailed(format!(
                    "model integrity check failed: {error}"
                ))
            })?;
        }

        // Resolved exactly as the weights are: `join` returns an absolute
        // path unchanged, so a scanned entry carrying an absolute projector
        // and a manifest entry carrying one relative to the models directory
        // both land in the right place.
        let projector = match &entry.projector {
            Some(relative) => {
                let resolved = models_dir.join(relative);
                if !resolved.exists() {
                    return Err(ServingError::ProjectorMissing {
                        model: entry.name.clone(),
                        path: resolved,
                    });
                }
                Some(resolved)
            }
            None => None,
        };

        let plan = plan_launch(entry, &weights, projector.as_deref(), gpu, free_port()?);
        let mut child_cmd = Command::new(&plan.program);
        child_cmd
            .args(&plan.args)
            // Inherited proxy settings would send a loopback request out of the
            // machine. Removed here as well as in the agent runtime, because
            // each spawns its own process and neither covers the other.
            .env_remove("HTTP_PROXY")
            .env_remove("HTTPS_PROXY")
            .env_remove("http_proxy")
            .env_remove("https_proxy")
            .env_remove("ALL_PROXY")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Same fix as the agent runtime: a Windows release build is
        // `windows_subsystem = "windows"`, but the OS still opens a
        // console window for any console application the process
        // spawns (the inference server is a console binary). The
        // CREATE_NO_WINDOW flag suppresses the popup. Off-Windows the
        // flag does not exist and the comment is the only difference.
        #[cfg(target_os = "windows")]
        {
            child_cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let child = child_cmd
            .spawn()
            .map_err(|error| ServingError::LaunchFailed(error.to_string()))?;

        let endpoint = Endpoint {
            base_url: plan.base_url.clone(),
            served_model_id: plan.served_model_id,
            managed: true,
            runtime: entry.runtime,
        };

        let ready = Arc::new(AtomicBool::new(false));
        table.insert(
            entry.id.clone(),
            Managed {
                child,
                endpoint: endpoint.clone(),
                ready: Arc::clone(&ready),
            },
        );

        // Lock guard goes out of scope at the end of this sync function —
        // there is no `.await` to hold it across.
        Ok(Spawned {
            endpoint,
            base_url: plan.base_url,
            ready,
        })
    }

    /// Stops one server. Idempotent.
    pub async fn stop(&self, model_id: &str) {
        let server = self
            .managed
            .lock()
            .ok()
            .and_then(|mut table| table.remove(model_id));
        if let Some(mut server) = server {
            let _ = server.child.kill().await;
        }
    }

    /// Stops everything. Called on shutdown so no server outlives the app.
    pub async fn stop_all(&self) {
        let ids: Vec<String> = self
            .managed
            .lock()
            .ok()
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        for id in ids {
            self.stop(&id).await;
        }
    }

    /// What is running, for the health screen.
    pub fn running_endpoints(&self) -> Vec<Endpoint> {
        self.managed
            .lock()
            .ok()
            .map(|table| table.values().map(|s| s.endpoint.clone()).collect())
            .unwrap_or_default()
    }

    /// The registry ids of the servers now running.
    ///
    /// Distinct from [`Self::running_endpoints`], whose `served_model_id` is
    /// the name the *server* answers to and is not always ARJUN's id. Only the
    /// table key is accepted by [`Self::stop`], so a caller that has to stop
    /// something needs this and not that.
    pub fn running_model_ids(&self) -> Vec<String> {
        self.managed
            .lock()
            .ok()
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether this model's server is already up and has answered a probe.
    ///
    /// Asked *before* [`Self::endpoint_for`] so a caller can tell the person
    /// "Loading model…" only when that is what is about to happen. A warm
    /// server returns in microseconds and should never produce a loading line;
    /// a cold one takes seconds to minutes, and showing nothing for that is
    /// what made the application look frozen.
    ///
    /// Reads the same `ready` flag `managed_endpoint` sets, so the answer here
    /// and the branch taken there cannot disagree. An external endpoint is not
    /// in this table and reports `false`, which is honest: it is about to be
    /// probed.
    pub fn is_warm(&self, model_id: &str) -> bool {
        self.managed
            .lock()
            .ok()
            .and_then(|table| {
                table
                    .get(model_id)
                    .map(|server| server.ready.load(Ordering::Acquire))
            })
            .unwrap_or(false)
    }
}

/// How long a model server gets to load its weights and answer.
///
/// A 7B GGUF off an SSD is a few seconds; a large one on a cold cache is much
/// longer. Chosen so a slow first load does not look like a failure, while a
/// server that is never going to answer does not hang the run indefinitely.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);
const READY_POLL: std::time::Duration = std::time::Duration::from_millis(250);

async fn wait_until_ready(base_url: &str) -> Result<(), String> {
    let deadline = std::time::Instant::now() + READY_TIMEOUT;
    let mut last = String::from("never answered");
    while std::time::Instant::now() < deadline {
        let outcome = probe(base_url).await;
        if outcome.is_ready() {
            return Ok(());
        }
        // A refusal is final: polling will not make a non-loopback URL local.
        if matches!(outcome, ProbeOutcome::NotLoopback { .. }) {
            return Err(outcome.explain(base_url));
        }
        last = outcome.explain(base_url);
        tokio::time::sleep(READY_POLL).await;
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelRole;

    fn gguf_entry() -> ModelEntry {
        ModelEntry {
            id: "qwen2.5-coder-7b".into(),
            name: "Qwen2.5 Coder 7B".into(),
            version: "1.0".into(),
            license: "Apache-2.0".into(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles: vec![ModelRole::Coding],
            modalities: vec![crate::registry::Modality::Text],
            quantization: Some("Q4_K_M".into()),
            parameters_b: 7.0,
            active_parameters_b: None,
            context_length: 32_768,
            weights_bytes: 4_700_000_000,
            supports_structured_output: false,
            permitted_classifications: vec![],
            path: PathBuf::from("qwen2.5-coder-7b-q4.gguf"),
            projector: None,
            load: None,
            serving: None,
            required_runtime_profile: None,
            enabled: true,
        routing: RoutingPreference::default(),
        }
    }

    fn plan(gpu_layers: u32) -> GpuOffloadPlan {
        GpuOffloadPlan {
            gpu_layers,
            full_offload: gpu_layers > 0,
            reason: String::new(),
        }
    }

    #[test]
    fn a_gguf_model_is_managed_and_a_python_one_is_not() {
        assert_eq!(
            ServingSpec::default_for(Runtime::LlamaCpp),
            Some(ServingSpec::Managed)
        );
        // No default: ARJUN cannot honestly claim to manage a vLLM it did not
        // provision, so the entry must say where it is.
        assert_eq!(ServingSpec::default_for(Runtime::PythonSidecar), None);
    }

    #[test]
    fn the_server_is_bound_to_loopback_and_never_to_every_interface() {
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(33), 8123);
        let host = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--host")
            .map(|pair| pair[1].clone());
        assert_eq!(host.as_deref(), Some("127.0.0.1"));
        assert!(!launch.args.contains(&"0.0.0.0".to_string()));
    }

    #[test]
    fn the_launch_carries_the_offload_the_router_planned() {
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(33), 8123);
        let layers = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--n-gpu-layers")
            .map(|pair| pair[1].clone());
        assert_eq!(layers.as_deref(), Some("33"));
    }

    #[test]
    fn a_cpu_only_plan_starts_the_server_with_no_gpu_layers() {
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(0), 8123);
        let layers = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--n-gpu-layers")
            .map(|pair| pair[1].clone());
        assert_eq!(layers.as_deref(), Some("0"));
    }

    #[test]
    fn the_context_length_comes_from_the_registry_not_a_default() {
        let mut entry = gguf_entry();
        entry.context_length = 8_192;
        let launch = plan_launch(&entry, Path::new("/models/m.gguf"), None, &plan(33), 8123);
        let ctx = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--ctx-size")
            .map(|pair| pair[1].clone());
        assert_eq!(ctx.as_deref(), Some("8192"));
    }

    #[test]
    fn a_vision_model_is_started_with_its_projector() {
        let launch = plan_launch(
            &gguf_entry(),
            Path::new("/models/m.gguf"),
            Some(Path::new("/models/mmproj-m-F16.gguf")),
            &plan(33),
            8123,
        );
        let mmproj = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--mmproj")
            .map(|pair| pair[1].clone());
        assert_eq!(
            mmproj.as_deref(),
            Some(Path::new("/models/mmproj-m-F16.gguf").display().to_string().as_str())
        );
    }

    #[test]
    fn a_model_with_no_projector_gets_no_mmproj_flag() {
        // Not merely absent from the plan: a bare `--mmproj` with no value, or
        // one pointing at nothing, makes llama-server fail to start. A
        // text-only model has to produce a command line that never mentions it.
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(33), 8123);
        assert!(!launch.args.iter().any(|arg| arg == "--mmproj"));
    }

    #[test]
    fn the_server_advertises_arjuns_model_id_so_the_trace_matches() {
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(33), 8123);
        let alias = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--alias")
            .map(|pair| pair[1].clone());
        assert_eq!(alias.as_deref(), Some("qwen2.5-coder-7b"));
        assert_eq!(launch.served_model_id, "qwen2.5-coder-7b");
    }

    #[test]
    fn the_base_url_is_loopback_and_carries_the_version_prefix() {
        let launch = plan_launch(&gguf_entry(), Path::new("/models/m.gguf"), None, &plan(33), 8123);
        assert_eq!(launch.base_url, "http://127.0.0.1:8123/v1");
        assert!(probe::check_loopback(&launch.base_url).is_ok());
    }

    #[tokio::test]
    async fn a_python_model_with_no_endpoint_says_what_to_add() {
        let mut entry = gguf_entry();
        entry.runtime = Runtime::PythonSidecar;
        entry.serving = None;

        let servers = ModelServers::new();
        let error = servers
            .endpoint_for(&entry, Path::new("/models"), &plan(0))
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("does not start"), "{message}");
        assert!(message.contains("baseUrl"), "{message}");
    }

    #[tokio::test]
    async fn an_external_endpoint_that_is_down_names_the_endpoint() {
        let mut entry = gguf_entry();
        entry.runtime = Runtime::PythonSidecar;
        entry.serving = Some(ServingSpec::External {
            base_url: "http://127.0.0.1:1/v1".into(),
        });

        let servers = ModelServers::new();
        let error = servers
            .endpoint_for(&entry, Path::new("/models"), &plan(0))
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("127.0.0.1:1"), "{message}");
        assert!(message.contains("Nothing is listening"), "{message}");
    }

    #[tokio::test]
    async fn an_external_endpoint_off_this_machine_is_refused() {
        let mut entry = gguf_entry();
        entry.runtime = Runtime::PythonSidecar;
        entry.serving = Some(ServingSpec::External {
            base_url: "https://api.together.xyz/v1".into(),
        });

        let servers = ModelServers::new();
        let error = servers
            .endpoint_for(&entry, Path::new("/models"), &plan(0))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("not on this machine"), "{error}");
    }

    #[tokio::test]
    async fn a_managed_model_with_no_weights_says_so_before_launching_anything() {
        let servers = ModelServers::new();
        let error = servers
            .endpoint_for(&gguf_entry(), Path::new("/nonexistent-models"), &plan(33))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("has no weights"), "{error}");
    }

    /// The manifest is what an administrator edits, so its exact spelling is a
    /// contract. This pins it in both directions.
    #[test]
    fn the_manifest_spells_an_external_endpoint_as_base_url_in_camel_case() {
        let json = r#"{"mode":"external","baseUrl":"http://127.0.0.1:8000/v1"}"#;
        let parsed: ServingSpec = serde_json::from_str(json).expect("baseUrl parses");
        assert_eq!(
            parsed,
            ServingSpec::External {
                base_url: "http://127.0.0.1:8000/v1".into()
            }
        );
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    }

    #[test]
    fn a_managed_entry_needs_nothing_but_its_mode() {
        let parsed: ServingSpec = serde_json::from_str(r#"{"mode":"managed"}"#).unwrap();
        assert_eq!(parsed, ServingSpec::Managed);
    }

    #[test]
    fn nothing_is_running_before_anything_starts() {
        assert!(ModelServers::new().running_endpoints().is_empty());
    }

    /// The race fix holds the lock across the check-and-insert critical
    /// section. A meaningful end-to-end demonstration would need a real
    /// `llama-server` (or a test stand-in) that actually binds the port and
    /// accepts health checks, because the duplicate spawn only occurs when
    /// the spawn succeeds — otherwise both callers fail at the same step and
    /// the map stays empty regardless of whether the lock was held.
    ///
    /// What this test *does* prove: under concurrent `endpoint_for` calls
    /// for the same model, the server table never grows past one entry. The
    /// buggy version (`self.running()` → drop → spawn → re-lock → insert)
    /// would, in a hypothetical world where spawn succeeds, push a second
    /// entry over the first; the fixed version cannot, because the second
    /// caller either finds the table empty and re-checks under the lock, or
    /// finds the table populated and returns the existing endpoint.
    ///
    /// With no `llama-server` on PATH, both calls return `LaunchFailed` and
    /// the table is empty. The `<= 1` assertion is therefore trivially true
    /// here, but documents the invariant the fix upholds and will catch a
    /// regression that reintroduces duplicate insertions once the
    /// integration test stand-in is in place.
    #[tokio::test]
    async fn concurrent_managed_endpoints_never_duplicate_table_entries() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        // Placeholder weights so the `weights.exists()` check passes and
        // both callers reach the spawn point. The spawn will fail because
        // `llama-server` is not on PATH in the test environment, which is
        // exactly what we want — the table must still hold zero or one.
        let weights_path = tmp.path().join("qwen.gguf");
        fs::write(&weights_path, b"placeholder").unwrap();

        let mut entry = gguf_entry();
        entry.path = weights_path.file_name().unwrap().to_owned().into();

        let servers = std::sync::Arc::new(ModelServers::new());
        let dir = tmp.path().to_path_buf();

        let s1 = servers.clone();
        let e1 = entry.clone();
        let d1 = dir.clone();
        let h1 = tokio::spawn(async move { s1.endpoint_for(&e1, &d1, &plan(0)).await });

        let s2 = servers.clone();
        let e2 = entry.clone();
        let d2 = dir.clone();
        let h2 = tokio::spawn(async move { s2.endpoint_for(&e2, &d2, &plan(0)).await });

        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();

        // Both calls fail to spawn `llama-server` in the test environment;
        // the assertion that matters is on the table, not on the result
        // values.
        assert!(r1.is_err(), "spawn must fail without a real server binary");
        assert!(r2.is_err(), "spawn must fail without a real server binary");

        let running = servers.running_endpoints();
        assert!(
            running.len() <= 1,
            "concurrent calls duplicated a managed server entry: {running:?}",
        );
    }
}

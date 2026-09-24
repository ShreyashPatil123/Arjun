//! The Laya sidecar process, and the engine that decides when to believe it.
//!
//! Intent analysis sits in front of every agent turn, so the property that
//! matters most here is not accuracy but that a slow, missing or broken
//! semantic reader can never make a turn slower than its deadline or stop one
//! from routing. Everything below serves that:
//!
//! - **Bounded.** A reader thread turns the child's stdout into a channel, so a
//!   classification waits at most [`LayaConfig::deadline`]. The graph sidecar
//!   blocks on `read_line` with no limit, which is right for a background index
//!   and wrong for something a person is waiting behind.
//! - **Never queued.** A turn that finds Laya still loading, busy with another
//!   turn, or still owing an earlier answer takes the keyword reading at once.
//!   Turns do not wait behind each other for a sidecar that answers one at a
//!   time.
//! - **Contained.** A sidecar that exits, stalls or answers a different question
//!   than the one asked is killed, and not respawned for a backoff interval, so a
//!   Python environment that cannot load the model costs one failed start a
//!   minute rather than one per turn.
//! - **Measured before trusted.** Laya decides routing only under a calibration
//!   fitted for the exact question bytes the sidecar loaded. Without one it runs
//!   in shadow: its verdict is logged beside the keyword verdict that routed.
//!
//! The sidecar speaks JSON-RPC over stdio (see `sidecars/intent_sidecar/main.py`).
//! There is no socket, so there is no HTTP client here for
//! `scripts/check-egress.mjs` to have to exempt.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::capability::classifier::IntentClassifier;
use crate::capability::intent_analysis::{
    GatePolicy, IntentAnalysis, LayaGate, LayaVerdict, ShadowVerdict,
};
use crate::capability::language;

/// Where the Laya bundle lives inside the model library, beside the other
/// locally-served models (compare `knowledge::graph::relations::model_directory`).
pub fn model_directory(models_dir: &Path) -> PathBuf {
    models_dir.join("local").join("convaiinnovations_laya")
}

/// The calibration file, kept with the weights it was fitted on.
pub const CALIBRATION_FILE: &str = "arjun-intent-calibration.json";

/// Schema version of [`LayaCalibration`].
pub const CALIBRATION_SCHEMA: u32 = 1;

/// How long a turn waits for Laya before reading intent from keywords instead.
///
/// Laya's own measurements of one `choice` question with the checkpoint already
/// resident: 193–464 ms on CPU (README, "Production Preload & Memory"), and on a
/// laptop Ryzen 9 6900HX 329 ms at 8 threads rising to 910 ms at one
/// (BENCHMARKS.md). 1.5 s clears the slowest of those with room for a loaded
/// machine, and caps what a turn can pay for intent analysis before the model is
/// even chosen. `ARJUN_LAYA_DEADLINE_MS` overrides it; the calibration run
/// reports the latency actually measured on the machine in front of it.
const DEFAULT_DEADLINE: Duration = Duration::from_millis(1500);

/// A request older than this is not late but lost; the sidecar is restarted.
const STALL_AFTER: Duration = Duration::from_secs(20);

/// How long a failed sidecar is left down before another start is attempted.
///
/// Doubles with each consecutive failure up to [`MAX_BACKOFF_DOUBLINGS`]: a
/// Python environment that cannot import torch at all would otherwise pay the
/// seconds of a torch import every minute for as long as ARJUN runs.
const RESPAWN_BACKOFF: Duration = Duration::from_secs(60);
const MAX_BACKOFF_DOUBLINGS: u32 = 5;

fn backoff(failures: u32) -> Duration {
    RESPAWN_BACKOFF * 2u32.pow(failures.saturating_sub(1).min(MAX_BACKOFF_DOUBLINGS))
}

/// What the sidecar reports about itself once loaded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SidecarStatus {
    pub loaded: bool,
    pub installed: Vec<String>,
    pub device: Option<String>,
    pub threads: Option<u32>,
    pub load_ms: Option<f64>,
    pub laya_version: Option<String>,
    pub question_version: Option<String>,
    pub question_fingerprint: Option<String>,
    pub rss_mb: Option<f64>,
    pub peak_rss_mb: Option<f64>,
    pub cuda_allocated_mb: Option<f64>,
    pub cuda_peak_mb: Option<f64>,
}

/// A gate, the policy it runs under, and the measurements that justify both.
///
/// Written by the calibration run in [`crate::capability::intent_eval`] and
/// read at startup. It names the question fingerprint it was fitted against
/// because Laya's probabilities are a property of the exact option wording: a
/// gate fitted on one wording says nothing about another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaCalibration {
    pub schema: u32,
    pub question_version: String,
    pub question_fingerprint: String,
    pub laya_version: Option<String>,
    pub device: String,
    /// Checkpoints that answered part of the fitting data. A gate says nothing
    /// about a checkpoint it never saw.
    pub checkpoints: Vec<String>,
    pub gate: LayaGate,
    pub policy: GatePolicy,
    /// Which dataset, by path and content hash.
    pub dataset: String,
    pub fitted_at: String,
    /// The measured comparison the gate was chosen from, verbatim.
    pub evidence: serde_json::Value,
}

impl LayaCalibration {
    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| {
            format!("no intent calibration at {} ({error})", path.display())
        })?;
        let calibration: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("the intent calibration at {} is unreadable: {error}", path.display()))?;
        if calibration.schema != CALIBRATION_SCHEMA {
            return Err(format!(
                "the intent calibration at {} is schema {}, and this build reads {}",
                path.display(),
                calibration.schema,
                CALIBRATION_SCHEMA
            ));
        }
        Ok(calibration)
    }

    /// Why this calibration does not cover an answer, if it does not.
    pub fn applies_to(&self, fingerprint: Option<&str>, checkpoint: &str) -> Result<(), String> {
        match fingerprint {
            Some(f) if f == self.question_fingerprint => {}
            Some(_) => {
                return Err(format!(
                    "the calibration was fitted for a different intent question than the one loaded \
                     ({} was calibrated)",
                    self.question_version
                ))
            }
            None => return Err("the sidecar did not report which intent question it loaded".into()),
        }
        if !self.checkpoints.iter().any(|c| c == checkpoint) {
            return Err(format!(
                "the {checkpoint} checkpoint answered, and the calibration only measured {}",
                self.checkpoints.join(", ")
            ));
        }
        Ok(())
    }
}

/// How to start the sidecar. Production runs the bundled script under the
/// deployment's Python; tests substitute a stand-in that speaks the protocol.
#[derive(Debug, Clone)]
pub struct Launch {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub pythonpath: Option<PathBuf>,
}

impl Launch {
    pub fn bundled() -> Result<Self, String> {
        let script = crate::deployment::require_path("intent-sidecar")?;
        Ok(Self {
            program: crate::deployment::program("python").into(),
            pythonpath: script.parent().map(Path::to_path_buf),
            args: vec![script.into_os_string()],
        })
    }
}

/// Everything the engine needs to know before its first turn.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub enabled: bool,
    pub model_dir: PathBuf,
    pub calibration_path: PathBuf,
    /// `cpu` unless an operator opts into `cuda`.
    pub device: String,
    pub deadline: Duration,
}

impl LayaConfig {
    /// Reads the operator's overrides from the environment.
    ///
    /// `ARJUN_LAYA=off` disables the semantic reader outright, `ARJUN_LAYA_DIR`
    /// moves the weights, `ARJUN_LAYA_DEVICE` opts into a GPU,
    /// `ARJUN_LAYA_DEADLINE_MS` changes the wait and `ARJUN_LAYA_CALIBRATION`
    /// points at a calibration file elsewhere.
    pub fn from_environment(models_dir: &Path) -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        let model_dir = var("ARJUN_LAYA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_directory(models_dir));
        let calibration_path = var("ARJUN_LAYA_CALIBRATION")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join(CALIBRATION_FILE));
        Self {
            enabled: !matches!(
                var("ARJUN_LAYA").as_deref().map(str::to_ascii_lowercase).as_deref(),
                Some("off" | "0" | "false" | "no")
            ),
            model_dir,
            calibration_path,
            device: var("ARJUN_LAYA_DEVICE").unwrap_or_else(|| "cpu".to_string()),
            deadline: var("ARJUN_LAYA_DEADLINE_MS")
                .and_then(|ms| ms.parse::<u64>().ok())
                .filter(|ms| *ms > 0)
                .map(Duration::from_millis)
                .unwrap_or(DEFAULT_DEADLINE),
        }
    }
}

/// Why a classification did not produce a verdict.
#[derive(Debug, Clone, PartialEq)]
enum Failure {
    /// Transient; the sidecar stays up.
    Unavailable(String),
    /// The sidecar is gone or cannot be trusted; restart it after a backoff.
    Fatal(String),
}

/// A running sidecar and the pipes to talk to it.
struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Result<serde_json::Value, String>>,
    next_id: u64,
    load_id: u64,
    status: Option<SidecarStatus>,
    /// A classification that timed out and has not been answered yet.
    outstanding: Option<(u64, Instant)>,
}

impl Sidecar {
    fn spawn(launch: &Launch, config: &LayaConfig) -> Result<Self, String> {
        let mut command = crate::system_analyzer::process_utils::create_hidden_command(&launch.program);
        command.args(&launch.args);
        if let Some(path) = &launch.pythonpath {
            command.env("PYTHONPATH", path);
        }
        command.env("ARJUN_LAYA_DIR", &config.model_dir);
        command.env("ARJUN_LAYA_DEVICE", &config.device);
        // Set here as well as in main.py, so an interpreter started some other
        // way still cannot reach the Hub.
        command.env("HF_HUB_OFFLINE", "1");
        command.env("TRANSFORMERS_OFFLINE", "1");
        // Prompts are written as UTF-8. Without these, Python on a Windows
        // machine with a legacy code page decodes stdin as cp1252 and every
        // Hindi prompt arrives garbled.
        command.env("PYTHONUTF8", "1");
        command.env("PYTHONIOENCODING", "utf-8");
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        // Inherited, not piped: transformers writes progress to stderr, and a
        // pipe nobody reads fills and deadlocks the child mid-load.
        command.stderr(Stdio::inherit());

        let mut child = command
            .spawn()
            .map_err(|error| format!("the intent sidecar could not start: {error}"))?;
        let stdin = child.stdin.take().ok_or("the intent sidecar has no stdin")?;
        let stdout = child.stdout.take().ok_or("the intent sidecar has no stdout")?;

        let (sender, responses) = mpsc::channel();
        std::thread::Builder::new()
            .name("laya-intent-reader".into())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => {
                            let _ = sender.send(Err("the intent sidecar exited".to_string()));
                            return;
                        }
                        Ok(_) if line.trim().is_empty() => continue,
                        Ok(_) => {
                            let parsed = serde_json::from_str(&line).map_err(|error| {
                                format!("the intent sidecar wrote a line that is not JSON: {error}")
                            });
                            if sender.send(parsed).is_err() {
                                return;
                            }
                        }
                    }
                }
            })
            .map_err(|error| format!("the intent sidecar's reader could not start: {error}"))?;

        let mut sidecar = Self {
            child,
            stdin,
            responses,
            next_id: 0,
            load_id: 0,
            status: None,
            outstanding: None,
        };
        // Loading takes seconds. It is requested now and answered whenever it
        // finishes; turns before then read intent from keywords.
        sidecar.load_id = sidecar.send("intent.load", serde_json::json!({}))?;
        Ok(sidecar)
    }

    fn send(&mut self, method: &str, params: serde_json::Value) -> Result<u64, String> {
        self.next_id += 1;
        let mut line = serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params,
        }))
        .map_err(|error| error.to_string())?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("the intent sidecar stopped reading ({error})"))?;
        Ok(self.next_id)
    }

    /// Consumes one response: settles the load, a late answer, or reports death.
    fn settle(&mut self, response: Result<serde_json::Value, String>) -> Result<Option<serde_json::Value>, Failure> {
        let value = response.map_err(Failure::Fatal)?;
        let id = value.get("id").and_then(serde_json::Value::as_u64);
        if id == Some(self.load_id) && self.status.is_none() {
            if let Some(error) = value.get("error") {
                return Err(Failure::Fatal(format!(
                    "the Laya model could not be loaded: {}",
                    error.get("message").and_then(serde_json::Value::as_str).unwrap_or("no message")
                )));
            }
            let status: SidecarStatus = serde_json::from_value(value.get("result").cloned().unwrap_or_default())
                .map_err(|error| Failure::Fatal(format!("the intent sidecar's status is unreadable: {error}")))?;
            log::info!(
                "[INTENT] Laya loaded: checkpoints={:?} device={} threads={:?} load_ms={:?} laya={} question={} rss_mb={:?} cuda_mb={:?}",
                status.installed,
                status.device.as_deref().unwrap_or("?"),
                status.threads,
                status.load_ms,
                status.laya_version.as_deref().unwrap_or("?"),
                status.question_version.as_deref().unwrap_or("?"),
                status.rss_mb,
                status.cuda_allocated_mb,
            );
            self.status = Some(status);
            return Ok(None);
        }
        if let Some((late, _)) = self.outstanding {
            if id == Some(late) {
                self.outstanding = None;
                return Ok(None);
            }
        }
        Ok(Some(value))
    }

    /// Drains whatever has arrived without waiting.
    fn poll(&mut self) -> Result<(), Failure> {
        loop {
            match self.responses.try_recv() {
                Ok(response) => {
                    self.settle(response)?;
                }
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(Failure::Fatal("the intent sidecar exited".into()))
                }
            }
        }
    }

    fn classify(
        &mut self,
        prompt: &str,
        checkpoint: Option<&str>,
        deadline: Duration,
    ) -> Result<LayaVerdict, Failure> {
        self.poll()?;
        let Some(status) = self.status.as_ref() else {
            return Err(Failure::Unavailable("Laya is still loading".into()));
        };
        if let Some(wanted) = checkpoint {
            if !status.installed.iter().any(|c| c == wanted) {
                return Err(Failure::Unavailable(format!(
                    "the {wanted} Laya checkpoint is not installed"
                )));
            }
        }
        if let Some((_, since)) = self.outstanding {
            if since.elapsed() > STALL_AFTER {
                return Err(Failure::Fatal(format!(
                    "Laya has not answered a request for {} s",
                    since.elapsed().as_secs()
                )));
            }
            return Err(Failure::Unavailable("Laya is still answering an earlier turn".into()));
        }

        let id = self
            .send(
                "intent.classify",
                serde_json::json!({ "prompt": prompt, "checkpoint": checkpoint }),
            )
            .map_err(Failure::Fatal)?;
        let started = Instant::now();
        loop {
            let remaining = deadline.saturating_sub(started.elapsed());
            match self.responses.recv_timeout(remaining) {
                Ok(response) => {
                    let Some(value) = self.settle(response)? else { continue };
                    if value.get("id").and_then(serde_json::Value::as_u64) != Some(id) {
                        continue;
                    }
                    if let Some(error) = value.get("error") {
                        // The sidecar is alive and said why; a bad prompt is not
                        // a reason to restart it.
                        return Err(Failure::Unavailable(format!(
                            "Laya refused this prompt: {}",
                            error.get("message").and_then(serde_json::Value::as_str).unwrap_or("no message")
                        )));
                    }
                    return serde_json::from_value(value.get("result").cloned().unwrap_or_default())
                        .map_err(|error| Failure::Fatal(format!("Laya's answer is unreadable: {error}")));
                }
                Err(RecvTimeoutError::Timeout) => {
                    self.outstanding = Some((id, started));
                    return Err(Failure::Unavailable(format!(
                        "Laya did not answer within {} ms",
                        deadline.as_millis()
                    )));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(Failure::Fatal("the intent sidecar exited".into()))
                }
            }
        }
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // Up to two checkpoints resident: 743M parameters, about 3 GB of
        // weights at fp32 before the process's own overhead. Never left behind.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum State {
    /// Not running; the next attempt is allowed at the instant given.
    Down { retry_at: Instant, reason: String },
    Up(Box<Sidecar>),
}

/// The semantic intent reader, with the keyword classifier behind it.
pub struct IntentEngine {
    config: LayaConfig,
    launch: Result<Launch, String>,
    calibration: Result<LayaCalibration, String>,
    state: Mutex<State>,
    /// Consecutive failed starts, for the backoff. Reset by a successful load.
    failures: std::sync::atomic::AtomicU32,
}

impl IntentEngine {
    /// The engine for this installation: bundled script, library weights.
    pub fn from_environment(models_dir: &Path) -> Self {
        Self::new(LayaConfig::from_environment(models_dir), Launch::bundled())
    }

    pub fn new(config: LayaConfig, launch: Result<Launch, String>) -> Self {
        let calibration = LayaCalibration::read(&config.calibration_path);
        match &calibration {
            Ok(c) => log::info!(
                "[INTENT] Laya calibration {}: gate p>={:.2} margin>={:.2} policy={:?} fitted {}",
                config.calibration_path.display(),
                c.gate.min_probability,
                c.gate.min_margin,
                c.policy,
                c.fitted_at
            ),
            Err(reason) if config.enabled && config.model_dir.is_dir() => log::warn!(
                "[INTENT] Laya will run in shadow mode — its verdicts are logged but do not route — because {reason}"
            ),
            Err(_) => {}
        }
        Self {
            config,
            launch,
            calibration,
            state: Mutex::new(State::Down {
                retry_at: Instant::now(),
                reason: "Laya has not been started".into(),
            }),
            failures: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Starts the sidecar loading, if it can run at all. Returns at once.
    ///
    /// Called at startup so the seconds a checkpoint takes to load are spent
    /// before the first turn rather than inside it.
    pub fn start(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.ensure_up(&mut state);
    }

    /// Blocks until the sidecar has loaded or failed. For the calibration
    /// harness and tests; a turn never waits for this.
    pub fn wait_until_ready(&self, limit: Duration) -> Result<SidecarStatus, String> {
        let started = Instant::now();
        loop {
            {
                let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                self.ensure_up(&mut state);
                match &mut *state {
                    State::Down { reason, .. } => return Err(reason.clone()),
                    State::Up(sidecar) => {
                        if let Err(failure) = sidecar.poll() {
                            let reason = self.take_down(&mut state, failure);
                            return Err(reason);
                        }
                        if let State::Up(sidecar) = &*state {
                            if let Some(status) = &sidecar.status {
                                self.loaded();
                                return Ok(status.clone());
                            }
                        }
                    }
                }
            }
            if started.elapsed() > limit {
                return Err(format!("Laya did not load within {} s", limit.as_secs()));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn ensure_up(&self, state: &mut State) {
        let State::Down { retry_at, .. } = state else { return };
        if Instant::now() < *retry_at {
            return;
        }
        let reason = if !self.config.enabled {
            Some("the semantic intent model is switched off (ARJUN_LAYA=off)".to_string())
        } else if !self.config.model_dir.is_dir() {
            Some(format!(
                "the Laya intent model is not installed at {}",
                self.config.model_dir.display()
            ))
        } else {
            None
        };
        let outcome = match (reason, &self.launch) {
            (Some(reason), _) => Err(reason),
            (None, Err(reason)) => Err(reason.clone()),
            (None, Ok(launch)) => Sidecar::spawn(launch, &self.config),
        };
        *state = match outcome {
            Ok(sidecar) => State::Up(Box::new(sidecar)),
            // Not installed, or switched off, is a steady state rather than a
            // failure: it is re-checked every interval without escalating.
            Err(reason) if !self.config.enabled || !self.config.model_dir.is_dir() => {
                State::Down { retry_at: Instant::now() + RESPAWN_BACKOFF, reason }
            }
            Err(reason) => {
                let failures = self.failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                State::Down { retry_at: Instant::now() + backoff(failures), reason }
            }
        };
    }

    /// Records that the sidecar loaded, which ends any backoff.
    fn loaded(&self) {
        self.failures.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stops a failed sidecar, and returns the reason to report.
    fn take_down(&self, state: &mut State, failure: Failure) -> String {
        let reason = match failure {
            Failure::Fatal(reason) | Failure::Unavailable(reason) => reason,
        };
        let failures = self.failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let wait = backoff(failures);
        log::warn!(
            "[INTENT] Laya stopped ({reason}); intent is read from keywords, and a restart is tried in {} s",
            wait.as_secs()
        );
        *state = State::Down { retry_at: Instant::now() + wait, reason: reason.clone() };
        reason
    }

    /// Reads one turn's intent. Never fails and never waits past the deadline.
    pub fn analyze(&self, prompt: &str) -> IntentAnalysis {
        let analysis = self.analyze_inner(prompt);
        log::info!("{}", analysis.log_line());
        analysis
    }

    fn analyze_inner(&self, prompt: &str) -> IntentAnalysis {
        let started = Instant::now();
        let language = language::detect(prompt);

        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => {
                return IntentAnalysis::keyword_fallback(prompt, "Laya was answering another turn", None)
            }
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        self.ensure_up(&mut state);
        let sidecar = match &mut *state {
            State::Down { reason, .. } => {
                return IntentAnalysis::keyword_fallback(prompt, reason.clone(), None)
            }
            State::Up(sidecar) => sidecar,
        };

        let verdict = match sidecar.classify(prompt, language.forced_checkpoint(), self.config.deadline) {
            Ok(verdict) => verdict,
            Err(Failure::Unavailable(reason)) => {
                return IntentAnalysis::keyword_fallback(prompt, reason, None)
            }
            Err(failure) => {
                let reason = self.take_down(&mut state, failure);
                return IntentAnalysis::keyword_fallback(prompt, reason, None);
            }
        };
        let fingerprint = sidecar
            .status
            .as_ref()
            .and_then(|s| s.question_fingerprint.clone());
        self.loaded();
        // Checked before the lock is released, so a sidecar answering a
        // different question is the one taken down.
        let ranked = match verdict.ranked() {
            Ok(ranked) => ranked,
            Err(reason) => {
                let reason = self.take_down(&mut state, Failure::Fatal(reason));
                return IntentAnalysis::keyword_fallback(prompt, reason, None);
            }
        };
        drop(state);
        let latency_ms = started.elapsed().as_secs_f32() * 1000.0;

        let calibration = self
            .calibration
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|c| c.applies_to(fingerprint.as_deref(), &verdict.checkpoint).map(|()| c));
        match calibration {
            Ok(calibration) => IntentAnalysis::from_laya(
                &verdict,
                &IntentClassifier::classify(prompt),
                calibration.gate,
                calibration.policy,
                language,
                latency_ms,
            )
            .unwrap_or_else(|reason| IntentAnalysis::keyword_fallback(prompt, reason, None)),
            Err(reason) => {
                let mut analysis = IntentAnalysis::keyword_fallback(
                    prompt,
                    format!("Laya is in shadow mode: {reason}"),
                    Some(ShadowVerdict::from_ranked(&ranked, &verdict.checkpoint)),
                );
                analysis.latency_ms = latency_ms;
                analysis
            }
        }
    }

    /// Laya's raw answer for one prompt, bypassing the gate. For the
    /// calibration harness, which needs the ungated distribution.
    pub fn raw_verdict(&self, prompt: &str) -> Result<(LayaVerdict, f32), String> {
        let started = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.ensure_up(&mut state);
        let State::Up(sidecar) = &mut *state else {
            return Err("Laya is not running".into());
        };
        let checkpoint = language::detect(prompt).forced_checkpoint();
        match sidecar.classify(prompt, checkpoint, self.config.deadline.max(Duration::from_secs(30))) {
            Ok(verdict) => Ok((verdict, started.elapsed().as_secs_f32() * 1000.0)),
            Err(Failure::Unavailable(reason)) => Err(reason),
            Err(failure) => Err(self.take_down(&mut state, failure)),
        }
    }

    /// The sidecar's current self-report, including memory.
    pub fn status(&self) -> Option<SidecarStatus> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let State::Up(sidecar) = &mut *state else { return None };
        let id = sidecar.send("intent.status", serde_json::json!({})).ok()?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while let Ok(response) = sidecar.responses.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            if let Ok(Some(value)) = sidecar.settle(response) {
                if value.get("id").and_then(serde_json::Value::as_u64) == Some(id) {
                    return serde_json::from_value(value.get("result").cloned()?).ok();
                }
            }
        }
        None
    }

    pub fn calibration(&self) -> Result<&LayaCalibration, &String> {
        self.calibration.as_ref()
    }

    pub fn config(&self) -> &LayaConfig {
        &self.config
    }
}

/// The engine the application manages, if it started one.
pub fn managed(app: &tauri::AppHandle) -> Option<Arc<IntentEngine>> {
    use tauri::Manager;
    app.try_state::<Arc<IntentEngine>>().map(|state| state.inner().clone())
}

/// Reads a turn's intent without blocking the async runtime.
///
/// `engine` is `None` wherever no engine is managed — tests, and any build that
/// never started one — and then the keyword reading is exactly what routing
/// used before this module existed.
pub async fn analyze_for_turn(engine: Option<Arc<IntentEngine>>, prompt: &str) -> IntentAnalysis {
    let Some(engine) = engine else {
        return IntentAnalysis::keyword(prompt);
    };
    let owned = prompt.to_string();
    tokio::task::spawn_blocking(move || engine.analyze(&owned))
        .await
        .unwrap_or_else(|error| {
            IntentAnalysis::keyword_fallback(prompt, format!("intent analysis failed: {error}"), None)
        })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::capability::intent_analysis::{IntentSource, INTENT_LABELS};
    use crate::model_intelligence::intent::PromptIntent;

    const FINGERPRINT: &str = "test-fingerprint";

    fn status_line() -> String {
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"loaded": true, "installed": ["english", "multilingual"], "device": "cpu",
                       "questionVersion": "arjun-intent-v1", "questionFingerprint": FINGERPRINT}
        })
        .to_string()
    }

    fn answer_line(id: u64, leader: &str, p: f32) -> String {
        let rest = (1.0 - p) / 5.0;
        let probabilities: serde_json::Map<String, serde_json::Value> = INTENT_LABELS
            .iter()
            .map(|l| (l.to_string(), serde_json::json!(if *l == leader { p } else { rest })))
            .collect();
        serde_json::json!({
            "jsonrpc": "2.0", "id": id,
            "result": {"choice": leader, "probabilities": probabilities, "answerConfidence": p,
                       "checkpoint": "english", "latencyMs": 12.0}
        })
        .to_string()
    }

    /// A stand-in for the Python sidecar: a shell script that answers the
    /// protocol by rote. It tests the client's handling of timing and failure,
    /// which is what cannot be tested against the real model.
    fn stand_in(script: &str) -> Launch {
        Launch {
            program: "sh".into(),
            args: vec!["-c".into(), script.into()],
            pythonpath: None,
        }
    }

    fn config(dir: &Path, calibration: Option<&LayaCalibration>, deadline_ms: u64) -> LayaConfig {
        let calibration_path = dir.join(CALIBRATION_FILE);
        if let Some(c) = calibration {
            std::fs::write(&calibration_path, serde_json::to_vec(c).unwrap()).unwrap();
        }
        LayaConfig {
            enabled: true,
            model_dir: dir.to_path_buf(),
            calibration_path,
            device: "cpu".into(),
            deadline: Duration::from_millis(deadline_ms),
        }
    }

    fn calibration() -> LayaCalibration {
        LayaCalibration {
            schema: CALIBRATION_SCHEMA,
            question_version: "arjun-intent-v1".into(),
            question_fingerprint: FINGERPRINT.into(),
            laya_version: Some("0.3.20".into()),
            device: "cpu".into(),
            checkpoints: vec!["english".into()],
            gate: LayaGate { min_probability: 0.5, min_margin: 0.2 },
            policy: GatePolicy::Laya,
            dataset: "test".into(),
            fitted_at: "test".into(),
            evidence: serde_json::Value::Null,
        }
    }

    fn engine(script: String, calibration: Option<&LayaCalibration>, deadline_ms: u64) -> (IntentEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let engine = IntentEngine::new(config(dir.path(), calibration, deadline_ms), Ok(stand_in(&script)));
        (engine, dir)
    }

    fn answers(lines: &[String]) -> String {
        let mut script = String::new();
        for line in lines {
            script.push_str(&format!("read -r _; printf '%s\\n' '{line}'; "));
        }
        script.push_str("sleep 5");
        script
    }

    #[test]
    fn a_calibrated_answer_routes_on_laya() {
        let cal = calibration();
        let (engine, _dir) = engine(answers(&[status_line(), answer_line(2, "research", 0.8)]), Some(&cal), 2000);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let analysis = engine.analyze("summarise the turnaround report");
        assert_eq!(analysis.source, IntentSource::Laya, "{analysis:?}");
        assert_eq!(analysis.primary_intent, PromptIntent::Research);
        assert!(!analysis.fallback_used);
        assert_eq!(analysis.laya_model.as_deref(), Some("english"));
    }

    /// No measured gate, no routing: Laya's verdict is kept for the log only.
    #[test]
    fn an_uncalibrated_answer_is_shadowed_by_the_keyword_reading() {
        let (engine, _dir) = engine(answers(&[status_line(), answer_line(2, "research", 0.8)]), None, 2000);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let analysis = engine.analyze("summarise the turnaround report");
        assert_eq!(analysis.source, IntentSource::Keyword);
        assert!(analysis.fallback_used);
        assert!(analysis.fallback_reason.as_deref().unwrap().contains("shadow mode"), "{analysis:?}");
        assert_eq!(analysis.laya_shadow.as_ref().unwrap().intent, PromptIntent::Research);
    }

    #[test]
    fn a_calibration_for_other_question_wording_does_not_apply() {
        let mut cal = calibration();
        cal.question_fingerprint = "an-older-wording".into();
        let (engine, _dir) = engine(answers(&[status_line(), answer_line(2, "coding", 0.9)]), Some(&cal), 2000);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let analysis = engine.analyze("write a parser");
        assert_eq!(analysis.source, IntentSource::Keyword);
        assert!(analysis.fallback_reason.unwrap().contains("different intent question"));
    }

    #[test]
    fn a_sidecar_that_does_not_answer_costs_the_deadline_and_no_more() {
        let cal = calibration();
        let (engine, _dir) = engine(answers(&[status_line()]), Some(&cal), 200);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let started = Instant::now();
        let analysis = engine.analyze("Refactor this Python function and fix the stack trace");
        let waited = started.elapsed();
        assert!(waited < Duration::from_millis(1000), "waited {waited:?}");
        assert!(analysis.fallback_reason.as_deref().unwrap().contains("did not answer within 200 ms"));
        // The keyword reading still routes the turn.
        assert_eq!(analysis.primary_intent, PromptIntent::Coding);

        // The next turn does not queue behind the lost request.
        let started = Instant::now();
        let next = engine.analyze("hello");
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(next.fallback_reason.unwrap().contains("earlier turn"));
    }

    #[test]
    fn a_sidecar_that_dies_falls_back_and_backs_off() {
        let (engine, _dir) = engine("exit 3".into(), Some(&calibration()), 500);
        assert!(engine.wait_until_ready(Duration::from_secs(5)).is_err());

        let analysis = engine.analyze("write a parser");
        assert_eq!(analysis.source, IntentSource::Keyword);
        assert!(analysis.fallback_used);
        // Which of the two is seen first is a race between the child exiting
        // and the load request being written; both are the same death.
        let reason = analysis.fallback_reason.unwrap();
        assert!(reason.contains("exited") || reason.contains("stopped reading"), "{reason}");
    }

    #[test]
    fn a_model_that_fails_to_load_says_why() {
        let error = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"No module named laya"}}"#;
        let (engine, _dir) = engine(answers(&[error.to_string()]), Some(&calibration()), 500);
        let reason = engine.wait_until_ready(Duration::from_secs(5)).unwrap_err();
        assert!(reason.contains("No module named laya"), "{reason}");
        let analysis = engine.analyze("write a parser");
        assert!(analysis.fallback_reason.unwrap().contains("No module named laya"));
    }

    #[test]
    fn an_answer_to_a_different_question_is_a_fault_not_a_reading() {
        let bad = r#"{"jsonrpc":"2.0","id":2,"result":{"choice":"billing","probabilities":{"billing":0.9,"tech":0.1},"checkpoint":"english","latencyMs":5}}"#;
        let (engine, _dir) = engine(answers(&[status_line(), bad.to_string()]), Some(&calibration()), 2000);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let analysis = engine.analyze("write a parser");
        assert_eq!(analysis.source, IntentSource::Keyword);
        assert!(analysis.fallback_reason.unwrap().contains("options where the intent question has 6"));
    }

    #[test]
    fn hindi_asks_for_a_checkpoint_the_sidecar_must_have() {
        let only_english = serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"loaded": true, "installed": ["english"], "questionFingerprint": FINGERPRINT}
        })
        .to_string();
        let (engine, _dir) = engine(answers(&[only_english]), Some(&calibration()), 2000);
        engine.wait_until_ready(Duration::from_secs(5)).unwrap();

        let analysis = engine.analyze("is report ka summary do");
        assert_eq!(analysis.language, "hi-en");
        assert!(analysis.fallback_reason.unwrap().contains("multilingual Laya checkpoint is not installed"));
    }

    #[test]
    fn nothing_is_spawned_when_the_model_is_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config(dir.path(), None, 500);
        config.model_dir = dir.path().join("not-installed");
        // A launch that would fail loudly if it were ever run.
        let engine = IntentEngine::new(config, Ok(stand_in("echo spawned >&2; exit 9")));
        let analysis = engine.analyze("write a parser");
        assert!(analysis.fallback_reason.unwrap().contains("not installed"));
    }

    /// The real sidecar script, resolved through the deployment table, not a
    /// stand-in. With no checkpoint in its directory it refuses to load and
    /// says why before importing torch, so this runs anywhere Python does.
    #[test]
    fn the_real_sidecar_refuses_an_empty_model_directory_by_name() {
        let script = crate::deployment::require_path("intent-sidecar").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let launch = Launch {
            program: "python3".into(),
            pythonpath: script.parent().map(Path::to_path_buf),
            args: vec![script.into_os_string()],
        };
        let engine = IntentEngine::new(config(dir.path(), None, 1000), Ok(launch));

        let reason = engine.wait_until_ready(Duration::from_secs(60)).unwrap_err();
        assert!(reason.contains("rl_agent_config.json"), "{reason}");

        let analysis = engine.analyze("Refactor this Python function and fix the stack trace");
        assert!(analysis.fallback_used);
        assert_eq!(analysis.primary_intent, PromptIntent::Coding, "the keyword reading still routes");
    }

    #[test]
    fn repeated_failures_back_off_further_each_time() {
        assert_eq!(backoff(1), Duration::from_secs(60));
        assert_eq!(backoff(2), Duration::from_secs(120));
        assert_eq!(backoff(3), Duration::from_secs(240));
        assert_eq!(backoff(50), Duration::from_secs(60 * 32), "capped");
    }

    #[test]
    fn switched_off_means_keywords() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config(dir.path(), None, 500);
        config.enabled = false;
        let engine = IntentEngine::new(config, Ok(stand_in("exit 9")));
        assert!(engine.analyze("x").fallback_reason.unwrap().contains("switched off"));
    }
}

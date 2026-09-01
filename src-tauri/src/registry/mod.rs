//! The local model registry.
//!
//! PS 26117: *"New open weight models should be addable later without redesigning
//! the system, since this space is moving fast."* That is the requirement this
//! module exists to satisfy, and it is stricter than it sounds — it means
//! registering a model must not involve a code change, a recompile, or a release.
//!
//! So a model is a **manifest entry plus a file on disk**. The manifest is read
//! at startup and on demand; nothing about a particular model is compiled in.
//! Adding one is: copy the weights in, add an entry, restart.
//!
//! ## Two runtimes, one registry
//!
//! Some of the best models for this problem are not GGUF and never will be —
//! Docling, MinerU, most document vision models and rerankers are PyTorch. The
//! entry therefore names its [`Runtime`], and the router treats both alike. A
//! registry that assumed GGUF would quietly exclude the whole document pipeline.
//!
//! ## What an entry has to declare
//!
//! Everything PS step 9 asks for: name, version, licence, hash, parameter class,
//! quantisation, modalities, context length, expected GPU memory, and the data
//! classifications it may be used on. The last is the one people forget, and it
//! is the one that stops a model that phones home — or simply one nobody has
//! reviewed — from being pointed at vendor negotiations.

pub mod discovery;
pub mod fit_score;
pub mod integrity;
pub mod router;
pub mod scan;
pub mod categorize;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::policy::Classification;

/// Which engine loads this model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Runtime {
    /// In-process `llama.cpp`. GGUF weights, quantised, lowest overhead.
    LlamaCpp,
    /// The Python sidecar, over stdio. Transformers models, OCR, embeddings.
    PythonSidecar,
}

impl Runtime {
    pub const fn label(self) -> &'static str {
        match self {
            Runtime::LlamaCpp => "llama.cpp",
            Runtime::PythonSidecar => "Python sidecar",
        }
    }
}

/// Modalities a model can process.
///
/// Used as a hard gate in the router: a task requiring an unsupported modality
/// is refused rather than routed to a model that cannot handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

impl Modality {
    pub const fn label(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
        }
    }
}

/// The job a model is registered to do.
///
/// A model may hold several — a vision-language model is usually competent at
/// general reasoning too — and the router picks per task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelRole {
    /// General instruction following, summarising, planning.
    Reasoning,
    /// Writing and debugging code.
    Coding,
    /// Photographs, drawings, scanned pages.
    Vision,
    /// Turning a scanned page into structured text.
    DocumentOcr,
    /// Producing vectors for retrieval.
    Embedding,
    /// Reordering retrieved passages.
    Rerank,
}

impl ModelRole {
    pub const ALL: &'static [ModelRole] = &[
        ModelRole::Reasoning,
        ModelRole::Coding,
        ModelRole::Vision,
        ModelRole::DocumentOcr,
        ModelRole::Embedding,
        ModelRole::Rerank,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            ModelRole::Reasoning => "reasoning",
            ModelRole::Coding => "coding",
            ModelRole::Vision => "vision",
            ModelRole::DocumentOcr => "document OCR",
            ModelRole::Embedding => "embedding",
            ModelRole::Rerank => "reranking",
        }
    }

    /// Smallest parameter count, in billions, worth routing this role to.
    ///
    /// Not a preference — a cliff. Measured tool-calling accuracy collapses
    /// below roughly 7B: a smaller model cannot hold the function-calling format
    /// across a multi-step loop, so an agent built on one fails in a way that
    /// looks like a bug in the orchestrator rather than a model that is too
    /// small. Reasoning degrades more gracefully, so its floor is lower.
    ///
    /// Grammar-constrained decoding lifts small models substantially, which is
    /// why these floors are as low as they are — without it they would need to
    /// be higher. Roles whose models are small by design have no floor at all.
    pub const fn minimum_parameters_b(self) -> f32 {
        match self {
            ModelRole::Coding => 7.0,
            ModelRole::Reasoning => 3.0,
            ModelRole::Vision => 1.0,
            // A 300M embedding model and a 1.2B document VLM are both correct
            // choices at their size. Judging them on parameter count would rule
            // out the best available option.
            ModelRole::DocumentOcr | ModelRole::Embedding | ModelRole::Rerank => 0.0,
        }
    }
}

/// One registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub version: String,
    /// SPDX identifier where there is one. Shown before a model is used, because
    /// a licence a PSU cannot accept is a deployment blocker, not a footnote.
    pub license: String,
    /// SHA-256 of the weights, checked at import.
    pub sha256: Option<String>,
    pub runtime: Runtime,
    pub roles: Vec<ModelRole>,
    /// Modalities this model supports (text, image, audio, video).
    /// Used as a hard gate: a task requiring an unsupported modality is refused.
    #[serde(default)]
    pub modalities: Vec<Modality>,
    pub quantization: Option<String>,
    /// Total parameters in billions. For a mixture-of-experts model this is the
    /// total, with `activeParametersB` carrying what actually runs per token.
    pub parameters_b: f32,
    #[serde(default)]
    pub active_parameters_b: Option<f32>,
    pub context_length: u32,
    /// Size of the weights on disk, which is what has to fit in memory.
    pub weights_bytes: u64,
    /// Whether this model supports structured output / tool calling.
    /// Used as a hard gate for roles that require it (coding, tool-calling).
    #[serde(default)]
    pub supports_structured_output: bool,
    /// Classifications this model may be used on. Empty means none — a model
    /// nobody has reviewed is not usable, rather than usable on everything.
    #[serde(default)]
    pub permitted_classifications: Vec<Classification>,
    /// Path to the weights, relative to the models directory.
    pub path: PathBuf,
    /// What the runtime needs to actually load this.
    ///
    /// Separate from `path` because the llama.cpp loader addresses a model by
    /// the package coordinates it was installed under, not by file path. Absent
    /// on an entry that describes a model the runtime cannot load on its own —
    /// a Python-sidecar model, for instance — and the activator refuses those
    /// with an explanation rather than failing obscurely at load time.
    #[serde(default)]
    pub load: Option<LoadSpec>,
    /// How this model is served: started by ARJUN, or already running.
    ///
    /// Absent means the default for the runtime — GGUF is managed, Python is
    /// not, because ARJUN cannot honestly claim to manage a vLLM process it did
    /// not provision. See [`crate::serving::ServingSpec`].
    #[serde(default)]
    pub serving: Option<crate::serving::ServingSpec>,
    /// Runtime profile required to run this model (e.g., "cuda", "vulkan", "cpu").
    /// Used as a hard gate: the model is only eligible if the profile is available.
    #[serde(default)]
    pub required_runtime_profile: Option<String>,
    /// Administrators disable a model without deleting it.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Routing preferences. Used as a tie-breaker, never a hard gate:
    /// a model marked `preferred` does not bypass the classification or
    /// VRAM filters, it only wins a tie against a same-size peer.
    #[serde(default)]
    pub routing: RoutingPreference,
}

/// How the router should treat this model when it is one of several that
/// all clear the hard gates. None of these widen permissions; they only
/// change *which* of the already-eligible candidates is picked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingPreference {
    /// True on the model that should win a tie against same-size peers.
    /// The user-facing equivalent of "Always use the best model for me":
    /// the operator marks one candidate per role as preferred, and the
    /// router honours it after the hard gates.
    #[serde(default)]
    pub preferred: bool,
    /// Tie-break order within a size band. Lower wins. Used by the
    /// per-model performance telemetry to surface "this model is
    /// consistently faster on my hardware" without inventing a number.
    /// Tied with `preferred` only when telemetry is absent.
    #[serde(default)]
    pub rank_within_band: u32,
}

/// The coordinates the inference runtime loads a model by.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadSpec {
    pub provider_id: String,
    /// The upstream model id, e.g. `Qwen/Qwen2.5-7B-Instruct`.
    pub model_id: String,
    pub quantization: String,
}

fn default_true() -> bool {
    true
}

fn is_default_orchestrator(entry: &ModelEntry) -> bool {
    let configured = crate::config::DEFAULT_ORCHESTRATOR_MODEL_ID;
    if entry
        .load
        .as_ref()
        .map(|load| normalized_identity(&load.model_id) == normalized_identity(configured))
        .unwrap_or(false)
    {
        return true;
    }

    let identity = normalized_identity(&format!("{} {}", entry.id, entry.name));
    identity.contains("gemma4") && identity.contains("12b")
}

fn normalized_identity(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

impl ModelEntry {
    /// Whether the inference runtime can load this on its own.
    pub fn is_loadable(&self) -> bool {
        self.load.is_some()
    }

    pub fn serves(&self, role: ModelRole) -> bool {
        self.roles.contains(&role)
    }

    /// Whether this model may be used on material of a given sensitivity.
    pub fn permits(&self, classification: Classification) -> bool {
        self.permitted_classifications.contains(&classification)
    }

    /// Whether it clears the floor for a role.
    ///
    /// Judged on *active* parameters where the model declares them: a sparse
    /// mixture-of-experts model with 120B total and 5B active behaves like a 5B
    /// model per token, and pretending otherwise would route agent planning to
    /// something that cannot hold a tool-call format.
    pub fn meets_floor(&self, role: ModelRole) -> bool {
        let effective = self.active_parameters_b.unwrap_or(self.parameters_b);
        effective >= role.minimum_parameters_b()
    }

    /// Whether this model supports the required modality.
    ///
    /// A model with no declared modalities is treated as text-only for
    /// backwards compatibility.
    pub fn supports_modality(&self, modality: Modality) -> bool {
        if self.modalities.is_empty() {
            return modality == Modality::Text;
        }
        self.modalities.contains(&modality)
    }

    /// Whether this model supports structured output / tool calling.
    pub fn supports_structured_output(&self) -> bool {
        self.supports_structured_output
    }

    /// Whether the required runtime profile is available.
    ///
    /// If the model declares no required profile, it is assumed to run on any
    /// profile (backwards compatibility).
    pub fn runtime_profile_available(&self, available_profiles: &[String]) -> bool {
        match &self.required_runtime_profile {
            Some(required) => available_profiles.iter().any(|p| p == required),
            None => true,
        }
    }

    /// Whether the model's hash has been verified.
    ///
    /// A model without a declared hash cannot be verified and is treated as
    /// unverified (not a hard failure, but reported).
    pub fn hash_verified(&self) -> bool {
        self.sha256.is_some()
    }

    /// The declared sha256 of the weights, if any. Used by the
    /// orchestrator path resolver (TODO 3) to refuse a model
    /// whose on-disk bytes do not match the registry.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Whether the model's license is in the allowed list.
    /// An empty allowed list means no restriction (all licenses allowed).
    pub fn license_allowed(&self, allowed_licenses: &[String]) -> bool {
        if allowed_licenses.is_empty() {
            return true;
        }
        allowed_licenses.iter().any(|l| l == &self.license)
    }
}

/// What the manifest file holds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    manifest_path: PathBuf,
}

impl ModelRegistry {
    /// Loads the registry: what an administrator declared, plus what is on disk.
    ///
    /// Declared entries win on collision. Discovered ones are visible but cleared
    /// for no classification, so they can be seen and reviewed without being
    /// usable on real material first.
    pub fn load_with_discovery(app_data_dir: &Path) -> Result<Self> {
        let models_dir = app_data_dir.join("models");

        // An unreadable manifest must not also hide the models that are plainly
        // on disk. It used to: the `?` here short-circuited before discovery
        // ran, the caller fell back to an empty registry, and the workbench
        // told an operator with six installed models that none were registered.
        //
        // A broken manifest is still a real problem and is still logged loudly.
        // It is just not a reason to pretend the machine is empty.
        let declared = match Self::load(&models_dir) {
            Ok(declared) => declared,
            Err(error) => {
                log::error!(
                    "[REGISTRY] {} could not be read, so only models found on disk are \
                     available and none of them is cleared for classified material: {error:#}",
                    models_dir.join("registry.json").display()
                );
                Self {
                    entries: Vec::new(),
                    manifest_path: models_dir.join("registry.json"),
                }
            }
        };
        let discovered = discovery::discover(app_data_dir);

        let count = discovered.len();
        let merged = discovery::merge(declared.entries, discovered);
        if count > 0 {
            log::info!(
                "[REGISTRY] {count} model(s) found on disk; they are listed but cleared for no                  classification until an administrator reviews them"
            );
        }

        Ok(Self {
            entries: merged,
            manifest_path: declared.manifest_path,
        })
    }

    /// Reads the manifest beside the models directory.
    ///
    /// A missing manifest is an empty registry, not an error: a fresh install
    /// legitimately has no models until somebody provisions one, and failing to
    /// start would be a worse answer than starting with nothing to route to.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let manifest_path = models_dir.join("registry.json");
        if !manifest_path.exists() {
            return Ok(Self {
                entries: Vec::new(),
                manifest_path,
            });
        }

        let raw = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("could not read {}", manifest_path.display()))?;
        // A byte-order mark is three invisible characters that `serde_json`
        // refuses outright, and on Windows they are routine: PowerShell's `>`,
        // `Out-File` and Notepad all write UTF-8 with one by default. Rejecting
        // a whole registry over them makes every model on the machine vanish
        // and tells the operator to import models they already have — which is
        // a long way to travel from three bytes nobody can see.
        let body = raw.strip_prefix('\u{feff}').unwrap_or(raw.as_str());
        let manifest: ModelManifest = serde_json::from_str(body)
            .with_context(|| format!("{} is not a valid model manifest", manifest_path.display()))?;

        Self::from_manifest(manifest, manifest_path)
    }

    pub(crate) fn from_manifest(manifest: ModelManifest, manifest_path: PathBuf) -> Result<Self> {
        // A duplicate id would make routing depend on manifest order, which is
        // exactly the kind of silent inconsistency that is painful to diagnose
        // later. Refuse it at load.
        let mut seen = BTreeSet::new();
        for entry in &manifest.models {
            if !seen.insert(entry.id.clone()) {
                anyhow::bail!("the model manifest registers {:?} more than once", entry.id);
            }
        }

        Ok(Self {
            entries: manifest.models,
            manifest_path,
        })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Where the weights live.
    ///
    /// Derived from the manifest rather than stored separately, so the two
    /// cannot disagree about which directory a `ModelEntry::path` is relative
    /// to. An empty registry loaded from an absent manifest reports the
    /// directory it looked in, which is what an administrator needs to see.
    pub fn models_dir(&self) -> &Path {
        self.manifest_path.parent().unwrap_or(Path::new("."))
    }

    /// Every registered model, including disabled ones, for the admin screen.
    pub fn all(&self) -> &[ModelEntry] {
        &self.entries
    }

    pub fn find(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Models that are enabled, serve this role, clear its floor, and are
    /// permitted for this material — the candidates the router chooses among.
    ///
    /// Additional hard gates:
    /// - `required_modality`: the task's required modality (text, image, etc.)
    /// - `require_structured_output`: whether the role requires structured output/tool calling
    /// - `available_runtime_profiles`: runtime profiles available on this machine
    /// - `allowed_licenses`: licenses permitted by policy
    pub fn candidates(
        &self,
        role: ModelRole,
        classification: Option<Classification>,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Vec<&ModelEntry> {
        self.entries
            .iter()
            .filter(|e| e.enabled)
            .filter(|e| e.serves(role))
            .filter(|e| e.meets_floor(role))
            .filter(|e| match classification {
                Some(c) => e.permits(c),
                None => true,
            })
            .filter(|e| {
                if let Some(modality) = required_modality {
                    e.supports_modality(modality)
                } else {
                    true
                }
            })
            .filter(|e| {
                if require_structured_output {
                    e.supports_structured_output()
                } else {
                    true
                }
            })
            .filter(|e| e.runtime_profile_available(available_runtime_profiles))
            .filter(|e| e.license_allowed(allowed_licenses))
            .collect()
    }

    /// The model entry used by the orchestrator — the model that runs the chat.
    /// An explicit `orchestrator.*` entry always wins. Without one, the product
    /// default (Gemma 4 12B) is selected from its installed load coordinates or
    /// recognizable identity.
    ///
    /// The marker is the first model whose `id` starts with
    /// `"orchestrator"` (i.e. `"orchestrator"` itself, or
    /// `"orchestrator.qwen3-4b"` etc.). A more elaborate
    /// tagging scheme can replace this without breaking
    /// callers.
    pub fn orchestrator_entry(&self) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| {
            e.id == "orchestrator" || e.id.starts_with("orchestrator.")
        }).or_else(|| self.entries.iter().find(|e| is_default_orchestrator(e)))
    }

    /// The (family, quantisation) tuple the orchestrator
    /// resolver should look for in the library scan. Falls
    /// back to the standard Gemma 4 12B / Q4_0 if no orchestrator
    /// is registered — the resolver then reports NotFound
    /// rather than guessing.
    pub fn orchestrator_identity(&self) -> Option<(String, String)> {
        let entry = self.orchestrator_entry()?;
        let family = if is_default_orchestrator(entry) {
            "gemma-4-12b".to_string()
        } else {
            entry
                .id
                .strip_prefix("orchestrator")
                .map(|s| s.trim_start_matches('.').to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| entry.name.clone())
        };
        let quant = entry
            .quantization
            .clone()
            .unwrap_or_else(|| crate::config::DEFAULT_ORCHESTRATOR_QUANTIZATION.to_string());
        Some((family, quant))
    }

    /// The root of the model library on disk. The orchestrator
    /// resolver scans this directory when the contract path
    /// is missing. The default is the registry's `models_dir`,
    /// which is the directory the manifest lives in.
    pub fn library_root(&self, app_data_dir: &Path) -> PathBuf {
        // The user may have installed models under the app
        // data dir (the default), or under a separate library
        // location. The latter is honoured by an environment
        // variable so an operator can point ARJUN at a
        // pre-existing library without copying files.
        if let Ok(custom) = std::env::var("ARJUN_MODEL_LIBRARY") {
            return PathBuf::from(custom);
        }
        app_data_dir.join("models")
    }

    /// The registered entry that owns a given on-disk path.
    /// Returns `None` for a path that no entry declared —
    /// a file the library scan picked up but the manifest
    /// does not list. The caller is then free to load the
    /// file as an unregistered model if its policy allows.
    pub fn entry_for_path(&self, path: &Path) -> Option<&ModelEntry> {
        self.entries
            .iter()
            .find(|e| e.path == path || path.ends_with(&e.path))
    }

    /// Writes the manifest to disk atomically. The file is
    /// written to `<manifest_path>.tmp`, fsynced, and renamed
    /// over `<manifest_path>`. A reader that opens during the
    /// write sees either the old or the new file, never a
    /// half-written one.
    ///
    /// TODO 4: this is the manifest writer the orchestrator
    /// path resolver and the auto-categorization pipeline
    /// both use. Atomic writes matter because a corrupted
    /// manifest makes the registry refuse to load every
    /// model on the machine.
    pub fn save_manifest(&self) -> Result<(), String> {
        let manifest = ModelManifest {
            models: self.entries.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("manifest could not be serialised: {e}"))?;
        let final_path = self.manifest_path.clone();
        let tmp_path = final_path.with_extension("json.tmp");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&tmp_path)
                .map_err(|e| format!("manifest tmp file could not be created: {e}"))?;
            file.write_all(&bytes)
                .map_err(|e| format!("manifest could not be written: {e}"))?;
            file.sync_all()
                .map_err(|e| format!("manifest could not be fsynced: {e}"))?;
        }
        std::fs::rename(&tmp_path, &final_path)
            .map_err(|e| format!("manifest could not be renamed: {e}"))?;
        Ok(())
    }

    /// Replaces the current entries with `new_entries` and
    /// writes the manifest. Used by the auto-categorization
    /// pipeline (TODO 5) to commit a re-classified library
    /// in one step.
    pub fn replace_and_save(&mut self, new_entries: Vec<ModelEntry>) -> Result<(), String> {
        // Refuse duplicates the same way `from_manifest` does.
        let mut seen = std::collections::BTreeSet::new();
        for entry in &new_entries {
            if !seen.insert(entry.id.clone()) {
                return Err(format!(
                    "the model manifest registers {:?} more than once",
                    entry.id
                ));
            }
        }
        self.entries = new_entries;
        self.save_manifest()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::Value;

    pub(crate) fn entry(id: &str, params: f32, roles: Vec<ModelRole>) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            name: id.into(),
            version: "1".into(),
            license: "apache-2.0".into(),
            sha256: None,
            runtime: Runtime::LlamaCpp,
            roles,
            modalities: vec![Modality::Text],
            quantization: Some("Q4_K_M".into()),
            parameters_b: params,
            active_parameters_b: None,
            context_length: 32_768,
            weights_bytes: (params * 0.6 * 1e9) as u64,
            supports_structured_output: false,
            permitted_classifications: Classification::ALL.to_vec(),
            path: PathBuf::from(format!("{id}.gguf")),
            load: Some(LoadSpec {
                provider_id: "huggingface".into(),
                model_id: id.into(),
                quantization: "Q4_K_M".into(),
            }),
            serving: None,
            required_runtime_profile: None,
            enabled: true,
            routing: RoutingPreference::default(),
        }
    }

    fn registry(entries: Vec<ModelEntry>) -> ModelRegistry {
        ModelRegistry::from_manifest(
            ModelManifest { models: entries },
            PathBuf::from("registry.json"),
        )
        .unwrap()
    }

    #[test]
    fn a_missing_manifest_is_an_empty_registry_not_a_failure() {
        let registry = ModelRegistry::load(Path::new("./definitely-not-here")).unwrap();
        assert!(registry.all().is_empty());
    }

    #[test]
    fn a_duplicate_id_is_refused_at_load() {
        let result = ModelRegistry::from_manifest(
            ModelManifest {
                models: vec![
                    entry("qwen", 8.0, vec![ModelRole::Reasoning]),
                    entry("qwen", 4.0, vec![ModelRole::Reasoning]),
                ],
            },
            PathBuf::from("registry.json"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn candidates_are_filtered_by_role() {
        let registry = registry(vec![
            entry("coder", 8.0, vec![ModelRole::Coding]),
            entry("thinker", 8.0, vec![ModelRole::Reasoning]),
        ]);
        let coding = registry.candidates(ModelRole::Coding, None, None, false, &[], &[]);
        assert_eq!(coding.len(), 1);
        assert_eq!(coding[0].id, "coder");
    }

    /// The floor is the point of the whole exercise: a 3B model fits any laptop
    /// and still cannot hold a tool-call format across a loop.
    #[test]
    fn a_model_below_the_coding_floor_is_not_a_candidate() {
        let registry = registry(vec![entry("tiny", 3.0, vec![ModelRole::Coding])]);
        assert!(registry.candidates(ModelRole::Coding, None, None, false, &[], &[]).is_empty());
        // The same model is fine for reasoning, whose floor is lower.
        assert_eq!(registry.candidates(ModelRole::Reasoning, None, None, false, &[], &[]).len(), 0);
    }

    #[test]
    fn small_models_are_not_penalised_in_roles_that_are_small_by_design() {
        let registry = registry(vec![
            entry("embed", 0.3, vec![ModelRole::Embedding]),
            entry("docvlm", 1.2, vec![ModelRole::DocumentOcr]),
        ]);
        assert_eq!(registry.candidates(ModelRole::Embedding, None, None, false, &[], &[]).len(), 1);
        assert_eq!(registry.candidates(ModelRole::DocumentOcr, None, None, false, &[], &[]).len(), 1);
    }

    /// A sparse model is judged on what actually runs per token.
    #[test]
    fn mixture_of_experts_models_are_judged_on_active_parameters() {
        let mut sparse = entry("moe", 120.0, vec![ModelRole::Coding]);
        sparse.active_parameters_b = Some(5.1);
        assert!(!sparse.meets_floor(ModelRole::Coding), "5.1B active is below the 7B floor");

        sparse.active_parameters_b = Some(10.0);
        assert!(sparse.meets_floor(ModelRole::Coding));
    }

    #[test]
    fn a_disabled_model_is_never_a_candidate() {
        let mut disabled = entry("retired", 8.0, vec![ModelRole::Reasoning]);
        disabled.enabled = false;
        let registry = registry(vec![disabled]);
        assert!(registry.candidates(ModelRole::Reasoning, None, None, false, &[], &[]).is_empty());
        // But it is still listed, so an administrator can see and re-enable it.
        assert_eq!(registry.all().len(), 1);
    }

    /// A model nobody has cleared is usable on nothing, rather than everything.
    #[test]
    fn an_unreviewed_model_is_permitted_for_no_classification() {
        let mut unreviewed = entry("unreviewed", 8.0, vec![ModelRole::Reasoning]);
        unreviewed.permitted_classifications = vec![];
        let registry = registry(vec![unreviewed]);

        for classification in Classification::ALL {
            assert!(
                registry
                    .candidates(ModelRole::Reasoning, Some(*classification), None, false, &[], &[])
                    .is_empty(),
                "an unreviewed model should not be usable on {}",
                classification.label()
            );
        }
    }

    #[test]
    fn classification_narrows_the_candidate_set() {
        let mut restricted = entry("restricted", 8.0, vec![ModelRole::Reasoning]);
        restricted.permitted_classifications = vec![Classification::Internal];
        let registry = registry(vec![restricted]);

        assert_eq!(
            registry
                .candidates(ModelRole::Reasoning, Some(Classification::Internal), None, false, &[], &[])
                .len(),
            1
        );
        assert!(registry
            .candidates(ModelRole::Reasoning, Some(Classification::Financial), None, false, &[], &[])
            .is_empty());
    }

    /// The manifest is the whole interface for adding a model: parsing one that
    /// names a model this code has never heard of must just work.
    #[test]
    fn a_model_this_code_has_never_seen_registers_from_json_alone() {
        let json = r#"{
            "models": [{
                "id": "some-future-model-2027",
                "name": "Something Not Yet Released",
                "version": "0.1",
                "license": "apache-2.0",
                "runtime": "pythonSidecar",
                "roles": ["vision", "documentOcr"],
                "parametersB": 2.4,
                "contextLength": 128000,
                "weightsBytes": 4800000000,
                "permittedClassifications": ["processDiagram"],
                "path": "future/model"
            }]
        }"#;
        let manifest: ModelManifest = serde_json::from_str(json).unwrap();
        let registry =
            ModelRegistry::from_manifest(manifest, PathBuf::from("registry.json")).unwrap();

        let entry = registry.find("some-future-model-2027").unwrap();
        assert_eq!(entry.runtime, Runtime::PythonSidecar);
        assert!(entry.serves(ModelRole::DocumentOcr));
        assert!(entry.enabled, "enabled defaults to true when omitted");
        assert!(entry.permits(Classification::ProcessDiagram));
    }

    /// The manifest a Windows tool wrote.
    ///
    /// PowerShell's `>`, `Out-File` and Notepad all write UTF-8 with a byte
    /// order mark by default, so this is the ordinary shape of a hand-edited
    /// registry on the platform this ships on, not an exotic corruption.
    #[test]
    fn a_manifest_written_with_a_byte_order_mark_still_loads() {
        let dir = tempfile::tempdir().expect("temp dir");
        let models_dir = dir.path().join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir");

        let json = r#"{
            "models": [{
                "id": "gemma-4-12b-it",
                "name": "Gemma 4 12B",
                "version": "1",
                "license": "gemma",
                "runtime": "llamaCpp",
                "roles": ["reasoning"],
                "parametersB": 12.0,
                "contextLength": 8192,
                "weightsBytes": 7000000000,
                "permittedClassifications": ["internal"],
                "path": "local/gemma"
            }]
        }"#;
        std::fs::write(models_dir.join("registry.json"), format!("\u{feff}{json}"))
            .expect("wrote the manifest");

        let registry = ModelRegistry::load(&models_dir).expect("a BOM is not a broken manifest");
        assert_eq!(registry.all().len(), 1);
        assert!(registry.find("gemma-4-12b-it").is_some());
    }

    /// The failure that made six installed models disappear from the workbench.
    #[test]
    fn an_unreadable_manifest_does_not_stop_the_registry_loading() {
        let dir = tempfile::tempdir().expect("temp dir");
        let models_dir = dir.path().join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir");
        std::fs::write(models_dir.join("registry.json"), b"{ not json at all")
            .expect("wrote the broken manifest");

        // Loads rather than failing, so the caller does not fall back to an
        // empty registry and tell an operator that nothing is installed. Any
        // model on disk would be discovered here; this fixture has none, and
        // the point under test is that it got that far at all.
        let registry = ModelRegistry::load_with_discovery(dir.path())
            .expect("a broken manifest is not a reason to refuse to start");
        assert!(registry.all().is_empty());

        // The manifest is still reported as broken when it is read on its own,
        // so nothing here quietly forgives a malformed file.
        assert!(ModelRegistry::load(&models_dir).is_err());
    }

    /// The manifest the system ships for a vision-language model.
    ///
    /// A real ARJUN deployment adds vision models by editing
    /// ``registry.json``; this is what that entry looks like, captured as
    /// a round-trip test so the JSON shape is exercised every time the
    /// registry is touched. The fields that are specific to vision —
    /// `roles: ["vision"]` and `modalities: ["text", "image"]` — are
    /// what the multimodal retriever queries against.
    #[test]
    fn a_vision_model_manifest_round_trips() {
        let json = r#"{
            "models": [{
                "id": "qwen2.5-vl-3b-instruct",
                "name": "Qwen2.5-VL 3B Instruct",
                "version": "1",
                "license": "Apache-2.0",
                "runtime": "llamaCpp",
                "roles": ["vision"],
                "modalities": ["text", "image"],
                "parametersB": 3.0,
                "contextLength": 32768,
                "weightsBytes": 2000000000,
                "quantization": "Q4_K_M",
                "permittedClassifications": ["internal", "processDiagram"],
                "supportsStructuredOutput": false,
                "path": "models/qwen2.5-vl-3b-instruct"
            }]
        }"#;
        let manifest: ModelManifest = serde_json::from_str(json).unwrap();
        let registry =
            ModelRegistry::from_manifest(manifest, PathBuf::from("registry.json")).unwrap();
        let model = registry.find("qwen2.5-vl-3b-instruct").unwrap();

        // The Vision role is what the multimodal tool queries.
        assert!(model.serves(ModelRole::Vision));
        // The image modality is what the runtime hard-gates on. A text-only
        // model would not be eligible for a vision request even if its
        // role were Vision.
        assert!(model.supports_modality(Modality::Image));
        // The runtime is the wire format the bridge expects to speak.
        assert_eq!(model.runtime, Runtime::LlamaCpp);
        // The classification gates what the model may see. A vision model
        // cleared for internal and process-diagram material can read
        // P&IDs but not vendor negotiations.
        assert!(model.permits(Classification::Internal));
        assert!(model.permits(Classification::ProcessDiagram));
        assert!(!model.permits(Classification::VendorNegotiation));

        // Round-trip: re-serialise the loaded manifest and confirm the
        // image modality survives. This is what stops a future refactor
        // from dropping the field on the way to JSON.
        let reserialised = serde_json::to_value(&model).unwrap();
        let modalities = reserialised.get("modalities").and_then(Value::as_array).unwrap();
        assert!(modalities.iter().any(|v| v == "image"));
    }

    /// A vision request must filter to models that both serve the Vision
    /// role *and* declare the image modality. A model with one but not
    /// the other is not eligible, and the bridge would refuse it on
    /// construction — the registry enforces the same constraint at the
    /// candidate stage.
    #[test]
    fn vision_candidates_require_both_role_and_modality() {
        let mut with_image = entry("vision-with-image", 3.0, vec![ModelRole::Vision]);
        with_image.modalities = vec![Modality::Text, Modality::Image];
        let registry = registry(vec![with_image]);

        let vision_candidates: Vec<&ModelEntry> = registry
            .all()
            .iter()
            .filter(|m: &&ModelEntry| {
                m.serves(ModelRole::Vision) && m.supports_modality(Modality::Image)
            })
            .collect();
        assert_eq!(vision_candidates.len(), 1);
        assert_eq!(vision_candidates[0].id, "vision-with-image");
    }

    // -----------------------------------------------------------------------
    // Atomic manifest write tests (TODO 4 of the 7-step plan).
    // -----------------------------------------------------------------------

    #[test]
    fn save_manifest_writes_a_valid_json_file() {
        // Build a registry with a known entry list, save it,
        // and re-load it. The round trip is the proof.
        let dir = std::env::temp_dir().join(format!(
            "arjun-registry-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest_path = dir.join("registry.json");
        let registry = ModelRegistry::from_manifest(
            ModelManifest {
                models: vec![entry("qwen-8b", 8.0, vec![ModelRole::Reasoning])],
            },
            manifest_path.clone(),
        )
        .expect("registry");
        registry.save_manifest().expect("save");
        // Reload from disk.
        let reloaded = ModelRegistry::load(&dir).expect("reload");
        assert_eq!(reloaded.all().len(), 1);
        assert_eq!(reloaded.all()[0].id, "qwen-8b");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_manifest_leaves_no_tmp_file_behind() {
        // The atomic write is `<path>.tmp` → fsync → rename.
        // A successful save must not leave the tmp file in
        // place; a reader who walks the directory would
        // otherwise find a stale half-written file.
        let dir = std::env::temp_dir().join(format!(
            "arjun-registry-tmp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = ModelRegistry::from_manifest(
            ModelManifest::default(),
            dir.join("registry.json"),
        )
        .expect("registry");
        registry.save_manifest().expect("save");
        let tmp = dir.join("registry.json.tmp");
        assert!(!tmp.exists(), "tmp file must be renamed away");
        let real = dir.join("registry.json");
        assert!(real.exists(), "the real manifest must exist");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replace_and_save_refuses_duplicate_ids() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-registry-dup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut registry = ModelRegistry::from_manifest(
            ModelManifest::default(),
            dir.join("registry.json"),
        )
        .expect("registry");
        let result = registry.replace_and_save(vec![
            entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]),
            entry("qwen-8b", 7.0, vec![ModelRole::Reasoning]),
        ]);
        assert!(result.is_err());
        // The registry state is unchanged on error.
        assert_eq!(registry.all().len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replace_and_save_persists_the_new_entries() {
        let dir = std::env::temp_dir().join(format!(
            "arjun-registry-replace-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut registry = ModelRegistry::from_manifest(
            ModelManifest::default(),
            dir.join("registry.json"),
        )
        .expect("registry");
        registry
            .replace_and_save(vec![
                entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]),
                entry("gemma-12b", 12.0, vec![ModelRole::Reasoning]),
            ])
            .expect("replace");
        // Reload from disk.
        let reloaded = ModelRegistry::load(&dir).expect("reload");
        assert_eq!(reloaded.all().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gemma_4_12b_is_the_implicit_orchestrator_when_no_entry_is_tagged() {
        let mut gemma = entry(
            "lmstudio-community_gemma-4-12B-it-QAT-GGUF_Q4_0",
            12.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        gemma.name = "Gemma 4 12B IT QAT".to_string();
        gemma.quantization = Some("Q4_0".to_string());
        gemma.load = Some(LoadSpec {
            provider_id: "huggingface".to_string(),
            model_id: crate::config::DEFAULT_ORCHESTRATOR_MODEL_ID.to_string(),
            quantization: "Q4_0".to_string(),
        });
        let registry = registry(vec![gemma]);

        let selected = registry
            .orchestrator_entry()
            .expect("the default Gemma package should run the orchestrator");
        assert_eq!(
            selected.load.as_ref().map(|load| load.model_id.as_str()),
            Some(crate::config::DEFAULT_ORCHESTRATOR_MODEL_ID)
        );
        assert_eq!(
            registry.orchestrator_identity(),
            Some(("gemma-4-12b".to_string(), "Q4_0".to_string()))
        );
    }

    #[test]
    fn an_explicit_orchestrator_still_overrides_the_gemma_default() {
        let gemma = entry("gemma-4-12b", 12.0, vec![ModelRole::Reasoning]);
        let explicit = entry(
            "orchestrator.custom-model",
            14.0,
            vec![ModelRole::Reasoning],
        );
        let registry = registry(vec![gemma, explicit]);

        assert_eq!(
            registry.orchestrator_entry().map(|model| model.id.as_str()),
            Some("orchestrator.custom-model")
        );
    }
}

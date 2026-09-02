//! Finding skills, deciding which are usable, and loading one when it is.
//!
//! ## The two-stage read, and why it is two stages
//!
//! Requirement 4: *metadata-only discovery. Do not load every full `SKILL.md`
//! into every model prompt.* A registry that read whole skills at start would
//! put every skill's instructions in front of every model, which is both
//! wasteful and — more importantly — the thing that makes a poisoned skill
//! effective without anybody choosing to use it.
//!
//! So discovery reads each file, hashes it, keeps the frontmatter, and **drops
//! the body**. [`Snapshot`] has nowhere to put a body; the type will not hold
//! one. The body is read again, from disk, only when [`SkillRegistry::load`] is
//! called for a specific skill that has passed its checks.
//!
//! Hashing does require reading the whole file once. That is not the same
//! thing: bytes that pass through a hasher and are dropped are not bytes in a
//! prompt, and the hash is what makes the trust check possible at all.
//!
//! ## What "signed" means here, stated plainly
//!
//! It means: *this exact content is on an operator-maintained allowlist.* The
//! trust list is `skills/trusted.json`, and it pairs a skill name with the
//! SHA-256 of its `SKILL.md`.
//!
//! That is an integrity control, not a cryptographic signature. It detects a
//! skill that was changed after somebody reviewed it, and it requires a
//! deliberate act to trust a new one. It does **not** prove who wrote the
//! skill, and it is only as good as the file permissions on the trust list —
//! an attacker who can write `SKILL.md` can usually write `trusted.json` too.
//!
//! Saying so is the point. A real signature needs a key to verify against and
//! somewhere safe to keep it, neither of which exists here yet; calling a hash
//! allowlist a signature would claim a property this does not have.
//!
//! ## Hot reload
//!
//! [`SkillRegistry::reload`] builds a whole new snapshot and swaps it in one
//! write. A run that has loaded a skill holds an `Arc<LoadedSkill>` — the
//! definition it started with, pinned for as long as it needs it. So a reload
//! during a run cannot change what that run is executing, and the boundary is
//! structural rather than a rule about when to call reload.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::Session;
use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;
use crate::sovereignty::mode::OperatingMode;

use super::containment;
use super::frontmatter;
use super::manifest::{self, Quarantine, SkillCard, SkillManifest};
use super::narrowing::{self, Narrowed};

/// Largest `SKILL.md` that will be read at all.
///
/// A skill is instructions, not a corpus. The limit exists because discovery
/// walks every directory on the machine, and one enormous file should not be
/// able to make that slow.
pub const MAX_SKILL_BYTES: u64 = 512 * 1024;

/// Largest reference, script or asset that will be read.
pub const MAX_REFERENCE_BYTES: u64 = 2 * 1024 * 1024;

/// The file an operator edits to trust a skill.
pub const TRUST_FILE: &str = "trusted.json";

/// One entry in the operator's trust list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustedSkill {
    pub name: String,
    /// SHA-256 of the whole `SKILL.md`, lowercase hex.
    pub sha256: String,
    /// Free text: who reviewed it and when. Never consulted for a decision.
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustList {
    #[serde(default)]
    pub trusted: Vec<TrustedSkill>,
}

impl TrustList {
    fn expected_for(&self, name: &str) -> Option<&str> {
        self.trusted
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.sha256.trim())
    }
}

/// A skill as discovery left it: metadata, and no body.
#[derive(Debug, Clone)]
struct Entry {
    manifest: SkillManifest,
    root: PathBuf,
    /// Set when the skill is unusable regardless of who is asking.
    quarantine: Option<Quarantine>,
}

/// Everything known about the skills on this machine at one moment.
///
/// Deliberately has no field that could hold a skill's body.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub discovered_at: DateTime<Utc>,
    pub root: PathBuf,
    entries: BTreeMap<String, Entry>,
    /// Directories that looked like skills and could not be read at all.
    pub unreadable: Vec<(String, Quarantine)>,
}

impl Snapshot {
    /// Cards for everything found, quarantined or not.
    ///
    /// Includes the directories whose `SKILL.md` did not validate. They have no
    /// manifest to describe them and they are listed anyway: an operator with a
    /// broken skill needs to see that it is there.
    pub fn cards(&self) -> Vec<SkillCard> {
        self.entries
            .values()
            .map(|entry| SkillCard::of(&entry.manifest, entry.quarantine.clone()))
            .chain(
                self.unreadable
                    .iter()
                    .map(|(folder, reason)| SkillCard::unreadable(folder, reason.clone())),
            )
            .collect()
    }

    /// How many skill directories were found, readable or not.
    pub fn count(&self) -> usize {
        self.entries.len() + self.unreadable.len()
    }

    /// How many are usable as they stand, ignoring who is asking.
    pub fn available(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.quarantine.is_none())
            .count()
    }
}

/// Who is asking, and what the machine is doing right now.
///
/// Passed in rather than reached for, so eligibility is a pure function of its
/// inputs and every combination can be driven in a test.
pub struct SkillContext<'a> {
    pub session: &'a Session,
    pub mode: OperatingMode,
    /// Tools the run already permits. A skill can only narrow this.
    pub run_permits: &'a [ToolName],
}

/// Why a skill could not be used right now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "refusal")]
pub enum LoadRefusal {
    /// No skill of that name was found.
    Unknown { name: String },
    /// It is quarantined. The reason says what would resolve it.
    Quarantined { reason: Quarantine },
    /// The signed-in person is not cleared for the material it is written for.
    NotCleared { classification: String },
    /// Its file changed between discovery and this load.
    ChangedOnDisk { expected: String, found: String },
    /// It could not be read from disk.
    Unreadable { detail: String },
}

impl LoadRefusal {
    pub fn explain(&self) -> String {
        match self {
            LoadRefusal::Unknown { name } => format!("There is no skill called {name:?}."),
            LoadRefusal::Quarantined { reason } => reason.explain(),
            LoadRefusal::NotCleared { classification } => format!(
                "That skill is written for {classification} material, which you are not cleared \
                 for."
            ),
            LoadRefusal::ChangedOnDisk { .. } => {
                "That skill's file changed after it was checked. It was not loaded. Reload the \
                 skills and review the change."
                    .to_string()
            }
            LoadRefusal::Unreadable { detail } => {
                format!("That skill could not be read: {detail}")
            }
        }
    }
}

/// A skill, loaded, with the body and the narrowing it produced.
///
/// Held behind an `Arc` by whatever is using it, so a reload cannot change the
/// definition a run is part-way through.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    pub manifest: SkillManifest,
    /// The instructions, verbatim. **Untrusted text**: it is guidance for the
    /// model and is never consulted for a decision. See the module note on
    /// [`super`].
    pub body: String,
    pub root: PathBuf,
    /// What the run may use while this is loaded. Never wider than what it
    /// permitted before.
    pub narrowed: Narrowed,
    pub loaded_at: DateTime<Utc>,
}

impl LoadedSkill {
    /// What goes in the run manifest. Requirement 7, in one place so a caller
    /// cannot record half of it.
    pub fn use_record(&self) -> SkillUse {
        SkillUse {
            name: self.manifest.name.clone(),
            version: self.manifest.version.clone(),
            sha256: self.manifest.sha256.clone(),
            license: self.manifest.license.clone(),
            author: self.manifest.author.clone(),
            classification: self.manifest.classification,
            network: self.manifest.network.as_str().to_string(),
            approval_class: self.manifest.approval_class.as_str().to_string(),
            requires_binaries: self.manifest.requires_binaries.clone(),
            signature: Signature::TrustedHash,
            tools_granted: self
                .narrowed
                .tools
                .iter()
                .map(|tool| tool.as_str().to_string())
                .collect(),
            tools_refused: self
                .narrowed
                .refused
                .iter()
                .map(|tool| tool.as_str().to_string())
                .collect(),
            loaded_at: self.loaded_at,
        }
    }
}

/// How a skill's integrity was established.
///
/// One variant today, named for what it actually is rather than for what a
/// reader might assume. See the module note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Signature {
    /// Its content hash is on the operator's trust list.
    TrustedHash,
}

/// One skill's use, as the run manifest records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillUse {
    pub name: String,
    pub version: String,
    pub sha256: String,
    pub license: String,
    pub author: String,
    pub classification: Classification,
    pub network: String,
    pub approval_class: String,
    pub requires_binaries: Vec<String>,
    pub signature: Signature,
    /// What the skill was actually allowed to use, after narrowing.
    pub tools_granted: Vec<String>,
    /// What it asked for and did not get.
    pub tools_refused: Vec<String>,
    pub loaded_at: DateTime<Utc>,
}

/// The skills on this machine.
pub struct SkillRegistry {
    root: PathBuf,
    snapshot: RwLock<Arc<Snapshot>>,
}

impl SkillRegistry {
    /// Discovers what is in `root` and holds it.
    ///
    /// A missing directory is an empty registry, not a failure: a deployment
    /// with no skills is a legitimate deployment.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let snapshot = Arc::new(discover(&root));
        Self {
            root,
            snapshot: RwLock::new(snapshot),
        }
    }

    /// The current snapshot. Cheap, and safe to hold across a reload.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot
            .read()
            .map(|guard| Arc::clone(&guard))
            .unwrap_or_else(|poisoned| Arc::clone(&poisoned.into_inner()))
    }

    /// Re-reads the directory and swaps the snapshot in.
    ///
    /// Safe at any moment *because* of what it does not do: it never mutates
    /// the snapshot a caller is holding. A run part-way through a tool call is
    /// working from an `Arc<LoadedSkill>` it took earlier, and that definition
    /// does not change underneath it.
    pub fn reload(&self) -> Arc<Snapshot> {
        let fresh = Arc::new(discover(&self.root));
        if let Ok(mut guard) = self.snapshot.write() {
            *guard = Arc::clone(&fresh);
        }
        fresh
    }

    /// Concise metadata for the skills this person may use right now.
    ///
    /// The whole of requirement 10: enough to choose, and no instructions. A
    /// caller decides from these and then asks for one by name.
    ///
    /// `query` is matched against the name and description, case-insensitively.
    /// An empty query returns everything eligible.
    pub fn search(&self, query: &str, context: &SkillContext<'_>) -> Vec<SkillCard> {
        let needle = query.trim().to_lowercase();
        let snapshot = self.snapshot();

        let mut found: Vec<SkillCard> = snapshot
            .entries
            .values()
            .filter(|entry| cleared_for(context.session, entry.manifest.classification))
            .filter(|entry| {
                needle.is_empty()
                    || entry.manifest.name.to_lowercase().contains(&needle)
                    || entry.manifest.description.to_lowercase().contains(&needle)
            })
            .map(|entry| {
                // The contextual checks are applied here rather than baked into
                // the snapshot, because they depend on things that change
                // without the files changing — the mode, and who is signed in.
                let quarantine = entry
                    .quarantine
                    .clone()
                    .or_else(|| contextual_quarantine(&entry.manifest, context.mode));
                SkillCard::of(&entry.manifest, quarantine)
            })
            .chain(
                // Directories that did not validate. Matched on the folder name
                // alone, because that is all that is known about them.
                snapshot
                    .unreadable
                    .iter()
                    .filter(|_| cleared_for(context.session, Classification::Internal))
                    .filter(|(folder, _)| {
                        needle.is_empty() || folder.to_lowercase().contains(&needle)
                    })
                    .map(|(folder, reason)| SkillCard::unreadable(folder, reason.clone())),
            )
            .collect();

        // Available first, then alphabetical. A quarantined skill is still
        // listed — an operator needs to see that it exists and why it cannot be
        // used, and hiding it would look like the skill was never installed.
        found.sort_by(|a, b| {
            b.is_available()
                .cmp(&a.is_available())
                .then_with(|| a.name.cmp(&b.name))
        });
        found
    }

    /// Loads one skill, after checking it may be used.
    ///
    /// Re-reads the file and re-checks its hash: discovery may have been
    /// minutes ago, and the property being relied on is that the bytes about to
    /// be put in front of a model are the bytes somebody trusted.
    pub fn load(
        &self,
        name: &str,
        context: &SkillContext<'_>,
    ) -> Result<Arc<LoadedSkill>, LoadRefusal> {
        let snapshot = self.snapshot();
        let entry = snapshot
            .entries
            .get(name)
            .ok_or_else(|| LoadRefusal::Unknown {
                name: name.to_string(),
            })?;

        if let Some(reason) = entry.quarantine.clone() {
            return Err(LoadRefusal::Quarantined { reason });
        }
        if let Some(reason) = contextual_quarantine(&entry.manifest, context.mode) {
            return Err(LoadRefusal::Quarantined { reason });
        }
        if !cleared_for(context.session, entry.manifest.classification) {
            return Err(LoadRefusal::NotCleared {
                classification: entry.manifest.classification.label().to_string(),
            });
        }

        let path = entry.root.join("SKILL.md");
        let source = read_capped(&path, MAX_SKILL_BYTES)
            .map_err(|detail| LoadRefusal::Unreadable { detail })?;
        let found = sha256_of(&source);
        if found != entry.manifest.sha256 {
            return Err(LoadRefusal::ChangedOnDisk {
                expected: entry.manifest.sha256.clone(),
                found,
            });
        }

        let split = frontmatter::split(&source).map_err(|error| LoadRefusal::Unreadable {
            detail: error.to_string(),
        })?;

        Ok(Arc::new(LoadedSkill {
            manifest: entry.manifest.clone(),
            body: split.body.to_string(),
            root: entry.root.clone(),
            narrowed: narrowing::narrow(context.run_permits, &entry.manifest.allowed_tools),
            loaded_at: Utc::now(),
        }))
    }

    /// Reads one of a loaded skill's own files.
    ///
    /// Requirement 6: only when needed, and only from inside the skill. The
    /// containment check is [`containment::resolve`]; this adds the size cap
    /// and the reading.
    pub fn read_reference(&self, skill: &LoadedSkill, named: &str) -> Result<String, String> {
        let path = containment::resolve(&skill.root, named).map_err(|refusal| {
            format!("{named:?} was not read: {refusal}")
        })?;
        read_capped(&path, MAX_REFERENCE_BYTES)
    }
}

/// Whether the signed-in person may see material of this kind.
fn cleared_for(session: &Session, classification: Classification) -> bool {
    classification
        .cleared_roles()
        .iter()
        .any(|role| session.user.roles.contains(role))
}

/// Checks that depend on the moment rather than on the file.
fn contextual_quarantine(
    manifest: &SkillManifest,
    mode: OperatingMode,
) -> Option<Quarantine> {
    // A skill that wants the network is not usable while confidential work is
    // permitted. Not because it could reach anywhere — the broker refuses
    // regardless — but because a skill whose author thought it needed the
    // network is one somebody should look at before it touches this material.
    if manifest.network != super::manifest::NetworkNeed::None
        && mode.permits_confidential_data()
    {
        return Some(Quarantine::RequiresNetwork {
            need: manifest.network.as_str().to_string(),
        });
    }
    None
}

/// Walks the skills directory once.
fn discover(root: &Path) -> Snapshot {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    let mut unreadable: Vec<(String, Quarantine)> = Vec::new();

    let trust = read_trust_list(root);
    let running = env!("CARGO_PKG_VERSION");

    let Ok(directory) = std::fs::read_dir(root) else {
        // No directory at all is an empty registry. A deployment with no skills
        // is legitimate, and it is not this module's business to create one.
        return Snapshot {
            discovered_at: Utc::now(),
            root: root.to_path_buf(),
            entries,
            unreadable,
        };
    };

    for child in directory.filter_map(Result::ok) {
        let path = child.path();
        if !path.is_dir() {
            continue;
        }
        let Some(folder) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        // A directory with no SKILL.md is not a skill and not an error: it may
        // be anything an operator keeps beside them.
        let definition = path.join("SKILL.md");
        if !definition.is_file() {
            continue;
        }

        match read_one(&definition, folder, running, &trust) {
            Ok((manifest, quarantine)) => {
                entries.insert(
                    manifest.name.clone(),
                    Entry {
                        manifest,
                        root: path.clone(),
                        quarantine,
                    },
                );
            }
            Err(reason) => unreadable.push((folder.to_string(), reason)),
        }
    }

    Snapshot {
        discovered_at: Utc::now(),
        root: root.to_path_buf(),
        entries,
        unreadable,
    }
}

/// Reads one skill's metadata, and decides whether it is quarantined.
///
/// The body is read into memory to be hashed and is then dropped. Nothing that
/// leaves this function can hold it.
fn read_one(
    definition: &Path,
    folder: &str,
    running: &str,
    trust: &TrustList,
) -> Result<(SkillManifest, Option<Quarantine>), Quarantine> {
    let source = read_capped(definition, MAX_SKILL_BYTES)
        .map_err(|detail| Quarantine::Malformed { detail })?;
    let sha256 = sha256_of(&source);

    let split = frontmatter::split(&source).map_err(|error| Quarantine::Malformed {
        detail: error.to_string(),
    })?;
    let document = frontmatter::parse(split.frontmatter).map_err(|error| Quarantine::Malformed {
        detail: error.to_string(),
    })?;

    // `split.body` goes out of scope here along with `source`. There is
    // deliberately no path from this function to a caller that carries it.
    let manifest = manifest::validate(&document, folder, &sha256)?;

    Ok((manifest.clone(), static_quarantine(&manifest, running, trust)))
}

/// Checks that depend only on the file and the machine.
fn static_quarantine(
    manifest: &SkillManifest,
    running: &str,
    trust: &TrustList,
) -> Option<Quarantine> {
    if !manifest::satisfies(&manifest.requires_arjun, running) {
        return Some(Quarantine::Incompatible {
            requires: manifest.requires_arjun.clone(),
            running: running.to_string(),
        });
    }

    for binary in &manifest.requires_binaries {
        if !on_path(binary) {
            return Some(Quarantine::MissingBinary {
                binary: binary.clone(),
            });
        }
    }

    match trust.expected_for(&manifest.name) {
        None => Some(Quarantine::Unsigned {
            sha256: manifest.sha256.clone(),
        }),
        Some(expected) if expected.eq_ignore_ascii_case(&manifest.sha256) => None,
        Some(expected) => Some(Quarantine::Tampered {
            expected: expected.to_string(),
            found: manifest.sha256.clone(),
        }),
    }
}

/// Reads the operator's trust list.
///
/// A missing or unreadable list means nothing is trusted, which quarantines
/// every skill. That is the safe direction: the alternative reading — an
/// unreadable list means everything is fine — is how a deleted file becomes a
/// silent grant.
fn read_trust_list(root: &Path) -> TrustList {
    std::fs::read_to_string(root.join(TRUST_FILE))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Whether a binary is on `PATH`.
///
/// Looked up rather than executed. Running `--version` on every declared binary
/// at discovery would start processes named by an untrusted file, which is a
/// large thing to do to answer a small question.
fn on_path(binary: &str) -> bool {
    // A name with a separator in it is not a binary name; treat it as absent
    // rather than resolving it, so a skill cannot use this to probe the disk.
    if binary.contains('/') || binary.contains('\\') {
        return false;
    }
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    let extensions: Vec<String> = std::env::var("PATHEXT")
        .map(|raw| raw.split(';').map(|e| e.to_lowercase()).collect())
        .unwrap_or_else(|_| vec![String::new()]);

    std::env::split_paths(&path).any(|directory| {
        extensions.iter().any(|extension| {
            directory
                .join(format!("{binary}{extension}"))
                .is_file()
        })
    })
}

/// Reads a file, refusing one above the cap without reading it.
fn read_capped(path: &Path, cap: u64) -> Result<String, String> {
    let size = std::fs::metadata(path)
        .map_err(|error| format!("{} could not be opened: {error}", path.display()))?
        .len();
    if size > cap {
        return Err(format!(
            "{} is {size} bytes, above the {cap} byte limit",
            path.display()
        ));
    }
    std::fs::read_to_string(path)
        .map_err(|error| format!("{} could not be read: {error}", path.display()))
}

/// SHA-256 of some text, lowercase hex, line endings normalised first.
///
/// The normalisation is what makes this a hash of the *skill* rather than of
/// the checkout it came from. Git is routinely configured to write CRLF into a
/// Windows working copy and LF everywhere else, so hashing the bytes as they
/// sit on disk gives one answer on a developer's machine and a different one
/// in CI — for the same reviewed content, with no edit in between.
///
/// That is not hypothetical. The shipped trust list ended up with five skills
/// hashed as LF and five as CRLF, and on a Windows checkout the LF five were
/// quarantined with "its contents changed after it was trusted". Nothing had
/// changed. Regenerating the list against the local bytes would only have
/// moved the failure to the other five on the next Linux checkout.
///
/// A carriage return carries no meaning in a Markdown skill definition, so
/// dropping it before hashing loses nothing an operator reviewed.
pub fn sha256_of(text: &str) -> String {
    let mut hasher = Sha256::new();
    for (index, line) in text.split("\r\n").enumerate() {
        if index > 0 {
            hasher.update(b"\n");
        }
        hasher.update(line.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

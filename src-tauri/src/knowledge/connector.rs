//! Where the organisation's documents actually live.
//!
//! PS 26117 asks for a *connector*, not a file uploader. The distinction is the
//! whole point: a refinery's procedures sit on a network share that half the
//! plant already has mapped, and a product that can only read what somebody
//! dragged into it will be permanently out of date through that friction alone.
//!
//! ## Two sources, one code path
//!
//! A Windows UNC path — `\\\\plant-fs\\engineering\\SOPs` — is reachable through
//! the ordinary filesystem API when the signed-in user has access. So the share
//! connector is the folder connector with a different root, and the difference
//! is one of *policy*, not mechanism: a share is somebody else's data, so it is
//! opened read-only and never written back to.
//!
//! Worth stating plainly because it looks like a gap: reading a share does not
//! go through the network broker. The broker owns outbound HTTP, and a file read
//! over SMB is not HTTP — it is the operating system's own file access, to a host
//! inside the plant, using the user's existing credentials. Routing it through an
//! HTTP chokepoint would prove nothing and break the thing that makes it work.
//!
//! ## Read-only, always
//!
//! Nothing here writes to a source. A connector that could modify the share
//! would make ARJUN a way to alter controlled documents, which is a much larger
//! claim than "reads them", and one no site would sign off.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::identity::Role;
use crate::policy::Classification;

/// Where a collection's documents come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SourceKind {
    /// A directory on this machine.
    LocalFolder,
    /// A UNC path to a share inside the organisation.
    NetworkShare,
}

impl SourceKind {
    pub const fn label(self) -> &'static str {
        match self {
            SourceKind::LocalFolder => "local folder",
            SourceKind::NetworkShare => "network share",
        }
    }
}

/// File types worth reading. Anything else in a folder is ignored rather than
/// failed on — a share full of CAD files and spreadsheets should not stop a
/// sync because ARJUN cannot read a `.dwg` yet.
const READABLE_EXTENSIONS: &[&str] = &["pdf", "docx", "txt", "md", "html", "htm"];

/// A set of documents with one owner and one sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Collection {
    pub id: String,
    /// What people call it: "Maintenance SOPs", "Inspection reports 2026".
    pub name: String,
    pub kind: SourceKind,
    pub root: PathBuf,
    /// Who is accountable for what is in here.
    pub owner: String,
    /// Applies to every document in the collection. A collection is the unit of
    /// sensitivity because a folder of vendor contracts is uniformly commercial,
    /// and asking somebody to classify a thousand files individually guarantees
    /// they will not.
    pub classification: Classification,
    /// Roles permitted to search this collection at all. Narrower than the
    /// classification's own clearance where a site wants it narrower; never
    /// wider, which [`Collection::effective_roles`] enforces.
    #[serde(default)]
    pub restricted_to_roles: Vec<Role>,
    /// How long documents are kept. `None` means indefinitely.
    #[serde(default)]
    pub retention_days: Option<u32>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl Collection {
    /// Roles that may actually search this collection.
    ///
    /// The intersection of the classification's clearance and any extra
    /// restriction on the collection. Intersection rather than union, so a
    /// collection setting can only ever narrow access — a misconfigured
    /// collection cannot hand out clearance the classification withholds.
    pub fn effective_roles(&self) -> Vec<Role> {
        let cleared = self.classification.cleared_roles();
        if self.restricted_to_roles.is_empty() {
            return cleared.to_vec();
        }
        cleared
            .iter()
            .filter(|role| self.restricted_to_roles.contains(role))
            .copied()
            .collect()
    }

    pub fn readable_by(&self, roles: &[Role]) -> bool {
        self.effective_roles().iter().any(|r| roles.contains(r))
    }
}

/// One file found in a collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredFile {
    pub path: PathBuf,
    /// Relative to the collection root, for display and for stable identity.
    pub relative_path: String,
    pub byte_size: u64,
    /// Seconds since the epoch. Compared against the last sync to spot changes.
    pub modified_at: u64,
}

/// What changed since the last time a collection was read.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncPlan {
    pub added: Vec<DiscoveredFile>,
    pub changed: Vec<DiscoveredFile>,
    /// Present last time, gone now. Their passages should be retired.
    pub removed: Vec<String>,
    pub unchanged: usize,
    /// Files skipped because nothing here can read them.
    pub skipped: Vec<String>,
}

impl SyncPlan {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.changed.is_empty() && self.removed.is_empty()
    }

    /// One line for the collection screen.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return format!("Nothing has changed. {} document(s) held.", self.unchanged);
        }
        let mut parts = Vec::new();
        if !self.added.is_empty() {
            parts.push(format!("{} new", self.added.len()));
        }
        if !self.changed.is_empty() {
            parts.push(format!("{} updated", self.changed.len()));
        }
        if !self.removed.is_empty() {
            parts.push(format!("{} removed", self.removed.len()));
        }
        format!("{}. {} unchanged.", parts.join(", "), self.unchanged)
    }
}

/// True when `path` stays inside `root` once `..` is resolved.
///
/// A collection root is a boundary, and a link or a crafted relative path that
/// walks out of it would quietly widen what the collection contains — including
/// past whatever classification the collection carries.
fn stays_within(root: &Path, path: &Path) -> bool {
    fn normalise(path: &Path) -> Option<PathBuf> {
        let mut out = PathBuf::new();
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    if !out.pop() {
                        return None;
                    }
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        Some(out)
    }

    match (normalise(root), normalise(path)) {
        (Some(root), Some(path)) => path.starts_with(&root),
        _ => false,
    }
}

fn is_readable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| READABLE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn modified_seconds(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Walks a collection and reports what is in it.
///
/// Read-only, and bounded: symbolic links are not followed, because a link into
/// a user's home directory would silently pull personal files into a collection
/// classified for engineering procedures.
pub fn discover(collection: &Collection) -> Result<(Vec<DiscoveredFile>, Vec<String>)> {
    let root = &collection.root;
    if !root.is_dir() {
        anyhow::bail!(
            "{} is not reachable. For a {}, check the path exists and that you are signed in \
             with access to it.",
            root.display(),
            collection.kind.label()
        );
    }

    let mut found = Vec::new();
    let mut skipped = Vec::new();
    let mut queue = vec![root.clone()];

    while let Some(directory) = queue.pop() {
        let entries = std::fs::read_dir(&directory)
            .with_context(|| format!("could not read {}", directory.display()))?;

        for entry in entries.flatten() {
            let path = entry.path();

            // Checked before anything else: a link out of the collection would
            // widen it past its own classification.
            if !stays_within(root, &path) {
                skipped.push(format!("{} (outside the collection)", path.display()));
                continue;
            }

            let Ok(metadata) = entry.metadata() else {
                continue;
            };

            if metadata.is_symlink() {
                skipped.push(format!("{} (symbolic link, not followed)", path.display()));
                continue;
            }

            if metadata.is_dir() {
                queue.push(path);
                continue;
            }

            if !is_readable(&path) {
                skipped.push(path.display().to_string());
                continue;
            }

            let relative = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| path.to_string_lossy().to_string());

            found.push(DiscoveredFile {
                relative_path: relative,
                byte_size: metadata.len(),
                modified_at: modified_seconds(&metadata),
                path,
            });
        }
    }

    found.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    skipped.sort();
    Ok((found, skipped))
}

/// What was held after the previous sync: relative path to (size, modified).
pub type PreviousState = BTreeMap<String, (u64, u64)>;

/// Works out what has to be re-read.
///
/// Compared on size and modification time rather than by hashing everything: a
/// share holding ten thousand documents cannot be fully hashed on every sync,
/// and a file whose size and timestamp are both unchanged has almost certainly
/// not changed. Content is hashed later, when a changed file is actually read,
/// so an identical re-save still resolves to the same document in the store.
pub fn plan_sync(discovered: &[DiscoveredFile], previous: &PreviousState) -> SyncPlan {
    let mut plan = SyncPlan::default();

    for file in discovered {
        match previous.get(&file.relative_path) {
            None => plan.added.push(file.clone()),
            Some((size, modified)) => {
                if *size != file.byte_size || *modified != file.modified_at {
                    plan.changed.push(file.clone());
                } else {
                    plan.unchanged += 1;
                }
            }
        }
    }

    let present: std::collections::BTreeSet<&String> =
        discovered.iter().map(|f| &f.relative_path).collect();
    for known in previous.keys() {
        if !present.contains(known) {
            plan.removed.push(known.clone());
        }
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collection(root: &Path, classification: Classification) -> Collection {
        Collection {
            id: "sops".into(),
            name: "Maintenance SOPs".into(),
            kind: SourceKind::LocalFolder,
            root: root.to_path_buf(),
            owner: "A. Fernandes".into(),
            classification,
            restricted_to_roles: Vec::new(),
            retention_days: None,
            enabled: true,
        }
    }

    fn write(root: &Path, relative: &str, contents: &[u8]) -> PathBuf {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn a_folder_of_documents_is_discovered_recursively() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "sop-4.pdf", b"x");
        write(dir.path(), "2026/inspection-114.pdf", b"x");
        write(dir.path(), "2026/notes.md", b"x");

        let (found, _) = discover(&collection(dir.path(), Classification::Internal)).unwrap();
        let names: Vec<_> = found.iter().map(|f| f.relative_path.as_str()).collect();

        assert_eq!(names, vec!["2026/inspection-114.pdf", "2026/notes.md", "sop-4.pdf"]);
    }

    /// A share full of CAD files should not fail a sync.
    #[test]
    fn unreadable_file_types_are_skipped_not_failed_on() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "drawing.dwg", b"x");
        write(dir.path(), "sop.pdf", b"x");

        let (found, skipped) = discover(&collection(dir.path(), Classification::Internal)).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].contains("drawing.dwg"));
    }

    #[test]
    fn an_unreachable_root_says_what_to_check() {
        let missing = collection(Path::new("//no-such-server/share"), Classification::Internal);
        let error = discover(&missing).unwrap_err().to_string();
        assert!(error.contains("not reachable"), "{error}");
    }

    #[test]
    fn a_path_that_walks_out_of_the_collection_is_refused() {
        let root = Path::new("C:/plant/sops");
        assert!(stays_within(root, Path::new("C:/plant/sops/4/sop.pdf")));
        assert!(!stays_within(root, Path::new("C:/plant/sops/../vendor/deal.pdf")));
        assert!(!stays_within(root, Path::new("C:/plant/sops-archive/old.pdf")));
    }

    // ── Sync planning ────────────────────────────────────────────────────

    fn file(name: &str, size: u64, modified: u64) -> DiscoveredFile {
        DiscoveredFile {
            path: PathBuf::from(name),
            relative_path: name.into(),
            byte_size: size,
            modified_at: modified,
        }
    }

    #[test]
    fn a_first_sync_treats_everything_as_new() {
        let plan = plan_sync(&[file("a.pdf", 10, 100)], &PreviousState::new());
        assert_eq!(plan.added.len(), 1);
        assert_eq!(plan.unchanged, 0);
    }

    #[test]
    fn an_untouched_file_is_not_re_read() {
        let mut previous = PreviousState::new();
        previous.insert("a.pdf".into(), (10, 100));

        let plan = plan_sync(&[file("a.pdf", 10, 100)], &previous);
        assert!(plan.is_empty());
        assert_eq!(plan.unchanged, 1);
    }

    #[test]
    fn a_changed_size_or_timestamp_marks_the_file_for_re_reading() {
        let mut previous = PreviousState::new();
        previous.insert("a.pdf".into(), (10, 100));

        assert_eq!(plan_sync(&[file("a.pdf", 11, 100)], &previous).changed.len(), 1);
        assert_eq!(plan_sync(&[file("a.pdf", 10, 200)], &previous).changed.len(), 1);
    }

    /// A document taken off the share should not keep answering questions.
    #[test]
    fn a_file_that_disappeared_is_reported_for_retirement() {
        let mut previous = PreviousState::new();
        previous.insert("gone.pdf".into(), (10, 100));

        let plan = plan_sync(&[], &previous);
        assert_eq!(plan.removed, vec!["gone.pdf"]);
    }

    #[test]
    fn the_summary_reads_plainly() {
        let mut previous = PreviousState::new();
        previous.insert("kept.pdf".into(), (10, 100));
        previous.insert("gone.pdf".into(), (10, 100));

        let plan = plan_sync(&[file("kept.pdf", 10, 100), file("new.pdf", 5, 50)], &previous);
        assert_eq!(plan.summary(), "1 new, 1 removed. 1 unchanged.");

        // A genuinely quiet sync: everything previously held is still there.
        let mut settled = PreviousState::new();
        settled.insert("kept.pdf".into(), (10, 100));
        let quiet = plan_sync(&[file("kept.pdf", 10, 100)], &settled);
        assert!(quiet.summary().starts_with("Nothing has changed"), "{}", quiet.summary());
    }

    // ── Access ───────────────────────────────────────────────────────────

    #[test]
    fn a_collection_inherits_its_classifications_clearance() {
        let dir = tempfile::tempdir().unwrap();
        let c = collection(dir.path(), Classification::VendorNegotiation);
        assert!(c.readable_by(&[Role::User]));
        assert!(!c.readable_by(&[Role::KnowledgeAdministrator]));
    }

    /// A collection setting narrows access; it can never widen it.
    #[test]
    fn a_collection_restriction_only_ever_narrows() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = collection(dir.path(), Classification::Internal);

        // Internal is readable by user, knowledge admin and reviewer.
        assert_eq!(c.effective_roles().len(), 3);

        c.restricted_to_roles = vec![Role::Reviewer];
        assert_eq!(c.effective_roles(), vec![Role::Reviewer]);

        // Naming a role the classification does not clear grants nothing.
        c.restricted_to_roles = vec![Role::Auditor];
        assert!(c.effective_roles().is_empty());
        assert!(!c.readable_by(&[Role::Auditor]));
    }

    #[test]
    fn a_network_share_is_described_as_such_when_it_cannot_be_reached() {
        let mut c = collection(Path::new("//plant-fs/engineering"), Classification::Internal);
        c.kind = SourceKind::NetworkShare;
        let error = discover(&c).unwrap_err().to_string();
        assert!(error.contains("network share"), "{error}");
    }
}

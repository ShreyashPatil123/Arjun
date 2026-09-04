//! Which folders and shares this installation reads, and what it read last time.
//!
//! ## Why this exists
//!
//! Everything needed to take an organisation's documents into the index was
//! already here and none of it could be reached. [`super::connector::discover`]
//! walks a collection, [`super::connector::plan_sync`] works out what changed,
//! [`super::chunking::chunk_document`] splits it, [`super::index::KnowledgeIndex`]
//! stores and searches it, and [`super::ingest::ingest_collection`] drives the
//! whole sequence. Not one of them had a caller outside its own tests. The
//! Knowledge screen offered a search box over an index nothing could fill, so a
//! fresh installation had no way to connect its SOPs and manuals at all.
//!
//! Two things were genuinely missing, and both live here:
//!
//! - **The collections themselves.** [`Collection`] described one but nothing
//!   wrote one down, so there was nothing to sync.
//! - **What the last sync saw.** `plan_sync` compares against a
//!   [`PreviousState`] — relative path to size and modification time — and no
//!   code anywhere produced one. Without it every sync is a first sync: every
//!   file re-read, re-extracted and re-indexed, and nothing ever recognised as
//!   removed.
//!
//! ## Why size and time rather than hashes
//!
//! The state file holds what `plan_sync` compares on, which is deliberately not
//! a hash. A share of ten thousand documents cannot be hashed on every sync;
//! content is hashed later, only for the files that look changed, which is what
//! makes an identical re-save still resolve to the same document. Storing what
//! the comparison actually uses keeps this file small and the sync cheap.
//!
//! ## Why two files rather than one
//!
//! The collection list is small, edited by a person, and worth reading when
//! something is wrong. The sync state is machine-written, grows with the share,
//! and is disposable — deleting it costs a full re-read and nothing else. Kept
//! together, the second would make the first unreadable.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::connector::{Collection, DiscoveredFile, PreviousState};

/// Bumped when the on-disk shape changes in a way an older build cannot read.
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectionsFile {
    #[serde(default = "default_schema")]
    schema_version: u32,
    #[serde(default)]
    collections: Vec<Collection>,
}

const fn default_schema() -> u32 {
    SCHEMA_VERSION
}

impl Default for CollectionsFile {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            collections: Vec::new(),
        }
    }
}

/// What one collection looked like at the end of the last sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SyncStateFile {
    #[serde(default = "default_schema")]
    schema_version: u32,
    /// When the sync that wrote this finished. Shown on the screen so an
    /// operator can tell a share that is up to date from one nobody has read
    /// since the documents changed.
    synced_at: String,
    /// Relative path to (bytes, modified-at seconds). Exactly what `plan_sync`
    /// compares, and nothing more.
    #[serde(default)]
    files: BTreeMap<String, (u64, u64)>,
}

/// The collections this installation reads, and their sync state.
pub struct CollectionStore {
    root: PathBuf,
}

impl CollectionStore {
    /// Opens the store, creating its directory if this is a fresh install.
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        let root = app_data_dir.join("knowledge");
        std::fs::create_dir_all(root.join("state"))
            .with_context(|| format!("could not create {}", root.display()))?;
        Ok(Self { root })
    }

    fn collections_path(&self) -> PathBuf {
        self.root.join("collections.json")
    }

    fn state_path(&self, collection_id: &str) -> PathBuf {
        self.root.join("state").join(format!("{collection_id}.json"))
    }

    /// Every collection an administrator has connected.
    ///
    /// A missing file is an empty list rather than an error: that is what a
    /// fresh installation looks like, and refusing to start there would be
    /// worse than starting with nothing connected.
    pub fn list(&self) -> Result<Vec<Collection>> {
        let path = self.collections_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let bytes =
            std::fs::read(&path).with_context(|| format!("could not read {}", path.display()))?;
        // A UTF-8 BOM is what a Windows text editor leaves behind, and serde
        // refuses it. An administrator editing this file by hand should not
        // have to know that.
        let text = String::from_utf8_lossy(&bytes);
        let text = text.trim_start_matches('\u{feff}');
        let file: CollectionsFile = serde_json::from_str(text)
            .with_context(|| format!("{} is not valid JSON", path.display()))?;
        Ok(file.collections)
    }

    /// Adds a collection, or replaces the one with the same id.
    pub fn upsert(&self, collection: Collection) -> Result<()> {
        let mut all = self.list()?;
        match all.iter_mut().find(|c| c.id == collection.id) {
            Some(existing) => *existing = collection,
            None => all.push(collection),
        }
        self.write(all)
    }

    /// Forgets a collection. Returns whether one was there to forget.
    ///
    /// Deliberately does not touch the index. Passages already taken in stay
    /// searchable until they are retired explicitly, because a citation made
    /// in an earlier conversation must still resolve — disconnecting a share
    /// is not the same act as withdrawing what it said.
    pub fn remove(&self, collection_id: &str) -> Result<bool> {
        let mut all = self.list()?;
        let before = all.len();
        all.retain(|c| c.id != collection_id);
        let removed = all.len() != before;
        if removed {
            self.write(all)?;
            // The sync state is meaningless without its collection, and leaving
            // it behind would silently make a re-added collection of the same
            // id look already synced.
            let _ = std::fs::remove_file(self.state_path(collection_id));
        }
        Ok(removed)
    }

    /// One collection by id.
    pub fn get(&self, collection_id: &str) -> Result<Option<Collection>> {
        Ok(self.list()?.into_iter().find(|c| c.id == collection_id))
    }

    /// What the last sync of this collection saw.
    ///
    /// Empty when there has never been one, which makes the next `plan_sync`
    /// treat every file as new — the correct reading of "never synced".
    pub fn previous_state(&self, collection_id: &str) -> PreviousState {
        let path = self.state_path(collection_id);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return PreviousState::new();
        };
        match serde_json::from_str::<SyncStateFile>(text.trim_start_matches('\u{feff}')) {
            Ok(state) => state.files,
            Err(error) => {
                // Unreadable state is treated as no state: the next sync
                // re-reads everything, which is slow but correct. Silently
                // trusting a half-parsed file would be neither.
                log::warn!(
                    "[KNOWLEDGE] {} could not be read, so the next sync re-reads the whole \
                     collection: {error}",
                    path.display()
                );
                PreviousState::new()
            }
        }
    }

    /// Records what this sync saw, for the next one to compare against.
    ///
    /// Written from what was *discovered*, not from what was successfully
    /// ingested. A file that failed to extract is still present on the share,
    /// and recording it as absent would make the next sync re-attempt it as
    /// though it were new — the same failure, forever, on every sync.
    pub fn record_sync(&self, collection_id: &str, discovered: &[DiscoveredFile]) -> Result<()> {
        let state = SyncStateFile {
            schema_version: SCHEMA_VERSION,
            synced_at: chrono::Utc::now().to_rfc3339(),
            files: discovered
                .iter()
                .map(|file| {
                    (
                        file.relative_path.clone(),
                        (file.byte_size, file.modified_at),
                    )
                })
                .collect(),
        };
        let path = self.state_path(collection_id);
        let json = serde_json::to_string_pretty(&state)?;
        write_atomically(&path, &json)
    }

    /// When this collection was last synced, if it ever was.
    pub fn last_synced_at(&self, collection_id: &str) -> Option<String> {
        let text = std::fs::read_to_string(self.state_path(collection_id)).ok()?;
        serde_json::from_str::<SyncStateFile>(text.trim_start_matches('\u{feff}'))
            .ok()
            .map(|state| state.synced_at)
    }

    fn write(&self, collections: Vec<Collection>) -> Result<()> {
        let file = CollectionsFile {
            schema_version: SCHEMA_VERSION,
            collections,
        };
        let json = serde_json::to_string_pretty(&file)?;
        write_atomically(&self.collections_path(), &json)
    }
}

/// Writes through a temporary file so a crash cannot leave a half-written one.
///
/// The collection list is what tells the product which shares it may read. A
/// truncated one is worse than an absent one: absent reads as "nothing
/// connected", truncated reads as "these are all the collections", and the
/// difference is a share that silently stops being searched.
fn write_atomically(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, contents)
        .with_context(|| format!("could not write {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("could not replace {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::connector::SourceKind;
    use crate::policy::Classification;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arjun-collections-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn collection(id: &str, root: &Path) -> Collection {
        Collection {
            id: id.to_string(),
            name: "Maintenance SOPs".into(),
            kind: SourceKind::LocalFolder,
            root: root.to_path_buf(),
            owner: "modeladmin".into(),
            classification: Classification::Internal,
            restricted_to_roles: Vec::new(),
            retention_days: None,
            enabled: true,
        }
    }

    fn discovered(path: &str, size: u64, modified: u64) -> DiscoveredFile {
        DiscoveredFile {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            byte_size: size,
            modified_at: modified,
        }
    }

    #[test]
    fn a_fresh_installation_has_no_collections_rather_than_an_error() {
        let dir = temp_dir("fresh");
        let store = CollectionStore::open(&dir).expect("open");
        assert!(store.list().expect("list").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_connected_collection_survives_a_restart() {
        let dir = temp_dir("roundtrip");
        let store = CollectionStore::open(&dir).expect("open");
        store
            .upsert(collection("c-1", Path::new("D:/shares/sop")))
            .expect("upsert");

        let reopened = CollectionStore::open(&dir).expect("reopen");
        let all = reopened.list().expect("list");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "Maintenance SOPs");
        assert_eq!(all[0].classification, Classification::Internal);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn upserting_the_same_id_replaces_rather_than_duplicates() {
        let dir = temp_dir("upsert");
        let store = CollectionStore::open(&dir).expect("open");
        store
            .upsert(collection("c-1", Path::new("D:/shares/sop")))
            .expect("first");
        let mut renamed = collection("c-1", Path::new("D:/shares/sop"));
        renamed.name = "Inspection reports".into();
        store.upsert(renamed).expect("second");

        let all = store.list().expect("list");
        assert_eq!(all.len(), 1, "one id is one collection");
        assert_eq!(all[0].name, "Inspection reports");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The missing half of `plan_sync`. Without a recorded state every sync is
    /// a first sync: everything re-read and nothing ever seen as removed.
    #[test]
    fn a_recorded_sync_is_what_the_next_one_compares_against() {
        let dir = temp_dir("state");
        let store = CollectionStore::open(&dir).expect("open");
        assert!(
            store.previous_state("c-1").is_empty(),
            "never synced means everything is new"
        );

        store
            .record_sync(
                "c-1",
                &[discovered("procedures/p-01.pdf", 142_093, 1_788_400_000)],
            )
            .expect("record");

        let previous = store.previous_state("c-1");
        assert_eq!(
            previous.get("procedures/p-01.pdf"),
            Some(&(142_093, 1_788_400_000))
        );
        assert!(store.last_synced_at("c-1").is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A re-added collection must not inherit the previous one's sync state, or
    /// it would look fully synced while nothing had been read.
    #[test]
    fn removing_a_collection_forgets_what_it_had_synced() {
        let dir = temp_dir("remove");
        let store = CollectionStore::open(&dir).expect("open");
        store
            .upsert(collection("c-1", Path::new("D:/shares/sop")))
            .expect("upsert");
        store
            .record_sync("c-1", &[discovered("a.pdf", 10, 20)])
            .expect("record");

        assert!(store.remove("c-1").expect("remove"));
        assert!(!store.remove("c-1").expect("remove again"));
        assert!(
            store.previous_state("c-1").is_empty(),
            "a re-added collection must start from nothing"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Unreadable state must not be trusted. Re-reading a share is slow;
    /// indexing against a half-parsed comparison is wrong.
    #[test]
    fn corrupt_sync_state_falls_back_to_a_full_re_read() {
        let dir = temp_dir("corrupt");
        let store = CollectionStore::open(&dir).expect("open");
        std::fs::write(store.state_path("c-1"), "{ not json").expect("write");
        assert!(store.previous_state("c-1").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file hand-edited on Windows arrives with a byte-order mark, and serde
    /// refuses it. An administrator should not have to know that.
    #[test]
    fn a_byte_order_mark_is_not_a_broken_file() {
        let dir = temp_dir("bom");
        let store = CollectionStore::open(&dir).expect("open");
        store
            .upsert(collection("c-1", Path::new("D:/shares/sop")))
            .expect("upsert");

        let path = store.collections_path();
        let text = std::fs::read_to_string(&path).expect("read");
        std::fs::write(&path, format!("\u{feff}{text}")).expect("rewrite with BOM");

        assert_eq!(store.list().expect("list").len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}

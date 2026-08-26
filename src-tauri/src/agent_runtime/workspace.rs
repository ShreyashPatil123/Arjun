//! The one directory a run may touch.
//!
//! Five of the eight tools take a path, and the gateway resolves every one of
//! them against a list of permitted roots. Until now that list was empty, which
//! made those five tools refuse unconditionally — correct as a default, useless
//! as a product.
//!
//! ## Why a directory per run and not one per user
//!
//! A shared scratch directory would let a task read what an earlier, unrelated
//! task left behind. That is not a hypothetical: a run that drafts an approval
//! note for one department and then reads a spreadsheet another department's
//! run produced has crossed a boundary nobody agreed to, and the audit record
//! would show a legitimate read of a permitted path.
//!
//! One directory per run makes the boundary the same shape as the work. A run
//! writes its deliverables where it can see them and nowhere else, and the
//! isolation is a property of the filesystem rather than of everyone
//! remembering to be careful.
//!
//! ## Why it is not cleaned up when the run ends
//!
//! The deliverable is in there. A run that produces an approval note and then
//! deletes it has produced nothing. Retention is an administrator's decision,
//! taken against a directory they can see, not something this module does on
//! their behalf.

use std::path::{Path, PathBuf};

/// Where a run's files live.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    #[error("the workspace for this run could not be created at {path}: {source}")]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl Workspace {
    /// Creates the directory for one run under the application's data directory.
    ///
    /// `run_id` is a UUID this process generated, so it cannot contain a path
    /// separator or a parent reference — but it is still joined as a single
    /// component rather than interpolated into a string, so a future change to
    /// how run ids are made cannot turn this into a traversal.
    pub fn create(app_data_dir: &Path, run_id: &str) -> Result<Self, WorkspaceError> {
        let root = app_data_dir.join("runs").join(run_id);
        std::fs::create_dir_all(&root).map_err(|source| WorkspaceError::Create {
            path: root.clone(),
            source,
        })?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The roots to hand the gateway. Exactly one, always.
    pub fn roots(&self) -> Vec<PathBuf> {
        vec![self.root.clone()]
    }

    /// What the model is told about where it may write.
    ///
    /// Included in the system prompt, because a model that does not know it has
    /// a workspace writes to plausible-sounding absolute paths and collects
    /// refusals instead of doing the work.
    pub fn describe(&self) -> String {
        format!(
            "You have a working directory for this task at {}. Read and write files only there; \
             paths outside it are refused. Use relative names such as \"approval-note.docx\" \
             rather than absolute paths.",
            self.root.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_gets_its_own_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let one = Workspace::create(dir.path(), "run-1").expect("created");
        let two = Workspace::create(dir.path(), "run-2").expect("created");

        assert!(one.root().is_dir());
        assert!(two.root().is_dir());
        assert_ne!(one.root(), two.root());
    }

    #[test]
    fn one_runs_directory_is_not_inside_anothers() {
        // The isolation claim, stated as the property that would be violated.
        let dir = tempfile::tempdir().expect("temp dir");
        let one = Workspace::create(dir.path(), "run-1").expect("created");
        let two = Workspace::create(dir.path(), "run-2").expect("created");

        assert!(!one.root().starts_with(two.root()));
        assert!(!two.root().starts_with(one.root()));
    }

    #[test]
    fn exactly_one_root_is_granted() {
        // More than one would mean a run could reach somewhere it did not create.
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = Workspace::create(dir.path(), "run-1").expect("created");
        assert_eq!(workspace.roots(), vec![workspace.root().to_path_buf()]);
    }

    #[test]
    fn the_model_is_told_where_it_may_write_and_how_to_name_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = Workspace::create(dir.path(), "run-1").expect("created");
        let described = workspace.describe();

        assert!(described.contains(&workspace.root().display().to_string()));
        // Both halves matter: where, and that absolute paths will be refused.
        assert!(described.contains("refused"));
        assert!(described.contains("relative"));
    }

    #[test]
    fn creating_the_same_run_twice_is_not_an_error() {
        // A retried run must not fail because its directory already exists.
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(Workspace::create(dir.path(), "run-1").is_ok());
        assert!(Workspace::create(dir.path(), "run-1").is_ok());
    }
}

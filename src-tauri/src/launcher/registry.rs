//! Tool registry — the shipped set plus whatever the user adds.
//!
//! User entries live in `tools.json` in the app data directory. They may add
//! new tools or replace a shipped one by reusing its id, so a stale built-in
//! definition never blocks anyone.
//!
//! A malformed user file is reported and skipped rather than fatal: losing the
//! shipped tools because one hand-written entry has a typo would be a poor
//! trade.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::launcher::spec::{builtin_tools, ToolSpec};

/// File users edit to add their own tools.
pub const USER_TOOLS_FILE: &str = "tools.json";

/// Outcome of loading the registry, including anything that was wrong with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Registry {
    pub tools: Vec<ToolSpec>,
    /// Problems found while loading, shown in the UI rather than swallowed.
    pub warnings: Vec<String>,
}

pub fn user_tools_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(USER_TOOLS_FILE)
}

/// Merges user entries over the shipped ones.
///
/// Matching ids replace; new ids are appended. Invalid entries are dropped with
/// a warning naming the entry and the reason.
pub fn merge(builtin: Vec<ToolSpec>, user: Vec<ToolSpec>) -> Registry {
    let mut tools = builtin;
    let mut warnings = Vec::new();

    for mut entry in user {
        entry.user_defined = true;

        if let Err(reason) = entry.validate() {
            warnings.push(format!("Ignored a tool in {USER_TOOLS_FILE}: {reason}"));
            continue;
        }

        match tools.iter().position(|t| t.id == entry.id) {
            Some(i) => {
                warnings.push(format!(
                    "'{}' from {USER_TOOLS_FILE} replaced the built-in definition",
                    entry.id
                ));
                tools[i] = entry;
            }
            None => tools.push(entry),
        }
    }

    Registry { tools, warnings }
}

/// Loads the registry for a given app data directory.
///
/// A missing user file is normal and produces no warning.
pub fn load(app_data_dir: &Path) -> Registry {
    let path = user_tools_path(app_data_dir);
    if !path.is_file() {
        return Registry { tools: builtin_tools(), warnings: Vec::new() };
    }

    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => {
            return Registry {
                tools: builtin_tools(),
                warnings: vec![format!("Could not read {}: {e}", path.display())],
            }
        }
    };

    match serde_json::from_str::<Vec<ToolSpec>>(&raw) {
        Ok(user) => merge(builtin_tools(), user),
        Err(e) => Registry {
            tools: builtin_tools(),
            warnings: vec![format!(
                "{} is not valid JSON ({e}). Using the built-in tools only.",
                path.display()
            )],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::spec::{DetectSpec, LaunchSpec, Protocol};
    use std::collections::HashMap;

    fn user_entry(id: &str) -> ToolSpec {
        ToolSpec {
            id: id.into(),
            name: format!("Tool {id}"),
            description: String::new(),
            protocol: Protocol::OpenAi,
            detect: DetectSpec {
                command: id.into(),
                version_arg: "--version".into(),
                expect: id.into(),
            },
            install: None,
            launch: LaunchSpec {
                command: id.into(),
                args: vec![],
                env: HashMap::new(),
                env_remove: vec![],
                client_config: None,
            },
            user_defined: false,
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_registry_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_new_user_entry_is_added_alongside_the_builtins() {
        let builtin_count = builtin_tools().len();

        let reg = merge(builtin_tools(), vec![user_entry("mytool")]);

        assert_eq!(reg.tools.len(), builtin_count + 1);
        assert!(reg.tools.iter().any(|t| t.id == "mytool"));
        assert!(reg.warnings.is_empty(), "adding a tool is not noteworthy");
    }

    #[test]
    fn a_user_entry_can_replace_a_builtin_by_id() {
        // A shipped definition going stale must never block the user.
        let mut replacement = user_entry("claude-code");
        replacement.name = "My Claude Code".into();

        let reg = merge(builtin_tools(), vec![replacement]);

        let found = reg.tools.iter().find(|t| t.id == "claude-code").unwrap();
        assert_eq!(found.name, "My Claude Code");
        assert_eq!(reg.tools.len(), builtin_tools().len(), "replaced, not appended");
        assert!(
            reg.warnings.iter().any(|w| w.contains("replaced")),
            "an override should be visible, not silent"
        );
    }

    #[test]
    fn user_entries_are_marked_so_the_ui_can_flag_them_unverified() {
        let reg = merge(builtin_tools(), vec![user_entry("mytool")]);

        let mine = reg.tools.iter().find(|t| t.id == "mytool").unwrap();
        assert!(mine.user_defined);
        assert!(reg.tools.iter().filter(|t| !t.user_defined).count() >= 1);
    }

    #[test]
    fn an_invalid_user_entry_is_skipped_with_a_reason() {
        let mut broken = user_entry("broken");
        broken.detect.expect = String::new();

        let reg = merge(builtin_tools(), vec![broken, user_entry("fine")]);

        assert!(!reg.tools.iter().any(|t| t.id == "broken"));
        assert!(reg.tools.iter().any(|t| t.id == "fine"), "one bad entry must not block others");
        assert_eq!(reg.warnings.len(), 1);
        assert!(reg.warnings[0].contains("version output"), "warning names the cause");
    }

    #[test]
    fn a_missing_user_file_is_normal() {
        let reg = load(&temp_dir("missing"));

        assert_eq!(reg.tools.len(), builtin_tools().len());
        assert!(reg.warnings.is_empty(), "not having custom tools is not a problem");
    }

    #[test]
    fn malformed_json_keeps_the_builtin_tools_working() {
        let dir = temp_dir("malformed");
        std::fs::write(user_tools_path(&dir), "{ this is not json").unwrap();

        let reg = load(&dir);

        assert_eq!(reg.tools.len(), builtin_tools().len(), "shipped tools still available");
        assert_eq!(reg.warnings.len(), 1);
        assert!(reg.warnings[0].contains("not valid JSON"));
    }

    #[test]
    fn a_valid_user_file_is_loaded_from_disk() {
        let dir = temp_dir("valid");
        let json = serde_json::to_string(&vec![user_entry("fromdisk")]).unwrap();
        std::fs::write(user_tools_path(&dir), json).unwrap();

        let reg = load(&dir);

        let loaded = reg.tools.iter().find(|t| t.id == "fromdisk").expect("should load");
        assert!(loaded.user_defined);
    }

    #[test]
    fn later_user_entries_win_over_earlier_ones() {
        // Two entries with the same id in one file: the last is the intent.
        let mut first = user_entry("dup");
        first.name = "First".into();
        let mut second = user_entry("dup");
        second.name = "Second".into();

        let reg = merge(builtin_tools(), vec![first, second]);

        let found = reg.tools.iter().find(|t| t.id == "dup").unwrap();
        assert_eq!(found.name, "Second");
        assert_eq!(reg.tools.iter().filter(|t| t.id == "dup").count(), 1);
    }
}

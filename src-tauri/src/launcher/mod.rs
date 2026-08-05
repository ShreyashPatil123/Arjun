//! Launch section — start a coding tool already connected to the gateway.
//!
//! The point is that a non-technical user never copies a URL or opens a
//! terminal: press Launch and the tool comes up talking to the local model.
//!
//! Connection is made **at launch only**, by handing the child process the
//! gateway address in its environment. No config file is written, so nothing
//! of the user's is modified and there is nothing to undo. The trade-off is
//! deliberate: opening the tool by other means will not be connected.

pub mod registry;
pub mod spec;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::launcher::spec::{output_identifies_tool, resolve_env, PackageManager, ToolSpec};

/// What state a tool is in, and therefore what the card offers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum ToolState {
    /// Checked and absent — offer Install.
    NotInstalled,
    /// Present and identified — offer Launch.
    Ready { version: String },
    /// Started by Sarathi and still alive — offer Stop.
    Running { pid: u32 },
    /// Something is wrong that the user must see, e.g. the package manager is
    /// missing. Kept separate from `NotInstalled` because the fix differs.
    Problem { reason: String },
}

/// A tool plus its current state, as sent to the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolStatus {
    pub id: String,
    pub name: String,
    pub description: String,
    pub protocol: spec::Protocol,
    pub user_defined: bool,
    /// Command line Install would run, for the confirmation prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_command: Option<String>,
    #[serde(flatten)]
    pub state: ToolState,
}

/// Tracks processes Sarathi started, so cards can show Running and offer Stop.
#[derive(Default)]
pub struct LaunchedProcesses {
    inner: Mutex<HashMap<String, u32>>,
}

impl LaunchedProcesses {
    pub fn record(&self, tool_id: &str, pid: u32) {
        if let Ok(mut m) = self.inner.lock() {
            m.insert(tool_id.to_string(), pid);
        }
    }

    pub fn forget(&self, tool_id: &str) {
        if let Ok(mut m) = self.inner.lock() {
            m.remove(tool_id);
        }
    }

    pub fn pid_of(&self, tool_id: &str) -> Option<u32> {
        self.inner.lock().ok()?.get(tool_id).copied()
    }

    pub fn running_count(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }
}

/// Locates a real executable for `command`.
///
/// Uses `where` on Windows and `command -v` elsewhere, then requires the result
/// to be an actual file. A shell built-in resolves to no path, which is how the
/// `continue` false positive is excluded before anything is run.
fn resolve_executable(command: &str) -> Option<PathBuf> {
    let (finder, args) = if cfg!(windows) {
        ("where", vec![command])
    } else {
        ("sh", vec!["-c"])
    };

    let output = if cfg!(windows) {
        std::process::Command::new(finder).args(&args).output().ok()?
    } else {
        std::process::Command::new("sh")
            .args(["-c", &format!("command -v {command}")])
            .output()
            .ok()?
    };

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next()?.trim();
    if first.is_empty() {
        return None;
    }

    let path = PathBuf::from(first);
    path.is_file().then_some(path)
}

/// Checks whether a tool is genuinely installed.
///
/// Three steps, all of which must pass:
/// 1. a real executable exists — rules out shell built-ins;
/// 2. running it with its version argument succeeds;
/// 3. the output names the tool — rules out an unrelated program of the same name.
pub fn detect(spec: &ToolSpec) -> ToolState {
    if resolve_executable(&spec.detect.command).is_none() {
        return ToolState::NotInstalled;
    }

    let output = std::process::Command::new(&spec.detect.command)
        .arg(&spec.detect.version_arg)
        .stdin(Stdio::null())
        .output();

    let Ok(out) = output else {
        return ToolState::NotInstalled;
    };

    // Some tools print their version to stderr.
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    if !output_identifies_tool(&combined, &spec.detect.expect) {
        return ToolState::NotInstalled;
    }

    let version = combined.lines().next().unwrap_or("").trim().to_string();
    ToolState::Ready { version }
}

/// True when the package manager an entry needs is available.
pub fn manager_available(manager: PackageManager) -> bool {
    resolve_executable(manager.program()).is_some()
}

/// Installs a tool through its package manager.
///
/// Sarathi never downloads an installer itself — it delegates to a manager the
/// user already has, which verifies its own sources. The caller is responsible
/// for showing [`PackageManager::command_line`] and getting consent first.
pub fn install(spec: &ToolSpec) -> Result<(), String> {
    let Some(install) = &spec.install else {
        return Err(format!("{} cannot be installed by Sarathi", spec.name));
    };

    if !manager_available(install.manager) {
        return Err(format!(
            "{} is needed to install {}, but it is not available on this machine.",
            install.manager.program(),
            spec.name
        ));
    }

    let output = std::process::Command::new(install.manager.program())
        .args(install.manager.install_args(&install.package))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not run {}: {e}", install.manager.program()))?;

    if output.status.success() {
        return Ok(());
    }

    // Surface the manager's own message — it usually says exactly what is wrong.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" ");
    Err(format!(
        "{} failed to install {}: {}",
        install.manager.program(),
        spec.name,
        if tail.trim().is_empty() { "no details reported".into() } else { tail }
    ))
}

/// Starts a tool with the gateway address in its environment.
///
/// Returns the child process id. The child is detached — closing Sarathi does
/// not kill a terminal the user is working in.
pub fn launch(spec: &ToolSpec, gateway_port: u16) -> Result<u32, String> {
    let env = resolve_env(&spec.launch.env, gateway_port);

    let mut cmd = std::process::Command::new(&spec.launch.command);
    cmd.args(&spec.launch.args);
    for (k, v) in &env {
        cmd.env(k, v);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", spec.name))?;

    Ok(child.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::spec::{builtin_tools, DetectSpec, LaunchSpec, Protocol};

    fn spec_for(command: &str, expect: &str) -> ToolSpec {
        ToolSpec {
            id: command.into(),
            name: command.into(),
            description: String::new(),
            protocol: Protocol::OpenAi,
            detect: DetectSpec {
                command: command.into(),
                version_arg: "--version".into(),
                expect: expect.into(),
            },
            install: None,
            launch: LaunchSpec { command: command.into(), args: vec![], env: HashMap::new() },
            user_defined: false,
        }
    }

    #[test]
    fn a_command_that_does_not_exist_is_not_installed() {
        let state = detect(&spec_for("definitely-not-a-real-program-xyz", "xyz"));
        assert_eq!(state, ToolState::NotInstalled);
    }

    /// Regression: a naive PATH lookup reported Continue as installed, because
    /// `continue` is a shell keyword. Requiring a real executable file excludes it.
    #[test]
    fn shell_builtins_are_not_mistaken_for_installed_tools() {
        for builtin in ["continue", "break", "return"] {
            assert_eq!(
                detect(&spec_for(builtin, builtin)),
                ToolState::NotInstalled,
                "'{builtin}' is a shell keyword, not an installed tool"
            );
        }
    }

    #[test]
    fn a_real_program_with_the_wrong_identity_is_not_installed() {
        // A program exists at this name, but it is not the tool we want.
        let cmd = if cfg!(windows) { "cmd" } else { "sh" };
        let state = detect(&spec_for(cmd, "definitely-not-in-that-output"));

        assert_eq!(state, ToolState::NotInstalled, "identity check must reject it");
    }

    #[test]
    fn npm_is_detected_on_this_machine() {
        // npm is present here, so this exercises the success path end to end.
        assert!(manager_available(PackageManager::Npm));
    }

    #[test]
    fn a_tool_with_no_install_route_says_so_rather_than_failing_oddly() {
        let spec = spec_for("something", "something");
        assert!(spec.install.is_none());

        let err = install(&spec).unwrap_err();
        assert!(err.contains("cannot be installed"), "got: {err}");
    }

    #[test]
    fn launching_a_missing_program_reports_the_tool_name() {
        let spec = spec_for("definitely-not-a-real-program-xyz", "xyz");

        let err = launch(&spec, 11435).unwrap_err();
        assert!(err.contains("could not start"), "got: {err}");
        assert!(err.contains(&spec.name), "the message should name the tool: {err}");
    }

    #[test]
    fn running_processes_are_tracked_per_tool() {
        let procs = LaunchedProcesses::default();
        assert_eq!(procs.running_count(), 0);
        assert_eq!(procs.pid_of("claude-code"), None);

        procs.record("claude-code", 1234);
        assert_eq!(procs.pid_of("claude-code"), Some(1234));
        assert_eq!(procs.running_count(), 1);

        procs.forget("claude-code");
        assert_eq!(procs.pid_of("claude-code"), None);
        assert_eq!(procs.running_count(), 0);
    }

    #[test]
    fn detecting_every_shipped_tool_never_panics() {
        // Runs the real detection path against the real machine. Claude Code is
        // installed here and opencode is not, so both branches are exercised.
        for spec in builtin_tools() {
            let state = detect(&spec);
            match state {
                ToolState::Ready { ref version } => {
                    assert!(!version.is_empty(), "{} reported ready with no version", spec.id)
                }
                ToolState::NotInstalled => {}
                other => panic!("unexpected state for {}: {other:?}", spec.id),
            }
        }
    }
}

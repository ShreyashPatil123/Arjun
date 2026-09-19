//! Utility functions for silently launching background child processes on Windows
//! with strict timeouts and popup error suppression.

use std::process::{Command, Output};
use std::time::Duration;
use std::sync::mpsc;
use std::thread;
use std::path::Path;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// Creates a `Command` configured to execute without spawning a visible console window on Windows,
/// and with OS error dialog popups suppressed.
pub fn create_hidden_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
        // Suppress Windows Error Reporting dialog popups (e.g. 0xc0000142 / crash popups)
        cmd.env("SEM_NOGPFAULTERRORBOX", "1");
    }
    cmd
}

/// Executes a hidden command with a strict timeout.
/// If the process hangs, shows an OS error modal dialog, or exceeds the timeout,
/// it is terminated immediately and returns an Err.
pub fn run_command_with_timeout(mut cmd: Command, timeout: Duration) -> Result<Output, String> {
    let (tx, rx) = mpsc::channel();

    // Piped, not inherited. `wait_with_output` can only collect a stream it was
    // given a pipe for: spawned with the default stdio the child writes
    // straight to *our* console and `Output.stdout` comes back empty every
    // time.
    //
    // That was not theoretical. Every line the software detector logged read
    // `version=""` — for Rust, Node, npm, Git, all of them — because the
    // version string it parsed was always an empty buffer, while the real
    // output appeared in ARJUN's own stdout. The same emptiness is why Python
    // was reported missing on a machine with three copies of it: the condition
    // that would have rescued a non-zero exit (`!stdout.is_empty()`) could
    // never be true, so the Windows Store alias's refusal was the whole answer.
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let child = cmd.spawn().map_err(|e| format!("Failed to spawn process: {}", e))?;
    let child_id = child.id();

    thread::spawn(move || {
        let res = child.wait_with_output();
        let _ = tx.send(res);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(e)) => Err(format!("Process execution error: {}", e)),
        Err(_) => {
            // Timed out — kill the child process if it's hanging or showing a modal dialog
            #[cfg(target_os = "windows")]
            {
                let _ = Command::new("taskkill")
                    .args(["/F", "/PID", &child_id.to_string()])
                    .creation_flags(0x08000000)
                    .output();
            }
            Err(format!("Process timed out after {:?}", timeout))
        }
    }
}

/// Resolves the absolute path of a binary on PATH purely in native Rust.
/// Avoids spawning `where.exe` or `which` child processes, preventing mini terminal windows.
pub fn resolve_binary_path_natively(bin: &str) -> Option<String> {
    // If it's already an absolute path and exists
    let direct = Path::new(bin);
    if direct.is_absolute() && direct.exists() {
        return Some(direct.to_string_lossy().to_string());
    }

    if let Some(paths) = std::env::var_os("PATH") {
        for path in std::env::split_paths(&paths) {
            let p = path.join(bin);
            if p.exists() && p.is_file() {
                return Some(p.to_string_lossy().to_string());
            }

            #[cfg(target_os = "windows")]
            {
                let exts = [".exe", ".cmd", ".bat"];
                for ext in &exts {
                    let p_ext = path.join(format!("{}{}", bin, ext));
                    if p_ext.exists() && p_ext.is_file() {
                        return Some(p_ext.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The output of a command that ran must actually come back.
    ///
    /// This is the regression that mattered: `run_command_with_timeout` spawned
    /// with the default stdio, so `wait_with_output` had no pipe to read and
    /// returned empty buffers every time. Nothing failed loudly. The software
    /// panel simply reported `version=""` for every tool on the machine, and
    /// Python — whose probe depends on reading the output when the Windows
    /// Store alias answers first — was reported missing on a machine with three
    /// interpreters installed.
    #[test]
    fn a_commands_stdout_is_captured_rather_than_inherited() {
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "echo captured-marker"]);
        } else {
            cmd.args(["-c", "echo captured-marker"]);
        }

        let output = run_command_with_timeout(cmd, Duration::from_secs(10))
            .expect("the command runs");

        assert!(output.status.success(), "the command should have succeeded");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            text.contains("captured-marker"),
            "stdout must be captured, not written to this process console. Got {text:?}"
        );
    }

    /// stderr is captured too, and separately.
    ///
    /// Both halves are needed: some tools print their version to stderr, and the
    /// Store alias's refusal arrives there as well.
    #[test]
    fn a_commands_stderr_is_captured_separately_from_stdout() {
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "echo err-marker 1>&2"]);
        } else {
            cmd.args(["-c", "echo err-marker 1>&2"]);
        }

        let output = run_command_with_timeout(cmd, Duration::from_secs(10))
            .expect("the command runs");

        let err = String::from_utf8_lossy(&output.stderr);
        let out = String::from_utf8_lossy(&output.stdout);
        assert!(err.contains("err-marker"), "stderr must be captured, got {err:?}");
        assert!(
            !out.contains("err-marker"),
            "stderr must not be folded into stdout, got {out:?}"
        );
    }
}

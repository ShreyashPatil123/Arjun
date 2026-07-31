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
pub fn create_hidden_command(program: &str) -> Command {
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

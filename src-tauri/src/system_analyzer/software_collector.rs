//! Developer software dependencies collector using native PATH and registry checks
//! Zero child process spawns for missing binaries (prevents mini terminal window popups).

use std::time::Duration;
use crate::system_analyzer::process_utils::{create_hidden_command, run_command_with_timeout, resolve_binary_path_natively};
use crate::system_analyzer::traits::{SoftwareDetectorInfo, SoftwareEnvironment};

/// Detects system software environment dependencies
pub fn detect_software() -> SoftwareEnvironment {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 Software Collector Started");

    let python = check_executable("Python", &["python", "python3"], &["--version"]);
    let rust = check_executable("Rust", &["rustc"], &["--version"]);
    let cargo = check_executable("Cargo", &["cargo"], &["--version"]);
    let git = check_executable("Git", &["git"], &["--version"]);
    let nodejs = check_executable("Node.js", &["node"], &["--version"]);
    let npm = check_executable("npm", &["npm.cmd", "npm"], &["--version"]);
    let pnpm = check_executable("pnpm", &["pnpm.cmd", "pnpm"], &["--version"]);
    let ollama = check_executable("Ollama", &["ollama"], &["--version"]);
    let cuda_toolkit = check_cuda_toolkit();
    let vc_redistributable = check_vc_redistributable();

    let docker = check_executable("Docker", &["docker"], &["--version"]);
    let rocm = check_executable("ROCm", &["rocm-smi"], &["--version"]);

    let mut additional = Vec::new();
    additional.push(docker);
    additional.push(rocm);

    log::info!("[SYSTEM ANALYZER DEBUG] ✓ Software Detection Summary: Python={}, Rust={}, Git={}, Node={}, Ollama={}",
        python.installed, rust.installed, git.installed, nodejs.installed, ollama.installed);

    SoftwareEnvironment {
        python,
        rust,
        cargo,
        git,
        nodejs,
        npm,
        pnpm,
        ollama,
        cuda_toolkit,
        vc_redistributable,
        additional,
    }
}

fn check_executable(display_name: &str, binary_names: &[&str], version_args: &[&str]) -> SoftwareDetectorInfo {
    for bin in binary_names {
        // First check natively in Rust if the binary exists on PATH.
        // If it doesn't exist natively, do NOT spawn a child process!
        if let Some(resolved_path) = resolve_binary_path_natively(bin) {
            let mut cmd = create_hidden_command(&resolved_path);
            cmd.args(version_args);

            if let Ok(output) = run_command_with_timeout(cmd, Duration::from_secs(2)) {
                if output.status.success() || !output.stdout.is_empty() || !output.stderr.is_empty() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let raw_out = if !stdout.trim().is_empty() { stdout } else { stderr };
                    let version = parse_version_string(&raw_out);

                    log::info!("[SYSTEM ANALYZER DEBUG] ✓ Found {}: version={:?}, path={:?}", display_name, version, resolved_path);

                    return SoftwareDetectorInfo {
                        name: display_name.to_string(),
                        installed: true,
                        version: Some(version),
                        path: Some(resolved_path),
                    };
                }
            }
        }
    }

    log::info!("[SYSTEM ANALYZER DEBUG] ℹ️ {} not found on PATH", display_name);

    SoftwareDetectorInfo {
        name: display_name.to_string(),
        installed: false,
        version: None,
        path: None,
    }
}

fn parse_version_string(raw: &str) -> String {
    let line = raw.lines().next().unwrap_or(raw).trim();
    line.to_string()
}

fn check_cuda_toolkit() -> SoftwareDetectorInfo {
    if let Some(resolved_path) = resolve_binary_path_natively("nvcc") {
        let mut cmd = create_hidden_command(&resolved_path);
        cmd.arg("--version");
        if let Ok(output) = run_command_with_timeout(cmd, Duration::from_secs(2)) {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let version = stdout
                    .lines()
                    .find(|l| l.contains("release"))
                    .unwrap_or("CUDA Toolkit")
                    .trim()
                    .to_string();
                return SoftwareDetectorInfo {
                    name: "CUDA Toolkit".to_string(),
                    installed: true,
                    version: Some(version),
                    path: Some(resolved_path),
                };
            }
        }
    }

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        return SoftwareDetectorInfo {
            name: "CUDA Toolkit".to_string(),
            installed: true,
            version: Some("Detected via CUDA_PATH".to_string()),
            path: Some(cuda_path),
        };
    }

    SoftwareDetectorInfo {
        name: "CUDA Toolkit".to_string(),
        installed: false,
        version: None,
        path: None,
    }
}

fn check_vc_redistributable() -> SoftwareDetectorInfo {
    #[cfg(target_os = "windows")]
    {
        let system32 = std::path::Path::new("C:\\Windows\\System32\\vcruntime140.dll");
        if system32.exists() {
            return SoftwareDetectorInfo {
                name: "Visual C++ Redistributable".to_string(),
                installed: true,
                version: Some("v140 (2015-2022)".to_string()),
                path: Some(system32.to_string_lossy().to_string()),
            };
        }
    }

    SoftwareDetectorInfo {
        name: "Visual C++ Redistributable".to_string(),
        installed: false,
        version: None,
        path: None,
    }
}

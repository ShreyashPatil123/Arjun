//! Developer software dependencies collector using PATH and registry checks

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::{SoftwareDetectorInfo, SoftwareEnvironment};

/// Detects system software environment dependencies
pub fn detect_software() -> SoftwareEnvironment {
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
        additional: Vec::new(),
    }
}

fn check_executable(display_name: &str, binary_names: &[&str], version_args: &[&str]) -> SoftwareDetectorInfo {
    for bin in binary_names {
        if let Ok(output) = create_hidden_command(bin).args(version_args).output() {
            if output.status.success() || !output.stdout.is_empty() || !output.stderr.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let raw_out = if !stdout.trim().is_empty() { stdout } else { stderr };
                let version = parse_version_string(&raw_out);

                let path = resolve_binary_path(bin);

                return SoftwareDetectorInfo {
                    name: display_name.to_string(),
                    installed: true,
                    version: Some(version),
                    path,
                };
            }
        }
    }

    SoftwareDetectorInfo {
        name: display_name.to_string(),
        installed: false,
        version: None,
        path: None,
    }
}

fn resolve_binary_path(bin: &str) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = create_hidden_command("where").arg(bin).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    return Some(first_line.trim().to_string());
                }
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(output) = create_hidden_command("which").arg(bin).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    return Some(first_line.trim().to_string());
                }
            }
        }
    }

    None
}

fn parse_version_string(raw: &str) -> String {
    let line = raw.lines().next().unwrap_or(raw).trim();
    line.to_string()
}

fn check_cuda_toolkit() -> SoftwareDetectorInfo {
    if let Ok(output) = create_hidden_command("nvcc").arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let version = stdout
                .lines()
                .find(|l| l.contains("release"))
                .unwrap_or("CUDA Toolkit")
                .trim()
                .to_string();
            let path = resolve_binary_path("nvcc");
            return SoftwareDetectorInfo {
                name: "CUDA Toolkit".to_string(),
                installed: true,
                version: Some(version),
                path,
            };
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
        // Check system32 for vcruntime140.dll
        let system32 = std::path::Path::new("C:\\Windows\\System32\\vcruntime140.dll");
        if system32.exists() {
            return SoftwareDetectorInfo {
                name: "Visual C++ Redistributable".to_string(),
                installed: true,
                version: Some("v140 (2015-2022)".to_string()),
                path: Some(system32.to_string_lossy().to_string()),
            };
        }

        // Check Windows registry via reg query
        if let Ok(output) = create_hidden_command("reg")
            .args([
                "query",
                "HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64",
                "/v",
                "Installed",
            ])
            .output()
        {
            if output.status.success() {
                return SoftwareDetectorInfo {
                    name: "Visual C++ Redistributable".to_string(),
                    installed: true,
                    version: Some("v140 x64".to_string()),
                    path: None,
                };
            }
        }
    }

    SoftwareDetectorInfo {
        name: "Visual C++ Redistributable".to_string(),
        installed: false,
        version: None,
        path: None,
    }
}

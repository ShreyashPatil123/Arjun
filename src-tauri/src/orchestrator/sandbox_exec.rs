//! Actually running model-written code, inside a container.
//!
//! [`super::sandbox`] decides *whether* code may run and what a given tier
//! honestly promises. This module is the part that runs it, and it implements
//! exactly one backend: a rootless container with the network switched off.
//!
//! ## Why only one backend
//!
//! ARJUN design rule 28 asks for seven properties — read-only base image, no
//! host credentials, no unrestricted host mounts, blocked network, limited
//! CPU/RAM, a timeout, and a restricted output directory. A container keeps all
//! seven. WSL2 shares the host network stack and a Windows job object shares
//! everything but CPU and memory, so neither can keep the one that matters on a
//! product whose claim is that nothing leaves the machine.
//!
//! An administrator can accept that risk for the *assessment*
//! (`SandboxAssessment::ReadyWithAcceptedRisk`), and that is their call to make.
//! It is not a backend this module knows how to drive, so it says so rather than
//! quietly running the program somewhere weaker than the caller believes.
//!
//! ## The air-gap detail that is easy to miss
//!
//! `docker run` fetches a missing image from a registry. On a machine whose
//! entire claim is zero egress, the first sandboxed execution would therefore
//! reach the internet — through a subprocess, so neither the broker nor the
//! egress gate would see it.
//!
//! So the image is checked for locally first, and the run passes `--pull=never`
//! as a second lock. A site provisions images the way it provisions weights:
//! deliberately, in advance, and offline.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::sandbox::{SandboxPolicy, SandboxTier};
use crate::system_analyzer::process_utils::create_hidden_command;

/// How long to wait for a killed container to actually go away.
const KILL_GRACE: Duration = Duration::from_secs(5);

/// Polling interval while waiting on the child.
const POLL: Duration = Duration::from_millis(50);

/// A language the sandbox knows how to run, and what it runs it with.
///
/// A fixed table rather than anything derived from the call. The model names a
/// language; it does not name an image, an interpreter, or a command line. That
/// is the difference between "run this Python" and "run this container", and
/// only the first is a thing a model is allowed to ask for.
struct Runtime {
    /// Image tag, which must already be present locally.
    image: &'static str,
    /// File the source is written to inside the mounted directory.
    filename: &'static str,
    /// Argv inside the container.
    argv: &'static [&'static str],
}

fn runtime_for(language: &str) -> Option<Runtime> {
    match language.trim().to_ascii_lowercase().as_str() {
        "python" | "python3" | "py" => Some(Runtime {
            image: "python:3.11-slim",
            filename: "main.py",
            argv: &["python", "/work/main.py"],
        }),
        "javascript" | "js" | "node" => Some(Runtime {
            image: "node:20-slim",
            filename: "main.js",
            argv: &["node", "/work/main.js"],
        }),
        _ => None,
    }
}

/// The languages a refusal can offer, so the model is not left guessing.
pub const SUPPORTED_LANGUAGES: &str = "python, javascript";

/// What happened when code ran.
#[derive(Debug, Clone)]
pub struct Execution {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration: Duration,
    /// Bytes dropped from the streams, if the cap was hit.
    pub truncated_bytes: usize,
}

impl Execution {
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// Which container runtime to drive, if either is usable.
fn container_binary() -> Option<&'static str> {
    for candidate in ["podman", "docker"] {
        let usable = create_hidden_command(candidate)
            .arg("info")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if usable {
            return Some(candidate);
        }
    }
    None
}

/// Whether an image is already on this machine.
///
/// Asked before the run, and asked again by `--pull=never`, because the two
/// answer different questions: this one produces a message a person can act on,
/// and the flag makes certain that a race between them cannot end in a fetch.
fn image_present(binary: &str, image: &str) -> bool {
    create_hidden_command(binary)
        .args(["image", "inspect", image])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Runs `source` in a container and returns what it did.
///
/// `workspace` is the run's own directory. A per-execution subdirectory is
/// created inside it and mounted as `/work`; nothing else on the host is
/// visible to the program.
pub fn run_in_container(
    tier: SandboxTier,
    policy: &SandboxPolicy,
    workspace: &Path,
    language: &str,
    source: &str,
) -> Result<Execution, String> {
    if tier != SandboxTier::Container {
        return Err(format!(
            "code execution is implemented only for a container, and the sandbox available here \
             is a {}. That tier cannot block a child process from reaching the network, and ARJUN \
             does not run model-written code on a tier that cannot. Install Podman, or start \
             Docker. Nothing was executed.",
            tier.label()
        ));
    }

    let runtime = runtime_for(language).ok_or_else(|| {
        format!(
            "{language:?} is not a language this sandbox runs. Supported: {SUPPORTED_LANGUAGES}. \
             Nothing was executed."
        )
    })?;

    let binary = container_binary().ok_or(
        "no container runtime is responding. A CLI may be installed while its daemon is stopped — \
         start Podman or Docker Desktop and try again. Nothing was executed.",
    )?;

    if !image_present(binary, runtime.image) {
        return Err(format!(
            "the image {} is not present on this machine, and ARJUN will not fetch it: pulling an \
             image is an outbound call made by a subprocess, which the broker does not see. Have \
             an administrator load it while the machine is connected (`{} pull {}`), or import a \
             saved copy with `{} load`. Nothing was executed.",
            runtime.image, binary, runtime.image, binary
        ));
    }

    // A directory per execution, so two concurrent runs cannot read or overwrite
    // each other's source, and so the mount can be scoped to one program.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = format!("arjun-sandbox-{}-{stamp}", std::process::id());
    let dir: PathBuf = workspace.join("sandbox-runs").join(&name);

    std::fs::create_dir_all(&dir).map_err(|e| {
        format!("could not prepare the execution directory {}: {e}", dir.display())
    })?;

    let source_path = dir.join(runtime.filename);
    std::fs::write(&source_path, source)
        .map_err(|e| format!("could not write the program to {}: {e}", source_path.display()))?;

    let mount = format!("{}:/work", dir.display());
    let memory = format!("{}b", policy.max_memory_bytes);
    let cpus = policy.max_cpus.to_string();

    let mut command = create_hidden_command(binary);
    command
        .arg("run")
        // Leave nothing behind; the mounted directory is the only output.
        .args(["--rm", "--name", &name])
        // The promise that matters most.
        .arg("--network=none")
        // Never reach a registry. See the module docs.
        .arg("--pull=never")
        // Root filesystem read-only; /work is the one writable place, and it is
        // the restricted output directory the design rule asks for.
        .arg("--read-only")
        .args(["--tmpfs", "/tmp:rw,size=64m,mode=1777"])
        .args(["--volume", &mount])
        .args(["--workdir", "/work"])
        // Resource caps.
        .args(["--memory", &memory])
        .args(["--cpus", &cpus])
        .args(["--pids-limit", "128"])
        // Drop every capability and forbid regaining privilege.
        .args(["--cap-drop", "ALL"])
        .args(["--security-opt", "no-new-privileges"])
        // Not root inside the container.
        .args(["--user", "1000:1000"])
        .arg(runtime.image)
        .args(runtime.argv)
        // The host environment is not inherited: no tokens, no paths, and not
        // the signed-in person's credentials.
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("could not start {binary}: {e}. Nothing was executed."))?;

    // `Command` has no timeout, and a program that never exits must not hold the
    // agent loop open. Poll, then kill the *container* by name — killing the
    // client process would leave the container running.
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if started.elapsed() >= policy.timeout {
                    timed_out = true;
                    let _ = create_hidden_command(binary)
                        .args(["kill", &name])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                    let killed_at = Instant::now();
                    while killed_at.elapsed() < KILL_GRACE {
                        if matches!(child.try_wait(), Ok(Some(_))) {
                            break;
                        }
                        std::thread::sleep(POLL);
                    }
                    let _ = child.kill();
                    break;
                }
                std::thread::sleep(POLL);
            }
            Err(e) => return Err(format!("lost track of the container: {e}")),
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("could not collect the container's output: {e}"))?;

    let cap = usize::try_from(policy.max_output_bytes).unwrap_or(usize::MAX);
    let (stdout, dropped_out) = cap_stream(&output.stdout, cap);
    let (stderr, dropped_err) = cap_stream(&output.stderr, cap);

    Ok(Execution {
        exit_code: if timed_out { None } else { output.status.code() },
        stdout,
        stderr,
        timed_out,
        duration: started.elapsed(),
        truncated_bytes: dropped_out + dropped_err,
    })
}

/// Truncates a stream at the cap, reporting how much went.
///
/// Silent truncation is how a model comes to treat the first half of an error as
/// the whole of it, so the count is returned and the caller says so.
fn cap_stream(raw: &[u8], cap: usize) -> (String, usize) {
    if raw.len() <= cap {
        return (String::from_utf8_lossy(raw).into_owned(), 0);
    }
    let kept = String::from_utf8_lossy(&raw[..cap]).into_owned();
    (kept, raw.len() - cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SandboxPolicy {
        SandboxPolicy::default()
    }

    #[test]
    fn a_tier_that_cannot_block_the_network_is_never_executed_on() {
        for tier in [SandboxTier::Wsl2, SandboxTier::JobObject, SandboxTier::None] {
            let refusal =
                run_in_container(tier, &policy(), Path::new("."), "python", "print(1)")
                    .expect_err("a tier below Container must refuse");

            assert!(
                refusal.contains("container"),
                "the refusal must name what is missing, got: {refusal}"
            );
            assert!(
                refusal.contains("Nothing was executed"),
                "a refusal must say nothing ran, got: {refusal}"
            );
        }
    }

    #[test]
    fn an_unknown_language_refuses_and_lists_what_it_does_run() {
        let refusal = run_in_container(
            SandboxTier::Container,
            &policy(),
            Path::new("."),
            "brainfuck",
            "+++.",
        )
        .expect_err("an unsupported language must refuse");

        assert!(refusal.contains("python"), "got: {refusal}");
        assert!(refusal.contains("javascript"), "got: {refusal}");
    }

    #[test]
    fn the_language_table_does_not_let_a_caller_choose_an_image() {
        // The model names a language. Anything that looks like an image, a flag
        // or a command must simply not resolve.
        for hostile in ["python:3.11-slim", "--privileged", "alpine", "sh -c id", ""] {
            assert!(
                runtime_for(hostile).is_none(),
                "{hostile:?} must not resolve to a runtime"
            );
        }
    }

    #[test]
    fn language_matching_is_case_and_whitespace_insensitive() {
        for spelling in ["Python", "  py  ", "PY", "Node", "JavaScript"] {
            assert!(runtime_for(spelling).is_some(), "{spelling:?} should resolve");
        }
    }

    #[test]
    fn every_runtime_runs_the_file_it_writes() {
        for language in ["python", "javascript"] {
            let rt = runtime_for(language).expect("known language");
            let target = format!("/work/{}", rt.filename);
            assert!(
                rt.argv.iter().any(|a| *a == target),
                "{language}: argv {:?} never names {target}",
                rt.argv
            );
        }
    }

    #[test]
    fn capping_reports_what_it_dropped() {
        let (kept, dropped) = cap_stream(b"abcdefghij", 4);
        assert_eq!(kept, "abcd");
        assert_eq!(dropped, 6);

        let (whole, none) = cap_stream(b"abc", 10);
        assert_eq!(whole, "abc");
        assert_eq!(none, 0);
    }

    #[test]
    fn a_timed_out_execution_is_never_reported_as_success() {
        let execution = Execution {
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: true,
            duration: Duration::from_secs(60),
            truncated_bytes: 0,
        };
        assert!(!execution.succeeded());
    }

    #[test]
    fn a_nonzero_exit_is_never_reported_as_success() {
        let execution = Execution {
            exit_code: Some(1),
            stdout: String::new(),
            stderr: "Traceback".to_string(),
            timed_out: false,
            duration: Duration::from_millis(10),
            truncated_bytes: 0,
        };
        assert!(!execution.succeeded());
    }
}

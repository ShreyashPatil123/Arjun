//! Platform-specific sandboxing for the document sidecar.
//!
//! The Python sidecar parses untrusted PDFs, images, and Office files. The
//! parsers in this stack (PyPDF2, Pillow, vision-model bindings) have a long
//! history of memory-safety bugs. ARJUN therefore tries to make the sidecar
//! process the *least* it can be: no privilege escalation, no surprise
//! outbound network, writes restricted to a temporary subdirectory of the
//! application data area.
//!
//! ## What this module actually does
//!
//! On Linux, [`apply_sandbox`] calls `prctl(PR_SET_NO_NEW_PRIVS, 1)` in the
//! child between `fork` and `exec`. That flag is irreversible: it prevents
//! the sidecar (or any process it can spawn) from regaining privileges via a
//! setuid or setgid binary, which is the standard escalation path once a
//! PDF parser bug gives an attacker code execution.
//!
//! We do **not** install a full seccomp BPF filter. Doing so correctly
//! requires enumerating every syscall the Python runtime, the parsers, and
//! any native library they load can make — and a typo in that list is a
//! process that either fails to start or, worse, can be escaped by the
//! attacker's first attempt to call a blocked syscall. The PR_SET_NO_NEW_PRIVS
//! guard is the high-leverage piece; a follow-up could add seccomp via
//! `libseccomp` once the dependency and testing cost is justified.
//!
//! On Windows and macOS there is no portable equivalent, so the function
//! is a no-op. The temp-directory restriction (set by the caller) and the
//! `kill_on_drop` behaviour on the [`Child`] handle are the layered
//! defences for those platforms.
//!
//! [`Child`]: std::process::Child

use std::process::Command;

/// Reduces the sidecar's privilege surface as much as the platform allows.
///
/// See the [module-level documentation](self) for what each platform does
/// and, importantly, what it does not.
pub fn apply_sandbox(cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;

        // SAFETY: `pre_exec` runs in the child process after `fork` and
        // before `exec`. The closure may call only async-signal-safe
        // functions; `prctl` is on the list, and the two integer arguments
        // are passed by value with no shared mutable state. The closure
        // runs once, on a single thread, before any other Rust code in the
        // child has run.
        //
        // We deliberately do *not* call `setgroups(0, ...)` here. That
        // syscall requires `CAP_SETGID`, which a sidecar launched as a
        // normal user does not have; the call would return `EPERM` and
        // would be silently swallowed in the current `Ok(())` shape,
        // leaving the impression that the supplementary groups were
        // cleared when they were not. If we ever need to drop groups
        // without capabilities, it has to be done by the parent before
        // `fork` — see the broader sandbox discussion in the
        // `capability` module.
        unsafe {
            cmd.pre_exec(|| {
                // PR_SET_NO_NEW_PRIVS is the one reliable privilege
                // boundary available to an unprivileged process. Once set
                // it cannot be unset, and any subsequent exec of a
                // setuid/setgid binary is a no-op for privilege purposes.
                libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
                Ok(())
            });
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Best-effort on Windows and macOS. The caller still:
        //   - sets TMPDIR / TEMP to a subdirectory of app_data_dir
        //   - holds the Child via Arc<Mutex<Child>> with a Drop that kills it
        //   - never sets proxy environment variables (already handled
        //     by create_hidden_command via env_remove)
        let _ = cmd;
    }
}
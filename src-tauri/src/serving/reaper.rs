//! Making sure no inference server outlives the application that started it.
//!
//! ## The failure this exists to stop
//!
//! `llama-server` is a child process holding gigabytes of VRAM. Three things
//! were supposed to clean it up and none of them covered the case that
//! actually happens:
//!
//! - `kill_on_drop(true)` on the [`tokio::process::Command`]. Only runs if the
//!   `Child` is *dropped*, which means the process unwound normally. A window
//!   closed with the X button, a force-kill from Task Manager, a panic, or a
//!   developer rebuilding over a running binary all skip every destructor.
//! - `ModelServers::stop_all`, whose own documentation says "Called on shutdown
//!   so no server outlives the app". Nothing called it. The comment described
//!   an intention that was never wired to an event.
//! - The operating system. It does not do this. An orphan is reparented and
//!   keeps running.
//!
//! Measured on a development machine after a few days of iterating: **eight**
//! orphaned `llama-server` processes — five copies of one chat model, three of
//! the OCR model — holding **7.4 GB of an 8 GB card** between them. The next
//! model ARJUN loaded had about 670 MB to fit in, so `plan_gpu_offload`'s
//! arithmetic was correct and irrelevant: llama.cpp could not place the layers
//! and the model ran on the CPU. The symptom reported was "the model loading
//! speed is slow", and the answer was that the GPU had already been given away
//! to copies of itself nobody could see.
//!
//! ## Why a job object rather than a tidier exit handler
//!
//! An exit handler is still worth having and is wired separately — see
//! `stop_all`'s caller. But an exit handler is a promise the process makes, and
//! the orphans above were created precisely by the paths where a process makes
//! no promises. A Windows job object is enforced by the kernel: a child
//! assigned to a job with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is terminated
//! when the last handle to that job closes, and the handle closes when the
//! process holding it dies — however it dies.
//!
//! So the two are layered deliberately. `stop_all` is the polite shutdown that
//! lets a server exit on its own; the job object is what happens when nothing
//! polite runs.
//!
//! ## Elsewhere
//!
//! Non-Windows builds get a no-op that reports honestly. The equivalent there
//! is a process group and `prctl(PR_SET_PDEATHSIG)`, which is a different
//! mechanism with different edge cases; claiming a guarantee this module does
//! not provide would be worse than saying it provides none.

/// Puts a spawned child under the application's lifetime.
///
/// Best-effort by design: a failure here costs the guarantee, not the feature.
/// A model server that runs without being adopted is a model server that runs,
/// and refusing to start one because it could not be enrolled would turn a
/// cleanup problem into an availability problem. The reason is logged so an
/// operator who later finds an orphan can tell whether this was why.
pub fn adopt(pid: u32) {
    #[cfg(target_os = "windows")]
    windows_impl::adopt(pid);

    #[cfg(not(target_os = "windows"))]
    {
        let _ = pid;
    }
}

/// Whether children are actually enrolled on this platform.
///
/// Reported rather than assumed: the health screen should be able to say "these
/// servers will not outlive the app" only where that is true.
pub fn guarantees_cleanup() -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::job_handle().is_some()
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

#[cfg(target_os = "windows")]
mod windows_impl {
    use std::sync::OnceLock;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE};

    /// The job every inference server is enrolled in.
    ///
    /// A raw handle rather than an owned wrapper, and deliberately never
    /// closed: the whole mechanism is "when this handle closes, kill the
    /// children", so the handle must live exactly as long as the process. A
    /// `Drop` that tidied it away at the wrong moment would kill every running
    /// model server, which is the opposite of the intent.
    struct Job(HANDLE);

    // The handle is used only by `AssignProcessToJobObject`, which is
    // thread-safe, and is never mutated after creation.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    static JOB: OnceLock<Option<Job>> = OnceLock::new();

    pub(super) fn job_handle() -> Option<HANDLE> {
        JOB.get_or_init(create).as_ref().map(|job| job.0)
    }

    fn create() -> Option<Job> {
        unsafe {
            // Unnamed: a named job would be shared with any other process that
            // guessed the name, and two ARJUN windows must not be able to kill
            // each other's servers.
            let handle = match CreateJobObjectW(None, None) {
                Ok(handle) if !handle.is_invalid() => handle,
                Ok(_) => {
                    log::warn!(
                        "[SERVING] the job object for model servers came back invalid; \
                         servers will not be cleaned up if this process is force-killed"
                    );
                    return None;
                }
                Err(error) => {
                    log::warn!(
                        "[SERVING] could not create the job object for model servers ({error}); \
                         servers will not be cleaned up if this process is force-killed"
                    );
                    return None;
                }
            };

            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            let set = SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const core::ffi::c_void,
                u32::try_from(std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                    .unwrap_or(0),
            );

            if let Err(error) = set {
                // A job with no kill-on-close limit is worse than none: it
                // would adopt children and guarantee nothing, and the log would
                // read as though cleanup were handled.
                log::warn!(
                    "[SERVING] could not set kill-on-close on the model-server job ({error}); \
                     servers will not be cleaned up if this process is force-killed"
                );
                let _ = CloseHandle(handle);
                return None;
            }

            log::info!(
                "[SERVING] model servers are enrolled in a job object; none can outlive this \
                 process, however it exits"
            );
            Some(Job(handle))
        }
    }

    pub(super) fn adopt(pid: u32) {
        let Some(job) = job_handle() else {
            return;
        };

        unsafe {
            // `PROCESS_SET_QUOTA | PROCESS_TERMINATE` is exactly what
            // `AssignProcessToJobObject` documents as required, and nothing
            // more: this handle should not be able to read the child's memory.
            let process = match OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, false, pid) {
                Ok(handle) => handle,
                Err(error) => {
                    log::warn!("[SERVING] could not open model server {pid} to enrol it ({error})");
                    return;
                }
            };

            if let Err(error) = AssignProcessToJobObject(job, process) {
                log::warn!("[SERVING] could not enrol model server {pid} in the job ({error})");
            }

            // The process handle is ours and is finished with. The *job* keeps
            // its own reference to the child, so closing this does not release
            // it from the job.
            let _ = CloseHandle(process);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guarantee is reported, not assumed.
    ///
    /// On Windows this also exercises the whole creation path — the job is
    /// built, the kill-on-close limit is set, and a failure at either step is
    /// reported as "no guarantee" rather than as success.
    #[test]
    fn the_cleanup_guarantee_is_reported_honestly_for_this_platform() {
        let guaranteed = guarantees_cleanup();

        if cfg!(target_os = "windows") {
            assert!(
                guaranteed,
                "Windows can enforce this and the build says it cannot; a model server that \
                 outlives the app holds VRAM nothing can reclaim"
            );
        } else {
            assert!(
                !guaranteed,
                "no job object exists off Windows, so no guarantee may be claimed"
            );
        }
    }

    /// Enrolling something that is not there must not bring the app down.
    ///
    /// A server can exit between `spawn` returning and this being called —
    /// a bad `--model` path is enough — and the pid is then stale. Best-effort
    /// means best-effort.
    #[test]
    fn adopting_a_process_that_has_already_gone_is_survivable() {
        // A pid that cannot be a live process on any platform this ships to.
        adopt(u32::MAX);
    }
}

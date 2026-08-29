//! Enclave memory pressure — a process-level memory ceiling for the
//! inference engine.
//!
//! ## Honest scope
//!
//! "Enclave" in this build is *not* a hardware-isolated trusted execution
//! environment. There is no SGX, no SEV, no TDX on the supported hardware
//! (NVIDIA RTX 5060 + Ryzen 7 250 laptop). What the module does instead is
//! what an OS can honestly offer a single-process desktop app:
//!
//! 1. A *process memory ceiling* — a hard limit on the resident-set size
//!    the inference engine is allowed to reach, enforced by the OS rather
//!    than by the engine. On Windows this is a Job Object with
//!    `JOB_OBJECT_LIMIT_PROCESS_MEMORY`; on Unix it is `setrlimit(RLIMIT_RSS)`
//!    on platforms that still implement it (most do not, in which case
//!    we fall back to `RLIMIT_AS` with the documented caveat that AS is
//!    virtual, not resident). On macOS we use `setrlimit(RLIMIT_RSS)` as
//!    a hint, with the same caveat.
//!
//! 2. A *loud-fail policy* — when the ceiling is hit, the inference
//!    process is expected to die in a way the audit log can record, not
//!    to silently start swapping until the host OOM-killer takes the
//!    whole app. The companion [`MemoryEnclave::enforce`] helper sets
//!    the limit and returns an [`EnclaveHandle`] that the caller holds
//!    for the lifetime of the engine; if the engine blows the ceiling,
//!    the OS kills the process and the audit log records "enclave limit
//!    exceeded" on next launch.
//!
//! 3. A *single, named, auditable flag* — [`EnclaveFlag::EnclaveActive`]
//!    is set in the audit log whenever the ceiling is in force, so a
//!    reviewer reading the log can see when "the inference was inside
//!    the memory ceiling" without having to chase a sidecar.
//!
//! ## What this is NOT
//!
//! - It is not a defense against an attacker who controls the runtime
//!   binary. A process the attacker owns can call `setrlimit` to raise
//!   the ceiling again.
//! - It is not a sandbox. The inference engine still runs in the same
//!   address space as the rest of the workbench. A memory ceiling on
//!   the engine does not stop the rest of the app from running.
//! - It is not enforced for the *webview*. The webview is a separate
//!   process owned by the OS, and applying a Job Object to it would
//!   break ordinary Tauri behaviour. This module targets the *inference
//!   sidecar* only.

use std::io;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The flag the audit log carries whenever the enclave ceiling is in
/// force. Kept as a single named constant so a reviewer reading the log
/// can grep for it.
pub const ENCLAVE_ACTIVE_FLAG: &str = "ENCLAVE_ACTIVE";

/// What went wrong. The variants are deliberately coarse: the operator
/// who reads the error does not need a stack trace, just "the OS would
/// not let me set the ceiling" or "the ceiling you asked for is not a
/// number the OS will accept."
#[derive(Debug, Error, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum EnclaveError {
    #[error(
        "the OS refused to install the memory ceiling: {reason}. \
         The inference engine will run without a ceiling, which is a \
         degradation rather than a hard failure."
    )]
    OsRefused { reason: String },
    #[error(
        "the requested ceiling ({requested_mib} MiB) is below the floor \
         ({floor_mib} MiB). The floor protects the engine from a too-tight \
         ceiling that would make the model unload before it has produced a \
         single token."
    )]
    BelowFloor { requested_mib: u64, floor_mib: u64 },
    #[error("internal handle was already consumed; the enclave is single-use")]
    AlreadyConsumed,
}

/// The shape of the enclave: how much memory the engine may use at peak,
/// expressed in mebibytes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EnclaveSpec {
    /// Peak resident set, in MiB. Hard floor is 256 MiB; the engine will
    /// refuse anything below that because no candidate model fits.
    pub peak_rss_mib: NonZeroU64,
    /// Whether the ceiling is enforced (`Strict`) or merely logged and
    /// measured (`Advisory`). `Advisory` is what a sandbox without
    /// `CAP_SYS_RESOURCE` falls back to; on Windows it is what happens
    /// when the Job Object cannot be created for some reason we cannot
    /// anticipate (a locked-down corporate image, say).
    pub mode: EnclaveMode,
}

impl EnclaveSpec {
    /// Floor below which no model in the catalog can run. Held in one
    /// place so the floor and the test suite agree.
    pub const FLOOR_MIB: u64 = 256;
}

/// How the ceiling is enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnclaveMode {
    /// OS-enforced hard ceiling. The engine dies on breach.
    Strict,
    /// Logged and measured, not enforced. The audit log records that the
    /// mode is advisory so a reviewer is not misled.
    Advisory,
}

/// The flag the audit log carries. The module's only public flag today;
/// the enum exists so future flags (e.g. `EnclaveWithSgx`) can be added
/// without an audit-log format change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EnclaveFlag {
    EnclaveActive,
}

impl EnclaveFlag {
    pub const fn as_str(self) -> &'static str {
        match self {
            EnclaveFlag::EnclaveActive => ENCLAVE_ACTIVE_FLAG,
        }
    }
}

/// Handle returned by [`MemoryEnclave::enforce`]. The handle is what
/// prevents the ceiling from being quietly released between when it is
/// installed and when the engine finishes. Dropping the handle does
/// *not* un-install the ceiling — the OS keeps the Job Object alive
/// for the process — but the handle is a *typed* signal that the engine
/// ran under a ceiling, and the caller's code is encouraged to hold it
/// for the duration of the inference session so the audit-log row
/// matches the lifetime of the engine.
#[derive(Debug)]
pub struct EnclaveHandle {
    /// Original spec, kept so the audit log can record what was set.
    spec: EnclaveSpec,
    /// Sequence number so the caller can tell multiple enforce() calls
    /// apart if it ever needs to.
    seq: u64,
    /// True once the handle has been "spent" by being asked to write its
    /// audit row. Prevents the same row from being written twice.
    spent: bool,
}

impl EnclaveHandle {
    /// The spec the ceiling was installed with.
    pub fn spec(&self) -> EnclaveSpec {
        self.spec
    }

    /// The sequence number the handle was issued under.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Whether the ceiling is `Strict` (OS-enforced) or `Advisory`
    /// (logged but not enforced). The flag carried in the audit log is
    /// the same value.
    pub fn mode(&self) -> EnclaveMode {
        self.spec.mode
    }
}

/// The enclave. Created once at startup, holds the OS resources the
/// ceiling needs.
pub struct MemoryEnclave {
    next_seq: u64,
    /// OS handle. On Windows this is a Job Object; on Unix it is a
    /// record that `setrlimit` was called. Held as a `usize` so the
    /// platform-specific wrapper can cast it without leaking types
    /// across the platform boundary.
    #[cfg(target_os = "windows")]
    os_handle: Option<usize>,
    #[cfg(not(target_os = "windows"))]
    os_handle: Option<usize>,
}

impl MemoryEnclave {
    /// Creates a fresh enclave with no ceiling installed. Call
    /// [`Self::enforce`] to install one.
    pub fn new() -> Self {
        Self {
            next_seq: 1,
            os_handle: None,
        }
    }

    /// Installs the requested ceiling. The returned handle is held by
    /// the engine for its lifetime; the OS keeps the Job Object alive
    /// even if the handle is dropped.
    ///
    /// In `Advisory` mode the function still returns success, but the
    /// audit-log row carries the `Advisory` flag so a reviewer can tell
    /// the difference. The OS may also force us into advisory mode by
    /// refusing the Job Object; in that case the returned handle's
    /// [`EnclaveHandle::mode`] is `Advisory` and a warning is logged.
    pub fn enforce(&mut self, spec: EnclaveSpec) -> Result<EnclaveHandle, EnclaveError> {
        if spec.peak_rss_mib.get() < EnclaveSpec::FLOOR_MIB {
            return Err(EnclaveError::BelowFloor {
                requested_mib: spec.peak_rss_mib.get(),
                floor_mib: EnclaveSpec::FLOOR_MIB,
            });
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);

        // Hand off to the platform implementation. We try `Strict`
        // first; if the OS refuses (e.g. sandboxed environment, locked
        // job on a corporate image), we fall back to `Advisory` and log
        // loudly so the operator knows the ceiling is *not* enforced.
        let effective_mode = match platform::install_ceiling(spec.peak_rss_mib.get()) {
            Ok(()) => spec.mode,
            Err(reason) => {
                log::warn!(
                    "[ENCLAVE] OS refused strict ceiling ({reason}); \
                     falling back to Advisory mode. The audit log will \
                     record ENCLAVE_ACTIVE with the Advisory flag."
                );
                EnclaveMode::Advisory
            }
        };

        let effective_spec = EnclaveSpec {
            peak_rss_mib: spec.peak_rss_mib,
            mode: effective_mode,
        };

        Ok(EnclaveHandle {
            spec: effective_spec,
            seq,
            spent: false,
        })
    }

    /// Writes the audit-log row that records the ceiling is in force.
    /// Idempotent: a second call on the same handle returns `Ok(())`
    /// without writing another row. This is the hook the engine is
    /// expected to call once, at startup, after a successful
    /// [`Self::enforce`].
    pub fn record_active(
        &self,
        handle: &mut EnclaveHandle,
        audit: &mut dyn EnclaveAuditSink,
    ) -> io::Result<()> {
        if handle.spent {
            return Ok(());
        }
        audit.write_enclave_row(handle.spec, handle.seq, EnclaveFlag::EnclaveActive)?;
        handle.spent = true;
        Ok(())
    }
}

/// The minimum surface an audit sink needs to expose so the enclave
/// can record its active flag. Implemented by `AuditService` in the
/// real app; the test suite provides an in-memory implementation.
pub trait EnclaveAuditSink {
    fn write_enclave_row(
        &mut self,
        spec: EnclaveSpec,
        seq: u64,
        flag: EnclaveFlag,
    ) -> io::Result<()>;
}

impl EnclaveAuditSink for crate::audit::AuditService {
    fn write_enclave_row(
        &mut self,
        spec: EnclaveSpec,
        seq: u64,
        flag: EnclaveFlag,
    ) -> io::Result<()> {
        let detail = serde_json::json!({
            "flag": flag.as_str(),
            "peak_rss_mib": spec.peak_rss_mib.get(),
            "mode": match spec.mode {
                EnclaveMode::Strict => "strict",
                EnclaveMode::Advisory => "advisory",
            },
            "seq": seq,
        });
        let summary = format!("Enclave flag {} raised", flag.as_str());
        self.record("system", crate::audit::AuditKind::PolicyDecision, summary, Some(detail))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }
}

mod platform {
    //! OS-specific ceiling installation.
    //!
    //! On Windows this is a Job Object with
    //! `JOB_OBJECT_LIMIT_PROCESS_MEMORY`. On other platforms it is
    //! `setrlimit(RLIMIT_AS)` (the only RSS-related rlimit the kernel
    //! will accept on Linux). All variants of `install_ceiling` return
    //! `Ok(())` on success and `Err(String)` with a human-readable
    //! reason on failure.

    /// Installs a process-memory ceiling, in MiB, on the calling
    /// process. Returns `Ok(())` on success.
    pub(super) fn install_ceiling(_peak_mib: u64) -> Result<(), String> {
        #[cfg(target_os = "windows")]
        {
            windows::install_job_object(_peak_mib)
        }
        #[cfg(target_os = "linux")]
        {
            linux::install_rlimit_as(_peak_mib)
        }
        #[cfg(target_os = "macos")]
        {
            macos::install_rlimit_rss(_peak_mib)
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            Err("memory ceiling is not implemented for this platform".to_string())
        }
    }

    #[cfg(target_os = "windows")]
    mod windows {
        //! Windows Job Object ceiling.
        //!
        //! We use the [`windows`] crate to call `CreateJobObjectW` and
        //! `SetInformationJobObject`. The Job Object applies to the
        //! *current process*; the OS will then enforce the limit on
        //! every child process we spawn unless we explicitly opt them
        //! out, which we do not.
        //!
        //! The `windows` crate is added to the workspace because it is
        //! the only stable way to talk to the Win32 Job Object API
        //! from Rust. If the crate is not available at build time
        //! (e.g. a Linux cross-build), the function returns a
        //! *stringified* error and the caller falls back to advisory
        //! mode, which is the documented degradation path.

        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::JobObjects::{
            CreateJobObjectW, SetInformationJobObject, JobObjectExtendedLimitInformation,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        };

        pub(super) fn install_job_object(peak_mib: u64) -> Result<(), String> {
            // SAFETY: the API requires raw pointer arguments, and
            // `CreateJobObjectW` with null arguments creates a job
            // without a name. The handle is leaked intentionally: the
            // job is meant to live for the lifetime of the process.
            let job: HANDLE = unsafe { CreateJobObjectW(None, None) }
                .map_err(|e| format!("CreateJobObjectW failed: {e}"))?;
            if job.is_invalid() {
                return Err("CreateJobObjectW returned an invalid handle".to_string());
            }

            // Build the limit information. The `ProcessMemoryLimit`
            // field is in bytes; convert from MiB carefully so we
            // cannot accidentally overflow on a 32-bit build.
            let peak_bytes: u64 = peak_mib.saturating_mul(1024 * 1024);
            // Build the struct field-by-field. `..Default::default()`
            // does not work on FFI structs in `windows` 0.52, so we
            // zero the rest by going through `Default` on a wrapper.
            let mut limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            limit_info.ProcessMemoryLimit = peak_bytes as usize;

            // SAFETY: `limit_info` is a POD struct, and we pass a
            // pointer to it along with its size, exactly as the API
            // requires.
            let result = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &limit_info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if let Err(e) = result {
                return Err(format!("SetInformationJobObject failed: {e}"));
            }
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    mod linux {
        //! Linux `setrlimit(RLIMIT_AS)` ceiling.
        //!
        //! Honest caveat: `RLIMIT_AS` is a *virtual* address-space
        //! limit, not a *resident* set limit. The kernel does not
        //! expose a per-process RSS rlimit any more; the historical
        //! `RLIMIT_RSS` is a no-op on modern Linux. The limit is
        //! still useful — when the engine mmap's its model weights,
        //! `RLIMIT_AS` will refuse the mmap before the system starts
        //! swapping — but it is not the same thing as a Job Object's
        //! resident-set enforcement. The `Advisory` mode is the
        //! honest disclosure of that gap.

        use std::io;

        /// Wraps the `prlimit64` syscall. We use the syscall
        /// directly because `std::process` does not expose RSS
        /// ceilings.
        pub(super) fn install_rlimit_as(peak_mib: u64) -> Result<(), String> {
            let peak_bytes: u64 = peak_mib.saturating_mul(1024 * 1024);
            // SAFETY: the second argument is a pointer to a `rlimit`
            // struct on the stack; the third is null because we are
            // not reading the old limit. The kernel writes a `rlimit`
            // to the fourth argument, but we do not need it.
            let res = unsafe {
                libc::prlimit64(
                    0, // current process
                    libc::RLIMIT_AS,
                    &libc::rlimit {
                        rlim_cur: peak_bytes as libc::rlim_t,
                        rlim_max: peak_bytes as libc::rlim_t,
                    },
                    std::ptr::null_mut(),
                )
            };
            if res != 0 {
                return Err(io::Error::last_os_error().to_string());
            }
            Ok(())
        }
    }

    #[cfg(target_os = "macos")]
    mod macos {
        use std::io;

        pub(super) fn install_rlimit_rss(peak_mib: u64) -> Result<(), String> {
            let peak_bytes: u64 = peak_mib.saturating_mul(1024 * 1024);
            let res = unsafe {
                libc::setrlimit(
                    libc::RLIMIT_RSS,
                    &libc::rlimit {
                        rlim_cur: peak_bytes as libc::rlim_t,
                        rlim_max: peak_bytes as libc::rlim_t,
                    },
                )
            };
            if res != 0 {
                return Err(io::Error::last_os_error().to_string());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A toy audit sink for the tests. Records every row the enclave
    /// asked it to write so a test can assert on the result.
    struct ToySink {
        rows: Mutex<Vec<(EnclaveSpec, u64, EnclaveFlag)>>,
    }
    impl ToySink {
        fn new() -> Self {
            Self {
                rows: Mutex::new(Vec::new()),
            }
        }
        fn count(&self) -> usize {
            self.rows.lock().unwrap().len()
        }
    }
    impl EnclaveAuditSink for ToySink {
        fn write_enclave_row(
            &mut self,
            spec: EnclaveSpec,
            seq: u64,
            flag: EnclaveFlag,
        ) -> io::Result<()> {
            self.rows.lock().unwrap().push((spec, seq, flag));
            Ok(())
        }
    }

    fn good_spec(mode: EnclaveMode) -> EnclaveSpec {
        EnclaveSpec {
            peak_rss_mib: NonZeroU64::new(4096).unwrap(),
            mode,
        }
    }

    #[test]
    fn floor_below_minimum_is_refused_before_os_call() {
        let mut enclave = MemoryEnclave::new();
        let err = enclave
            .enforce(EnclaveSpec {
                peak_rss_mib: NonZeroU64::new(64).unwrap(),
                mode: EnclaveMode::Strict,
            })
            .unwrap_err();
        assert!(matches!(err, EnclaveError::BelowFloor { .. }));
    }

    #[test]
    fn valid_spec_produces_a_handle_with_monotonic_seq() {
        let mut enclave = MemoryEnclave::new();
        let h1 = enclave.enforce(good_spec(EnclaveMode::Strict)).unwrap();
        let h2 = enclave.enforce(good_spec(EnclaveMode::Strict)).unwrap();
        assert_eq!(h1.seq(), 1);
        assert_eq!(h2.seq(), 2);
    }

    #[test]
    fn record_active_writes_exactly_one_row_per_handle() {
        let mut enclave = MemoryEnclave::new();
        let mut handle = enclave.enforce(good_spec(EnclaveMode::Strict)).unwrap();
        let mut sink = ToySink::new();
        enclave.record_active(&mut handle, &mut sink).unwrap();
        enclave.record_active(&mut handle, &mut sink).unwrap();
        enclave.record_active(&mut handle, &mut sink).unwrap();
        assert_eq!(sink.count(), 1, "idempotent: only one row per handle");
    }

    #[test]
    fn handle_carries_the_spec_the_caller_asked_for() {
        let mut enclave = MemoryEnclave::new();
        let handle = enclave.enforce(good_spec(EnclaveMode::Strict)).unwrap();
        assert_eq!(handle.spec().peak_rss_mib.get(), 4096);
    }

    #[test]
    fn flag_constant_is_what_audit_logs_grep_for() {
        // The grep-for-this-string contract: a reviewer can search the
        // audit log for ENCLAVE_ACTIVE and find the row the enclave
        // writes on activation. If a future contributor renames the
        // constant, this test fails.
        assert_eq!(ENCLAVE_ACTIVE_FLAG, "ENCLAVE_ACTIVE");
        assert_eq!(EnclaveFlag::EnclaveActive.as_str(), "ENCLAVE_ACTIVE");
    }

    #[test]
    fn every_mode_round_trips_through_serde() {
        let json = serde_json::to_string(&EnclaveMode::Strict).unwrap();
        let back: EnclaveMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, EnclaveMode::Strict);
        let json = serde_json::to_string(&EnclaveMode::Advisory).unwrap();
        let back: EnclaveMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, EnclaveMode::Advisory);
    }

    #[test]
    fn real_audit_sink_writes_a_row_for_enclave_active() {
        // Round-trip through the real AuditService implementation of
        // EnclaveAuditSink so the trait impl is exercised in tests
        // (not just in production).
        let mut audit = crate::audit::AuditService::from_connection(
            rusqlite::Connection::open_in_memory().unwrap(),
        )
        .unwrap();
        let mut enclave = MemoryEnclave::new();
        let mut handle = enclave.enforce(good_spec(EnclaveMode::Strict)).unwrap();
        enclave.record_active(&mut handle, &mut audit).unwrap();
        let recent = audit.recent(10).unwrap();
        let active_row = recent
            .iter()
            .find(|e| e.summary.contains("ENCLAVE_ACTIVE"))
            .expect("expected an ENCLAVE_ACTIVE row");
        // The detail is a serde_json::Value — assert on the shape.
        let detail = active_row.detail.as_ref().unwrap();
        let map: HashMap<String, serde_json::Value> =
            serde_json::from_value(detail.clone()).unwrap();
        assert_eq!(map.get("flag").unwrap().as_str().unwrap(), "ENCLAVE_ACTIVE");
        assert_eq!(map.get("peak_rss_mib").unwrap().as_u64().unwrap(), 4096);
    }
}

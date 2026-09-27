//! Independent observation — what the operating system says ARJUN is connected to.
//!
//! The broker in [`super::broker`] can report that it refused everything, and its
//! own tests prove the decision logic is right. Neither shows that *no other part
//! of the process* opened a socket: a stray library, a transitive dependency, or
//! a bug that bypassed the broker would be invisible to a monitor the broker
//! writes itself.
//!
//! So this module does not ask ARJUN anything. It asks Windows, through
//! `GetExtendedTcpTable`, for the TCP connections owned by ARJUN's processes,
//! and reports whatever comes back. If the broker is lying — or simply wrong —
//! the two views disagree, and that disagreement is the finding.
//!
//! "ARJUN's processes" means this one and every process it started: the
//! WebView2 browser that draws the window, the model servers, the Python
//! sidecars, the agent runtime. Watching only this process ID once showed a
//! clean table while the WebView2 browser process was connected to Microsoft
//! (see `super::webview`).
//!
//! Chosen over Sysmon deliberately: Sysmon needs an install and administrator
//! rights, which cannot be assumed on a demo laptop. This needs neither, and the
//! evidence is the same shape — a per-process connection list from the OS.
//!
//! Scope, stated plainly: this covers TCP for ARJUN's process tree. It does not see UDP,
//! raw sockets, or a connection opened and closed entirely between two polls. It
//! is corroboration from a second, independent vantage point, not a packet capture.

use serde::{Deserialize, Serialize};

/// One connection the operating system attributes to one of ARJUN's processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservedConnection {
    pub local: String,
    pub remote: String,
    /// False when the remote address leaves this machine — the thing that matters.
    pub loopback: bool,
    /// The process that owns the connection, so a finding names its source.
    pub pid: u32,
    pub process: String,
}

/// What the OS reports, and whether anything left the machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationReport {
    pub connections: Vec<ObservedConnection>,
    /// Connections whose remote address is not loopback.
    pub external_count: usize,
    /// The executable name of every process that was watched, this one first.
    pub processes: Vec<String>,
    /// Set when the platform cannot be queried, so the UI can say "unknown"
    /// rather than showing an empty list that looks like proof of nothing.
    pub unavailable_reason: Option<String>,
}

impl ObservationReport {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            connections: Vec::new(),
            external_count: 0,
            processes: Vec::new(),
            unavailable_reason: Some(reason.into()),
        }
    }

    fn from_connections(connections: Vec<ObservedConnection>, processes: Vec<String>) -> Self {
        let external_count = connections.iter().filter(|c| !c.loopback).count();
        Self {
            connections,
            external_count,
            processes,
            unavailable_reason: None,
        }
    }
}

/// One row of the operating system's process table.
#[derive(Debug, Clone)]
struct ProcessRow {
    pid: u32,
    parent: u32,
    name: String,
}

/// `root` and every process it started, directly or through a child, `root`
/// first.
///
/// Windows reuses process IDs, so a row's recorded parent ID can now belong to
/// one of ours while the row itself was started by the ID's previous owner.
/// Such a row predates its supposed parent, and `created` (a creation time for
/// a process ID) is how it is told apart. When either time is unknown the row
/// is kept: an extra process can only add connections to the report, never
/// hide one.
fn descendants(rows: &[ProcessRow], root: u32, created: impl Fn(u32) -> Option<u64>) -> Vec<&ProcessRow> {
    let mut tree: Vec<&ProcessRow> = rows.iter().filter(|r| r.pid == root).take(1).collect();
    let mut included: std::collections::HashSet<u32> = tree.iter().map(|r| r.pid).collect();
    let mut next = 0;
    while next < tree.len() {
        let parent = tree[next].pid;
        let parent_created = created(parent);
        for row in rows {
            if row.parent != parent || included.contains(&row.pid) {
                continue;
            }
            let predates_parent =
                matches!((parent_created, created(row.pid)), (Some(p), Some(c)) if c < p);
            if predates_parent {
                continue;
            }
            included.insert(row.pid);
            tree.push(row);
        }
        next += 1;
    }
    tree
}

/// True when an address never leaves the machine.
///
/// Covers IPv4 `127.0.0.0/8`, IPv6 `::1`, the unspecified addresses a listening
/// socket reports, and IPv4-mapped loopback (`::ffff:127.0.0.1`), which is what
/// a dual-stack listener on this machine actually shows up as.
fn is_loopback_addr(addr: &std::net::IpAddr) -> bool {
    match addr {
        std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified(),
        std::net::IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            match v6.to_ipv4_mapped() {
                Some(v4) => v4.is_loopback() || v4.is_unspecified(),
                None => false,
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::{descendants, is_loopback_addr, ObservationReport, ObservedConnection, ProcessRow};
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use windows::Win32::Foundation::{CloseHandle, FILETIME};
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    use windows::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCP6TABLE_OWNER_PID, MIB_TCPTABLE_OWNER_PID,
        TCP_TABLE_OWNER_PID_ALL,
    };
    use windows::Win32::Networking::WinSock::{AF_INET, AF_INET6};

    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

    /// Pulls the owner-PID TCP table for one address family into a byte buffer.
    ///
    /// The table is sized between calls — it can grow while we are asking — so
    /// the required size is requested first and the fetch retried a bounded
    /// number of times rather than looping until it happens to fit.
    fn fetch_table(family: u16) -> Result<Vec<u8>, String> {
        let mut size: u32 = 0;

        // SAFETY: a null table pointer with size 0 is the documented way to ask
        // for the required buffer size; the call only writes through `size`.
        let rc = unsafe {
            GetExtendedTcpTable(
                None,
                &mut size,
                false,
                family as u32,
                TCP_TABLE_OWNER_PID_ALL,
                0,
            )
        };
        if rc != ERROR_INSUFFICIENT_BUFFER && rc != 0 {
            return Err(format!("GetExtendedTcpTable size query failed ({rc})"));
        }

        for _ in 0..4 {
            let mut buffer = vec![0u8; size as usize];
            // SAFETY: `buffer` is `size` bytes and stays alive for the call;
            // the API writes at most `size` bytes and updates `size` if it needs
            // more, in which case we allocate again on the next iteration.
            let rc = unsafe {
                GetExtendedTcpTable(
                    Some(buffer.as_mut_ptr() as *mut _),
                    &mut size,
                    false,
                    family as u32,
                    TCP_TABLE_OWNER_PID_ALL,
                    0,
                )
            };
            match rc {
                0 => return Ok(buffer),
                ERROR_INSUFFICIENT_BUFFER => continue,
                other => return Err(format!("GetExtendedTcpTable failed ({other})")),
            }
        }
        Err("the TCP table kept growing between reads".to_string())
    }

    /// Every process on the machine, from one Toolhelp snapshot.
    fn process_table() -> Result<Vec<ProcessRow>, String> {
        // SAFETY: TH32CS_SNAPPROCESS with process ID 0 snapshots every process;
        // the handle is closed below on every path.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }
            .map_err(|e| format!("CreateToolhelp32Snapshot failed ({e})"))?;
        let mut rows = Vec::new();
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        // SAFETY: `entry` is a PROCESSENTRY32W with dwSize set, as both calls
        // require, and it outlives them.
        let mut more = unsafe { Process32FirstW(snapshot, &mut entry) }.is_ok();
        while more {
            let len = entry.szExeFile.iter().position(|&c| c == 0).unwrap_or(entry.szExeFile.len());
            rows.push(ProcessRow {
                pid: entry.th32ProcessID,
                parent: entry.th32ParentProcessID,
                name: String::from_utf16_lossy(&entry.szExeFile[..len]),
            });
            more = unsafe { Process32NextW(snapshot, &mut entry) }.is_ok();
        }
        // SAFETY: `snapshot` is a valid handle from CreateToolhelp32Snapshot.
        let _ = unsafe { CloseHandle(snapshot) };
        Ok(rows)
    }

    /// When a process started, in FILETIME ticks; `None` when it cannot be opened.
    fn creation_time(pid: u32) -> Option<u64> {
        // SAFETY: limited query access is all GetProcessTimes needs; the handle
        // is closed before returning.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
        let (mut created, mut exited, mut kernel, mut user) =
            (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
        // SAFETY: all four out-pointers are live FILETIMEs on this stack.
        let read = unsafe { GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) };
        let _ = unsafe { CloseHandle(handle) };
        read.ok()?;
        Some(((created.dwHighDateTime as u64) << 32) | created.dwLowDateTime as u64)
    }

    fn collect_ipv4(owners: &HashMap<u32, String>, out: &mut Vec<ObservedConnection>) -> Result<(), String> {
        let buffer = fetch_table(AF_INET.0)?;
        if buffer.len() < std::mem::size_of::<MIB_TCPTABLE_OWNER_PID>() {
            return Ok(());
        }

        // SAFETY: the buffer was filled by GetExtendedTcpTable for AF_INET with
        // TCP_TABLE_OWNER_PID_ALL, so it begins with a MIB_TCPTABLE_OWNER_PID
        // whose `table` field is the first of `dwNumEntries` rows.
        let table = unsafe { &*(buffer.as_ptr() as *const MIB_TCPTABLE_OWNER_PID) };
        let rows = unsafe {
            std::slice::from_raw_parts(table.table.as_ptr(), table.dwNumEntries as usize)
        };

        for row in rows {
            let Some(process) = owners.get(&row.dwOwningPid) else {
                continue;
            };
            // Addresses arrive in network byte order; ports arrive in the low
            // 16 bits, also network order.
            let local = IpAddr::V4(Ipv4Addr::from(row.dwLocalAddr.to_le_bytes()));
            let remote = IpAddr::V4(Ipv4Addr::from(row.dwRemoteAddr.to_le_bytes()));
            let local_port = u16::from_be((row.dwLocalPort & 0xFFFF) as u16);
            let remote_port = u16::from_be((row.dwRemotePort & 0xFFFF) as u16);

            out.push(ObservedConnection {
                local: format!("{local}:{local_port}"),
                remote: format!("{remote}:{remote_port}"),
                loopback: is_loopback_addr(&remote),
                pid: row.dwOwningPid,
                process: process.clone(),
            });
        }
        Ok(())
    }

    fn collect_ipv6(owners: &HashMap<u32, String>, out: &mut Vec<ObservedConnection>) -> Result<(), String> {
        let buffer = fetch_table(AF_INET6.0)?;
        if buffer.len() < std::mem::size_of::<MIB_TCP6TABLE_OWNER_PID>() {
            return Ok(());
        }

        // SAFETY: as above, for AF_INET6 the buffer begins with a
        // MIB_TCP6TABLE_OWNER_PID followed by `dwNumEntries` rows.
        let table = unsafe { &*(buffer.as_ptr() as *const MIB_TCP6TABLE_OWNER_PID) };
        let rows = unsafe {
            std::slice::from_raw_parts(table.table.as_ptr(), table.dwNumEntries as usize)
        };

        for row in rows {
            let Some(process) = owners.get(&row.dwOwningPid) else {
                continue;
            };
            let local = IpAddr::V6(Ipv6Addr::from(row.ucLocalAddr));
            let remote = IpAddr::V6(Ipv6Addr::from(row.ucRemoteAddr));
            let local_port = u16::from_be((row.dwLocalPort & 0xFFFF) as u16);
            let remote_port = u16::from_be((row.dwRemotePort & 0xFFFF) as u16);

            out.push(ObservedConnection {
                local: format!("[{local}]:{local_port}"),
                remote: format!("[{remote}]:{remote_port}"),
                loopback: is_loopback_addr(&remote),
                pid: row.dwOwningPid,
                process: process.clone(),
            });
        }
        Ok(())
    }

    pub fn observe() -> ObservationReport {
        let rows = match process_table() {
            Ok(rows) => rows,
            Err(e) => return ObservationReport::unavailable(e),
        };
        let tree = descendants(&rows, std::process::id(), creation_time);
        // Without this process in the table there is nothing to walk from, and
        // an empty tree would read as a clean result.
        if tree.is_empty() {
            return ObservationReport::unavailable("this process is missing from the process table");
        }
        let processes: Vec<String> = tree.iter().map(|r| r.name.clone()).collect();
        let owners: HashMap<u32, String> = tree.iter().map(|r| (r.pid, r.name.clone())).collect();
        let mut connections = Vec::new();

        // A failure on one family is reported rather than swallowed: an empty
        // list must never be mistaken for a clean result.
        if let Err(e) = collect_ipv4(&owners, &mut connections) {
            return ObservationReport::unavailable(e);
        }
        if let Err(e) = collect_ipv6(&owners, &mut connections) {
            return ObservationReport::unavailable(e);
        }

        ObservationReport::from_connections(connections, processes)
    }
}

#[cfg(not(windows))]
mod platform {
    use super::ObservationReport;

    pub fn observe() -> ObservationReport {
        ObservationReport::unavailable(
            "Independent connection observation is implemented for Windows only.",
        )
    }
}

/// Asks the operating system which TCP connections belong to ARJUN's processes:
/// this one and every process it started.
pub fn observe_own_connections() -> ObservationReport {
    platform::observe()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn loopback_forms_are_all_recognised() {
        for addr in [
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(127, 5, 5, 5)),
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            // What a dual-stack listener on this machine actually reports.
            IpAddr::V6(Ipv4Addr::new(127, 0, 0, 1).to_ipv6_mapped()),
        ] {
            assert!(is_loopback_addr(&addr), "{addr} should count as loopback");
        }
    }

    #[test]
    fn routable_addresses_are_not_loopback() {
        for addr in [
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)),
            IpAddr::V6(Ipv6Addr::new(0x2606, 0x4700, 0, 0, 0, 0, 0, 1)),
            IpAddr::V6(Ipv4Addr::new(8, 8, 8, 8).to_ipv6_mapped()),
        ] {
            assert!(!is_loopback_addr(&addr), "{addr} should not count as loopback");
        }
    }

    /// An unavailable observation must never look like a clean result.
    #[test]
    fn unavailable_is_distinguishable_from_no_connections() {
        let unavailable = ObservationReport::unavailable("no platform support");
        let clean = ObservationReport::from_connections(Vec::new(), vec!["sarathi.exe".into()]);

        assert!(unavailable.unavailable_reason.is_some());
        assert!(clean.unavailable_reason.is_none());
        assert_eq!(unavailable.connections.len(), clean.connections.len());
    }

    #[test]
    fn external_connections_are_counted() {
        let report = ObservationReport::from_connections(vec![
            ObservedConnection {
                local: "127.0.0.1:11435".into(),
                remote: "127.0.0.1:54321".into(),
                loopback: true,
                pid: 10,
                process: "llama-server.exe".into(),
            },
            ObservedConnection {
                local: "192.168.1.10:52000".into(),
                remote: "8.8.8.8:443".into(),
                loopback: false,
                pid: 11,
                process: "msedgewebview2.exe".into(),
            },
        ], vec!["sarathi.exe".into(), "llama-server.exe".into(), "msedgewebview2.exe".into()]);
        assert_eq!(report.external_count, 1);
    }

    fn row(pid: u32, parent: u32, name: &str) -> ProcessRow {
        ProcessRow { pid, parent, name: name.into() }
    }

    fn pids(tree: &[&ProcessRow]) -> Vec<u32> {
        tree.iter().map(|r| r.pid).collect()
    }

    /// The WebView2 browser is a child of ARJUN and its renderers are its
    /// children: a connection from any of them is ARJUN's.
    #[test]
    fn the_tree_reaches_grandchildren_and_nothing_else() {
        let rows = vec![
            row(1, 0, "explorer.exe"),
            row(10, 1, "sarathi.exe"),
            row(20, 10, "msedgewebview2.exe"),
            row(21, 20, "msedgewebview2.exe"),
            row(30, 10, "llama-server.exe"),
            row(40, 1, "chrome.exe"),
            row(41, 40, "chrome.exe"),
        ];
        let tree = descendants(&rows, 10, |_| None);
        assert_eq!(pids(&tree), vec![10, 20, 30, 21]);
    }

    /// Process 30 was started by an earlier owner of ID 10, before ARJUN had it.
    #[test]
    fn a_process_older_than_its_recorded_parent_belongs_to_someone_else() {
        let rows = vec![row(10, 1, "sarathi.exe"), row(20, 10, "msedgewebview2.exe"), row(30, 10, "svchost.exe")];
        let created = |pid: u32| match pid {
            10 => Some(1_000),
            20 => Some(2_000),
            30 => Some(500),
            _ => None,
        };
        assert_eq!(pids(&descendants(&rows, 10, created)), vec![10, 20]);
    }

    /// Not knowing when a process started must not remove it from the report.
    #[test]
    fn a_child_whose_start_time_is_unknown_is_still_watched() {
        let rows = vec![row(10, 1, "sarathi.exe"), row(20, 10, "msedgewebview2.exe")];
        let created = |pid: u32| if pid == 10 { Some(1_000) } else { None };
        assert_eq!(pids(&descendants(&rows, 10, created)), vec![10, 20]);
    }

    /// Windows reports the idle process as its own parent; a loop in the
    /// table must not loop the walk.
    #[test]
    fn a_process_that_is_its_own_parent_does_not_loop() {
        let rows = vec![row(0, 0, "[System Process]"), row(10, 0, "sarathi.exe")];
        assert_eq!(pids(&descendants(&rows, 0, |_| None)), vec![0, 10]);
        assert!(descendants(&rows, 99, |_| None).is_empty());
    }

    /// Runs against the live OS. It asserts only that the query returns a
    /// coherent answer — a machine with a real connection open is not a failure.
    #[test]
    fn observing_this_process_returns_a_coherent_report() {
        let report = observe_own_connections();
        if report.unavailable_reason.is_none() {
            assert_eq!(
                report.external_count,
                report.connections.iter().filter(|c| !c.loopback).count()
            );
            assert!(!report.processes.is_empty(), "the watched processes include this one");
        }
    }
}

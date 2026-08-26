//! Independent observation — what the operating system says ARJUN is connected to.
//!
//! The broker in [`super::broker`] can report that it refused everything, and its
//! own tests prove the decision logic is right. Neither shows that *no other part
//! of the process* opened a socket: a stray library, a transitive dependency, or
//! a bug that bypassed the broker would be invisible to a monitor the broker
//! writes itself.
//!
//! So this module does not ask ARJUN anything. It asks Windows, through
//! `GetExtendedTcpTable`, for the TCP connections owned by this process ID, and
//! reports whatever comes back. If the broker is lying — or simply wrong — the
//! two views disagree, and that disagreement is the finding.
//!
//! Chosen over Sysmon deliberately: Sysmon needs an install and administrator
//! rights, which cannot be assumed on a demo laptop. This needs neither, and the
//! evidence is the same shape — a per-process connection list from the OS.
//!
//! Scope, stated plainly: this covers TCP for this process. It does not see UDP,
//! raw sockets, or a connection opened and closed entirely between two polls. It
//! is corroboration from a second, independent vantage point, not a packet capture.

use serde::{Deserialize, Serialize};

/// One connection the operating system attributes to this process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservedConnection {
    pub local: String,
    pub remote: String,
    /// False when the remote address leaves this machine — the thing that matters.
    pub loopback: bool,
}

/// What the OS reports, and whether anything left the machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationReport {
    pub connections: Vec<ObservedConnection>,
    /// Connections whose remote address is not loopback.
    pub external_count: usize,
    /// Set when the platform cannot be queried, so the UI can say "unknown"
    /// rather than showing an empty list that looks like proof of nothing.
    pub unavailable_reason: Option<String>,
}

impl ObservationReport {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            connections: Vec::new(),
            external_count: 0,
            unavailable_reason: Some(reason.into()),
        }
    }

    fn from_connections(connections: Vec<ObservedConnection>) -> Self {
        let external_count = connections.iter().filter(|c| !c.loopback).count();
        Self {
            connections,
            external_count,
            unavailable_reason: None,
        }
    }
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
    use super::{is_loopback_addr, ObservationReport, ObservedConnection};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
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

    fn collect_ipv4(pid: u32, out: &mut Vec<ObservedConnection>) -> Result<(), String> {
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
            if row.dwOwningPid != pid {
                continue;
            }
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
            });
        }
        Ok(())
    }

    fn collect_ipv6(pid: u32, out: &mut Vec<ObservedConnection>) -> Result<(), String> {
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
            if row.dwOwningPid != pid {
                continue;
            }
            let local = IpAddr::V6(Ipv6Addr::from(row.ucLocalAddr));
            let remote = IpAddr::V6(Ipv6Addr::from(row.ucRemoteAddr));
            let local_port = u16::from_be((row.dwLocalPort & 0xFFFF) as u16);
            let remote_port = u16::from_be((row.dwRemotePort & 0xFFFF) as u16);

            out.push(ObservedConnection {
                local: format!("[{local}]:{local_port}"),
                remote: format!("[{remote}]:{remote_port}"),
                loopback: is_loopback_addr(&remote),
            });
        }
        Ok(())
    }

    pub fn observe() -> ObservationReport {
        let pid = std::process::id();
        let mut connections = Vec::new();

        // A failure on one family is reported rather than swallowed: an empty
        // list must never be mistaken for a clean result.
        if let Err(e) = collect_ipv4(pid, &mut connections) {
            return ObservationReport::unavailable(e);
        }
        if let Err(e) = collect_ipv6(pid, &mut connections) {
            return ObservationReport::unavailable(e);
        }

        ObservationReport::from_connections(connections)
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

/// Asks the operating system which TCP connections belong to this process.
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
        let clean = ObservationReport::from_connections(Vec::new());

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
            },
            ObservedConnection {
                local: "192.168.1.10:52000".into(),
                remote: "8.8.8.8:443".into(),
                loopback: false,
            },
        ]);
        assert_eq!(report.external_count, 1);
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
        }
    }
}

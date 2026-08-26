//! Operating mode — the switch that decides whether ARJUN may touch the network.
//!
//! PS 26117 requires proof that no external call is made *at any point*. That is
//! only defensible if there is exactly one condition under which a call is even
//! attemptable, and it is impossible for confidential data to be present while
//! that condition holds.
//!
//! Hence two modes and one invariant:
//!
//! - [`OperatingMode::Provisioning`] — the network is reachable, but only for the
//!   model catalog and weight download, and **no document may be opened, ingested,
//!   retrieved or processed**.
//! - [`OperatingMode::Work`] — every outbound call is refused. All confidential
//!   work happens here.
//!
//! The invariant is what makes the guarantee hold: *confidential data and network
//! access are never enabled at the same time*. Neither mode is "secure" on its
//! own; the safety comes from the two never overlapping.
//!
//! Work is the default. A fresh install that has never been configured must not
//! start in the mode that can reach the internet.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OperatingMode {
    /// Network permitted, strictly for model acquisition. Documents refused.
    Provisioning,
    /// Network refused. Confidential work permitted.
    Work,
}

impl Default for OperatingMode {
    /// Work, always. Defaulting to Provisioning would mean a misconfigured or
    /// freshly installed deployment silently comes up able to reach the internet.
    fn default() -> Self {
        OperatingMode::Work
    }
}

impl OperatingMode {
    /// Whether an outbound request may even be attempted in this mode.
    ///
    /// This is necessary, never sufficient — the broker still checks the host
    /// allowlist afterwards.
    pub const fn permits_network(self) -> bool {
        matches!(self, OperatingMode::Provisioning)
    }

    /// Whether confidential material may be opened, indexed or processed.
    ///
    /// The exact complement of [`Self::permits_network`]. If these two ever
    /// return true for the same mode, the sovereign guarantee is gone.
    pub const fn permits_confidential_data(self) -> bool {
        matches!(self, OperatingMode::Work)
    }

    /// Label for logs, audit records and the UI.
    pub const fn label(self) -> &'static str {
        match self {
            OperatingMode::Provisioning => "Provisioning",
            OperatingMode::Work => "Work",
        }
    }
}

impl fmt::Display for OperatingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Why an operation was refused, so the audit record explains itself rather
/// than recording a bare denial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Refusal {
    /// An outbound call attempted while in Work mode.
    NetworkInWorkMode { host: String },
    /// An outbound call to a host that is not on the allowlist, in any mode.
    HostNotAllowed { host: String },
    /// A URL that could not be parsed, or carried no host at all.
    UnparseableTarget { target: String },
    /// A non-HTTPS scheme. Model weights are large and integrity matters, and
    /// permitting plaintext would also permit a trivially spoofed allowlist host.
    InsecureScheme { scheme: String },
    /// Confidential data touched while the network was reachable.
    DataInProvisioningMode { operation: String },
}

impl Refusal {
    /// One-line explanation, suitable for the audit log and the UI.
    pub fn reason(&self) -> String {
        match self {
            Refusal::NetworkInWorkMode { host } => format!(
                "Refused a connection to {host}: ARJUN is in Work mode, which permits no outbound calls."
            ),
            Refusal::HostNotAllowed { host } => format!(
                "Refused a connection to {host}: not on the model-acquisition allowlist."
            ),
            Refusal::UnparseableTarget { target } => format!(
                "Refused a connection: {target:?} is not a URL with a host."
            ),
            Refusal::InsecureScheme { scheme } => format!(
                "Refused a connection over {scheme}: only https is permitted."
            ),
            Refusal::DataInProvisioningMode { operation } => format!(
                "Refused {operation}: ARJUN is in Provisioning mode, where the network is \
                 reachable, so confidential material may not be opened."
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_the_mode_that_cannot_reach_the_internet() {
        assert_eq!(OperatingMode::default(), OperatingMode::Work);
        assert!(!OperatingMode::default().permits_network());
    }

    /// The invariant, asserted directly: no mode may allow both.
    #[test]
    fn network_and_confidential_data_are_never_both_permitted() {
        for mode in [OperatingMode::Provisioning, OperatingMode::Work] {
            assert!(
                !(mode.permits_network() && mode.permits_confidential_data()),
                "{mode} permits network and confidential data at once"
            );
        }
    }

    /// And no mode may forbid both, which would make the app inert.
    #[test]
    fn every_mode_permits_exactly_one_of_the_two() {
        for mode in [OperatingMode::Provisioning, OperatingMode::Work] {
            assert_ne!(mode.permits_network(), mode.permits_confidential_data());
        }
    }
}

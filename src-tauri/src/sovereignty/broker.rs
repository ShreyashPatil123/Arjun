//! The network broker — the only place in ARJUN that may open an outbound socket.
//!
//! Sarathi reached the network from four different modules, each constructing its
//! own `reqwest::Client`. That is impossible to audit: proving "no external calls"
//! means proving a negative about every call site, and every new one silently
//! widens the claim.
//!
//! The broker replaces that with a single chokepoint. It owns the one outbound
//! client in the process, and a CI check fails the build if any other module
//! constructs one. Auditing the sovereign claim then means reading this file.
//!
//! ## What it enforces
//!
//! 1. **Mode.** Nothing leaves in [`OperatingMode::Work`].
//! 2. **Scheme.** https only — plaintext would let anyone on the path answer for
//!    an allowlisted host.
//! 3. **Host, from the parsed URL.** Checked against an exact-match allowlist,
//!    never a string prefix. A lookalike host carries the
//!    prefix but not the host, and prefix matching is how allowlists get bypassed.
//! 4. **Redirects.** Not followed at all. A permitted host that redirected to an
//!    arbitrary one would otherwise walk straight through checks 1 to 3.
//!
//! Every decision — permitted and refused alike — is recorded. A monitor that
//! logs only refusals cannot show that nothing was sent.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::mode::{OperatingMode, Refusal};
use crate::audit::{AuditKind, AuditService};

/// Hosts reachable in Provisioning mode. Exact matches only.
///
/// This is the entire external surface of the product. Adding an entry widens
/// what "sovereign" means here, so each one should be justified in review.
const MODEL_ACQUISITION_HOSTS: &[&str] = &[
    "huggingface.co",
    // Weight blobs are served from CDN hosts rather than the API host, so
    // omitting these would make every download fail at the last hop.
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
];

/// How many egress decisions are retained for the live monitor.
///
/// The monitor is a demo and triage surface, not the system of record — Phase 1
/// moves the durable copy into the append-only audit table.
const EVENT_RING_CAPACITY: usize = 512;

/// What the broker decided about one outbound attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EgressEvent {
    pub at: DateTime<Utc>,
    /// Host as parsed from the URL, never the raw string.
    pub host: String,
    pub mode: OperatingMode,
    pub permitted: bool,
    /// Why it was refused, or why it was allowed.
    pub reason: String,
    /// Set when the app was deliberately testing its own controls.
    pub canary: bool,
}

/// Errors surfaced to callers. A refusal is not exceptional — it is the expected
/// outcome in Work mode — so it carries its explanation with it.
#[derive(Debug, thiserror::Error)]
pub enum BrokerError {
    #[error("{}", .0.reason())]
    Refused(Refusal),
    #[error("network error: {0}")]
    Transport(#[from] reqwest::Error),
}

pub struct NetworkBroker {
    mode: RwLock<OperatingMode>,
    client: reqwest::Client,
    events: Mutex<VecDeque<EgressEvent>>,
    /// Attached once the application data directory is known.
    ///
    /// The in-memory ring above is what the live monitor reads — it has to be
    /// cheap enough to poll every two seconds. This is the durable copy, so a
    /// refusal deep inside the downloader still reaches the permanent record
    /// rather than only the panel someone happens to be looking at.
    audit: RwLock<Option<Arc<AuditService>>>,
}

/// Process-wide broker.
///
/// The call sites that need it — the catalog, the resolver, the downloader — are
/// deep in free functions that no `AppHandle` reaches, and threading one through
/// every layer would touch far more code than it protects. This mirrors the
/// existing [`crate::core::event_bus::get_event_bus`] idiom.
///
/// A singleton is also the right shape here: "there is exactly one way out" is
/// the property being enforced, and a per-caller broker would quietly undo it.
static GLOBAL_BROKER: OnceLock<Arc<NetworkBroker>> = OnceLock::new();

/// The one broker. Created on first use, so ordering against Tauri setup does
/// not matter and tests can reach it without booting the app.
pub fn global_broker() -> &'static Arc<NetworkBroker> {
    GLOBAL_BROKER.get_or_init(NetworkBroker::new)
}

impl NetworkBroker {
    pub fn new() -> Arc<Self> {
        // Redirects are refused outright rather than followed and re-checked.
        // Re-checking is possible, but "follow nothing" is a smaller thing to
        // prove correct, and the model hosts need no cross-host redirect that
        // the caller cannot handle explicitly.
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("ARJUN")
            .build()
            .expect("the outbound HTTP client must build");

        Arc::new(Self {
            mode: RwLock::new(OperatingMode::default()),
            client,
            events: Mutex::new(VecDeque::with_capacity(EVENT_RING_CAPACITY)),
            audit: RwLock::new(None),
        })
    }

    /// Starts writing decisions to the permanent record.
    ///
    /// Called once at startup, after the data directory exists. Decisions made
    /// before this still reach the ring and the log; they are simply not durable,
    /// which is the correct trade for the handful of events that can occur
    /// before storage is ready.
    pub fn attach_audit(&self, audit: Arc<AuditService>) {
        *self.audit.write().expect("audit lock poisoned") = Some(audit);
    }

    pub fn mode(&self) -> OperatingMode {
        *self.mode.read().expect("mode lock poisoned")
    }

    /// Switches mode, returning the previous one so the caller can record the
    /// transition. Entering Work mode is always safe; entering Provisioning is
    /// gated on the caller having checked the admin role and confirmed that no
    /// confidential material is open.
    pub fn set_mode(&self, next: OperatingMode) -> OperatingMode {
        let mut guard = self.mode.write().expect("mode lock poisoned");
        let previous = *guard;
        *guard = next;
        log::info!("[SOVEREIGNTY] mode {previous} -> {next}");
        previous
    }

    /// Decides whether a target may be contacted, without contacting it.
    ///
    /// Split out from [`Self::get`] so the decision is testable on its own, and
    /// so the canary exercises the real logic rather than a copy of it.
    pub fn evaluate(&self, target: &str) -> Result<String, Refusal> {
        let url = reqwest::Url::parse(target).map_err(|_| Refusal::UnparseableTarget {
            target: target.to_string(),
        })?;

        if url.scheme() != "https" {
            return Err(Refusal::InsecureScheme {
                scheme: url.scheme().to_string(),
            });
        }

        let host = url
            .host_str()
            .ok_or_else(|| Refusal::UnparseableTarget {
                target: target.to_string(),
            })?
            .to_ascii_lowercase();

        // Mode is checked before the allowlist so the refusal names the real
        // cause: in Work mode an allowlisted host is still refused, and saying
        // "not on the allowlist" would be actively misleading.
        if !self.mode().permits_network() {
            return Err(Refusal::NetworkInWorkMode { host });
        }

        if !MODEL_ACQUISITION_HOSTS.contains(&host.as_str()) {
            return Err(Refusal::HostNotAllowed { host });
        }

        Ok(host)
    }

    /// The only outbound GET in the process.
    pub async fn get(&self, target: &str) -> Result<reqwest::Response, BrokerError> {
        match self.evaluate(target) {
            Ok(host) => {
                let mode = self.mode();
                self.record(EgressEvent {
                    at: Utc::now(),
                    host: host.clone(),
                    mode,
                    permitted: true,
                    reason: format!("Permitted: {host} is on the model-acquisition allowlist."),
                    canary: false,
                });
                Ok(self.client.get(target).send().await?)
            }
            Err(refusal) => {
                self.record_refusal(&refusal, false);
                Err(BrokerError::Refused(refusal))
            }
        }
    }

    /// Checks a target, records the decision, and on success lends the broker's
    /// client so the caller can build its own request.
    ///
    /// [`Self::get`] covers the simple case. Model downloads cannot use it: they
    /// need range headers for resume, streamed bodies and their own timeouts.
    /// Rewriting that logic here would mean reimplementing the downloader inside
    /// the broker, so instead the caller keeps its request-building and the
    /// broker keeps the decision. The borrow is what enforces it — there is no
    /// way to obtain a client without passing the check first.
    ///
    /// The returned client follows no redirects, so a caller handling a 302 must
    /// pass the new location back through here rather than following it.
    pub fn authorize(&self, target: &str) -> Result<&reqwest::Client, BrokerError> {
        match self.evaluate(target) {
            Ok(host) => {
                let mode = self.mode();
                self.record(EgressEvent {
                    at: Utc::now(),
                    host: host.clone(),
                    mode,
                    permitted: true,
                    reason: format!("Permitted: {host} is on the model-acquisition allowlist."),
                    canary: false,
                });
                Ok(&self.client)
            }
            Err(refusal) => {
                self.record_refusal(&refusal, false);
                Err(BrokerError::Refused(refusal))
            }
        }
    }

    /// The other half of the invariant: refuses anything that would touch
    /// confidential material while the network is reachable.
    ///
    /// [`Self::evaluate`] stops data leaving. This stops data *arriving* into a
    /// process that can currently talk to the internet — which is the situation
    /// the whole two-mode design exists to prevent. Every entry point for
    /// confidential material calls this first: uploads, ingestion, retrieval and
    /// task execution.
    ///
    /// `operation` is named in the refusal, so the user is told what was refused
    /// rather than being handed a generic denial.
    pub fn guard_confidential(&self, operation: &str) -> Result<(), Refusal> {
        if self.mode().permits_confidential_data() {
            return Ok(());
        }

        let refusal = Refusal::DataInProvisioningMode {
            operation: operation.to_string(),
        };
        self.record_refusal(&refusal, false);
        Err(refusal)
    }

    /// Checks a target and returns a request builder already bound to the
    /// broker's client, so the caller can add its own headers and timeout.
    ///
    /// This is the form most existing call sites want: they need an auth header
    /// or a range header, but have no reason to own a client. Because the only
    /// route to a builder is through this check, adding a header cannot bypass
    /// the mode, scheme and host rules.
    pub fn authorized_get(&self, target: &str) -> Result<reqwest::RequestBuilder, BrokerError> {
        Ok(self.authorize(target)?.get(target))
    }

    /// Deliberately attempts a connection that must fail, demonstrating that the
    /// controls are active rather than merely configured (PS step 6).
    ///
    /// Runs the real [`Self::evaluate`] path, so a bug that opened the door shows
    /// up here as a pass instead of being hidden behind a stub.
    pub fn run_canary(&self) -> EgressEvent {
        const CANARY_TARGET: &str = "https://example.invalid/arjun-canary";

        let mode = self.mode();
        let event = match self.evaluate(CANARY_TARGET) {
            Err(refusal) => EgressEvent {
                at: Utc::now(),
                host: "example.invalid".to_string(),
                mode,
                permitted: false,
                reason: refusal.reason(),
                canary: true,
            },
            // Reaching this arm means the controls did not hold. Recorded as a
            // permitted egress so it is impossible to miss in the monitor.
            Ok(host) => EgressEvent {
                at: Utc::now(),
                host,
                mode,
                permitted: true,
                reason: "CANARY FAILED: the broker would have permitted this call. \
                         Do not process confidential material in this state."
                    .to_string(),
                canary: true,
            },
        };

        self.record(event.clone());
        event
    }

    /// Most recent decisions, newest first, for the live monitor.
    pub fn recent_events(&self) -> Vec<EgressEvent> {
        let events = self.events.lock().expect("event ring poisoned");
        events.iter().rev().cloned().collect()
    }

    fn record_refusal(&self, refusal: &Refusal, canary: bool) {
        let host = match refusal {
            Refusal::NetworkInWorkMode { host } | Refusal::HostNotAllowed { host } => host.clone(),
            Refusal::UnparseableTarget { target } => target.clone(),
            Refusal::InsecureScheme { scheme } => scheme.clone(),
            Refusal::DataInProvisioningMode { operation } => operation.clone(),
        };
        let mode = self.mode();
        self.record(EgressEvent {
            at: Utc::now(),
            host,
            mode,
            permitted: false,
            reason: refusal.reason(),
            canary,
        });
    }

    fn record(&self, event: EgressEvent) {
        if event.permitted {
            log::info!("[SOVEREIGNTY] {}", event.reason);
        } else {
            log::warn!("[SOVEREIGNTY] {}", event.reason);
        }

        if let Some(audit) = self.audit.read().expect("audit lock poisoned").as_ref() {
            // A failure to write the durable copy must not stop the operation
            // that produced it — losing the ability to make decisions because
            // the log is unavailable would be a worse failure than losing a log
            // line. It is logged loudly instead.
            if let Err(e) = audit.record(
                "system",
                AuditKind::EgressDecision,
                event.reason.clone(),
                Some(serde_json::json!({
                    "host": event.host,
                    "permitted": event.permitted,
                    "mode": event.mode,
                    "canary": event.canary,
                })),
            ) {
                log::error!("[SOVEREIGNTY] could not write the egress decision to the audit log: {e}");
            }
        }

        let mut events = self.events.lock().expect("event ring poisoned");
        if events.len() == EVENT_RING_CAPACITY {
            events.pop_front();
        }
        events.push_back(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_mode_refuses_even_an_allowlisted_host() {
        let broker = NetworkBroker::new();
        assert_eq!(broker.mode(), OperatingMode::Work);
        let err = broker.evaluate("https://huggingface.co/api/models").unwrap_err();
        assert!(matches!(err, Refusal::NetworkInWorkMode { .. }));
    }

    #[test]
    fn provisioning_mode_permits_an_allowlisted_host() {
        let broker = NetworkBroker::new();
        broker.set_mode(OperatingMode::Provisioning);
        assert_eq!(
            broker.evaluate("https://huggingface.co/api/models").unwrap(),
            "huggingface.co"
        );
    }

    /// The bypass a prefix-matching allowlist would wave through.
    #[test]
    fn a_lookalike_host_is_refused_despite_the_matching_prefix() {
        let broker = NetworkBroker::new();
        broker.set_mode(OperatingMode::Provisioning);
        for target in [
            "https://huggingface.co.evil.test/api/models",
            "https://evil.test/https://huggingface.co",
            "https://nothuggingface.co/api/models",
        ] {
            assert!(
                matches!(broker.evaluate(target), Err(Refusal::HostNotAllowed { .. })),
                "{target} should have been refused"
            );
        }
    }

    #[test]
    fn plaintext_is_refused_even_for_an_allowlisted_host() {
        let broker = NetworkBroker::new();
        broker.set_mode(OperatingMode::Provisioning);
        assert!(matches!(
            broker.evaluate("http://huggingface.co/api/models"),
            Err(Refusal::InsecureScheme { .. })
        ));
    }

    #[test]
    fn the_canary_is_refused_in_work_mode_and_is_recorded() {
        let broker = NetworkBroker::new();
        let event = broker.run_canary();
        assert!(!event.permitted, "canary must not be permitted in Work mode");
        assert!(event.canary);
        assert_eq!(broker.recent_events().len(), 1);
    }

    /// The monitor has to show that nothing was sent, which needs permitted
    /// calls on the record too, not only refusals.
    #[test]
    fn permitted_decisions_are_recorded_not_just_refusals() {
        let broker = NetworkBroker::new();
        broker.set_mode(OperatingMode::Provisioning);
        broker.record(EgressEvent {
            at: Utc::now(),
            host: "huggingface.co".into(),
            mode: broker.mode(),
            permitted: true,
            reason: "permitted".into(),
            canary: false,
        });
        assert!(broker.recent_events().iter().any(|e| e.permitted));
    }

    #[test]
    fn confidential_work_is_allowed_in_work_mode() {
        let broker = NetworkBroker::new();
        assert!(broker.guard_confidential("open a document").is_ok());
    }

    #[test]
    fn confidential_work_is_refused_while_the_network_is_reachable() {
        let broker = NetworkBroker::new();
        broker.set_mode(OperatingMode::Provisioning);
        let err = broker.guard_confidential("open a document").unwrap_err();
        assert!(matches!(err, Refusal::DataInProvisioningMode { .. }));
        // The refusal is evidence, so it belongs in the log like any other.
        assert!(broker.recent_events().iter().any(|e| !e.permitted));
    }

    /// The invariant, exercised through the real entry points rather than
    /// asserted about the enum: in either mode exactly one of the two is open.
    #[test]
    fn network_and_confidential_access_are_never_open_together() {
        let broker = NetworkBroker::new();
        for mode in [OperatingMode::Work, OperatingMode::Provisioning] {
            broker.set_mode(mode);
            let network_ok = broker.evaluate("https://huggingface.co/api/models").is_ok();
            let data_ok = broker.guard_confidential("ingest").is_ok();
            assert_ne!(network_ok, data_ok, "{mode} opened both or neither");
        }
    }

    #[test]
    fn the_ring_is_bounded() {
        let broker = NetworkBroker::new();
        for _ in 0..(EVENT_RING_CAPACITY + 50) {
            broker.run_canary();
        }
        assert_eq!(broker.recent_events().len(), EVENT_RING_CAPACITY);
    }
}

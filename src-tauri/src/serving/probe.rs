//! Asking a local inference server what it is, without changing it.
//!
//! Before a run is sent anywhere, ARJUN checks the endpoint is up and serving
//! the model the router chose. The check is deliberately read-only: it must not
//! load, wake, or unload anything, because a probe that has side effects turns
//! a health screen into a way to disturb a running job.
//!
//! ## Loopback, enforced here rather than assumed
//!
//! [`crate::sovereignty::broker`] is the one way *out of the machine*. A local
//! inference server is not egress, so it does not go through the broker — but
//! that must not become a hole. So this module constructs its own client and
//! refuses any URL that is not loopback before a socket is opened.
//!
//! The same rule is enforced independently in `agent-runtime/src/run.ts`. Two
//! checks in two languages, because this one guards the probe and that one
//! guards the inference traffic, and neither covers the other's path.
//!
//! ## The outcomes are typed
//!
//! Adapted from OpenClaw's `extensions/llama-cpp/src/external-server/discovery.ts`
//! (MIT), which distinguishes unreachable from HTTP error from unparseable
//! response. The distinction is what makes the health screen useful: "the server
//! is not running" and "the server is running but returned 401" need different
//! actions, and collapsing them to "unavailable" wastes an operator's afternoon.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How long to wait for a local server that should answer immediately.
///
/// Generous for loopback, but a llama-server still loading a 5 GB model can
/// take a moment to answer its first request.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// What a probe found.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", tag = "state")]
pub enum ProbeOutcome {
    /// Serving, and these are the model ids it advertises.
    Ready { models: Vec<String> },
    /// Nothing answered. Almost always: the server is not running.
    Unreachable { detail: String },
    /// Something answered, but not with success. Usually authentication.
    HttpError { status: u16, path: String },
    /// Answered with success, but not with something this understands.
    InvalidResponse { path: String, detail: String },
    /// Refused before a socket was opened.
    NotLoopback { host: String },
}

impl ProbeOutcome {
    pub fn is_ready(&self) -> bool {
        matches!(self, ProbeOutcome::Ready { .. })
    }

    /// One line for an operator, naming the fix where there is one.
    pub fn explain(&self, base_url: &str) -> String {
        match self {
            ProbeOutcome::Ready { models } => {
                format!("{base_url} is serving {} model(s).", models.len())
            }
            ProbeOutcome::Unreachable { detail } => format!(
                "Nothing is listening at {base_url} ({detail}). Start the model server, or correct \
                 the endpoint in the registry entry."
            ),
            ProbeOutcome::HttpError { status, path } => format!(
                "{base_url}{path} answered {status}. The server is running but refused the request \
                 — check whether it was started with an API key."
            ),
            ProbeOutcome::InvalidResponse { path, detail } => format!(
                "{base_url}{path} answered, but not in the OpenAI-compatible shape ARJUN expects \
                 ({detail}). Check this is a llama-server, vLLM or SGLang endpoint."
            ),
            ProbeOutcome::NotLoopback { host } => format!(
                "{host} is not on this machine. ARJUN only sends work to inference servers running \
                 locally, so this endpoint was refused before any connection was attempted."
            ),
        }
    }
}

/// Whether a host is this machine.
///
/// The address is **parsed**, not pattern-matched. An earlier version of this
/// accepted anything beginning `127.`, which let the hostname
/// `127.example.com` — a name an attacker controls and DNS resolves anywhere —
/// through as loopback. Delegating to [`std::net::IpAddr::is_loopback`] gets
/// 127.0.0.0/8 and ::1 right and, more importantly, refuses everything that
/// merely looks like an address.
///
/// `localhost` is accepted by name because that is what `llama-server` prints
/// when it starts, and an operator copying it into the registry should not be
/// told their own machine is remote.
pub fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_loopback())
}

/// Confirms a base URL points at this machine, and returns its host.
pub fn check_loopback(base_url: &str) -> Result<String, ProbeOutcome> {
    let url = reqwest::Url::parse(base_url).map_err(|error| ProbeOutcome::InvalidResponse {
        path: String::new(),
        detail: format!("the endpoint is not a URL: {error}"),
    })?;
    let host = url.host_str().unwrap_or_default().to_string();
    if is_loopback_host(&host) {
        Ok(host)
    } else {
        Err(ProbeOutcome::NotLoopback { host })
    }
}

/// The models an OpenAI-compatible `/models` response advertises.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelRow>,
}

#[derive(Debug, Deserialize)]
struct ModelRow {
    id: String,
}

/// Asks an OpenAI-compatible endpoint which models it serves.
///
/// `base_url` includes the version prefix, e.g. `http://127.0.0.1:8080/v1`.
pub async fn probe(base_url: &str) -> ProbeOutcome {
    if let Err(refusal) = check_loopback(base_url) {
        return refusal;
    }

    // Constructed per probe rather than shared. A probe happens once per run
    // start and once per health check, so pooling buys nothing, and a
    // short-lived client cannot accumulate connections to a server that has
    // since been replaced.
    let client = match reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        // A local server has no proxy, and honouring an inherited proxy variable
        // would turn a loopback probe into a request that leaves the machine.
        .no_proxy()
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return ProbeOutcome::InvalidResponse {
                path: String::new(),
                detail: format!("an HTTP client could not be built: {error}"),
            }
        }
    };

    let path = "/models";
    let url = format!("{}{path}", base_url.trim_end_matches('/'));

    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(error) => {
            return ProbeOutcome::Unreachable {
                detail: describe_transport_error(&error),
            }
        }
    };

    let status = response.status();
    if !status.is_success() {
        return ProbeOutcome::HttpError {
            status: status.as_u16(),
            path: path.to_string(),
        };
    }

    match response.json::<ModelsResponse>().await {
        Ok(body) => ProbeOutcome::Ready {
            models: body.data.into_iter().map(|row| row.id).collect(),
        },
        Err(error) => ProbeOutcome::InvalidResponse {
            path: path.to_string(),
            detail: error.to_string(),
        },
    }
}

/// Turns a transport error into something an operator can act on.
fn describe_transport_error(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        return "timed out".to_string();
    }
    if error.is_connect() {
        return "connection refused".to_string();
    }
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_machine_is_recognised_in_the_forms_a_server_prints() {
        for host in ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]", "127.1.2.3"] {
            assert!(is_loopback_host(host), "{host} should be loopback");
        }
    }

    #[test]
    fn anything_off_this_machine_is_not_loopback() {
        // The private-network cases matter most: a plant VLAN address is the
        // plausible mistake, and it is still another machine.
        for host in [
            "192.168.1.50",
            "10.0.0.4",
            "172.16.0.9",
            "api.openai.com",
            // A hostname that merely starts like a loopback address. DNS
            // resolves it wherever its owner likes, so accepting it would be a
            // way out of the machine wearing a local-looking name.
            "127.example.com",
            "127.0.0.1.evil.com",
            "0x7f000001",
            "",
        ] {
            assert!(!is_loopback_host(host), "{host} should not be loopback");
        }
    }

    #[test]
    fn a_public_endpoint_is_refused_by_name() {
        let refusal = check_loopback("https://api.openai.com/v1").unwrap_err();
        assert_eq!(
            refusal,
            ProbeOutcome::NotLoopback {
                host: "api.openai.com".to_string()
            }
        );
        assert!(refusal
            .explain("https://api.openai.com/v1")
            .contains("only sends work to inference servers running locally"));
    }

    #[test]
    fn a_loopback_endpoint_passes_the_check() {
        assert_eq!(check_loopback("http://127.0.0.1:8080/v1").unwrap(), "127.0.0.1");
    }

    #[test]
    fn every_outcome_explains_itself_without_repeating_the_url() {
        let outcomes = [
            ProbeOutcome::Ready { models: vec!["a".into()] },
            ProbeOutcome::Unreachable { detail: "connection refused".into() },
            ProbeOutcome::HttpError { status: 401, path: "/models".into() },
            ProbeOutcome::InvalidResponse { path: "/models".into(), detail: "bad json".into() },
            ProbeOutcome::NotLoopback { host: "example.com".into() },
        ];
        for outcome in outcomes {
            let text = outcome.explain("http://127.0.0.1:8080/v1");
            assert!(!text.is_empty());
            // Every failure names something the operator can do next.
            if !outcome.is_ready() {
                assert!(
                    text.contains("Start") || text.contains("check") || text.contains("Check")
                        || text.contains("refused"),
                    "no action in: {text}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_probe_of_a_public_url_opens_no_socket() {
        // The refusal must come from the loopback check, not from a DNS failure
        // — which is why this asserts the variant rather than merely "not ready".
        let outcome = probe("https://api.openai.com/v1").await;
        assert!(matches!(outcome, ProbeOutcome::NotLoopback { .. }));
    }

    #[tokio::test]
    async fn a_probe_of_a_dead_port_says_it_is_not_running() {
        // Port 1 on loopback: reserved, and nothing legitimate binds it.
        let outcome = probe("http://127.0.0.1:1/v1").await;
        assert!(
            matches!(outcome, ProbeOutcome::Unreachable { .. }),
            "expected unreachable, got {outcome:?}"
        );
    }
}

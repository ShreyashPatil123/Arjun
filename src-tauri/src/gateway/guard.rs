//! Origin guard — blocks browser-originated requests.
//!
//! The gateway runs unauthenticated on loopback, matching Ollama's model: paste
//! a URL into your tool and it works, no token to configure.
//!
//! That leaves one real hole. A web page you visit can issue requests to
//! `127.0.0.1` from your own browser, so a random site could drive your local
//! model and read whatever context Sarathi injects — without you ever knowing.
//!
//! Browsers attach an `Origin` header to cross-origin requests and forbid pages
//! from forging it. Command-line tools (Claude Code, opencode, curl) send none.
//! So rejecting requests that carry a foreign `Origin` blocks the drive-by case
//! while costing legitimate clients nothing — no token, no configuration.
//!
//! `Host` is checked too, which stops DNS-rebinding: an attacker's domain
//! resolving to 127.0.0.1 arrives with their hostname in `Host`, not ours.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Hostnames that legitimately address this machine.
const ALLOWED_HOSTS: &[&str] = &["127.0.0.1", "localhost", "[::1]", "::1"];

/// True when the `Host` header names this machine.
///
/// The port is ignored — only the hostname matters for rebinding.
fn host_is_local(host: &str) -> bool {
    let hostname = strip_port(host);
    ALLOWED_HOSTS.iter().any(|h| hostname.eq_ignore_ascii_case(h))
}

/// True when an `Origin` points at this machine (so our own dashboard, served
/// from localhost, is not locked out).
fn origin_is_local(origin: &str) -> bool {
    // Tauri's own webview uses these schemes rather than http://localhost.
    if origin.starts_with("tauri://") || origin.starts_with("http://tauri.localhost") {
        return true;
    }
    let after_scheme = match origin.split_once("://") {
        Some((_, rest)) => rest,
        None => return false,
    };
    host_is_local(after_scheme)
}

/// Splits a trailing `:port`, leaving IPv6 literals in brackets intact.
fn strip_port(host: &str) -> &str {
    if host.starts_with('[') {
        // [::1]:11435 -> [::1]
        return match host.find(']') {
            Some(end) => &host[..=end],
            None => host,
        };
    }
    match host.rsplit_once(':') {
        Some((name, port)) if port.chars().all(|c| c.is_ascii_digit()) => name,
        _ => host,
    }
}

/// Rejects requests that a browser could have made from another site.
pub async fn origin_guard(request: Request, next: Next) -> Response {
    let headers = request.headers();

    if let Some(host) = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok()) {
        if !host_is_local(host) {
            log::warn!("[GATEWAY] Rejected request with non-local Host header: {:?}", host);
            return (
                StatusCode::FORBIDDEN,
                "Sarathi only accepts requests addressed to localhost",
            )
                .into_response();
        }
    }

    if let Some(origin) = headers.get(axum::http::header::ORIGIN).and_then(|v| v.to_str().ok()) {
        if !origin_is_local(origin) {
            log::warn!("[GATEWAY] Rejected browser request from origin: {:?}", origin);
            return (
                StatusCode::FORBIDDEN,
                "Sarathi does not accept requests from web pages",
            )
                .into_response();
        }
    }

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_hosts_are_accepted_with_or_without_a_port() {
        for host in ["127.0.0.1", "127.0.0.1:11435", "localhost", "localhost:11435", "[::1]:11435"] {
            assert!(host_is_local(host), "{host} should be local");
        }
    }

    #[test]
    fn dns_rebinding_hostnames_are_rejected() {
        // Resolves to 127.0.0.1 but arrives carrying the attacker's hostname.
        for host in ["evil.com", "evil.com:11435", "localtest.me", "127.0.0.1.nip.io"] {
            assert!(!host_is_local(host), "{host} must be rejected");
        }
    }

    #[test]
    fn a_web_page_origin_is_rejected() {
        for origin in ["https://evil.com", "http://evil.com:8080", "https://chat.example.com"] {
            assert!(!origin_is_local(origin), "{origin} must be rejected");
        }
    }

    #[test]
    fn our_own_ui_origins_are_allowed() {
        for origin in [
            "http://localhost:1420",
            "http://127.0.0.1:11435",
            "tauri://localhost",
            "http://tauri.localhost",
        ] {
            assert!(origin_is_local(origin), "{origin} should be allowed");
        }
    }

    #[test]
    fn a_malformed_origin_is_rejected_rather_than_assumed_safe() {
        for origin in ["", "evil.com", "javascript:alert(1)", "null"] {
            assert!(!origin_is_local(origin), "{origin:?} must not be treated as local");
        }
    }

    #[test]
    fn ipv6_literals_keep_their_brackets_when_the_port_is_stripped() {
        assert_eq!(strip_port("[::1]:11435"), "[::1]");
        assert_eq!(strip_port("[::1]"), "[::1]");
        assert_eq!(strip_port("127.0.0.1:11435"), "127.0.0.1");
        // A bare hostname with a non-numeric suffix is not a port.
        assert_eq!(strip_port("localhost"), "localhost");
    }
}

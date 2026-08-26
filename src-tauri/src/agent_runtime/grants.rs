//! Single-use authorisation grants.
//!
//! ## The problem this solves
//!
//! The agent loop runs in a separate process, in another language. It asks this
//! side whether a tool call may proceed, and then — separately — asks this side
//! to perform it. Two messages, and between them a gap in which the runtime
//! could, through a bug or a compromise, execute something the gateway never saw:
//! call again without asking, ask about a cheap argument and execute an
//! expensive one, or reuse yesterday's yes.
//!
//! A boolean reply cannot close that gap. So an allow is not a boolean, it is a
//! token bound to the exact call it authorised, redeemable once.
//!
//! ## What a grant is bound to
//!
//! The run, the tool-call id, the tool name, and a fingerprint of the arguments.
//! Change any of them and the token does not match. That makes the four attacks
//! above structurally impossible rather than conventionally avoided:
//!
//! - calling without asking — there is no token to present;
//! - replaying — the token was consumed by the first redemption;
//! - argument swapping — the fingerprint is over the arguments themselves;
//! - a stale yes — grants expire, and a run's grants die with the run.
//!
//! ## What it does not solve
//!
//! A grant proves the gateway said yes to *this* call. It does not prove the
//! caller is honest about anything else, which is why `tool.execute` re-derives
//! the verdict independently. Two independent checks, one structural and one
//! semantic; the grant covers a compromised runtime, the re-check covers a bug
//! in the grant.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// How long an unredeemed grant stays good.
///
/// Generous relative to the gap it covers — authorise and execute are
/// consecutive messages on one pipe — because the only thing a short window buys
/// is flakiness under load. It is short relative to a session, which is what
/// matters: a token that leaks is useless within the minute.
const GRANT_TTL: Duration = Duration::from_secs(60);

/// What a token was issued for. Every field participates in matching.
#[derive(Debug, Clone)]
struct Grant {
    run_id: String,
    tool_call_id: String,
    tool: String,
    /// SHA-256 over the canonical form of the arguments.
    args_fingerprint: String,
    issued_at: Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RedeemError {
    /// No such token: never issued, already spent, or expired and swept.
    Unknown,
    /// Presented after its window closed.
    Expired,
    /// The token exists but describes a different call. The interesting failure:
    /// something authorised one thing and tried to do another.
    Mismatch { field: &'static str },
}

impl std::fmt::Display for RedeemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RedeemError::Unknown => write!(
                f,
                "no authorisation grant for this call. It was not put through the gateway, or its grant was already spent."
            ),
            RedeemError::Expired => write!(
                f,
                "the authorisation for this call has expired. Ask again before executing."
            ),
            RedeemError::Mismatch { field } => write!(
                f,
                "this call does not match what was authorised (differing {field}). Refused."
            ),
        }
    }
}

/// Fingerprints arguments so a grant is bound to them.
///
/// `serde_json`'s object representation is a `BTreeMap` in this build, so its
/// serialisation is key-sorted and therefore stable across two encodings of the
/// same logical value. `fingerprint_ignores_key_order` pins that property: if a
/// dependency change ever swapped in insertion-order maps, argument swapping
/// would silently become possible again, and a failing test is how that should
/// surface.
fn fingerprint(args: &Value) -> String {
    let canonical = serde_json::to_string(args).unwrap_or_default();
    let digest = Sha256::digest(canonical.as_bytes());
    format!("{digest:x}")
}

#[derive(Default)]
pub struct GrantLedger {
    grants: HashMap<String, Grant>,
}

impl GrantLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Issues a token for one authorised call.
    pub fn issue(
        &mut self,
        run_id: &str,
        tool_call_id: &str,
        tool: &str,
        args: &Value,
    ) -> String {
        self.sweep_expired();
        let token = uuid::Uuid::new_v4().to_string();
        self.grants.insert(
            token.clone(),
            Grant {
                run_id: run_id.to_string(),
                tool_call_id: tool_call_id.to_string(),
                tool: tool.to_string(),
                args_fingerprint: fingerprint(args),
                issued_at: Instant::now(),
            },
        );
        token
    }

    /// Spends a token, checking it describes exactly this call.
    ///
    /// The token is removed on *every* attempt, matching or not. A redemption is
    /// one attempt, not one success: leaving a rejected token in place would let
    /// a caller keep guessing, and there is no legitimate retry — the authorise
    /// step is what a genuine caller repeats.
    pub fn redeem(
        &mut self,
        token: &str,
        run_id: &str,
        tool_call_id: &str,
        tool: &str,
        args: &Value,
    ) -> Result<(), RedeemError> {
        let Some(grant) = self.grants.remove(token) else {
            return Err(RedeemError::Unknown);
        };
        if grant.issued_at.elapsed() > GRANT_TTL {
            return Err(RedeemError::Expired);
        }
        if grant.run_id != run_id {
            return Err(RedeemError::Mismatch { field: "run" });
        }
        if grant.tool_call_id != tool_call_id {
            return Err(RedeemError::Mismatch {
                field: "tool call id",
            });
        }
        if grant.tool != tool {
            return Err(RedeemError::Mismatch { field: "tool" });
        }
        if grant.args_fingerprint != fingerprint(args) {
            return Err(RedeemError::Mismatch { field: "arguments" });
        }
        Ok(())
    }

    /// Drops every grant belonging to a run.
    ///
    /// Called when a run ends however it ends. A grant that outlives its run is
    /// a yes with nothing left to say yes to.
    pub fn revoke_run(&mut self, run_id: &str) -> usize {
        let before = self.grants.len();
        self.grants.retain(|_, grant| grant.run_id != run_id);
        before - self.grants.len()
    }

    fn sweep_expired(&mut self) {
        self.grants
            .retain(|_, grant| grant.issued_at.elapsed() <= GRANT_TTL);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.grants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args() -> Value {
        json!({ "query": "seal specification" })
    }

    #[test]
    fn a_grant_admits_the_call_it_was_issued_for() {
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert_eq!(
            ledger.redeem(&token, "run-1", "tc-1", "search_documents", &args()),
            Ok(())
        );
    }

    #[test]
    fn a_grant_cannot_be_replayed() {
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert!(ledger
            .redeem(&token, "run-1", "tc-1", "search_documents", &args())
            .is_ok());
        assert_eq!(
            ledger.redeem(&token, "run-1", "tc-1", "search_documents", &args()),
            Err(RedeemError::Unknown)
        );
    }

    #[test]
    fn authorising_one_argument_does_not_authorise_another() {
        // The attack the fingerprint exists to stop: ask about a harmless path,
        // execute against a sensitive one.
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "read_scoped_file", &json!({ "path": "notes.txt" }));
        assert_eq!(
            ledger.redeem(
                &token,
                "run-1",
                "tc-1",
                "read_scoped_file",
                &json!({ "path": "/etc/shadow" })
            ),
            Err(RedeemError::Mismatch { field: "arguments" })
        );
    }

    #[test]
    fn a_grant_does_not_transfer_to_another_tool() {
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert_eq!(
            ledger.redeem(&token, "run-1", "tc-1", "execute_code", &args()),
            Err(RedeemError::Mismatch { field: "tool" })
        );
    }

    #[test]
    fn a_grant_does_not_transfer_to_another_run() {
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert_eq!(
            ledger.redeem(&token, "run-2", "tc-1", "search_documents", &args()),
            Err(RedeemError::Mismatch { field: "run" })
        );
    }

    #[test]
    fn a_grant_does_not_transfer_to_another_call_in_the_same_run() {
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert_eq!(
            ledger.redeem(&token, "run-1", "tc-2", "search_documents", &args()),
            Err(RedeemError::Mismatch {
                field: "tool call id"
            })
        );
    }

    #[test]
    fn an_invented_token_is_refused() {
        let mut ledger = GrantLedger::new();
        ledger.issue("run-1", "tc-1", "search_documents", &args());
        assert_eq!(
            ledger.redeem("made-up", "run-1", "tc-1", "search_documents", &args()),
            Err(RedeemError::Unknown)
        );
    }

    #[test]
    fn a_rejected_redemption_still_spends_the_token() {
        // One redemption is one attempt. Otherwise a caller holding a token can
        // keep guessing arguments until something matches.
        let mut ledger = GrantLedger::new();
        let token = ledger.issue("run-1", "tc-1", "search_documents", &args());
        let _ = ledger.redeem(&token, "run-1", "tc-1", "execute_code", &args());
        assert_eq!(
            ledger.redeem(&token, "run-1", "tc-1", "search_documents", &args()),
            Err(RedeemError::Unknown)
        );
    }

    #[test]
    fn ending_a_run_revokes_its_grants_and_leaves_others_alone() {
        let mut ledger = GrantLedger::new();
        let doomed = ledger.issue("run-1", "tc-1", "search_documents", &args());
        let other = ledger.issue("run-2", "tc-1", "search_documents", &args());

        assert_eq!(ledger.revoke_run("run-1"), 1);
        assert_eq!(
            ledger.redeem(&doomed, "run-1", "tc-1", "search_documents", &args()),
            Err(RedeemError::Unknown)
        );
        assert!(ledger
            .redeem(&other, "run-2", "tc-1", "search_documents", &args())
            .is_ok());
    }

    #[test]
    fn fingerprint_ignores_key_order() {
        // Pins the canonicalisation the argument binding relies on. If this
        // fails, `serde_json` has started preserving insertion order and
        // argument swapping is possible again by reordering keys.
        let one: Value = serde_json::from_str(r#"{"a":1,"b":2}"#).unwrap();
        let other: Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        assert_eq!(fingerprint(&one), fingerprint(&other));
    }

    #[test]
    fn fingerprint_separates_values_that_differ() {
        assert_ne!(
            fingerprint(&json!({ "path": "a.txt" })),
            fingerprint(&json!({ "path": "b.txt" }))
        );
    }

    #[test]
    fn issuing_sweeps_nothing_it_should_not() {
        let mut ledger = GrantLedger::new();
        ledger.issue("run-1", "tc-1", "search_documents", &args());
        ledger.issue("run-1", "tc-2", "search_documents", &args());
        assert_eq!(ledger.len(), 2);
    }
}

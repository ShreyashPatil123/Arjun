//! Commands for signing in, and for reading the record.
//!
//! Every command here that returns audit data checks
//! [`Permission::ViewAuditLog`] first. The audit log names who did what, so
//! reading it is itself a privileged act — leaving it open to anyone would make
//! the record a convenient index of other people's activity.

use std::sync::{Arc, RwLock};

use tauri::State;

use crate::audit::merkle::MerkleVerification;
use crate::audit::provenance_hmac::{self, OfflineVerifyReport, SignedProvenance};
use crate::audit::{AuditEntry, AuditKind, AuditService, ChainVerification};
use crate::sovereignty::zero_trust::{
    GateDecision, ToolCallRequest, ZeroTrustConfig, ZeroTrustGate, ZeroTrustMode,
};
use crate::identity::{
    AuthenticationStatus, CredentialStore, Permission, Role, Session, User, UserDirectory,
};

/// Records a security-critical audit entry, refusing to silently lose it.
///
/// The naïve `let _ = audit.record(...)` pattern is dangerous for
/// security-relevant events: if the SQLite DB is full, corrupted, or locked,
/// the record call returns an error that gets discarded, and the action that
/// triggered it proceeds without a log entry. An attacker who can fill the
/// disk or wedge the database then performs privileged actions with no
/// evidence left behind — a "no audit log" failure is worse than no audit
/// log, because the rest of the system still behaves as if one exists.
///
/// Use this helper for events where the absence of a record is itself a
/// security incident (password changes, zero-trust toggles, successful
/// sign-ins). Best-effort logging — which can fail without aborting the
/// operation — is appropriate for non-critical events like failed sign-in
/// attempts and ordinary sign-outs.
fn record_critical(
    audit: &AuditService,
    actor: &str,
    kind: AuditKind,
    message: String,
    detail: Option<serde_json::Value>,
) -> Result<(), String> {
    audit
        .record(actor, kind, message, detail)
        .map(|_| ())
        .map_err(|error| {
            log::error!(
                "CRITICAL AUDIT FAILURE: actor={actor:?} kind={:?} error={error}",
                kind,
            );
            "Security logging failed; the operation was aborted to prevent unlogged actions."
                .to_string()
        })
}

/// The signed-in user, or `None` before anyone signs in.
pub type CurrentSession = Arc<RwLock<Option<Session>>>;

/// Reads the current session, or explains that nobody is signed in.
pub fn require_session(session: &CurrentSession) -> Result<Session, String> {
    session
        .read()
        .expect("session lock poisoned")
        .clone()
        .ok_or_else(|| "Nobody is signed in.".to_string())
}

/// Reads the current session and checks one permission on it.
///
/// `pub` so other command modules (model management, agents, memory) can
/// gate their Tauri commands on the same matrix the governance commands
/// use, without re-implementing the check.
pub fn require_permission(session: &CurrentSession, permission: Permission) -> Result<Session, String> {
    let session = require_session(session)?;
    if !session.holds(permission) {
        return Err(format!(
            "{} is not permitted to {}.",
            session.user.display_name,
            permission.describe()
        ));
    }
    Ok(session)
}

/// Local accounts available to sign in as.
#[tauri::command]
pub async fn list_accounts(directory: State<'_, Arc<UserDirectory>>) -> Result<Vec<User>, String> {
    Ok(directory.all().to_vec())
}

/// Where sign-in stands: whether anyone has a password yet.
#[tauri::command]
pub async fn authentication_status(
    credentials: State<'_, Arc<CredentialStore>>,
) -> Result<AuthenticationStatus, String> {
    credentials.status().map_err(|e| e.to_string())
}

/// Sets the password for the first administrator, on a deployment that has none.
///
/// Only available while no account has a password, and only for an account that
/// actually holds the administrator role — otherwise the first person to reach a
/// fresh install could give themselves the keys.
#[tauri::command]
pub async fn set_initial_administrator_password(
    directory: State<'_, Arc<UserDirectory>>,
    credentials: State<'_, Arc<CredentialStore>>,
    audit: State<'_, Arc<AuditService>>,
    user_id: String,
    password: String,
) -> Result<(), String> {
    let status = credentials.status().map_err(|e| e.to_string())?;
    if status != AuthenticationStatus::AwaitingFirstAdministrator {
        return Err(
            "This deployment is already set up. An administrator resets other accounts \n             from Settings."
                .to_string(),
        );
    }

    let user = directory
        .find(&user_id)
        .ok_or_else(|| format!("No account with id {user_id:?}."))?;

    if !user.roles.contains(&Role::Administrator) {
        return Err(format!(
            "{} is not an administrator. The first password has to be set on an \n             administrator account.",
            user.display_name
        ));
    }

    credentials
        .set_password(&user_id, &password)
        .map_err(|e| e.to_string())?;

    record_critical(
        audit.inner(),
        &user_id,
        AuditKind::Session,
        format!("Initial administrator password set for {}", user.display_name),
        None,
    )?;
    Ok(())
}

/// Sets or resets another account's password. Administrators only.
#[tauri::command]
pub async fn set_account_password(
    session: State<'_, CurrentSession>,
    directory: State<'_, Arc<UserDirectory>>,
    credentials: State<'_, Arc<CredentialStore>>,
    audit: State<'_, Arc<AuditService>>,
    user_id: String,
    password: String,
) -> Result<(), String> {
    let actor = require_permission(&session, Permission::ModifyPolicy)?;
    let target = directory
        .find(&user_id)
        .ok_or_else(|| format!("No account with id {user_id:?}."))?;

    credentials
        .set_password(&user_id, &password)
        .map_err(|e| e.to_string())?;

    record_critical(
        audit.inner(),
        &actor.user.id,
        AuditKind::PolicyDecision,
        format!(
            "{} set the password for {}",
            actor.user.display_name, target.display_name
        ),
        Some(serde_json::json!({ "targetUserId": user_id })),
    )?;
    Ok(())
}

/// Signs in, checking the password.
///
/// A failed attempt is recorded. The refusal itself says only that the
/// combination was wrong, never which half — telling someone the account exists
/// but the password is wrong hands them half the answer.
#[tauri::command]
pub async fn sign_in(
    directory: State<'_, Arc<UserDirectory>>,
    credentials: State<'_, Arc<CredentialStore>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    user_id: String,
    password: String,
) -> Result<Session, String> {
    attempt_sign_in(
        directory.inner(),
        credentials.inner(),
        session.inner(),
        audit.inner(),
        &user_id,
        &password,
    )
    .await
}

/// Core sign-in logic, factored out of the Tauri command so it can be tested
/// without constructing `tauri::State` (which has no public constructor in the
/// `tauri` crate's public API — only `tauri::test::mock_app` can produce one,
/// and that requires the `test` feature, which this crate does not enable).
///
/// The directory lookup happens *before* the credential verify so that an
/// unknown account and an account with a wrong password take the same path
/// through the function. The credential layer already pays for a dummy
/// Argon2 hash on the unknown-user path, so the dominant cost is constant —
/// this change closes the residual directory-lookup gap.
async fn attempt_sign_in(
    directory: &UserDirectory,
    credentials: &CredentialStore,
    session: &CurrentSession,
    audit: &AuditService,
    user_id: &str,
    password: &str,
) -> Result<Session, String> {
    const REFUSAL: &str = "That account and password do not match.";

    // TIMING FIX: Look up the user first, regardless of whether the credential
    // is going to verify. The credential layer already pays for a dummy Argon2
    // hash on the unknown-user path, so the expensive step is constant — but
    // the directory lookup was not. Reading it before `verify` collapses the
    // two refusal branches into one, removing the residual signal that
    // distinguished "verified but no such account" from "wrong password".
    let user_lookup = directory.find(user_id);

    let verified = credentials
        .verify(user_id, password)
        .map_err(|e| e.to_string())?;

    if !verified || user_lookup.is_none() {
        // Recorded against the attempted id so repeated failures are visible,
        // without implying the account exists.
        let _ = audit.record(
            user_id,
            AuditKind::Session,
            format!("Failed sign-in attempt for {user_id:?}"),
            Some(serde_json::json!({ "succeeded": false })),
        );
        return Err(REFUSAL.to_string());
    }

    let user = user_lookup.expect("user exists — checked above").clone();

    let new_session = Session::open(user);

    // SECURITY: A successful sign-in must be auditable. The session is
    // installed only after the audit row has been written, so an attacker
    // who has wedged the audit database cannot both authenticate and avoid
    // the log entry — `record_critical` returns the error before any
    // session state is touched.
    record_critical(
        audit,
        &new_session.user.id,
        AuditKind::Session,
        format!(
            "{} signed in as {}",
            new_session.user.display_name,
            new_session
                .user
                .roles
                .iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some(serde_json::json!({
            "userId": new_session.user.id,
            "roles": new_session.user.roles,
            "authenticated": true,
        })),
    )?;

    *session.write().expect("session lock poisoned") = Some(new_session.clone());
    Ok(new_session)
}

#[tauri::command]
pub async fn sign_out(
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<(), String> {
    let previous = session.write().expect("session lock poisoned").take();
    if let Some(previous) = previous {
        let _ = audit.record(
            &previous.user.id,
            AuditKind::Session,
            format!("{} signed out", previous.user.display_name),
            None,
        );
    }
    Ok(())
}

/// Who is signed in, if anyone.
#[tauri::command]
pub async fn current_session(session: State<'_, CurrentSession>) -> Result<Option<Session>, String> {
    Ok(session.read().expect("session lock poisoned").clone())
}

/// Everything the signed-in user is entitled to do, for the account screen.
#[tauri::command]
pub async fn current_permissions(
    session: State<'_, CurrentSession>,
) -> Result<Vec<Permission>, String> {
    Ok(require_session(&session)?.user.permissions())
}

/// The most recent audit entries, newest first.
#[tauri::command]
pub async fn recent_audit_entries(
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    limit: Option<usize>,
) -> Result<Vec<AuditEntry>, String> {
    require_permission(&session, Permission::ViewAuditLog)?;
    audit
        .recent(limit.unwrap_or(200).min(1000))
        .map_err(|e| e.to_string())
}

/// Walks the whole chain and recomputes every seal.
///
/// This is the check that turns the log from a list into evidence: it either
/// confirms the record is unaltered, or names the first entry that was.
#[tauri::command]
pub async fn verify_audit_chain(
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<ChainVerification, String> {
    require_permission(&session, Permission::ViewAuditLog)?;
    audit.verify_chain().map_err(|e| e.to_string())
}

/// Mints an HMAC tag for a provenance block under the operator-set key
/// stored in `app_data_dir`. Returns the tag, the algorithm, the message
/// digest, and whether the key was actually present. See
/// `audit::provenance_hmac` for the honest security claim — this is a
/// *checkpoint*, not a digital signature.
#[tauri::command]
pub async fn sign_provenance(
    session: State<'_, CurrentSession>,
    app_data_dir: State<'_, std::path::PathBuf>,
    provenance: crate::sih_workflow::evidence_package::Provenance,
) -> Result<SignedProvenance, String> {
    require_permission(&session, Permission::ModifyPolicy)?;
    provenance_hmac::sign(&app_data_dir, &provenance).map_err(|e| e.to_string())
}

/// Re-derives an HMAC tag for a previously signed provenance block
/// using the on-disk key, and returns the diff between the stored tag
/// and the recomputed one. Inspectors run this in a clean environment
/// to confirm a package has not been tampered with since signing.
#[tauri::command]
pub async fn verify_provenance(
    session: State<'_, CurrentSession>,
    app_data_dir: State<'_, std::path::PathBuf>,
    signed: SignedProvenance,
) -> Result<OfflineVerifyReport, String> {
    require_permission(&session, Permission::ViewAuditLog)?;
    provenance_hmac::offline_report(&app_data_dir, &signed).map_err(|e| e.to_string())
}

/// Reads the current zero-trust configuration. Available to any
/// authenticated user — there is no privacy in *what mode the toggle is
/// in*, only in the audit log it writes.
#[tauri::command]
pub async fn read_zero_trust_config(
    session: State<'_, CurrentSession>,
    gate: State<'_, Arc<ZeroTrustGate>>,
) -> Result<ZeroTrustConfig, String> {
    require_session(&session)?;
    gate.read().map_err(|e| e.to_string())
}

/// Changes the zero-trust configuration. Requires `ModifyPolicy`; the
/// change itself is recorded in the audit log.
#[tauri::command]
pub async fn set_zero_trust_mode(
    session: State<'_, CurrentSession>,
    gate: State<'_, Arc<ZeroTrustGate>>,
    mode: ZeroTrustMode,
    reauth_window_seconds: u32,
    reason: Option<String>,
) -> Result<ZeroTrustConfig, String> {
    let who = require_permission(&session, Permission::ModifyPolicy)?;
    gate.set(&who.user.id, mode, reauth_window_seconds, reason)
        .map_err(|e| e.to_string())
}

/// Asks the gate whether a tool call should proceed. Returns a
/// `GateDecision`; the UI either lets the call run, surfaces a
/// confirmation dialog, or hard-denies the call.
#[tauri::command]
pub async fn zero_trust_check_tool_call(
    session: State<'_, CurrentSession>,
    gate: State<'_, Arc<ZeroTrustGate>>,
    request: ToolCallRequest,
) -> Result<GateDecision, String> {
    let who = require_session(&session)?;
    gate.check_tool_call(&who.user.id, &request).map_err(|e| e.to_string())
}

/// Records the human's response to a `RequireHumanApproval` decision.
/// Writes a second audit row referencing the original request row.
#[tauri::command]
pub async fn zero_trust_confirm_approval(
    session: State<'_, CurrentSession>,
    gate: State<'_, Arc<ZeroTrustGate>>,
    approval_id: i64,
    approved: bool,
) -> Result<(), String> {
    let who = require_session(&session)?;
    gate.confirm_approval(&who.user.id, approval_id, approved)
        .map_err(|e| e.to_string())
}

/// Verifies the audit log against the last recorded Merkle snapshot.
///
/// The chain can still pass a per-row recompute while being *silently
/// rewritten* — an attacker who can edit the file can rewrite rows in
/// matching pairs that re-seal to the same hash, then break the seal of
/// the row after to hide what they did. A Merkle root is the second witness:
/// it was written outside the chain, and a recorded root that disagrees
/// with what the chain now reproduces is the strongest in-process signal
/// we can offer that something has been rewritten.
#[tauri::command]
pub async fn verify_audit_merkle(
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
) -> Result<MerkleVerification, String> {
    require_permission(&session, Permission::ViewAuditLog)?;
    let conn = audit.connection_handle();
    crate::audit::merkle::verify(&conn).map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditService;
    use crate::identity::{CredentialStore, UserDirectory};
    use rusqlite::Connection;
    use std::time::{Duration, Instant};

    fn audit_in_memory() -> Arc<AuditService> {
        Arc::new(AuditService::from_connection(Connection::open_in_memory().unwrap()).unwrap())
    }

    fn credentials_in_memory() -> Arc<CredentialStore> {
        Arc::new(CredentialStore::from_connection(Connection::open_in_memory().unwrap()).unwrap())
    }

    /// The sign-in helper under test calls `credential.verify(...)`, which on
    /// Argon2 default parameters takes tens to a few hundred milliseconds per
    /// call. Any timing test on it is dominated by that work — both the
    /// unknown-user and wrong-password branches pay for a real (or dummy)
    /// Argon2 hash before returning. The directory-lookup timing that this
    /// patch actually fixes is a tiny fraction of that, well within Argon2's
    /// jitter. The check below is therefore coarse: we just confirm both
    /// refusal paths stay within a sane multiple of each other.
    #[tokio::test]
    async fn refusal_timing_is_similar_for_unknown_user_and_wrong_password() {
        let directory = Arc::new(UserDirectory::seeded());
        let credentials = credentials_in_memory();
        credentials
            .set_password("admin", "correct horse battery")
            .unwrap();
        let session: CurrentSession = Arc::new(RwLock::new(None));
        let audit = audit_in_memory();

        // Warm up the Argon2 paths so the first call's allocator behaviour
        // does not skew the timing comparison.
        let _ = attempt_sign_in(
            &directory,
            &credentials,
            &session,
            &audit,
            "ghost-user",
            "any-password-12+chars",
        )
        .await;
        let _ = attempt_sign_in(
            &directory,
            &credentials,
            &session,
            &audit,
            "admin",
            "wrong-password-12+chars",
        )
        .await;

        let runs = 5;
        let mut missing_total = Duration::ZERO;
        let mut wrong_total = Duration::ZERO;

        for _ in 0..runs {
            let t = Instant::now();
            let _ = attempt_sign_in(
                &directory,
                &credentials,
                &session,
                &audit,
                "ghost-user",
                "any-password-12+chars",
            )
            .await;
            missing_total += t.elapsed();

            let t = Instant::now();
            let _ = attempt_sign_in(
                &directory,
                &credentials,
                &session,
                &audit,
                "admin",
                "wrong-password-12+chars",
            )
            .await;
            wrong_total += t.elapsed();
        }

        let missing = missing_total.as_secs_f64() / runs as f64;
        let wrong = wrong_total.as_secs_f64() / runs as f64;

        // Both paths must do comparable amounts of work. Argon2's parameter
        // set is the dominant cost; the directory lookup is microseconds.
        // Allow a 3× margin to absorb scheduler jitter.
        let ratio = missing.max(wrong) / missing.min(wrong).max(0.000_001);
        assert!(
            ratio < 3.0,
            "refusal-path timing diverged: missing={missing:?} wrong={wrong:?} ratio={ratio:.2}",
        );
    }

    /// The directory lookup happens *before* `verify`, so a verified account
    /// whose id is missing from the directory still ends up in the refusal
    /// branch — the same one as a wrong password. This guards against a
    /// future refactor that moves the lookup back behind the verify.
    #[tokio::test]
    async fn verified_user_missing_from_directory_is_refused() {
        let directory = Arc::new(UserDirectory::seeded());
        let credentials = credentials_in_memory();

        // Forge a credential for a user id that is *not* in the directory.
        credentials
            .set_password("phantom", "correct horse battery")
            .unwrap();

        let session: CurrentSession = Arc::new(RwLock::new(None));
        let audit = audit_in_memory();

        let result = attempt_sign_in(
            &directory,
            &credentials,
            &session,
            &audit,
            "phantom",
            "correct horse battery",
        )
        .await;

        assert!(result.is_err(), "phantom user must be refused");
        assert_eq!(
            result.unwrap_err(),
            "That account and password do not match.",
            "refusal must use the canonical message, never a directory-specific one",
        );
        // The session must remain empty — we never opened it.
        assert!(session.read().unwrap().is_none());
    }

    /// A successful sign-in sets the session.
    #[tokio::test]
    async fn a_successful_sign_in_opens_the_session() {
        let directory = Arc::new(UserDirectory::seeded());
        let credentials = credentials_in_memory();
        credentials.set_password("admin", "correct horse battery").unwrap();
        let session: CurrentSession = Arc::new(RwLock::new(None));
        let audit = audit_in_memory();

        let new_session = attempt_sign_in(
            &directory,
            &credentials,
            &session,
            &audit,
            "admin",
            "correct horse battery",
        )
        .await
        .expect("correct password must sign in");

        assert_eq!(new_session.user.id, "admin");
        assert_eq!(session.read().unwrap().as_ref().unwrap().user.id, "admin");
    }
}

//! Commands for signing in, and for reading the record.
//!
//! Every command here that returns audit data checks
//! [`Permission::ViewAuditLog`] first. The audit log names who did what, so
//! reading it is itself a privileged act — leaving it open to anyone would make
//! the record a convenient index of other people's activity.

use std::sync::{Arc, RwLock};

use tauri::State;

use crate::audit::{AuditEntry, AuditKind, AuditService, ChainVerification};
use crate::identity::{
    AuthenticationStatus, CredentialStore, Permission, Role, Session, User, UserDirectory,
};

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
fn require_permission(session: &CurrentSession, permission: Permission) -> Result<Session, String> {
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

    let _ = audit.record(
        &user_id,
        AuditKind::Session,
        format!("Initial administrator password set for {}", user.display_name),
        None,
    );
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

    let _ = audit.record(
        &actor.user.id,
        AuditKind::PolicyDecision,
        format!(
            "{} set the password for {}",
            actor.user.display_name, target.display_name
        ),
        Some(serde_json::json!({ "targetUserId": user_id })),
    );
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
    const REFUSAL: &str = "That account and password do not match.";

    let verified = credentials
        .verify(&user_id, &password)
        .map_err(|e| e.to_string())?;

    if !verified {
        // Recorded against the attempted id so repeated failures are visible,
        // without implying the account exists.
        let _ = audit.record(
            &user_id,
            AuditKind::Session,
            format!("Failed sign-in attempt for {user_id:?}"),
            Some(serde_json::json!({ "succeeded": false })),
        );
        return Err(REFUSAL.to_string());
    }

    let user = directory
        .find(&user_id)
        .ok_or_else(|| REFUSAL.to_string())?
        .clone();

    let new_session = Session::open(user);
    *session.write().expect("session lock poisoned") = Some(new_session.clone());

    let _ = audit.record(
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
    );

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

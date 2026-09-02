//! Commands backing the Audit & Network surface.
//!
//! These are what makes the sovereign claim checkable from the UI rather than
//! only from a log file: the current mode, the record of every egress decision,
//! and the canary that demonstrates the controls are live.

use std::sync::Arc;
use tauri::State;

use crate::audit::{AuditKind, AuditService};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;
use crate::sovereignty::{
    observe_own_connections, EgressEvent, NetworkBroker, ObservationReport, OperatingMode,
};

/// The mode ARJUN is currently in.
#[tauri::command]
pub async fn get_operating_mode(
    broker: State<'_, Arc<NetworkBroker>>,
    session: State<'_, CurrentSession>,
) -> Result<OperatingMode, String> {
    // Read-only inspection of the current mode. The matrix does not
    // gate this beyond sign-in.
    require_session(&session)?;
    Ok(broker.mode())
}

/// Switches operating mode.
///
/// Entering Work mode only ever removes capability, so anyone may do it and it
/// is never refused — a user who suspects something is wrong must always be able
/// to seal the machine.
///
/// Entering Provisioning is the direction that needs guarding: it makes the
/// network reachable. It requires [`Permission::EnterProvisioning`], which only
/// the administrator and model-administrator roles hold.
#[tauri::command]
pub async fn set_operating_mode(
    broker: State<'_, Arc<NetworkBroker>>,
    session: State<'_, CurrentSession>,
    audit: State<'_, Arc<AuditService>>,
    mode: OperatingMode,
) -> Result<OperatingMode, String> {
    let actor = if mode == OperatingMode::Provisioning {
        let signed_in = require_session(&session)?;
        if !signed_in.holds(Permission::EnterProvisioning) {
            let refusal = format!(
                "{} is not permitted to {}.",
                signed_in.user.display_name,
                Permission::EnterProvisioning.describe()
            );
            let _ = audit.record(
                &signed_in.user.id,
                AuditKind::PolicyDecision,
                refusal.clone(),
                Some(serde_json::json!({ "requested": "provisioning", "allowed": false })),
            );
            return Err(refusal);
        }
        signed_in.user.id
    } else {
        // Sealing the machine is always permitted, including before sign-in.
        require_session(&session)
            .map(|s| s.user.id)
            .unwrap_or_else(|_| "system".to_string())
    };

    let previous = broker.set_mode(mode);

    // Entering Work mode immediately proves the controls took effect, rather
    // than trusting that the switch did what it said.
    if mode == OperatingMode::Work {
        let canary = broker.run_canary();
        if canary.permitted {
            return Err(
                "Switched to Work mode, but the canary was still permitted.                  The egress controls are not holding — do not process                  confidential material until this is resolved."
                    .to_string(),
            );
        }
    }

    let _ = audit.record(
        &actor,
        AuditKind::ModeChanged,
        format!("Mode changed from {previous} to {mode}"),
        Some(serde_json::json!({ "from": previous, "to": mode })),
    );

    log::info!("[SOVEREIGNTY] mode change confirmed: {previous} -> {mode}");
    Ok(previous)
}

/// Every egress decision the broker has made, newest first.
///
/// Includes permitted calls, not only refusals — a monitor that shows only
/// blocks cannot demonstrate that nothing was sent.
#[tauri::command]
pub async fn recent_egress_events(
    broker: State<'_, Arc<NetworkBroker>>,
    session: State<'_, CurrentSession>,
) -> Result<Vec<EgressEvent>, String> {
    // The egress log is part of the audit record. The matrix puts it
    // under `ViewAuditLog`. A `User` who wants to know "did anything
    // leave this machine" is the right question, but the answer is
    // for an auditor or reviewer.
    require_permission(&session, Permission::ViewAuditLog)?;
    Ok(broker.recent_events())
}

/// Deliberately attempts an external connection that must fail (ARJUN design rule 6).
///
/// The returned event is the evidence: in Work mode it must come back refused,
/// and it appears in the monitor alongside everything else.
#[tauri::command]
pub async fn run_egress_canary(
    broker: State<'_, Arc<NetworkBroker>>,
    session: State<'_, CurrentSession>,
) -> Result<EgressEvent, String> {
    // The canary is an audit/test operation. Same gate as the egress log.
    require_permission(&session, Permission::ViewAuditLog)?;
    Ok(broker.run_canary())
}

/// What the operating system says this process is connected to.
///
/// Deliberately does not consult the broker. The broker reporting that it
/// refused everything is ARJUN vouching for itself; this is a second, unrelated
/// vantage point, and the two disagreeing is exactly the finding worth having.
#[tauri::command]
pub async fn observe_process_connections(
    session: State<'_, CurrentSession>,
) -> Result<ObservationReport, String> {
    // Process observation is an audit-class read. Same gate as the
    // egress log: it answers "what left the machine" and that is
    // for the people who own the audit story.
    require_permission(&session, Permission::ViewAuditLog)?;
    Ok(observe_own_connections())
}

/// Asks whether confidential material may be handled right now.
///
/// Called before a file is attached, a collection is opened, or a task starts.
/// Returns the refusal text rather than a bare boolean so the UI can explain
/// what is blocked and why, instead of a control that is mysteriously inert.
#[tauri::command]
pub async fn assert_confidential_allowed(
    broker: State<'_, Arc<NetworkBroker>>,
    operation: String,
) -> Result<(), String> {
    broker
        .guard_confidential(&operation)
        .map_err(|refusal| refusal.reason())
}

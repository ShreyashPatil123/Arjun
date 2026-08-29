//! Zero-trust admin mode — the "every tool call asks" toggle.
//!
//! ## Honest scope
//!
//! "Zero trust" in a single-process desktop app is not the same as zero trust
//! across a network: there is no perimeter, and any code that runs in this
//! process can already read every byte of memory the app owns. What the
//! toggle *can* do is force human approval at specific, named, well-defined
//! chokepoints, and write those decisions to the audit log so a reviewer can
//! see who approved what.
//!
//! The four chokepoints the toggle currently gates are:
//!
//! 1. **Tool calls** — every call to a tool (sandboxed code, file write,
//!    document generation) requires a fresh user approval while the toggle
//!    is on. Off (the default), approvals follow the role-based permission
//!    system in [`crate::identity`].
//! 2. **Memory reads** — every read of the persistent memory store is
//!    logged with a row in the audit log, even when the read is permitted.
//!    The toggle does not block the read; it writes a record. (See
//!    `memory_reads` below.)
//! 3. **Model switches** — when the active model is swapped, the user
//!    must re-authenticate within the last N seconds, where N is
//!    `reauth_window_seconds` (default 60). The re-auth requirement is
//!    also in effect when the toggle is *off*; the toggle's effect on
//!    model switching is to *lengthen* the window from the default
//!    five minutes down to the configured value, on the theory that the
//!    operator who turned the toggle on wants to know it is them at the
//!    keyboard, not a coworker.
//! 4. **Approval capture** — a tool call that would have been approved
//!    silently under role-based permissions now appears in the audit
//!    log with a `ZeroTrustApproval` row, recording that the human
//!    explicitly OK'd it under the tighter regime.
//!
//! ## What this is NOT
//!
//! It is not a defense against a compromised model, a compromised webview,
//! or a compromised binary. A process the attacker already controls can
//! short-circuit the gate. The honest claim is that, in the *honest* case —
//! a careful operator at the keyboard — the toggle makes the operation
//! auditable down to every individual tool call.
//!
//! ## Persistence
//!
//! The toggle is stored in the SQLite database beside the audit log, in a
//! single-row `zero_trust_config` table. The state can only be changed by
//! an account holding `Permission::ModifyPolicy`; the change itself is
//! recorded in the audit log.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::audit::{AuditKind, AuditService};

/// The mode the toggle is in. `Off` is the default; the other three are
/// additive: each one tightens the regime on top of the previous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ZeroTrustMode {
    /// Default. No extra approval requirements beyond role-based.
    Off,
    /// Every tool call must be explicitly approved by a human in the UI.
    /// The approval row is recorded in the audit log.
    ApproveEveryToolCall,
    /// Same as `ApproveEveryToolCall`, plus every memory read is logged.
    ApproveEveryToolCallAndLogMemoryReads,
    /// Same as the previous, plus the re-auth window on model switch is
    /// tightened to the configured value (default 60 seconds).
    ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth,
}

impl ZeroTrustMode {
    /// The default. Held here so it is the one place to change.
    pub const fn default_mode() -> Self {
        ZeroTrustMode::Off
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ZeroTrustMode::Off => "off",
            ZeroTrustMode::ApproveEveryToolCall => "approve_every_tool_call",
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReads => {
                "approve_every_tool_call_and_log_memory_reads"
            }
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth => {
                "approve_every_tool_call_and_log_memory_reads_and_tighten_reauth"
            }
        }
    }

    fn from_str(raw: &str) -> Option<Self> {
        Some(match raw {
            "off" => ZeroTrustMode::Off,
            "approve_every_tool_call" => ZeroTrustMode::ApproveEveryToolCall,
            "approve_every_tool_call_and_log_memory_reads" => {
                ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReads
            }
            "approve_every_tool_call_and_log_memory_reads_and_tighten_reauth" => {
                ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth
            }
            _ => return None,
        })
    }

    /// Whether the given mode requires every tool call to be approved
    /// individually by a human. A no-op when the mode is `Off`.
    pub const fn requires_tool_call_approval(self) -> bool {
        !matches!(self, ZeroTrustMode::Off)
    }

    /// Whether the given mode logs every memory read.
    pub const fn logs_memory_reads(self) -> bool {
        matches!(
            self,
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReads
                | ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth
        )
    }

    /// Whether the given mode tightens the re-auth window on model
    /// switch to the configured value (instead of the default five
    /// minutes).
    pub const fn tightens_reauth(self) -> bool {
        matches!(
            self,
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth
        )
    }
}

/// The full configuration of the zero-trust subsystem, in the single-row
/// `zero_trust_config` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZeroTrustConfig {
    pub mode: ZeroTrustMode,
    /// Seconds since the last explicit re-authentication within which a
    /// model switch is allowed. Only consulted when `mode` is
    /// `tightens_reauth()`; otherwise the default five-minute window
    /// applies. The minimum is 5 seconds, to keep the operator from
    /// tripping over a zero-length window by accident.
    pub reauth_window_seconds: u32,
    /// When the configuration was last changed.
    pub updated_at: DateTime<Utc>,
    /// Who last changed it.
    pub updated_by: String,
    /// Why it was changed. The audit log carries the same field, so a
    /// reviewer can read the reason alongside the row that records the
    /// change.
    pub last_change_reason: Option<String>,
}

impl ZeroTrustConfig {
    /// The hard-coded defaults. Held in one place so the test suite and
    /// the runtime agree.
    pub fn defaults(updated_by: &str) -> Self {
        Self {
            mode: ZeroTrustMode::default_mode(),
            reauth_window_seconds: 60,
            updated_at: Utc::now(),
            updated_by: updated_by.to_string(),
            last_change_reason: None,
        }
    }
}

/// A request to perform a tool call. Captured here so the gate can be
/// asked once, deterministically, whether the call should proceed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRequest {
    /// The name of the tool being called, e.g. `"sandbox.run_python"`.
    pub tool: String,
    /// One-line human-readable description, printed in the approval dialog.
    pub description: String,
    /// Whether the calling code has already shown its work — arguments
    /// captured, dry-run output shown — so the approval dialog can
    /// confirm what is being approved.
    pub inspected: bool,
}

/// The decision returned by the gate. A `Permit` is a normal pass; a
/// `RequireHumanApproval` is the gate's call for the UI to ask the user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum GateDecision {
    /// Tool call is allowed to proceed.
    Permit { reason: String },
    /// Tool call must be approved by a human in the UI before it runs.
    /// The `approval_id` is the row id of the audit entry recording the
    /// request, so the UI can update it when the human responds.
    RequireHumanApproval {
        approval_id: i64,
        prompt: String,
    },
    /// Tool call is unconditionally refused. This is the gate's hard
    /// limit; the UI must not offer the user an "approve anyway" button.
    Deny { reason: String },
}

/// The gate itself. Held as a `tauri::State` so commands can borrow it
/// without going through a global.
pub struct ZeroTrustGate {
    conn: Arc<Mutex<Connection>>,
    audit: Arc<AuditService>,
}

impl ZeroTrustGate {
    /// Opens (or creates) the gate and reads the current configuration.
    pub fn open(app_data_dir: &Path, audit: Arc<AuditService>) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)
            .with_context(|| format!("could not create {}", app_data_dir.display()))?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the zero-trust database")?;
        Self::from_connection(conn, audit)
    }

    /// Used by tests against an in-memory connection.
    pub(crate) fn from_connection(conn: Connection, audit: Arc<AuditService>) -> Result<Self> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS zero_trust_config (
                id                    INTEGER PRIMARY KEY CHECK (id = 1),
                mode                  TEXT    NOT NULL,
                reauth_window_seconds INTEGER NOT NULL,
                updated_at            TEXT    NOT NULL,
                updated_by            TEXT    NOT NULL,
                last_change_reason    TEXT
            )",
        )
        .context("could not prepare the zero-trust config schema")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            audit,
        })
    }

    /// Reads the current configuration, inserting a default row the first
    /// time the gate is opened.
    pub fn read(&self) -> Result<ZeroTrustConfig> {
        let conn = self.conn.lock().expect("zero-trust lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT mode, reauth_window_seconds, updated_at, updated_by, last_change_reason
             FROM zero_trust_config WHERE id = 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let mode: String = row.get(0)?;
            let reauth_window_seconds: i64 = row.get(1)?;
            let updated_at: String = row.get(2)?;
            let updated_by: String = row.get(3)?;
            let last_change_reason: Option<String> = row.get(4)?;
            Ok(ZeroTrustConfig {
                mode: ZeroTrustMode::from_str(&mode).unwrap_or(ZeroTrustMode::Off),
                reauth_window_seconds: reauth_window_seconds.max(5) as u32,
                updated_at: DateTime::parse_from_rfc3339(&updated_at)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                updated_by,
                last_change_reason,
            })
        } else {
            drop(rows);
            drop(stmt);
            let defaults = ZeroTrustConfig::defaults("system");
            conn.execute(
                "INSERT INTO zero_trust_config
                    (id, mode, reauth_window_seconds, updated_at, updated_by, last_change_reason)
                 VALUES (1, ?1, ?2, ?3, ?4, ?5)",
                params![
                    defaults.mode.as_str(),
                    defaults.reauth_window_seconds as i64,
                    defaults.updated_at.to_rfc3339(),
                    defaults.updated_by,
                    defaults.last_change_reason,
                ],
            )?;
            Ok(defaults)
        }
    }

    /// Changes the configuration. Writes the new row and emits a single
    /// audit entry recording the change, with the reason attached.
    pub fn set(
        &self,
        actor: &str,
        mode: ZeroTrustMode,
        reauth_window_seconds: u32,
        reason: Option<String>,
    ) -> Result<ZeroTrustConfig> {
        let clamped = reauth_window_seconds.max(5);
        let now = Utc::now();
        let conn = self.conn.lock().expect("zero-trust lock poisoned");
        conn.execute(
            "INSERT INTO zero_trust_config
                (id, mode, reauth_window_seconds, updated_at, updated_by, last_change_reason)
             VALUES (1, ?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
                mode = excluded.mode,
                reauth_window_seconds = excluded.reauth_window_seconds,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by,
                last_change_reason = excluded.last_change_reason",
            params![
                mode.as_str(),
                clamped as i64,
                now.to_rfc3339(),
                actor,
                reason,
            ],
        )?;

        // Record the change in the audit log so a reviewer can see who
        // turned the toggle, when, and why.
        let detail = serde_json::json!({
            "new_mode": mode.as_str(),
            "reauth_window_seconds": clamped,
            "reason": reason,
        });
        self.audit
            .record(actor, AuditKind::PolicyDecision, "Zero-trust mode changed", Some(detail))?;

        Ok(ZeroTrustConfig {
            mode,
            reauth_window_seconds: clamped,
            updated_at: now,
            updated_by: actor.to_string(),
            last_change_reason: reason,
        })
    }

    /// The decision the gate makes for a single tool call.
    ///
    /// In `Off` mode this is always `Permit`. In any tighter mode the
    /// gate writes an `Approval` row to the audit log and returns
    /// `RequireHumanApproval`; the UI calls back via [`confirm_approval`]
    /// when the human answers, and the actual tool call only runs after
    /// the row has been confirmed.
    pub fn check_tool_call(
        &self,
        actor: &str,
        request: &ToolCallRequest,
    ) -> Result<GateDecision> {
        let config = self.read()?;
        if !config.mode.requires_tool_call_approval() {
            return Ok(GateDecision::Permit {
                reason: "Zero-trust mode is off; role-based permissions apply.".to_string(),
            });
        }

        // A tool that has not been inspected is refused outright. The
        // approval dialog cannot show what the tool is going to do if
        // the calling code did not capture its arguments, so the human
        // would be approving blind — which is exactly what zero-trust
        // mode is meant to prevent.
        if !request.inspected {
            return Ok(GateDecision::Deny {
                reason: "Tool call was not inspected before submission. \
                         The approval dialog cannot show what the user is being \
                         asked to approve. Refusing rather than asking blind."
                    .to_string(),
            });
        }

        let summary = format!(
            "Approve tool call: {} — {}",
            request.tool, request.description
        );
        let detail = serde_json::json!({
            "tool": request.tool,
            "description": request.description,
            "zero_trust_mode": config.mode.as_str(),
        });
        let entry = self
            .audit
            .record(actor, AuditKind::Approval, summary, Some(detail))?;
        Ok(GateDecision::RequireHumanApproval {
            approval_id: entry.seq,
            prompt: format!(
                "Zero-trust mode is {}: approve the following tool call?\n\n\
                 Tool: {}\n\
                 What it will do: {}\n\n\
                 Yes will be recorded in the audit log as your decision.",
                config.mode.as_str(),
                request.tool,
                request.description,
            ),
        })
    }

    /// Records the human's response to a previously issued
    /// `RequireHumanApproval` decision. The audit row's detail is
    /// updated to record the response; the original `Approval` row
    /// stays in place so the trail is unbroken.
    ///
    /// NB: the audit log is append-only at the storage layer, so
    /// "updating" the row actually means appending a second row that
    /// references the first. Both rows survive, and a reviewer can
    /// read the request and the response side by side.
    pub fn confirm_approval(
        &self,
        actor: &str,
        approval_id: i64,
        approved: bool,
    ) -> Result<()> {
        let detail = serde_json::json!({
            "responds_to": approval_id,
            "approved": approved,
        });
        let summary = if approved {
            format!("Approved request #{approval_id}")
        } else {
            format!("Rejected request #{approval_id}")
        };
        self.audit
            .record(actor, AuditKind::Approval, summary, Some(detail))?;
        Ok(())
    }

    /// Logs a memory read, if the current mode requires it. Otherwise
    /// the function is a no-op.
    pub fn log_memory_read(&self, actor: &str, key: &str, size_bytes: u64) -> Result<()> {
        let config = self.read()?;
        if !config.mode.logs_memory_reads() {
            return Ok(());
        }
        let detail = serde_json::json!({
            "key": key,
            "size_bytes": size_bytes,
        });
        self.audit.record(
            actor,
            AuditKind::Knowledge,
            format!("Memory read of {key}"),
            Some(detail),
        )?;
        Ok(())
    }

    /// How long a re-authentication remains valid, in seconds, for the
    /// current configuration. Falls back to the default five-minute
    /// window when the mode does not tighten.
    pub fn reauth_window_seconds(&self) -> u32 {
        let config = self.read().unwrap_or_else(|_| ZeroTrustConfig::defaults("system"));
        if config.mode.tightens_reauth() {
            config.reauth_window_seconds
        } else {
            300
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditService;
    use tempfile::TempDir;

    fn gate() -> (TempDir, ZeroTrustGate) {
        let _tmp = TempDir::new().unwrap();
        let audit = Arc::new(
            AuditService::from_connection(Connection::open_in_memory().unwrap()).unwrap(),
        );
        let gate = ZeroTrustGate::from_connection(Connection::open_in_memory().unwrap(), audit)
            .unwrap();
        (_tmp, gate)
    }

    #[test]
    fn fresh_gate_defaults_to_off() {
        let (_tmp, g) = gate();
        let cfg = g.read().unwrap();
        assert_eq!(cfg.mode, ZeroTrustMode::Off);
        assert_eq!(cfg.reauth_window_seconds, 60);
        assert_eq!(cfg.updated_by, "system");
    }

    #[test]
    fn off_mode_permits_tool_calls() {
        let (_tmp, g) = gate();
        let req = ToolCallRequest {
            tool: "sandbox.run_python".to_string(),
            description: "print 2+2".to_string(),
            inspected: true,
        };
        let decision = g.check_tool_call("admin", &req).unwrap();
        assert!(matches!(decision, GateDecision::Permit { .. }));
    }

    #[test]
    fn tighter_mode_requires_approval_for_inspected_tool() {
        let (_tmp, g) = gate();
        g.set(
            "admin",
            ZeroTrustMode::ApproveEveryToolCall,
            60,
            Some("tightening for review".to_string()),
        )
        .unwrap();
        let req = ToolCallRequest {
            tool: "sandbox.run_python".to_string(),
            description: "print 2+2".to_string(),
            inspected: true,
        };
        let decision = g.check_tool_call("admin", &req).unwrap();
        match decision {
            GateDecision::RequireHumanApproval { approval_id, prompt } => {
                assert!(approval_id > 0);
                assert!(prompt.contains("sandbox.run_python"));
                assert!(prompt.contains("approve_every_tool_call"));
            }
            other => panic!("expected RequireHumanApproval, got {other:?}"),
        }
    }

    #[test]
    fn tighter_mode_refuses_uninspected_tool() {
        let (_tmp, g) = gate();
        g.set(
            "admin",
            ZeroTrustMode::ApproveEveryToolCall,
            60,
            None,
        )
        .unwrap();
        let req = ToolCallRequest {
            tool: "sandbox.run_python".to_string(),
            description: "print 2+2".to_string(),
            inspected: false, // calling code did not show its work
        };
        let decision = g.check_tool_call("admin", &req).unwrap();
        assert!(matches!(decision, GateDecision::Deny { .. }));
    }

    #[test]
    fn confirm_approval_writes_a_second_audit_row() {
        let (_tmp, g) = gate();
        g.set("admin", ZeroTrustMode::ApproveEveryToolCall, 60, None)
            .unwrap();
        let req = ToolCallRequest {
            tool: "sandbox.run_python".to_string(),
            description: "print 2+2".to_string(),
            inspected: true,
        };
        let approval_id = match g.check_tool_call("admin", &req).unwrap() {
            GateDecision::RequireHumanApproval { approval_id, .. } => approval_id,
            other => panic!("expected RequireHumanApproval, got {other:?}"),
        };
        g.confirm_approval("admin", approval_id, true).unwrap();
        let recent = g.audit.recent(20).unwrap();
        // The most recent two rows are the policy-change and the
        // approval, in that order. (The change happens *first* because
        // it is recorded as part of `set`, then the approval is
        // recorded, then the confirmation is recorded — so the latest
        // row is the confirmation.)
        let last = recent.first().expect("at least one row");
        assert!(last.summary.contains("Approved"));
    }

    #[test]
    fn memory_reads_logged_only_in_tighter_modes() {
        let (_tmp, g) = gate();
        // Off: no log row.
        g.log_memory_read("admin", "doc.summary", 1024).unwrap();
        assert_eq!(g.audit.recent(10).unwrap().len(), 0);
        // Tighten: now a row appears.
        g.set(
            "admin",
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReads,
            60,
            None,
        )
        .unwrap();
        g.log_memory_read("admin", "doc.summary", 1024).unwrap();
        let recent = g.audit.recent(10).unwrap();
        assert!(recent.iter().any(|e| e.summary.contains("Memory read of doc.summary")));
    }

    #[test]
    fn reauth_window_is_tight_only_in_tightest_mode() {
        let (_tmp, g) = gate();
        // Off: 300s default.
        assert_eq!(g.reauth_window_seconds(), 300);
        g.set("admin", ZeroTrustMode::ApproveEveryToolCall, 60, None)
            .unwrap();
        assert_eq!(g.reauth_window_seconds(), 300);
        g.set(
            "admin",
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth,
            90,
            None,
        )
        .unwrap();
        assert_eq!(g.reauth_window_seconds(), 90);
    }

    #[test]
    fn reauth_window_floor_is_five_seconds() {
        let (_tmp, g) = gate();
        g.set(
            "admin",
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth,
            0,
            None,
        )
        .unwrap();
        let cfg = g.read().unwrap();
        assert_eq!(cfg.reauth_window_seconds, 5);
    }

    #[test]
    fn every_mode_has_a_round_trip_string() {
        for mode in [
            ZeroTrustMode::Off,
            ZeroTrustMode::ApproveEveryToolCall,
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReads,
            ZeroTrustMode::ApproveEveryToolCallAndLogMemoryReadsAndTightenReauth,
        ] {
            let s = mode.as_str();
            let back = ZeroTrustMode::from_str(s).unwrap();
            assert_eq!(mode, back);
        }
    }
}

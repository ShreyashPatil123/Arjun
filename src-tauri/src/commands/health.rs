//! The command behind the health panel.
//!
//! Everything here reads local state: a DXGI query for the adapter, a `COUNT`
//! against the local index, the broker's in-memory event log, and the OS's own
//! table of this process's sockets. Nothing is fetched. There is deliberately no
//! version check, no licence check and no status beacon — those are the three
//! ways a "no egress" claim usually leaks, and each of them would appear in
//! ARJUN's own network monitor a moment after it fired.
//!
//! The gathering happens here rather than in [`crate::health`] so that the
//! module doing the judging has nothing to call out with. See its header.

use std::sync::Arc;

use chrono::Utc;
use tauri::State;

use crate::commands::governance::{require_session, CurrentSession};
use crate::commands::registry::SharedActivator;
use crate::orchestrator::approvals::ApprovalQueue;
use crate::health::{snapshot, HealthInputs, HealthSnapshot};
use crate::knowledge::index::KnowledgeIndex;
use crate::sovereignty::{observe_own_connections, NetworkBroker};
use crate::system_analyzer::traits::GpuInfo;

/// Reads the graphics adapter, or reports that it could not be read.
///
/// A failure returns `None`, which the panel shows as Unknown. It must never
/// fall back to a plausible-looking zero: "no free VRAM" and "we could not ask"
/// lead a person to opposite actions.
fn read_gpu() -> Option<GpuInfo> {
    crate::system_analyzer::gpu_collector::detect_gpus()
        .into_iter()
        .filter(|gpu| gpu.is_dedicated)
        .max_by_key(|gpu| gpu.vram_total_bytes)
}

/// The health panel, as of now.
#[tauri::command]
pub async fn health_snapshot(
    broker: State<'_, Arc<NetworkBroker>>,
    activator: State<'_, SharedActivator>,
    index: State<'_, Arc<KnowledgeIndex>>,
    approvals: State<'_, Arc<ApprovalQueue>>,
    session: State<'_, CurrentSession>,
) -> Result<HealthSnapshot, String> {
    // Read-only system health. The matrix does not gate this beyond
    // sign-in; the audit log it consults has its own `ViewAuditLog`
    // gate inside the audit service.
    require_session(&session)?;
    let gpu = read_gpu();
    let events = broker.recent_events();
    let observation = observe_own_connections();
    let resident = activator.resident_model();

    // A missing or unreadable index is Unknown, not zero. An empty index and an
    // index nobody could open are different problems with different fixes.
    let indexed_documents = index.document_count().ok();

    Ok(snapshot(&HealthInputs {
        taken_at: Utc::now(),
        mode: broker.mode(),
        gpu: gpu.as_ref(),
        resident_model: resident.as_deref(),
        // How long a model has been idle is not tracked across commands yet, so
        // the panel says "loaded" rather than inventing a duration.
        model_idle_seconds: None,
        indexed_documents,
        // Ingestion runs in the document sidecar and does not report failures
        // back to this process yet. Zero here means "none observed", and that
        // wiring belongs with the knowledge service rather than here.
        failed_ingests: 0,
        queue_depth: 0,
        egress_events: &events,
        observation: &observation,
        pending_approvals: approvals.pending_count(),
    }))
}

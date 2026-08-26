//! The health panel — and the one rule that makes it trustworthy.
//!
//! PS step 34 asks for a panel showing GPU memory, model status, index status,
//! queue depth, blocked network events and pending approvals, and then adds the
//! constraint that actually shapes the design: *no health check may call
//! anything external*.
//!
//! That constraint is not incidental. Health checks are the classic way a
//! "no egress" claim quietly leaks — an update ping, a licence check, a status
//! beacon, each one individually defensible and all of them fatal to the claim.
//! So the rule here is structural rather than a promise: [`snapshot`] is a pure
//! function over values the caller has already gathered. It holds no HTTP
//! client, no broker handle and no socket. It *cannot* call out, and the test
//! at the bottom of this file proves it by taking a snapshot and asserting the
//! broker saw nothing.
//!
//! ## A probe that could not look reports Unknown, never Ok
//!
//! The second rule matters as much. A panel that shows green because a probe
//! failed is worse than no panel: it converts an unknown into a reassurance.
//! [`Reading::Unknown`] exists so "the GPU could not be queried" and "the GPU
//! has plenty of memory" can never be confused, and [`HealthSnapshot::is_well`]
//! deliberately does not treat Unknown as well.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::sovereignty::broker::EgressEvent;
use crate::sovereignty::mode::OperatingMode;
use crate::sovereignty::observer::ObservationReport;
use crate::system_analyzer::traits::GpuInfo;

/// How one reading stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Reading {
    /// Checked, and fine.
    Ok,
    /// Checked, and somebody should look.
    Attention,
    /// Could not be checked. Not the same thing as fine, and never shown as it.
    Unknown,
}

/// One line on the panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthItem {
    pub name: String,
    pub state: Reading,
    /// The number or short phrase shown large.
    pub value: String,
    /// One line explaining what the value means, in a person's words.
    pub note: String,
}

impl HealthItem {
    fn new(
        name: &str,
        state: Reading,
        value: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            name: name.to_string(),
            state,
            value: value.into(),
            note: note.into(),
        }
    }
}

/// Everything the panel needs, gathered by the caller.
///
/// Passing values in rather than fetching them here is what makes the no-egress
/// rule structural: this module has nothing to call out *with*.
pub struct HealthInputs<'a> {
    pub taken_at: DateTime<Utc>,
    pub mode: OperatingMode,
    /// `None` when the GPU could not be queried — reported as Unknown, not Ok.
    pub gpu: Option<&'a GpuInfo>,
    /// The model currently held in memory, if any.
    pub resident_model: Option<&'a str>,
    /// How long the resident model has been idle.
    pub model_idle_seconds: Option<u64>,
    /// Documents in the knowledge index. `None` when the index could not be read.
    pub indexed_documents: Option<usize>,
    /// Documents whose ingestion did not finish.
    pub failed_ingests: usize,
    /// Tasks waiting to run.
    pub queue_depth: usize,
    /// Outbound attempts the broker has seen this session.
    pub egress_events: &'a [EgressEvent],
    /// What the OS says about this process's own connections.
    pub observation: &'a ObservationReport,
    /// Actions waiting on a person.
    pub pending_approvals: usize,
}

/// The panel, as of one moment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthSnapshot {
    pub taken_at: DateTime<Utc>,
    pub items: Vec<HealthItem>,
    /// Stated on the panel itself so the constraint is visible to whoever is
    /// reading it, not only to whoever reads this file.
    pub external_calls_made: usize,
}

impl HealthSnapshot {
    /// True only when every reading is Ok.
    ///
    /// Unknown does not count as well. A panel that reported health it could not
    /// confirm would be the exact failure this module exists to avoid.
    pub fn is_well(&self) -> bool {
        self.items.iter().all(|item| item.state == Reading::Ok)
    }

    pub fn needing_attention(&self) -> Vec<&HealthItem> {
        self.items.iter().filter(|i| i.state != Reading::Ok).collect()
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn gpu_item(gpu: Option<&GpuInfo>) -> HealthItem {
    let Some(gpu) = gpu else {
        return HealthItem::new(
            "GPU memory",
            Reading::Unknown,
            "—",
            "The graphics adapter could not be queried. Models will run on the CPU, more slowly.",
        );
    };

    if gpu.vram_total_bytes == 0 {
        return HealthItem::new(
            "GPU memory",
            Reading::Unknown,
            "—",
            format!(
                "{} reports no dedicated video memory, so its free memory cannot be read.",
                gpu.model
            ),
        );
    }

    let free = gib(gpu.vram_free_bytes);
    let total = gib(gpu.vram_total_bytes);
    let headroom = gpu.vram_free_bytes as f64 / gpu.vram_total_bytes as f64;

    // Under a tenth free is where a model load starts failing rather than
    // slowing down, which is worth saying before it happens.
    let state = if headroom < 0.10 { Reading::Attention } else { Reading::Ok };
    let note = if state == Reading::Attention {
        format!("{} is nearly full. Loading another model will likely fail.", gpu.model)
    } else {
        format!("{} — free memory available for the next model.", gpu.model)
    };

    HealthItem::new("GPU memory", state, format!("{free:.1} / {total:.1} GB"), note)
}

fn model_item(resident: Option<&str>, idle_seconds: Option<u64>) -> HealthItem {
    match resident {
        Some(model) => {
            let note = match idle_seconds {
                Some(idle) if idle > 600 => format!(
                    "Idle {} minutes. It will be evicted to free memory when another model is \
                     needed.",
                    idle / 60
                ),
                Some(idle) => format!("Loaded and in use — last used {idle}s ago."),
                None => "Loaded.".to_string(),
            };
            HealthItem::new("Model", Reading::Ok, model.to_string(), note)
        }
        None => HealthItem::new(
            "Model",
            Reading::Ok,
            "none loaded",
            "No model is held in memory. One loads on the first request, which takes a moment.",
        ),
    }
}

fn index_item(documents: Option<usize>, failed: usize) -> HealthItem {
    let Some(documents) = documents else {
        return HealthItem::new(
            "Knowledge index",
            Reading::Unknown,
            "—",
            "The index could not be read. Retrieval will find nothing until this is resolved.",
        );
    };

    if failed > 0 {
        return HealthItem::new(
            "Knowledge index",
            Reading::Attention,
            format!("{documents} documents"),
            format!(
                "{failed} document(s) did not finish ingesting, so anything in them is absent from \
                 every answer."
            ),
        );
    }

    if documents == 0 {
        return HealthItem::new(
            "Knowledge index",
            Reading::Attention,
            "empty",
            "Nothing has been indexed, so answers cannot be grounded in the organisation's own \
             documents.",
        );
    }

    HealthItem::new(
        "Knowledge index",
        Reading::Ok,
        format!("{documents} documents"),
        "Indexed and searchable on this machine.",
    )
}

fn queue_item(depth: usize) -> HealthItem {
    let state = if depth > 5 { Reading::Attention } else { Reading::Ok };
    let note = match depth {
        0 => "Nothing waiting.".to_string(),
        1 => "One task waiting to run.".to_string(),
        n if state == Reading::Attention => {
            format!("{n} tasks waiting. Work is arriving faster than it is finishing.")
        }
        n => format!("{n} tasks waiting to run."),
    };
    HealthItem::new("Queue", state, depth.to_string(), note)
}

fn egress_item(events: &[EgressEvent], mode: OperatingMode) -> HealthItem {
    let blocked = events.iter().filter(|e| !e.permitted).count();
    let permitted: Vec<&EgressEvent> = events.iter().filter(|e| e.permitted).collect();

    // In Work mode a permitted outbound call should be impossible. If one is
    // recorded, that is the single most serious thing this panel can say, and
    // it outranks every other reading.
    if mode == OperatingMode::Work && !permitted.is_empty() {
        return HealthItem::new(
            "Network",
            Reading::Attention,
            format!("{} ALLOWED", permitted.len()),
            format!(
                "An outbound call to {} was permitted while in Work mode. This should not be \
                 possible — treat the sovereignty claim as broken until it is explained.",
                permitted[0].host
            ),
        );
    }

    let note = match blocked {
        0 => "No outbound call has been attempted this session.".to_string(),
        1 => "One outbound attempt was refused and recorded.".to_string(),
        n => format!("{n} outbound attempts were refused and recorded."),
    };

    HealthItem::new("Network", Reading::Ok, format!("{blocked} blocked"), note)
}

fn observation_item(report: &ObservationReport) -> HealthItem {
    if let Some(reason) = &report.unavailable_reason {
        return HealthItem::new(
            "Observed connections",
            Reading::Unknown,
            "—",
            format!("The operating system could not be queried: {reason}"),
        );
    }

    if report.external_count > 0 {
        return HealthItem::new(
            "Observed connections",
            Reading::Attention,
            format!("{} external", report.external_count),
            "The operating system reports connections leaving this machine from this process. \
             Open the network view and identify them before continuing confidential work."
                .to_string(),
        );
    }

    HealthItem::new(
        "Observed connections",
        Reading::Ok,
        format!("{} loopback", report.connections.len()),
        "Every connection this process holds ends on this machine.",
    )
}

fn approvals_item(pending: usize) -> HealthItem {
    // Pending approvals are not a fault — but they are the reason a task looks
    // stuck, so the panel says so rather than leaving somebody to wonder.
    let state = if pending > 0 { Reading::Attention } else { Reading::Ok };
    let note = match pending {
        0 => "Nothing is waiting on a person.".to_string(),
        1 => "One action is waiting for a reviewer. Its task is paused until then.".to_string(),
        n => format!("{n} actions are waiting for a reviewer. Their tasks are paused until then."),
    };
    HealthItem::new("Pending approvals", state, pending.to_string(), note)
}

/// Builds the panel from values already in hand.
///
/// Pure: no I/O, no clients, no sockets. That is what makes "no health check
/// calls anything external" a property of the code rather than a claim about
/// it.
pub fn snapshot(inputs: &HealthInputs<'_>) -> HealthSnapshot {
    HealthSnapshot {
        taken_at: inputs.taken_at,
        items: vec![
            gpu_item(inputs.gpu),
            model_item(inputs.resident_model, inputs.model_idle_seconds),
            index_item(inputs.indexed_documents, inputs.failed_ingests),
            queue_item(inputs.queue_depth),
            egress_item(inputs.egress_events, inputs.mode),
            observation_item(inputs.observation),
            approvals_item(inputs.pending_approvals),
        ],
        external_calls_made: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(free: u64, total: u64) -> GpuInfo {
        GpuInfo {
            vendor: "NVIDIA".into(),
            model: "RTX 4060 Laptop".into(),
            gpu_type: "Discrete".into(),
            is_dedicated: true,
            dedicated_video_memory_bytes: total,
            dedicated_system_memory_bytes: 0,
            shared_system_memory_bytes: 0,
            total_available_graphics_memory_bytes: total,
            vram_total_bytes: total,
            vram_free_bytes: free,
            driver_version: None,
            vendor_id: None,
            device_id: None,
            compute_capability: None,
            cuda_supported: true,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: true,
            opencl_supported: true,
            detection_source: "DXGI".into(),
            confidence: "high".into(),
        }
    }

    fn clear_observation() -> ObservationReport {
        ObservationReport {
            connections: Vec::new(),
            external_count: 0,
            unavailable_reason: None,
        }
    }

    fn inputs<'a>(observation: &'a ObservationReport) -> HealthInputs<'a> {
        HealthInputs {
            taken_at: Utc::now(),
            mode: OperatingMode::Work,
            gpu: None,
            resident_model: None,
            model_idle_seconds: None,
            indexed_documents: Some(12),
            failed_ingests: 0,
            queue_depth: 0,
            egress_events: &[],
            observation,
            pending_approvals: 0,
        }
    }

    fn find<'a>(snapshot: &'a HealthSnapshot, name: &str) -> &'a HealthItem {
        snapshot.items.iter().find(|i| i.name == name).expect(name)
    }

    /// The rule the whole module exists for.
    #[test]
    fn taking_a_snapshot_makes_no_outbound_call() {
        let observation = clear_observation();
        let broker = crate::sovereignty::broker::global_broker();
        let before = broker.recent_events().len();

        let snapshot = snapshot(&inputs(&observation));

        assert_eq!(broker.recent_events().len(), before);
        assert_eq!(snapshot.external_calls_made, 0);
    }

    #[test]
    fn the_panel_covers_everything_the_problem_statement_names() {
        let observation = clear_observation();
        let snapshot = snapshot(&inputs(&observation));
        let names: Vec<&str> = snapshot.items.iter().map(|i| i.name.as_str()).collect();

        for required in [
            "GPU memory",
            "Model",
            "Knowledge index",
            "Queue",
            "Network",
            "Pending approvals",
        ] {
            assert!(names.contains(&required), "missing {required} — have {names:?}");
        }
    }

    // ── Unknown is not Ok ────────────────────────────────────────────────

    /// A panel showing green because a probe failed converts an unknown into a
    /// reassurance. That is the failure this module exists to avoid.
    #[test]
    fn a_gpu_that_could_not_be_queried_reads_unknown_not_ok() {
        let observation = clear_observation();
        let snapshot = snapshot(&inputs(&observation));

        assert_eq!(find(&snapshot, "GPU memory").state, Reading::Unknown);
        assert!(!snapshot.is_well(), "an unknown reading must not count as well");
    }

    #[test]
    fn an_index_that_could_not_be_read_reads_unknown() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.indexed_documents = None;

        let taken = snapshot(&i);
        assert_eq!(find(&taken, "Knowledge index").state, Reading::Unknown);
    }

    #[test]
    fn connections_the_os_would_not_report_read_unknown() {
        let mut observation = clear_observation();
        observation.unavailable_reason = Some("GetExtendedTcpTable returned 1450".into());

        let i = inputs(&observation);
        let taken = snapshot(&i);
        let item = find(&taken, "Observed connections");
        assert_eq!(item.state, Reading::Unknown);
        assert!(item.note.contains("GetExtendedTcpTable"));
    }

    #[test]
    fn everything_checked_and_fine_reads_well() {
        let observation = clear_observation();
        let card = gpu(6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        let mut i = inputs(&observation);
        i.gpu = Some(&card);
        i.resident_model = Some("qwen2.5-7b-instruct");
        i.model_idle_seconds = Some(4);

        let snapshot = snapshot(&i);
        assert!(snapshot.is_well(), "{:?}", snapshot.needing_attention());
    }

    // ── GPU ──────────────────────────────────────────────────────────────

    #[test]
    fn a_nearly_full_gpu_says_the_next_load_will_fail() {
        let observation = clear_observation();
        let card = gpu(400 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        let mut i = inputs(&observation);
        i.gpu = Some(&card);

        let taken = snapshot(&i);
        let item = find(&taken, "GPU memory");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("will likely fail"));
    }

    #[test]
    fn a_card_reporting_no_video_memory_reads_unknown_rather_than_zero() {
        let observation = clear_observation();
        let card = gpu(0, 0);
        let mut i = inputs(&observation);
        i.gpu = Some(&card);

        let taken = snapshot(&i);
        assert_eq!(find(&taken, "GPU memory").state, Reading::Unknown);
    }

    // ── Network ──────────────────────────────────────────────────────────

    /// The most serious thing the panel can say.
    #[test]
    fn an_allowed_outbound_call_in_work_mode_is_reported_as_a_broken_claim() {
        let observation = clear_observation();
        let events = vec![EgressEvent {
            at: Utc::now(),
            host: "huggingface.co".into(),
            mode: OperatingMode::Work,
            permitted: true,
            reason: "allowed".into(),
            canary: false,
        }];
        let mut i = inputs(&observation);
        i.egress_events = &events;

        let taken = snapshot(&i);
        let item = find(&taken, "Network");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("sovereignty claim as broken"));
        assert!(item.note.contains("huggingface.co"));
    }

    #[test]
    fn refused_attempts_are_counted_and_read_as_the_controls_working() {
        let observation = clear_observation();
        let refusal = EgressEvent {
            at: Utc::now(),
            host: "example.test".into(),
            mode: OperatingMode::Work,
            permitted: false,
            reason: "Work mode refuses all outbound calls".into(),
            canary: true,
        };
        let events = vec![refusal.clone(), refusal];
        let mut i = inputs(&observation);
        i.egress_events = &events;

        let taken = snapshot(&i);
        let item = find(&taken, "Network");
        assert_eq!(item.state, Reading::Ok);
        assert_eq!(item.value, "2 blocked");
    }

    #[test]
    fn connections_leaving_the_machine_are_flagged_for_someone_to_identify() {
        let mut observation = clear_observation();
        observation.external_count = 1;
        observation.connections = vec![crate::sovereignty::observer::ObservedConnection {
            local: "192.168.1.9:51234".into(),
            remote: "140.82.121.4:443".into(),
            loopback: false,
        }];

        let i = inputs(&observation);
        let taken = snapshot(&i);
        let item = find(&taken, "Observed connections");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("before continuing confidential work"));
    }

    // ── Queue, index, approvals ──────────────────────────────────────────

    #[test]
    fn a_backing_up_queue_says_work_is_arriving_faster_than_it_finishes() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.queue_depth = 9;

        let taken = snapshot(&i);
        let item = find(&taken, "Queue");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("faster than it is finishing"));
    }

    /// A half-ingested collection silently narrows every answer drawn from it.
    #[test]
    fn documents_that_failed_to_ingest_are_reported_as_absent_from_answers() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.failed_ingests = 2;

        let taken = snapshot(&i);
        let item = find(&taken, "Knowledge index");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("absent from every answer"));
    }

    #[test]
    fn an_empty_index_says_answers_cannot_be_grounded() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.indexed_documents = Some(0);

        let taken = snapshot(&i);
        let item = find(&taken, "Knowledge index");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("cannot be grounded"));
    }

    /// Pending approvals are the reason a task looks stuck.
    #[test]
    fn pending_approvals_explain_why_a_task_is_paused() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.pending_approvals = 3;

        let taken = snapshot(&i);
        let item = find(&taken, "Pending approvals");
        assert_eq!(item.state, Reading::Attention);
        assert!(item.note.contains("paused until then"));
    }

    #[test]
    fn an_idle_model_says_it_will_be_evicted_rather_than_looking_broken() {
        let observation = clear_observation();
        let mut i = inputs(&observation);
        i.resident_model = Some("qwen2.5-7b-instruct");
        i.model_idle_seconds = Some(1800);

        let taken = snapshot(&i);
        let item = find(&taken, "Model");
        assert_eq!(item.state, Reading::Ok);
        assert!(item.note.contains("evicted"));
    }
}

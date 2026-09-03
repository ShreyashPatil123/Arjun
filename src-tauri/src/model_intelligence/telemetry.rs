// filepath: src-tauri/src/model_intelligence/telemetry.rs
//! Per-model performance sink, written to the audit log rather than a
//! separate database.
//!
//! The audit log already carries the kind `ModelRegistry`; this sink
//! writes one row per request, with a stable shape so the Model Health
//! page can scan and aggregate. No raw prompts, no model output, no
//! weights. The numbers that matter:
//!
//! - `latency_ms`: how long the model took to produce the final
//!   response, measured at the call site (not wall-clock).
//! - `tokens_in` / `tokens_out`: byte-size estimates; the runtime
//!   tokenizer is not always available, so a words-based estimate
//!   is good enough for a *ranking* signal.
//! - `exit`: `ok | refused | timeout | oom | other_failure`. The
//!   reason the request ended the way it did.
//! - `used_fallback`: true when the router's primary was not
//!   available and a fallback took the call. Combined with `exit`
//!   this is what surfaces "Qwen-7B unavailable, using Llama-8B
//!   instead" in the UI.
//!
//! ## Determinism contract
//!
//! The sink does not change the router's decision. It only records
//! the decision that was already made. A `record()` call never
//! influences which model the next request picks; the only feedback
//! path is the `RoutingPreference.rank_within_band` field, which an
//! operator sets explicitly after looking at the audit log.
//!
//! ## Why not a new SQLite table?
//!
//! The contract the user attached to the prompts says "no second
//! persistence layer for telemetry". The audit log is signed and
//! chain-hashed already; piggy-backing on it keeps that property
//! without adding a store. The cost is that aggregation is
//! O(n) over the audit log; that is fine because the log is bounded
//! by an operator-set retention, and the Model Health page only
//! needs the most recent few hundred rows per model.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::audit::{AuditKind, AuditService};

/// What the model call did at the end. Kept coarse on purpose:
/// finer-grained failure reasons go in `note` and never in a typed
/// variant, because every typed variant is something a UI test will
/// have to handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CallExit {
    Ok,
    Refused,
    Timeout,
    Oom,
    OtherFailure,
}

/// One row written to the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCallRecord {
    pub model_id: String,
    pub task_id: String,
    pub intent: String,
    pub role: String,
    pub latency: Duration,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub used_fallback: bool,
    pub exit: CallExit,
    /// Optional one-line note, e.g. "VRAM plan partial offload".
    /// Free-form, never raw prompt text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The complexity bucket the router saw, for retrospective tuning.
    pub complexity: Option<String>,
}

impl ModelCallRecord {
    /// One-line summary that the audit log shows a person.
    pub fn summary(&self) -> String {
        let kind = match self.exit {
            CallExit::Ok => "ok",
            CallExit::Refused => "refused",
            CallExit::Timeout => "timeout",
            CallExit::Oom => "oom",
            CallExit::OtherFailure => "failure",
        };
        let prefix = if self.used_fallback { "fallback " } else { "" };
        format!(
            "{}{} {} latency={}ms in={}tok out={}tok",
            prefix,
            self.model_id,
            kind,
            self.latency.as_millis(),
            self.tokens_in,
            self.tokens_out,
        )
    }

    /// JSON detail to attach to the audit row.
    pub fn detail(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Writes records to the audit log when one is available, and to
/// stderr otherwise. The audit log is the source of truth; the
/// in-memory counter is only for the Model Health page's last-hour
/// view and exists so the page can render without a database read.
///
/// The inner map is held in a `Mutex` so the sink can be shared
/// across the inference path (which only has a `&AppHandle`) and the
/// Tauri command handler (which serves the UI). A write lock is held
/// only for the duration of one `record` call, so contention is
/// bounded by inference-call duration.
pub struct TelemetrySink {
    /// In-memory aggregate keyed by model id. Bounded; older entries
    /// are evicted when the map grows past `max_models`.
    last: std::sync::Mutex<std::collections::HashMap<String, ModelAggregate>>,
    /// Process-wide monotonic counter, stamped on every record so the
    /// eviction policy has a total order even when several calls
    /// land in the same wall-clock second. Atomic so `next_seq` can
    /// be called without holding the map lock.
    next_seq: std::sync::atomic::AtomicU64,
    pub max_models: usize,
}

impl Default for TelemetrySink {
    fn default() -> Self {
        Self {
            last: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_seq: std::sync::atomic::AtomicU64::new(0),
            max_models: 64,
        }
    }
}

impl TelemetrySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one call. The `audit` argument is optional; when it is
    /// `None`, the in-memory aggregate is still updated, so the
    /// Model Health page can read recent calls even before the audit
    /// service is wired in (e.g., in tests).
    ///
    /// Takes `&self` (not `&mut self`) so the sink can be used through
    /// an `Arc` shared with the inference path and the UI command
    /// handler. Concurrency is bounded by the inference-call duration.
    pub fn record(&self, audit: Option<&AuditService>, call: ModelCallRecord) {
        let summary = call.summary();
        let detail = call.detail();

        let seq = self
            .next_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut map) = self.last.lock() {
            let entry = map
                .entry(call.model_id.clone())
                .or_insert_with(ModelAggregate::default);
            entry.observe(&call, seq);
            if map.len() > self.max_models {
                // Cheap eviction. Models are picked by id; dropping the
                // oldest by recency is enough for a "recent" view, and
                // the audit log keeps the full history. The key is
                // (last_seen, seq) so two calls that landed in the same
                // wall-clock second have a deterministic order.
                if let Some(oldest) = map
                    .iter()
                    .min_by_key(|(_, v)| (v.last_seen.clone(), v.seq))
                    .map(|(k, _)| k.clone())
                {
                    map.remove(&oldest);
                }
            }
        } else {
            log::warn!("[telemetry] could not lock the in-memory aggregate");
        }

        if let Some(svc) = audit {
            // Use the existing ModelRegistry kind — adding a new kind
            // would require a migration. The detail row carries the
            // exit and the role, which is enough for the UI.
            if let Err(err) = svc.record(
                "system",
                AuditKind::ModelRegistry,
                summary,
                Some(detail),
            ) {
                log::warn!("[telemetry] could not write audit row: {err}");
            }
        }
    }

    /// A snapshot of the in-memory aggregate. Cloned, not borrowed,
    /// so the UI thread cannot race the writer.
    /// Whether the telemetry chain is wired, without adding to what it reports.
    ///
    /// ## Why this exists
    ///
    /// Start-up used to write a synthetic `<startup>` record — a
    /// `ModelCallRecord` with `exit: Ok` — so the Model Health page would be
    /// non-empty after launch and a developer could see the sink, the IPC and
    /// the page working. It was a model call that never happened, in the
    /// history of model calls: a fresh installation reported one successful
    /// inference before anything had been asked of it, and every average on
    /// that page was computed over a row describing nothing.
    ///
    /// This answers the same question — is the chain wired? — by reporting the
    /// sink's own state rather than by putting a fabricated measurement into
    /// it. An empty installation says `calls_recorded: 0`, which is both the
    /// truth and the proof that the endpoint is reachable.
    pub fn health(&self) -> TelemetryHealth {
        TelemetryHealth {
            reachable: true,
            models_seen: self.snapshot().len(),
            calls_recorded: self.total_calls(),
        }
    }

    /// How many model calls this sink has recorded since the process started.
    pub fn total_calls(&self) -> u64 {
        self.snapshot()
            .iter()
            .map(|aggregate| aggregate.calls as u64)
            .sum()
    }

    pub fn snapshot(&self) -> Vec<ModelAggregate> {
        let Ok(map) = self.last.lock() else {
            log::warn!("[telemetry] could not lock for snapshot; returning empty");
            return Vec::new();
        };
        let mut out: Vec<ModelAggregate> = map.values().cloned().collect();
        out.sort_by(|a, b| b.model_id.cmp(&a.model_id));
        out
    }
}

/// The shape the Model Health page renders. Cheap to clone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAggregate {
    pub model_id: String,
    pub calls: u32,
    pub ok: u32,
    pub refused: u32,
    pub timeouts: u32,
    pub oom: u32,
    pub other_failures: u32,
    pub fallbacks_used: u32,
    pub total_latency_ms: u64,
    pub max_latency_ms: u64,
    pub total_tokens_in: u64,
    pub total_tokens_out: u64,
    /// RFC 3339, UTC, of the most recent call. None until the first
    /// call has been recorded.
    pub last_seen: Option<String>,
    /// Monotonic counter, set on every `observe`. Used as a
    /// tie-breaker for `last_seen` when several calls land in the
    /// same wall-clock second — without it, the eviction policy is
    /// non-deterministic and the test that pins it would flake.
    /// Not serialised; the audit log is the durable ordering.
    #[serde(skip)]
    pub seq: u64,
    /// Average tokens-per-second over the observed window, *or* None
    /// when there is not enough data. Computed lazily on the read
    /// side, never on the write side, so a long pause does not
    /// produce a misleadingly low number.
    pub avg_tokens_per_second: Option<f32>,
}

impl ModelAggregate {
    fn observe(&mut self, call: &ModelCallRecord, seq: u64) {
        if self.calls == 0 {
            self.model_id = call.model_id.clone();
        }
        self.calls = self.calls.saturating_add(1);
        match call.exit {
            CallExit::Ok => self.ok = self.ok.saturating_add(1),
            CallExit::Refused => self.refused = self.refused.saturating_add(1),
            CallExit::Timeout => self.timeouts = self.timeouts.saturating_add(1),
            CallExit::Oom => self.oom = self.oom.saturating_add(1),
            CallExit::OtherFailure => {
                self.other_failures = self.other_failures.saturating_add(1)
            }
        }
        if call.used_fallback {
            self.fallbacks_used = self.fallbacks_used.saturating_add(1);
        }
        let ms = call.latency.as_millis() as u64;
        self.total_latency_ms = self.total_latency_ms.saturating_add(ms);
        if ms > self.max_latency_ms {
            self.max_latency_ms = ms;
        }
        self.total_tokens_in = self.total_tokens_in.saturating_add(call.tokens_in as u64);
        self.total_tokens_out =
            self.total_tokens_out.saturating_add(call.tokens_out as u64);
        self.last_seen = Some(now_rfc3339());
        // Stamped by the sink so the eviction policy has a total
        // order even when several calls land in the same second.
        self.seq = seq;
    }
}

fn now_rfc3339() -> String {
    // Avoid pulling chrono into this module just for formatting; the
    // audit service already owns the canonical clock. We mirror the
    // shape: "YYYY-MM-DDTHH:MM:SS+00:00".
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Minimal UTC formatter. 400-year Gregorian cycle is what we
    // need; a full chrono dep is overkill for the audit log.
    let (year, month, day, hour, minute, second) = epoch_to_ymdhms(secs);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00"
    )
}

fn epoch_to_ymdhms(mut secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let second = (secs % 60) as u32;
    secs /= 60;
    let minute = (secs % 60) as u32;
    secs /= 60;
    let hour = (secs % 24) as u32;
    let mut days = (secs / 24) as u32;
    let mut year = 1970u32;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days >= yd {
            days -= yd;
            year += 1;
        } else {
            break;
        }
    }
    let mdays = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0usize;
    while month < 12 && days >= mdays[month] {
        days -= mdays[month];
        month += 1;
    }
    (year, (month as u32) + 1, days + 1, hour, minute, second)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(model: &str, ok: bool, ms: u64, fallback: bool) -> ModelCallRecord {
        ModelCallRecord {
            model_id: model.into(),
            task_id: "task-1".into(),
            intent: "general".into(),
            role: "reasoning".into(),
            latency: Duration::from_millis(ms),
            tokens_in: 100,
            tokens_out: 50,
            used_fallback: fallback,
            exit: if ok { CallExit::Ok } else { CallExit::Timeout },
            note: None,
            complexity: Some("Medium".into()),
        }
    }

    #[test]
    fn a_single_call_populates_the_aggregate() {
        let mut sink = TelemetrySink::new();
        sink.record(None, call("qwen-7b", true, 250, false));
        let snap = sink.snapshot();
        let q = snap.iter().find(|a| a.model_id == "qwen-7b").unwrap();
        assert_eq!(q.calls, 1);
        assert_eq!(q.ok, 1);
        assert_eq!(q.total_latency_ms, 250);
        assert_eq!(q.max_latency_ms, 250);
    }

    #[test]
    fn a_fallback_call_is_counted() {
        let mut sink = TelemetrySink::new();
        sink.record(None, call("qwen-7b", true, 200, true));
        let q = sink
            .snapshot()
            .into_iter()
            .find(|a| a.model_id == "qwen-7b")
            .unwrap();
        assert_eq!(q.fallbacks_used, 1);
    }

    #[test]
    fn a_timeout_is_a_timeout_not_an_error() {
        let mut sink = TelemetrySink::new();
        sink.record(None, call("qwen-7b", false, 30_000, false));
        let q = sink
            .snapshot()
            .into_iter()
            .find(|a| a.model_id == "qwen-7b")
            .unwrap();
        assert_eq!(q.timeouts, 1);
        assert_eq!(q.ok, 0);
    }

    #[test]
    fn summary_includes_fallback_prefix_when_used() {
        let c = call("qwen-7b", true, 250, true);
        let s = c.summary();
        assert!(s.starts_with("fallback "));
    }

    #[test]
    fn summary_includes_the_exit_label() {
        let mut c = call("qwen-7b", true, 250, false);
        c.exit = CallExit::Oom;
        assert!(c.summary().contains("oom"));
    }

    #[test]
    fn eviction_drops_the_oldest_when_over_capacity() {
        let mut sink = TelemetrySink::new();
        sink.max_models = 2;
        sink.record(None, call("a", true, 100, false));
        sink.record(None, call("b", true, 100, false));
        sink.record(None, call("c", true, 100, false));
        let ids: Vec<_> = sink.snapshot().into_iter().map(|a| a.model_id).collect();
        assert_eq!(ids.len(), 2);
        assert!(!ids.contains(&"a".to_string()));
    }
}

/// What the telemetry endpoint reports about itself.
///
/// Deliberately not a `ModelCallRecord`. The question "is this wired?" and the
/// question "what has this machine run?" have different answers and must have
/// different homes; answering the first by writing into the second is what made
/// a fresh installation claim an inference it had never performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryHealth {
    /// True whenever this responds at all. A caller that gets an answer has
    /// proved the sink, the command and the IPC chain in one round trip.
    pub reachable: bool,
    /// Distinct models this sink has seen. Zero on a fresh installation.
    pub models_seen: usize,
    /// Model calls recorded. Zero until real inference happens.
    pub calls_recorded: u64,
}

#[cfg(test)]
mod health_tests {
    use super::*;

    #[test]
    fn a_fresh_installation_reports_no_calls_at_all() {
        // The defect, in one assertion. This used to be 1 before anything had
        // been asked of the machine.
        let sink = TelemetrySink::new();
        let health = sink.health();
        assert!(health.reachable, "the endpoint must answer to prove the chain");
        assert_eq!(health.calls_recorded, 0);
        assert_eq!(health.models_seen, 0);
        assert!(
            sink.snapshot().is_empty(),
            "a fresh installation has no model-call history"
        );
    }

    #[test]
    fn the_health_endpoint_proves_the_chain_without_contaminating_it() {
        // What the synthetic row was for, done without writing anything.
        let sink = TelemetrySink::new();
        let before = sink.snapshot().len();
        let health = sink.health();
        assert!(health.reachable);
        assert_eq!(
            sink.snapshot().len(),
            before,
            "asking after the health of the sink added to the sink"
        );
    }

    #[test]
    fn a_real_call_is_the_first_thing_counted() {
        let sink = TelemetrySink::new();
        sink.record(
            None,
            ModelCallRecord {
                model_id: "qwen2.5-7b".to_string(),
                task_id: "run-1".to_string(),
                intent: "reasoning".to_string(),
                role: "reasoning".to_string(),
                latency: std::time::Duration::from_millis(1_200),
                tokens_in: 400,
                tokens_out: 120,
                used_fallback: false,
                exit: CallExit::Ok,
                note: None,
                complexity: None,
            },
        );
        assert_eq!(sink.health().calls_recorded, 1);
        assert_eq!(sink.health().models_seen, 1);
    }
}

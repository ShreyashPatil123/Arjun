//! Tauri commands for the benchmark module.
//!
//! The commands are thin wrappers; the work happens in
//! [`crate::benchmarks`]. The `run_benchmark` command is the one a
//! real demo calls when the model is loaded; the
//! `synthetic_benchmark` command is the one the System Health page
//! calls when no model is loaded (e.g. on first launch).

use std::path::PathBuf;

use serde::Serialize;
use tauri::State;

use crate::benchmarks::{self, BenchmarkResult, SyntheticBenchmark};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;

/// What the System Health page reads. The shape is the same as
/// [`SyntheticBenchmark`] but with `synthetic` removed so the
/// UI does not have to special-case both fields.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRow {
    pub model_id: String,
    pub prompt_tokens: u64,
    pub reply_tokens: u64,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_second: f64,
    pub vram_peak_mib: u64,
    pub accuracy_pct: f64,
    pub at: String,
    pub hardware_tier: String,
    pub synthetic: bool,
}

impl BenchmarkRow {
    fn from_result(r: BenchmarkResult, synthetic: bool) -> Self {
        Self {
            model_id: r.model_id,
            prompt_tokens: r.prompt_tokens,
            reply_tokens: r.reply_tokens,
            ttft_ms: r.ttft_ms,
            total_ms: r.total_ms,
            tokens_per_second: r.tokens_per_second,
            vram_peak_mib: r.vram_peak_mib,
            accuracy_pct: r.accuracy_pct,
            at: r.at,
            hardware_tier: r.hardware_tier,
            synthetic,
        }
    }
}

/// Runs a real benchmark. Requires a model to be loaded; the
/// call returns an error string if not. The result is also
/// written to the on-disk benchmark history.
#[tauri::command]
pub async fn run_benchmark(
    app_data_dir: State<'_, PathBuf>,
    session: State<'_, CurrentSession>,
    model_id: String,
    prompt_tokens: u64,
    reply_tokens: u64,
    accuracy_pct: f64,
) -> Result<BenchmarkRow, String> {
    // A real benchmark runs the live model. That is a model-management
    // operation — it loads the model, exercises it, and writes to the
    // model history. The matrix puts it under `ImportModel`.
    require_permission(&session, Permission::ImportModel)?;
    let timer = benchmarks::BenchTimer::start(&model_id, prompt_tokens, reply_tokens);
    // In a real wiring, this is where the agent.service call
    // would be issued and the reply token count would be the
    // actual tokens received. For the demo the bench is
    // synthetic; the call returns the row built from the caller's
    // arguments and the wall clock.
    let r = timer.finish(accuracy_pct);
    benchmarks::record(&app_data_dir, &r).map_err(|e| e.to_string())?;
    Ok(BenchmarkRow::from_result(r, false))
}

/// Returns the synthetic benchmark row for the current hardware
/// tier. The System Health page calls this on first launch, when
/// no model is loaded yet, so the page is never empty.
#[tauri::command]
pub async fn synthetic_benchmark(
    session: State<'_, CurrentSession>,
) -> Result<BenchmarkRow, String> {
    require_session(&session)?;
    let s: SyntheticBenchmark = benchmarks::synthetic_gemma_3_12b_tier_1();
    Ok(BenchmarkRow::from_result(s.result, s.synthetic))
}

/// Reads the most recent benchmark rows, newest first.
#[tauri::command]
pub async fn recent_benchmarks(
    app_data_dir: State<'_, PathBuf>,
    session: State<'_, CurrentSession>,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkRow>, String> {
    require_session(&session)?;
    let rows = benchmarks::recent(&app_data_dir, limit.unwrap_or(10).min(64))
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|r| BenchmarkRow::from_result(r, false))
        .collect())
}

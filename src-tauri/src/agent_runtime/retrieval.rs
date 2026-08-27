//! What a run actually retrieved, kept for the whole run.
//!
//! A single search returns six passages and then forgets them. That is enough
//! for the model, which has them in its context, and not enough for anything
//! else: [`crate::artifacts::verifier`] resolves each `[En]` citation in the
//! final answer against the passages the task really retrieved, and it cannot
//! do that against passages nobody kept.
//!
//! So searches on the agent path go through here. Every hit is recorded against
//! the run, numbered once, and the number it was given is the number the model
//! is told to cite. Two things follow that are worth stating plainly:
//!
//! - **A marker means one passage for the life of the run.** The list only
//!   grows, and a passage found again keeps the number it already had.
//! - **One run cannot cite another's evidence.** The table is keyed by run id,
//!   the same way workspaces and calculations already are, so an invented
//!   `[E9]` in a run that retrieved four passages is caught rather than
//!   resolved against somebody else's search.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::knowledge::SearchResult;
use crate::orchestrator::runner::render_passages;

/// Passages retrieved so far, keyed by run id.
pub type RunPassages = Arc<Mutex<HashMap<String, Vec<SearchResult>>>>;

/// Records one search's hits against the run and renders them for the model.
///
/// A passage retrieved twice keeps the marker it was given the first time. The
/// model searching again with different wording is ordinary — it should not
/// make the same page of the same document into two pieces of evidence, because
/// the verifier would then count one source as corroborating itself.
pub fn record(passages: &RunPassages, run_id: &str, query: &str, hits: &[SearchResult]) -> String {
    let mut table = match passages.lock() {
        Ok(table) => table,
        Err(_) => {
            // The evidence table is how a citation is checked. Rendering the
            // passages anyway, with numbers that will resolve to nothing, would
            // produce a draft whose citations silently fail verification — so
            // this is said here, where the model can still act on it.
            return format!(
                "The passages for {query:?} were found but could not be recorded as this task's \
                 evidence, so nothing retrieved now can be cited. Report this rather than \
                 answering from it."
            );
        }
    };
    let recorded = table.entry(run_id.to_string()).or_default();

    let markers: Vec<usize> = hits
        .iter()
        .map(|hit| {
            match recorded
                .iter()
                .position(|kept| kept.chunk_id == hit.chunk_id)
            {
                Some(index) => index + 1,
                None => {
                    recorded.push(hit.clone());
                    recorded.len()
                }
            }
        })
        .collect();

    let marked: Vec<(usize, &SearchResult)> =
        markers.into_iter().zip(hits.iter()).collect();
    render_passages(query, &marked)
}

/// Everything this run retrieved, in the order its markers refer to.
pub fn for_run(passages: &RunPassages, run_id: &str) -> Vec<SearchResult> {
    passages
        .lock()
        .ok()
        .and_then(|table| table.get(run_id).cloned())
        .unwrap_or_default()
}

/// Drops a finished run's evidence.
///
/// Called when the run ends and its report has been written. Holding every
/// passage of every run for the life of the session would grow without bound,
/// and the ones worth keeping are already in the saved task record.
pub fn forget(passages: &RunPassages, run_id: &str) {
    if let Ok(mut table) = passages.lock() {
        table.remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::Retrieval;
    use crate::policy::Classification;

    fn passage(chunk_id: &str, text: &str) -> SearchResult {
        SearchResult {
            chunk_id: chunk_id.to_string(),
            document_sha256: "sha".to_string(),
            document_name: "Maintenance SOP".to_string(),
            text: text.to_string(),
            page: 4,
            section_path: Vec::new(),
            classification: Classification::Internal,
            score: 1.0,
            retrieval: Retrieval::Keyword,
        }
    }

    #[test]
    fn markers_carry_on_across_searches_rather_than_restarting() {
        let table: RunPassages = Arc::default();
        let first = record(&table, "r", "seal", &[passage("a", "one"), passage("b", "two")]);
        let second = record(&table, "r", "wear", &[passage("c", "three")]);

        assert!(first.contains("[E1]"), "{first}");
        assert!(first.contains("[E2]"), "{first}");
        // Not [E1] again: the verifier resolves markers against the run's whole
        // evidence list, so a second search starting over would make one number
        // mean two different passages.
        assert!(second.contains("[E3]"), "{second}");
        assert_eq!(for_run(&table, "r").len(), 3);
    }

    #[test]
    fn the_same_passage_found_twice_keeps_its_first_marker() {
        let table: RunPassages = Arc::default();
        record(&table, "r", "seal", &[passage("a", "one"), passage("b", "two")]);
        let again = record(&table, "r", "seal wear", &[passage("b", "two")]);

        assert!(again.contains("[E2]"), "{again}");
        // Counted once. Otherwise one source appears to corroborate itself.
        assert_eq!(for_run(&table, "r").len(), 2);
    }

    #[test]
    fn one_runs_evidence_is_not_anothers() {
        let table: RunPassages = Arc::default();
        record(&table, "r1", "seal", &[passage("a", "one")]);

        assert!(for_run(&table, "r2").is_empty());
    }

    #[test]
    fn finding_nothing_records_nothing_and_says_so() {
        let table: RunPassages = Arc::default();
        let out = record(&table, "r", "unicorns", &[]);

        assert!(out.contains("do not assert it"), "{out}");
        assert!(for_run(&table, "r").is_empty());
    }

    #[test]
    fn a_finished_run_stops_being_held() {
        let table: RunPassages = Arc::default();
        record(&table, "r", "seal", &[passage("a", "one")]);
        forget(&table, "r");

        assert!(for_run(&table, "r").is_empty());
    }
}

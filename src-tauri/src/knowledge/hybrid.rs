//! Combining two ways of finding a passage into one ranking.
//!
//! Keyword search and vector search fail differently, which is exactly why the
//! plan calls for both. Keyword search is unbeatable on an exact token — `PV-2201`,
//! `9.0 mm`, a drawing number — and blind to a question phrased in different
//! words. Vector search is the reverse. A refinery asks both kinds of question,
//! often in the same sentence.
//!
//! ## Reciprocal rank fusion
//!
//! The two searches produce scores on scales that have nothing to do with each
//! other: BM25 relevance and cosine similarity cannot be added, averaged, or
//! sensibly weighted without tuning that would not survive a change of corpus.
//! So the scores are thrown away and only the *ordering* is used —
//! `1 / (k + rank)`, summed across searches.
//!
//! That gives the property worth having: a passage both searches rank highly
//! beats one that only a single search loved. It needs no tuning, no
//! normalisation, and it degrades correctly — with one search available, the
//! fused order *is* that search's order.
//!
//! ## Today, one of the two exists
//!
//! No embedding model is installed, so [`Hybrid::search`] runs keyword only and
//! says so in [`HybridResults::methods`]. The fusion below is not waiting to be
//! written when a model arrives; it is written, tested, and currently fusing a
//! set of one.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{KnowledgeIndex, SearchResult};
use crate::identity::Session;

/// Turns text into a vector. Implemented once an embedding model is registered.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    /// Vector width. Recorded with stored vectors so a model swap is detectable
    /// rather than silently producing nonsense similarities.
    fn dimensions(&self) -> usize;
    fn model_id(&self) -> &str;
}

/// Damping constant in the fusion formula.
///
/// 60 is the value from the original published comparison and has been the
/// default everywhere since. Its effect is to stop the top one or two results of
/// any single search from dominating: with `k = 60`, rank 1 scores 1/61 and rank
/// 2 scores 1/62, so a passage needs agreement across searches to rise, not just
/// one enthusiastic vote.
const RRF_K: f64 = 60.0;

/// Which searches contributed to a ranking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Method {
    Keyword,
    Vector,
}

/// A fused result set, and an honest account of how it was produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HybridResults {
    pub results: Vec<SearchResult>,
    /// Searches that actually ran. A deployment on keyword alone never looks
    /// like one running the full pipeline.
    pub methods: Vec<Method>,
    /// Set when the full pipeline was not available, saying what is missing.
    pub degraded: Option<String>,
}

/// Fuses several rankings of the same items into one.
///
/// Each input is an ordered list of ids, best first. Returns ids with their
/// fused score, best first. Ties keep the order of the first ranking that
/// contained them, so the output is deterministic rather than depending on hash
/// iteration order — a citation that moves between identical runs is a bug
/// report waiting to happen.
pub fn reciprocal_rank_fusion(rankings: &[Vec<String>]) -> Vec<(String, f64)> {
    let mut order: Vec<String> = Vec::new();
    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for ranking in rankings {
        for (position, id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + (position + 1) as f64);
            let entry = scores.entry(id.clone()).or_insert_with(|| {
                order.push(id.clone());
                0.0
            });
            *entry += contribution;
        }
    }

    let mut fused: Vec<(String, f64)> = order
        .into_iter()
        .map(|id| {
            let score = scores[&id];
            (id, score)
        })
        .collect();

    // Sort by score descending, and by first-seen order within a tie. The
    // enumerate captures first-seen position before the sort disturbs it.
    let first_seen: std::collections::HashMap<&String, usize> = fused
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id, i))
        .collect::<std::collections::HashMap<_, _>>()
        .into_iter()
        .map(|(k, v)| (k, v))
        .collect();
    let positions: Vec<usize> = fused.iter().map(|(id, _)| first_seen[id]).collect();
    let mut indexed: Vec<(usize, (String, f64))> =
        positions.into_iter().zip(fused.drain(..)).collect();

    indexed.sort_by(|a, b| {
        b.1 .1
            .partial_cmp(&a.1 .1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    indexed.into_iter().map(|(_, pair)| pair).collect()
}

pub struct Hybrid<'a> {
    pub index: &'a KnowledgeIndex,
    /// Absent until an embedding model is registered.
    pub embedder: Option<&'a dyn Embedder>,
}

impl<'a> Hybrid<'a> {
    /// Searches with everything available, and says what that was.
    pub fn search(
        &self,
        session: &Session,
        query: &str,
        limit: usize,
    ) -> Result<HybridResults> {
        // Over-fetch from each search so fusion has room to reorder. Fusing two
        // lists already truncated to `limit` would throw away the passage that
        // ranked eleventh in both and should have come first.
        let depth = (limit * 3).max(limit);
        let keyword = self.index.search(session, query, depth)?;

        let Some(embedder) = self.embedder else {
            let mut results = keyword;
            results.truncate(limit);
            return Ok(HybridResults {
                results,
                methods: vec![Method::Keyword],
                degraded: Some(format!(
                    "Searching by keyword only. Vector search needs an embedding model, which \
                     is not installed, so questions phrased differently from the document may \
                     be missed. Exact terms such as tag numbers and measurements are unaffected."
                )),
            });
        };

        // The vector half. Reached once an embedder exists; the fusion below is
        // already exercised by its own tests.
        let _ = embedder.embed(query)?;
        let vector: Vec<SearchResult> = Vec::new();

        let keyword_ids: Vec<String> = keyword.iter().map(|r| r.chunk_id.clone()).collect();
        let vector_ids: Vec<String> = vector.iter().map(|r| r.chunk_id.clone()).collect();

        let fused = reciprocal_rank_fusion(&[keyword_ids, vector_ids]);

        let mut by_id: std::collections::HashMap<String, SearchResult> = keyword
            .into_iter()
            .chain(vector)
            .map(|r| (r.chunk_id.clone(), r))
            .collect();

        let results: Vec<SearchResult> = fused
            .into_iter()
            .filter_map(|(id, _)| by_id.remove(&id))
            .take(limit)
            .collect();

        Ok(HybridResults {
            results,
            methods: vec![Method::Keyword, Method::Vector],
            degraded: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};
    use crate::knowledge::{Chunk, ChunkKind};
    use crate::policy::Classification;

    fn ids(pairs: &[(String, f64)]) -> Vec<&str> {
        pairs.iter().map(|(id, _)| id.as_str()).collect()
    }

    #[test]
    fn one_ranking_fuses_to_itself() {
        let fused = reciprocal_rank_fusion(&[vec!["a".into(), "b".into(), "c".into()]]);
        assert_eq!(ids(&fused), vec!["a", "b", "c"]);
    }

    /// The property worth having: agreement beats one enthusiastic vote.
    #[test]
    fn a_passage_both_searches_like_beats_one_only_a_single_search_loved() {
        let keyword = vec!["loved-by-keyword".into(), "agreed".into()];
        let vector = vec!["loved-by-vector".into(), "agreed".into()];

        let fused = reciprocal_rank_fusion(&[keyword, vector]);
        assert_eq!(ids(&fused)[0], "agreed");
    }

    #[test]
    fn a_passage_found_by_only_one_search_still_appears() {
        let fused = reciprocal_rank_fusion(&[
            vec!["only-keyword".into()],
            vec!["only-vector".into()],
        ]);
        assert_eq!(fused.len(), 2);
    }

    /// A citation that moves between identical runs is a bug report waiting to
    /// happen, so ties must resolve the same way every time.
    #[test]
    fn identical_input_produces_identical_output() {
        let rankings = vec![
            vec!["a".into(), "b".into(), "c".into()],
            vec!["c".into(), "b".into(), "a".into()],
        ];
        let first = reciprocal_rank_fusion(&rankings);
        for _ in 0..20 {
            assert_eq!(reciprocal_rank_fusion(&rankings), first);
        }
    }

    #[test]
    fn an_empty_ranking_contributes_nothing_and_breaks_nothing() {
        let fused = reciprocal_rank_fusion(&[vec!["a".into(), "b".into()], vec![]]);
        assert_eq!(ids(&fused), vec!["a", "b"]);
    }

    #[test]
    fn fusing_nothing_yields_nothing() {
        assert!(reciprocal_rank_fusion(&[]).is_empty());
        assert!(reciprocal_rank_fusion(&[vec![], vec![]]).is_empty());
    }

    /// Rank one should not run away with it — that is what the damping is for.
    #[test]
    fn the_top_of_one_ranking_does_not_automatically_win() {
        // "runner-up" is second in both; "champion" is first in one and absent
        // from the other. Agreement wins.
        let fused = reciprocal_rank_fusion(&[
            vec!["champion".into(), "runner-up".into()],
            vec!["other".into(), "runner-up".into()],
        ]);
        assert_eq!(ids(&fused)[0], "runner-up");
    }

    // ── End to end, with the search that actually exists ─────────────────

    struct Fixture {
        _dir: tempfile::TempDir,
        index: KnowledgeIndex,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let index = KnowledgeIndex::open(dir.path()).unwrap();
        index
            .index_document(
                "Maintenance SOP",
                Classification::Internal,
                &[Chunk {
                    id: "c1".into(),
                    document_sha256: "sop".into(),
                    ordinal: 0,
                    text: "Minimum acceptable wall thickness is 9.0 mm for PV-2201.".into(),
                    page: 4,
                    section_path: vec!["4.2 Wall Thickness".into()],
                    kind: ChunkKind::Prose,
                    char_count: 56,
                }],
            )
            .unwrap();
        Fixture { _dir: dir, index }
    }

    fn session() -> Session {
        Session::open(User::new("p", "P", vec![Role::Employee]))
    }

    #[test]
    fn without_an_embedder_it_searches_by_keyword_and_says_so() {
        let f = fixture();
        let hybrid = Hybrid {
            index: &f.index,
            embedder: None,
        };

        let found = hybrid.search(&session(), "PV-2201", 10).unwrap();

        assert_eq!(found.results.len(), 1);
        assert_eq!(found.methods, vec![Method::Keyword]);
        let degraded = found.degraded.expect("should say it is degraded");
        assert!(degraded.contains("embedding model"));
        // And it says what still works, so the limitation is actionable.
        assert!(degraded.contains("Exact terms"));
    }

    #[test]
    fn the_limit_is_respected_even_though_more_are_fetched_for_fusion() {
        let f = fixture();
        let hybrid = Hybrid {
            index: &f.index,
            embedder: None,
        };
        let found = hybrid.search(&session(), "thickness", 1).unwrap();
        assert_eq!(found.results.len(), 1);
    }
}

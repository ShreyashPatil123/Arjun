//! How much of an OCR'd document is allowed into the prompt.
//!
//! ## Why this exists here as well as in the runtime
//!
//! The rule is written twice, and that is a cost worth naming rather than
//! hiding. `agent-runtime/src/ocr-budget.ts` holds the same thresholds because
//! the runtime needs them to explain its own ledger. This copy is the one that
//! **decides**, because the prompt is composed on this side — in
//! `compose_prompt_with_attachments`, before the runtime is handed anything at
//! all. Putting the decision in the runtime would mean shipping the whole
//! 40-page document across the wire in order to be told it did not fit.
//!
//! The two are kept in step by the constants below being asserted against the
//! same boundary in both test suites. If they ever disagree, the Rust side is
//! what actually happened and the TypeScript side is what the screen claimed —
//! which is the sort of divergence this repository has been bitten by before.
//!
//! ## The rule
//!
//! With `budget = window - committed` (what is genuinely free this turn) and
//! `document` the OCR'd text's size:
//!
//! - `document <= FULL_INCLUSION_SHARE * budget` → the whole text goes in.
//! - otherwise, if the resulting allowance is at least
//!   [`MINIMUM_USEFUL_ALLOWANCE`] → as much of the document as that allowance
//!   holds is injected, and the turn says plainly how much that was.
//! - otherwise → nothing of it goes in, and the turn says so plainly.
//!
//! ## Why half, and not all of the free budget
//!
//! A document permitted to fill the whole of what is free leaves nothing for
//! the answer or for a second turn. The person asks a follow-up, the run
//! compacts immediately, and the first thing it loses is the document it was
//! just given. Half leaves room for the conversation the document was attached
//! for.
//!
//! ## Why not summarisation
//!
//! It costs a second model call on a machine that may have nothing left after
//! the first, and it replaces the document's own words with a paraphrase — in a
//! system whose evidence is meant to be quotable back to the page it came from.
//! Taking a prefix keeps every word it does include verbatim.
//!
//! ## What this does not yet do
//!
//! The part that is included is the *beginning* of the document, not the part
//! most relevant to the question. Ranking passages against the question — the
//! knowledge index already has the machinery, in `KnowledgeIndex::search` — is
//! the obvious improvement and is deliberately not claimed here until it is
//! built: `explanation` is shown to the person verbatim, and a sentence saying
//! the relevant passages were chosen, over a prefix that was not, would be a
//! lie about the evidence an answer rests on. [`take_tokens`] is the seam.

use serde::{Deserialize, Serialize};

/// The share of free budget one document's text may occupy.
pub const FULL_INCLUSION_SHARE: f64 = 0.5;

/// The floor under a document's allowance, in tokens.
///
/// Below this, injecting "the document" means injecting three sentences of it.
/// A model reading that answers confidently from a fragment nobody has marked
/// as a fragment, which is worse than being told the document did not fit.
pub const MINIMUM_USEFUL_ALLOWANCE: u32 = 512;

/// Characters per token, for the pre-call estimate.
///
/// Four is the usual figure for English prose and is wrong for dense tables,
/// which is exactly what OCR output often is. It is labelled an estimate
/// wherever it travels, and the reconciliation on the next model turn replaces
/// it with what was actually charged. See
/// [`crate::ai_engine::token_reconciliation`].
const CHARS_PER_TOKEN: usize = 4;

/// A character-count estimate of a document's token cost.
///
/// Deliberately not called `count`: nothing here counted anything.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    ((chars / CHARS_PER_TOKEN) as u32).max(1)
}

/// What will be done with one document's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InjectionStrategy {
    /// The whole text goes into the prompt.
    Full,
    /// Only as much of the document as fits goes in; the rest is left out.
    Chunked,
    /// None of it fits. The document is named, nothing more.
    ReferenceOnly,
}

impl InjectionStrategy {
    pub fn label(self) -> &'static str {
        match self {
            InjectionStrategy::Full => "in full",
            InjectionStrategy::Chunked => "in part",
            InjectionStrategy::ReferenceOnly => "by reference only",
        }
    }
}

/// The decision, with the numbers it was made from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InjectionPlan {
    pub strategy: InjectionStrategy,
    /// Tokens this document may spend.
    pub allowance: u32,
    /// The document's full estimated size.
    pub document_tokens: u32,
    /// What was free before this document was considered.
    pub budget: u32,
    /// Why, in the words the person reads. Shown verbatim.
    pub explanation: String,
}

/// Decides how one document enters the prompt.
///
/// A `window` of zero means nobody told this process the model's context size.
/// That yields `Full` with an explanation saying so, rather than a refusal: an
/// unknown window is not evidence that the document does not fit, and refusing
/// on it would break every run against an unconfigured model. If it really does
/// not fit, the runtime's own tokenizer check refuses with an exact figure —
/// which is a better error than a guess made here.
pub fn plan(document_tokens: u32, committed: u32, window: u32) -> InjectionPlan {
    if window == 0 {
        return InjectionPlan {
            strategy: InjectionStrategy::Full,
            allowance: document_tokens,
            document_tokens,
            budget: 0,
            explanation:
                "The model's context window is not known here, so the whole document was included."
                    .to_string(),
        };
    }

    let budget = window.saturating_sub(committed);
    let allowance = (f64::from(budget) * FULL_INCLUSION_SHARE) as u32;

    if document_tokens <= allowance {
        return InjectionPlan {
            strategy: InjectionStrategy::Full,
            allowance,
            document_tokens,
            budget,
            explanation: format!(
                "The whole document was included — about {document_tokens} tokens, within the {allowance} available to it."
            ),
        };
    }

    if allowance < MINIMUM_USEFUL_ALLOWANCE {
        return InjectionPlan {
            strategy: InjectionStrategy::ReferenceOnly,
            allowance: 0,
            document_tokens,
            budget,
            explanation: format!(
                "There was no room for this document — about {document_tokens} tokens against {allowance} available. \
                 It was read, but none of its text is in this turn."
            ),
        };
    }

    let share = ((f64::from(allowance) / f64::from(document_tokens)) * 100.0).round() as u32;
    InjectionPlan {
        strategy: InjectionStrategy::Chunked,
        allowance,
        document_tokens,
        budget,
        explanation: format!(
            "This document is about {document_tokens} tokens and only {allowance} were available, so roughly \
             the first {share}% of it was included. The rest is not in this turn."
        ),
    }
}

/// Truncates text to an allowance, on a character boundary.
///
/// Used only for the `Chunked` strategy when no retrieval index is available
/// for the document. The caller must say in the prompt that this happened —
/// text that stops mid-document with nothing marking it is how a model comes to
/// believe it has read a whole page.
pub fn take_tokens(text: &str, allowance: u32) -> String {
    let budget_chars = allowance as usize * CHARS_PER_TOKEN;
    if text.chars().count() <= budget_chars {
        return text.to_string();
    }
    text.chars().take(budget_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_document_goes_in_whole() {
        let p = plan(1_000, 2_000, 32_000);
        assert_eq!(p.strategy, InjectionStrategy::Full);
    }

    #[test]
    fn a_large_document_is_chunked() {
        let p = plan(40_000, 4_000, 32_000);
        assert_eq!(p.strategy, InjectionStrategy::Chunked);
        assert_eq!(p.allowance, 14_000, "(32000 - 4000) * 0.5");
    }

    /// The boundary that must match `ocr-budget.test.ts` exactly. If these two
    /// ever disagree, the screen is describing a decision the prompt did not
    /// make.
    #[test]
    fn the_boundary_is_half_the_free_budget() {
        let free = 32_000 - 2_000;
        let exact = (f64::from(free) * FULL_INCLUSION_SHARE) as u32;
        assert_eq!(plan(exact, 2_000, 32_000).strategy, InjectionStrategy::Full);
        assert_eq!(
            plan(exact + 1, 2_000, 32_000).strategy,
            InjectionStrategy::Chunked
        );
    }

    #[test]
    fn a_document_never_takes_more_than_half_of_what_is_free() {
        for document in [1_000u32, 10_000, 100_000, 1_000_000] {
            let p = plan(document, 8_000, 32_000);
            assert!(
                p.allowance <= (32_000 - 8_000) / 2,
                "{document} was allowed {} of a 24000 free budget",
                p.allowance
            );
        }
    }

    #[test]
    fn a_nearly_full_turn_admits_nothing_rather_than_a_sliver() {
        let p = plan(50_000, 31_400, 32_000);
        assert_eq!(p.strategy, InjectionStrategy::ReferenceOnly);
        assert_eq!(p.allowance, 0);
        assert!(p.explanation.contains("no room"), "{}", p.explanation);
    }

    #[test]
    fn an_unknown_window_includes_everything_and_says_so() {
        let p = plan(90_000, 0, 0);
        assert_eq!(p.strategy, InjectionStrategy::Full);
        assert!(p.explanation.contains("not known"), "{}", p.explanation);
    }

    #[test]
    fn an_over_committed_turn_has_no_budget_rather_than_a_negative_one() {
        // `saturating_sub`, not a wrapping subtraction — which on u32 would
        // produce a budget of four billion and admit the document whole.
        let p = plan(500, 40_000, 32_000);
        assert_eq!(p.budget, 0);
        assert_eq!(p.strategy, InjectionStrategy::ReferenceOnly);
    }

    #[test]
    fn an_empty_document_estimates_to_zero_not_one() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1, "a non-empty document is never zero");
    }

    #[test]
    fn truncation_respects_character_boundaries() {
        // Multi-byte input must not be cut mid-character; slicing by bytes
        // would panic on the first non-ASCII page the OCR model produces.
        let text = "नमस्ते दुनिया".repeat(50);
        let taken = take_tokens(&text, 4);
        assert_eq!(taken.chars().count(), 16);
    }

    #[test]
    fn truncation_leaves_short_text_alone() {
        assert_eq!(take_tokens("short", 100), "short");
    }
}

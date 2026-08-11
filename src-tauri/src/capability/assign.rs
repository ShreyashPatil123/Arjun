//! Which capability slot an installed adapter fills.
//!
//! An adapter arrives from HuggingFace as a repository name and a bag of tags.
//! The capability layer, meanwhile, can only bind what it finds under a known
//! key — `coding`, `reasoning`, `tool-calling`, `mathematics`, `research`.
//! Something has to bridge the two, and this is it.
//!
//! ## Guessing, honestly
//!
//! The evidence is the same evidence [`crate::commands::adapter_details`]
//! already presents to the user, and it is read the same way: what the author
//! *declared* through tags outranks what the repository *name* merely hints at.
//! Those functions are reused rather than reimplemented so the panel explaining
//! an adapter and the slot it lands in can never disagree.
//!
//! When nothing matches, this returns [`None`] rather than picking a plausible
//! slot. That is deliberate. A coding adapter silently filed under `research`
//! would never activate, and the user would have no way to see why — the exact
//! failure the details module argues against, where a confident wrong answer is
//! worse than an uncertain right one. An unassigned adapter installs, says so,
//! and waits to be told.

use crate::commands::adapter_details::{stated_skills, suggested_skills};
use crate::model_providers::huggingface::adapter_provider::AdapterCapability;
use crate::model_providers::huggingface::card::ModelCategory;

/// Where an adapter's capability assignment came from.
///
/// Mirrors [`crate::commands::adapter_details::Confidence`], with the extra case
/// that only exists once an assignment can be corrected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentConfidence {
    /// The author's own tags said so.
    Stated,
    /// Read out of the repository name — a hint, not a statement.
    Suggested,
    /// The user chose it.
    Manual,
}

impl AssignmentConfidence {
    /// Manifest form. Kept lowercase and stable — it is persisted.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stated => "stated",
            Self::Suggested => "suggested",
            Self::Manual => "manual",
        }
    }
}

/// A capability slot, and how sure we are that it is the right one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAssignment {
    /// Manifest key: `coding`, `reasoning`, `tool-calling`, `mathematics`, or
    /// `research`.
    pub capability: String,
    pub confidence: AssignmentConfidence,
}

/// The capability a model category implies, when it implies one at all.
///
/// Matched exhaustively rather than with a wildcard, so adding a category to
/// [`ModelCategory`] fails to compile here instead of silently mapping to
/// nothing.
fn capability_for(category: ModelCategory) -> Option<&'static str> {
    match category {
        ModelCategory::Coding => Some(AdapterCapability::Coding.key()),
        ModelCategory::Reasoning => Some(AdapterCapability::Reasoning.key()),
        ModelCategory::Agentic => Some(AdapterCapability::ToolCalling.key()),
        ModelCategory::Math => Some(AdapterCapability::Mathematics.key()),

        // These describe a model's shape, modality, or lack of specialisation
        // rather than a task the capability layer can route to. `Research` is
        // absent from `ModelCategory` entirely and is handled separately below.
        ModelCategory::Vision
        | ModelCategory::Multilingual
        | ModelCategory::LongContext
        | ModelCategory::MixtureOfExperts
        | ModelCategory::SmallAndFast
        | ModelCategory::General
        | ModelCategory::LoraAdapter => None,
    }
}

/// Whole-word search.
///
/// Substring matching would let `research` be found inside unrelated words and,
/// worse, let short keywords like `api` match inside `rapid`. Splitting on
/// non-alphanumerics keeps `research-lora` matching while `researcherly` does
/// not.
fn mentions(haystack: &str, needles: &[&str]) -> bool {
    let lower = haystack.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    needles.iter().any(|n| words.contains(n))
}

/// `research` has no [`ModelCategory`], so it is matched from its own keywords.
fn looks_like_research(text: &str) -> bool {
    mentions(text, &AdapterCapability::Research.search_keywords())
}

/// The first capability implied by a list of categories.
///
/// `categorize` returns in a fixed order (coding, reasoning, math, agentic), so
/// an adapter tagged both `code` and `math` lands in the same slot every time
/// rather than depending on iteration order.
fn first_capability(categories: &[ModelCategory]) -> Option<String> {
    categories
        .iter()
        .find_map(|c| capability_for(*c))
        .map(String::from)
}

/// Infers which capability an adapter should fill.
///
/// `repo_id` is the full `owner/name`; only the name half is treated as a hint,
/// since an owner like `math-lm` says nothing about an individual adapter.
///
/// Returns [`None`] when neither the tags nor the name imply a capability the
/// runtime can route to.
pub fn infer(repo_id: &str, tags: &[String]) -> Option<CapabilityAssignment> {
    let short_name = repo_id.split('/').next_back().unwrap_or(repo_id);

    // 1. What the author declared. Tags are written by whoever trained the
    //    adapter, so a match here is a statement rather than a guess.
    if let Some(capability) = first_capability(&stated_skills(tags)) {
        return Some(CapabilityAssignment {
            capability,
            confidence: AssignmentConfidence::Stated,
        });
    }
    if looks_like_research(&tags.join(" ")) {
        return Some(CapabilityAssignment {
            capability: AdapterCapability::Research.key().to_string(),
            confidence: AssignmentConfidence::Stated,
        });
    }

    // 2. What the name suggests. A hint, and recorded as one.
    if let Some(capability) = first_capability(&suggested_skills(short_name)) {
        return Some(CapabilityAssignment {
            capability,
            confidence: AssignmentConfidence::Suggested,
        });
    }
    if looks_like_research(short_name) {
        return Some(CapabilityAssignment {
            capability: AdapterCapability::Research.key().to_string(),
            confidence: AssignmentConfidence::Suggested,
        });
    }

    // 3. Nothing legible. Say so instead of inventing a slot.
    None
}

/// True when `capability` is a slot the runtime can actually route to.
///
/// Guards the reassignment command against a manifest key nothing will ever
/// look up.
pub fn is_known_capability(capability: &str) -> bool {
    AdapterCapability::all()
        .iter()
        .any(|c| c.key() == capability)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn author_tags_outrank_the_repository_name() {
        // The name says maths, the author says code. The author wins, and the
        // result is recorded as stated rather than suggested.
        let got = infer("someone/gsm8k-math-lora", &tags(&["code", "peft"])).unwrap();

        assert_eq!(got.capability, "coding");
        assert_eq!(got.confidence, AssignmentConfidence::Stated);
    }

    #[test]
    fn a_name_alone_is_only_a_suggestion() {
        let got = infer("someone/llama-3-sql-coder-lora", &[]).unwrap();

        assert_eq!(got.capability, "coding");
        assert_eq!(
            got.confidence,
            AssignmentConfidence::Suggested,
            "a name is a hint and must never be reported as stated"
        );
    }

    #[test]
    fn agentic_maps_to_the_tool_calling_slot() {
        // The category and the capability key are spelled differently; this is
        // the mapping that keeps them in sync.
        let got = infer("someone/hermes-function-calling", &[]).unwrap();
        assert_eq!(got.capability, "tool-calling");
    }

    #[test]
    fn math_and_reasoning_reach_their_own_slots() {
        assert_eq!(infer("x/metamath-lora", &[]).unwrap().capability, "mathematics");
        assert_eq!(infer("x/qwq-cot-lora", &[]).unwrap().capability, "reasoning");
    }

    #[test]
    fn research_is_matched_from_its_own_keywords() {
        // `research` has no ModelCategory, so it can only come from the
        // AdapterCapability keyword list.
        let stated = infer("someone/some-lora", &tags(&["research"])).unwrap();
        assert_eq!(stated.capability, "research");
        assert_eq!(stated.confidence, AssignmentConfidence::Stated);

        let suggested = infer("someone/academic-paper-lora", &[]).unwrap();
        assert_eq!(suggested.capability, "research");
        assert_eq!(suggested.confidence, AssignmentConfidence::Suggested);
    }

    #[test]
    fn categories_with_no_routable_capability_yield_nothing() {
        // Vision, multilingual, long-context and MoE describe what a model is,
        // not a task the capability layer can route to.
        for repo in [
            "someone/llava-vision-lora",
            "someone/aya-multilingual-lora",
            "someone/mixtral-moe-lora",
        ] {
            assert!(
                infer(repo, &[]).is_none(),
                "{repo} should not be assigned a capability"
            );
        }
    }

    #[test]
    fn an_illegible_adapter_is_left_unassigned() {
        assert!(infer("someone/my-finetune-v2", &[]).is_none());
        assert!(infer("someone/experiment-3", &tags(&["peft", "lora"])).is_none());
    }

    #[test]
    fn the_owner_half_of_the_id_is_not_treated_as_a_hint() {
        // An owner named "coder" says nothing about this particular adapter.
        assert!(infer("coder/experiment-3", &[]).is_none());
    }

    #[test]
    fn keyword_matching_is_whole_word() {
        // "research" inside a longer word is not a research adapter.
        assert!(!looks_like_research("researcherly"));
        assert!(looks_like_research("deep-research-lora"));
    }

    #[test]
    fn every_inferred_capability_is_one_the_runtime_knows() {
        for repo in [
            "x/sql-coder-lora",
            "x/qwq-cot-lora",
            "x/metamath-lora",
            "x/hermes-function-calling",
            "x/academic-paper-lora",
        ] {
            let got = infer(repo, &[]).expect(repo);
            assert!(
                is_known_capability(&got.capability),
                "{repo} produced unroutable capability '{}'",
                got.capability
            );
        }
    }

    #[test]
    fn confidence_strings_are_stable() {
        // These are persisted in manifest.json; changing them silently would
        // orphan every existing record.
        assert_eq!(AssignmentConfidence::Stated.as_str(), "stated");
        assert_eq!(AssignmentConfidence::Suggested.as_str(), "suggested");
        assert_eq!(AssignmentConfidence::Manual.as_str(), "manual");
    }
}

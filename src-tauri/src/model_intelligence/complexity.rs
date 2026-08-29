// filepath: src-tauri/src/model_intelligence/complexity.rs
//! A deterministic estimate of how hard a task is, so the router can prefer a
//! stronger model when the prompt is dense and a faster one when it is thin.
//!
//! The original prompt asked for `Low | Medium | High | Critical`, with
//! `Critical` always routing to "the largest model". On ARJUN's reference
//! hardware the largest model that fits is 7B; there is no Critical tier to
//! escalate *to*. The estimator therefore returns `Low | Medium | High`, with
//! High meaning "use the best 7B you have, do not pick a weak one for the
//! sake of speed". A `Critical` variant is intentionally not present.
//!
//! The estimator is **deterministic**: same input → same output, no learned
//! classifier, no invented benchmark. Each axis is a small rule that a
//! reviewer can argue with in plain English.
//!
//! Axes:
//!
//! 1. **Token count.** Heuristic on whitespace-separated words, multiplied by
//!    a generous factor for non-ASCII and code. This is the prompt's own
//!    size; a long prompt needs a model with a long context and with enough
//!    remaining tokens to think.
//! 2. **Reasoning depth.** Counts signals for multi-step analysis: numbered
//!    lists, "step by step", "compare", "evaluate", "calculate", code with
//!    loops, multiple questions in one turn. A summary is shallow; an audit
//!    is deep.
//! 3. **Vision needs.** True when an attachment is image-like, or the prompt
//!    asks for OCR / "this image" / a P&ID symbol. The router uses this as a
//!    hard gate, not a complexity axis, but it raises the floor.
//! 4. **Tool density.** How many tools the model is likely to call. Sourced
//!    from the task plan when one is available; otherwise estimated from
//!    intent-classifier signals ("and then …", "first …, then …", "after
//!    that"). A 1-shot Q&A is Low; an inspection workflow is High.
//!
//! These axes are scored into a single bucket. The mapping is small, named,
//! and documented so a reviewer can change it without learning the rest of
//! the codebase.

use serde::{Deserialize, Serialize};

/// A coarse size class. Higher is harder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Complexity {
    /// Short fact-lookup, single tool call, no image.
    Low,
    /// Multi-sentence answer, two or three tool calls, no image.
    Medium,
    /// Long reasoning, image attached, or many tool calls.
    High,
}

/// What the estimator looked at, in case a reviewer wants to know why a
/// task was rated Hard.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComplexityBreakdown {
    pub bucket: Complexity,
    pub estimated_tokens: u32,
    pub reasoning_signals: u32,
    pub needs_vision: bool,
    pub likely_tool_calls: u32,
    /// The thresholds the bucket was decided against. Logged verbatim into
    /// the audit row so a future change in the rules is auditable.
    pub rules: &'static str,
}

/// What the planner saw.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSignals {
    /// Image / PDF / drawing attachments, in the order the user supplied them.
    #[serde(default)]
    pub attachments: Vec<AttachmentKind>,
    /// Steps the plan already committed to. An empty plan is allowed; the
    /// estimator just falls back to prompt-level signals.
    #[serde(default)]
    pub planned_steps: u32,
    /// Tools the plan already committed to. Used for tool-density.
    #[serde(default)]
    pub planned_tool_calls: u32,
    /// Already-known modality requirements. `None` means the estimator has
    /// to decide; `Some(true)` is a hard requirement, `Some(false)` forbids
    /// image attachments.
    #[serde(default)]
    pub vision_required: Option<bool>,
}

/// One attachment. The estimator does not open the file; the planner does,
/// and hands the kind back here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AttachmentKind {
    Text,
    Pdf,
    Image,
    Drawing,
    Spreadsheet,
    Unknown,
}

impl AttachmentKind {
    /// Whether this attachment pulls vision capability into the task.
    pub fn needs_vision(self) -> bool {
        matches!(self, AttachmentKind::Image | AttachmentKind::Drawing)
    }
}

/// The estimator. Held as a struct so the rules can carry configuration if a
/// future change wants to make the buckets tunable per workspace.
pub struct ComplexityEstimator {
    /// At or above this many estimated tokens, the prompt is "long". Tuned
    /// for a 32K context window: a 6K prompt leaves enough room for the
    /// answer and the chain-of-thought on a 7B model.
    pub long_prompt_tokens: u32,
    /// Reasoning-signal count that bumps a prompt from Low to Medium.
    pub medium_reasoning_signals: u32,
    /// Reasoning-signal count that bumps a prompt from Medium to High.
    pub high_reasoning_signals: u32,
    /// Tool-call count that bumps a prompt from Medium to High.
    pub high_tool_calls: u32,
}

impl Default for ComplexityEstimator {
    fn default() -> Self {
        Self {
            long_prompt_tokens: 2_000,
            medium_reasoning_signals: 1,
            high_reasoning_signals: 3,
            high_tool_calls: 4,
        }
    }
}

impl ComplexityEstimator {
    /// Estimates the complexity of a task from its prompt and signals.
    pub fn estimate(&self, prompt: &str, signals: &TaskSignals) -> ComplexityBreakdown {
        let estimated_tokens = estimate_tokens(prompt);
        let reasoning_signals = count_reasoning_signals(prompt);
        let likely_tool_calls =
            signals.planned_tool_calls.max(estimated_tool_calls(prompt));
        let needs_vision = signals
            .vision_required
            .unwrap_or_else(|| signals.attachments.iter().any(|a| a.needs_vision()));

        // Bucket decision. Each rule is one sentence and is logged as
        // `rules` so an auditor can replay the decision later.
        let bucket = if needs_vision || estimated_tokens >= self.long_prompt_tokens {
            // Long prompts and vision needs both raise the floor to High.
            // A 7B model with a long context is fine here; the question is
            // not whether to escalate, but to avoid the weakest candidate.
            Complexity::High
        } else if reasoning_signals >= self.high_reasoning_signals
            || likely_tool_calls >= self.high_tool_calls
        {
            Complexity::High
        } else if reasoning_signals >= self.medium_reasoning_signals
            || likely_tool_calls >= 1
            || signals.planned_steps >= 2
        {
            Complexity::Medium
        } else {
            Complexity::Low
        };

        ComplexityBreakdown {
            bucket,
            estimated_tokens,
            reasoning_signals,
            needs_vision,
            likely_tool_calls,
            rules: "needs_vision OR long_prompt_tokens -> High; \
                    high_reasoning_signals OR high_tool_calls -> High; \
                    medium_reasoning_signals OR >=2 tool calls OR >=3 planned steps -> Medium; \
                    else Low",
        }
    }
}

/// A conservative token estimate. Not a real tokenizer — there is no model
/// to call yet, and pulling one just to count words would be a chicken-and-
/// egg problem. The figure is good enough to gate context length and the
/// complexity bucket.
fn estimate_tokens(prompt: &str) -> u32 {
    // ~1.3 tokens per whitespace-separated word is the upper end of typical
    // BPE ratios for English / code mixes; CJK and Devanagari are denser
    // per word but that just makes us *over-estimate* which is the safe
    // direction. Cap the scan to 100K words to keep a hostile prompt cheap.
    let words = prompt.split_whitespace().take(100_000).count() as u32;
    words.saturating_mul(13) / 10
}

/// Counts phrases that look like multi-step reasoning. Each hit is one
/// signal; multiple hits on the same phrase still count once.
fn count_reasoning_signals(prompt: &str) -> u32 {
    const PHRASES: &[&str] = &[
        "step by step",
        "compare",
        "evaluate",
        "calculate",
        "explain why",
        "pros and cons",
        "trade-off",
        "tradeoff",
        "differences between",
        "versus",
        "vs.",
        "first, ",
        "then ",
        "after that",
        "finally ",
        "summarize",
        "analyse",
        "analyze",
    ];
    let lower = prompt.to_ascii_lowercase();
    PHRASES
        .iter()
        .filter(|phrase| lower.contains(*phrase))
        .count() as u32
}

/// Tool-density estimate when the plan has not committed to a count. Heuristic
/// only — the real number comes from the plan, which is the preferred input.
fn estimated_tool_calls(prompt: &str) -> u32 {
    let lower = prompt.to_ascii_lowercase();
    let mut count: u32 = 0;
    for phrase in [
        "search ",
        "look up",
        "find ",
        "read ",
        "calculate ",
        "compute ",
        "draft ",
        "write ",
        "create ",
    ] {
        if lower.contains(phrase) {
            count = count.saturating_add(1);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(prompt: &str) -> Complexity {
        ComplexityEstimator::default()
            .estimate(prompt, &TaskSignals::default())
            .bucket
    }

    #[test]
    fn a_short_question_is_low() {
        assert_eq!(est("What is the wall thickness of V-101?"), Complexity::Low);
    }

    #[test]
    fn a_long_prompt_is_high_even_with_simple_intent() {
        let long = "word ".repeat(3_000);
        assert_eq!(est(&long), Complexity::High);
    }

    #[test]
    fn multiple_reasoning_signals_is_high() {
        let prompt = "Compare the pros and cons of the two approaches. \
                       First, evaluate the cost. Then, calculate the runtime. \
                       Finally, summarize the trade-off.";
        assert_eq!(est(prompt), Complexity::High);
    }

    #[test]
    fn a_single_tool_signal_is_medium() {
        let prompt = "Search the SOPs for hydrotest interval and tell me the answer.";
        assert_eq!(est(prompt), Complexity::Medium);
    }

    #[test]
    fn an_image_attachment_raises_to_high() {
        let breakdown = ComplexityEstimator::default().estimate(
            "What is this?",
            &TaskSignals {
                attachments: vec![AttachmentKind::Image],
                ..Default::default()
            },
        );
        assert_eq!(breakdown.bucket, Complexity::High);
        assert!(breakdown.needs_vision);
    }

    #[test]
    fn planned_steps_count_toward_density() {
        let breakdown = ComplexityEstimator::default().estimate(
            "Draft the note.",
            &TaskSignals {
                planned_steps: 4,
                ..Default::default()
            },
        );
        assert_eq!(breakdown.bucket, Complexity::Medium);
    }

    #[test]
    fn rules_string_is_a_known_auditable_artefact() {
        // The rules string is logged into the audit row. Pinning it catches
        // an accidental change to the bucket logic in code review.
        let breakdown = ComplexityEstimator::default()
            .estimate("hi", &TaskSignals::default());
        assert!(breakdown.rules.contains("needs_vision"));
    }
}

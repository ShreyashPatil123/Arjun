//! How many tokens one generation may produce.
//!
//! A single number served this product for as long as there was one kind of
//! turn: `DEFAULT_MAX_TOKENS`, 4 096, sent with every request in
//! `commands::agent`. That is generous for a routing decision and short for a
//! plan that has to decompose a task and dispatch to sub-agents, and being
//! wrong in both directions at once has a cost on each side.
//!
//! Too small is the visible one. A generation that runs out mid-sentence ends
//! as [`crate::agent_runtime::outcome::RunOutcome::LengthLimited`] — a fragment
//! presented as an answer, which this codebase already treats as its own
//! ending rather than a completion.
//!
//! Too large is the quiet one, and on a small card it is the more expensive.
//! Every token of headroom is a token the compactor must reserve and cannot
//! spend on the conversation: `settingsForWindow` in the runtime's
//! `compaction.ts` sets the reserve to at least the output cap, so a 16 384
//! ceiling on a 65 536 window hands a quarter of the context to a reserve a
//! routing answer will never touch.
//!
//! ## What this is not
//!
//! It is not a task budget. The ceiling here bounds **one inference call**; a
//! task needing more than one goes on through
//! [`crate::ai_engine::continuation`], which checkpoints and resumes rather
//! than truncating. The two are easy to conflate, and conflating them is the
//! whole reason a long task used to come back cut in half.
//!
//! It is also not a forecast of what the model will produce — it is a ceiling
//! on what it is allowed to produce. That is why erring slightly high within a
//! band is right, and why a hard ceiling sits above every band.

use serde::{Deserialize, Serialize};

use crate::model_intelligence::complexity::Complexity;

/// The most any single generation may produce, whatever else is asked for.
///
/// A backstop, not a policy. The policy is the per-intent band below; this is
/// the number past which a request has stopped being a generation that got long
/// and become a model that has stopped converging — and the answer to that is
/// the continuation pipeline's convergence guard, not a larger buffer.
pub const HARD_CEILING_TOKENS: u32 = 16_384;

/// The floor under any band once the context window has been taken into
/// account.
///
/// A window whose half is below this cannot hold a useful answer anyway, and
/// clamping smaller would turn "this model's context is tight" into "every
/// answer is truncated" — which reads as a defect in the assistant rather than
/// a limit of the deployment.
pub const MIN_GENERATION_TOKENS: u32 = 512;

/// What a turn is being asked to do, as far as the budget is concerned.
///
/// Deliberately coarser than [`crate::model_intelligence::intent::PromptIntent`],
/// which classifies a prompt to pick a *prompt profile*. This classifies a call
/// to pick an *output ceiling*, and the two do not partition the same way: a
/// coding question and a research question want different system directives and
/// the same amount of room to answer in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GenerationKind {
    /// Picking a route, a capability, a classification. The answer is a choice
    /// and a short justification, and a model writing an essay here has
    /// misunderstood the question.
    Routing,
    /// The ordinary agent turn: some prose, possibly a tool call or several.
    /// The default, and what an unclassified turn gets.
    ToolCalling,
    /// Breaking a task into steps and dispatching them. Longer because the
    /// output is structured and enumerated — a plan truncated at step four of
    /// nine is worse than no plan, because the steps it did emit look complete.
    Decomposition,
    /// Sustained multi-step reasoning, or a long piece of writing. The largest
    /// band, and still below the hard ceiling.
    ComplexReasoning,
}

impl GenerationKind {
    /// The ceiling for this kind of call, before any window is considered.
    ///
    /// The upper end of each band rather than the lower: a ceiling is not a
    /// forecast, and being 1 000 tokens generous costs a slightly larger
    /// compaction reserve, while being 1 000 tokens mean costs an answer that
    /// stops mid-sentence.
    pub const fn ceiling(self) -> u32 {
        match self {
            GenerationKind::Routing => 2_048,
            GenerationKind::ToolCalling => 4_096,
            GenerationKind::Decomposition => 6_144,
            GenerationKind::ComplexReasoning => 8_192,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            GenerationKind::Routing => "routing",
            GenerationKind::ToolCalling => "tool-calling",
            GenerationKind::Decomposition => "decomposition",
            GenerationKind::ComplexReasoning => "complex-reasoning",
        }
    }
}

impl GenerationKind {
    /// Which band a turn falls into, from what ARJUN already knows about it.
    ///
    /// Three signals, and no fourth. All three are computed for every turn
    /// before this is asked, so nothing new is inferred and no extra model call
    /// is made to classify a model call:
    ///
    /// - `plan_steps` — how many steps the planner laid out. More than a couple
    ///   means the turn has to enumerate and dispatch, which is the output
    ///   shape that most often runs out of room mid-list.
    /// - `complexity` — [`crate::model_intelligence::complexity::Complexity`],
    ///   which already weighs prompt length, reasoning signals and
    ///   attachments.
    /// - `intent` — the router's own label for this turn.
    ///
    /// Order matters. Decomposition is checked before complexity because a
    /// multi-step plan is a claim about the *shape* of the output, while
    /// complexity is a claim about its difficulty, and a truncated plan is the
    /// worse failure: its emitted steps look like the whole list.
    pub fn classify(plan_steps: usize, complexity: Complexity, intent: &str) -> Self {
        // A plan that dispatches several steps has to name them all. Two is the
        // threshold rather than one because a single-step plan is an ordinary
        // turn wearing a plan.
        if plan_steps > 2 {
            return GenerationKind::Decomposition;
        }
        match complexity {
            Complexity::High => GenerationKind::ComplexReasoning,
            Complexity::Low if is_routing_intent(intent) => GenerationKind::Routing,
            _ => GenerationKind::ToolCalling,
        }
    }
}

/// Intents whose answer is a choice rather than a piece of work.
///
/// Matched on the router's own labels rather than on the prompt: the router has
/// already done this classification, and a second reading of the same prompt
/// here could disagree with the model that was actually picked.
fn is_routing_intent(intent: &str) -> bool {
    const ROUTING: &[&str] = &["routing", "classification", "capability", "triage"];
    let intent = intent.trim().to_ascii_lowercase();
    ROUTING.iter().any(|label| intent == *label)
}

/// A ceiling, and why it is that number.
///
/// The reason travels with the figure because this lands in the run trace, and
/// "4 096" on its own tells a person nothing about whether the answer in front
/// of them was cut short by policy or by the window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationBudget {
    pub max_tokens: u32,
    pub kind: GenerationKind,
    pub reason: String,
}

/// The ceiling for one generation, given what it is doing and what it has room
/// for.
///
/// `served_window` is what the server was **actually started with**, not what
/// the registry declares — see [`crate::serving::Endpoint::context_tokens`].
/// `None` means nobody knows, and the band applies unreduced, which is where
/// this was before the window was consulted at all.
///
/// The window matters because an output cap is charged twice: once as tokens
/// the model may emit, and once as the reserve the compactor keeps free for
/// them. A cap above half the window leaves a turn with less room for the
/// conversation than for the reply, so it is clamped there — and the clamp says
/// so, because a limit nobody can see is one they will read as a defect.
pub fn budget_for(kind: GenerationKind, served_window: Option<u32>) -> GenerationBudget {
    let band = kind.ceiling().min(HARD_CEILING_TOKENS);
    let Some(window) = served_window.filter(|window| *window > 0) else {
        return GenerationBudget {
            max_tokens: band,
            kind,
            reason: format!(
                "{} work is allowed {band} tokens for this generation",
                kind.label()
            ),
        };
    };

    let half = (window / 2).max(MIN_GENERATION_TOKENS);
    if band <= half {
        return GenerationBudget {
            max_tokens: band,
            kind,
            reason: format!(
                "{} work is allowed {band} tokens for this generation, within the \
                 {window}-token window this model is served at",
                kind.label()
            ),
        };
    }

    GenerationBudget {
        max_tokens: half,
        kind,
        reason: format!(
            "{} work would be allowed {band} tokens, lowered to {half} so the reply reserve \
             does not take more than half of the {window}-token window this model is served at",
            kind.label()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[GenerationKind] = &[
        GenerationKind::Routing,
        GenerationKind::ToolCalling,
        GenerationKind::Decomposition,
        GenerationKind::ComplexReasoning,
    ];

    #[test]
    fn every_band_sits_under_the_hard_ceiling() {
        for kind in ALL {
            assert!(
                kind.ceiling() <= HARD_CEILING_TOKENS,
                "{} exceeds the per-generation ceiling",
                kind.label()
            );
            assert!(kind.ceiling() >= MIN_GENERATION_TOKENS);
        }
    }

    /// The bands are ordered by how much room the work needs: a routing call
    /// must never be allowed as much as a decomposition.
    #[test]
    fn the_bands_are_ordered_by_the_work_they_describe() {
        assert!(GenerationKind::Routing.ceiling() < GenerationKind::ToolCalling.ceiling());
        assert!(GenerationKind::ToolCalling.ceiling() < GenerationKind::Decomposition.ceiling());
        assert!(
            GenerationKind::Decomposition.ceiling() < GenerationKind::ComplexReasoning.ceiling()
        );
    }

    /// The previous behaviour, preserved exactly for the default turn on a
    /// window with room to spare: 4 096, the figure `DEFAULT_MAX_TOKENS` held.
    #[test]
    fn an_ordinary_turn_still_gets_the_number_it_always_got() {
        let budget = budget_for(GenerationKind::ToolCalling, Some(65_536));
        assert_eq!(budget.max_tokens, 4_096);
    }

    /// A cap larger than half the window is the compactor's problem, not the
    /// model's: it reserves at least the output cap, so an unclamped ceiling on
    /// a small window starves the conversation it was meant to serve.
    #[test]
    fn a_cap_never_takes_more_than_half_the_window_it_is_served_in() {
        let budget = budget_for(GenerationKind::ComplexReasoning, Some(8_192));
        assert_eq!(budget.max_tokens, 4_096);
        assert!(
            budget.reason.contains("lowered"),
            "a clamp nobody can see is one they will read as a defect: {}",
            budget.reason
        );
    }

    /// An unknown window leaves the band alone rather than guessing at one.
    #[test]
    fn an_unknown_window_does_not_invent_a_clamp() {
        let budget = budget_for(GenerationKind::ComplexReasoning, None);
        assert_eq!(
            budget.max_tokens,
            GenerationKind::ComplexReasoning.ceiling()
        );
    }

    /// A window too small to halve usefully still yields a usable cap rather
    /// than one that truncates every answer.
    #[test]
    fn a_tiny_window_yields_the_floor_rather_than_something_unusable() {
        let budget = budget_for(GenerationKind::Routing, Some(256));
        assert_eq!(budget.max_tokens, MIN_GENERATION_TOKENS);
    }

    /// A multi-step plan is classified by its shape, not its difficulty.
    ///
    /// A truncated plan is the worse failure mode: the steps it managed to emit
    /// look like the complete list.
    #[test]
    fn a_multi_step_plan_outranks_the_complexity_estimate() {
        assert_eq!(
            GenerationKind::classify(5, Complexity::Low, "reasoning"),
            GenerationKind::Decomposition
        );
    }

    /// A single-step plan is an ordinary turn wearing a plan.
    #[test]
    fn a_short_plan_is_not_a_decomposition() {
        assert_eq!(
            GenerationKind::classify(1, Complexity::Medium, "reasoning"),
            GenerationKind::ToolCalling
        );
    }

    #[test]
    fn a_hard_task_gets_the_largest_band() {
        assert_eq!(
            GenerationKind::classify(0, Complexity::High, "reasoning"),
            GenerationKind::ComplexReasoning
        );
    }

    /// Only a simple turn that the router itself called routing gets the
    /// smallest band. A short prompt about real work does not.
    #[test]
    fn only_the_routers_own_label_earns_the_routing_band() {
        assert_eq!(
            GenerationKind::classify(0, Complexity::Low, "classification"),
            GenerationKind::Routing
        );
        assert_eq!(
            GenerationKind::classify(0, Complexity::Low, "coding"),
            GenerationKind::ToolCalling,
            "a short coding question is still work, not a routing decision"
        );
    }

    /// An unclassified turn lands on the default band, which is the figure the
    /// product shipped with.
    #[test]
    fn an_unknown_intent_falls_back_to_the_default_band() {
        let kind = GenerationKind::classify(0, Complexity::Medium, "");
        assert_eq!(kind, GenerationKind::ToolCalling);
        assert_eq!(kind.ceiling(), 4_096);
    }

    /// No band, on any window, may exceed the hard ceiling. The clamp only ever
    /// lowers.
    #[test]
    fn the_hard_ceiling_holds_on_every_window() {
        for kind in ALL {
            for window in [None, Some(1), Some(8_192), Some(65_536), Some(1_048_576)] {
                let budget = budget_for(*kind, window);
                assert!(budget.max_tokens <= HARD_CEILING_TOKENS);
                assert!(budget.max_tokens >= MIN_GENERATION_TOKENS);
                assert!(budget.max_tokens <= kind.ceiling());
            }
        }
    }
}

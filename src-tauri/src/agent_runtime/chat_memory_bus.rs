//! Model-agnostic chat memory bus.
//!
//! ## The problem this solves
//!
//! Every prior turn of context was fitted against a single model's physical
//! window, and anything that did not fit was permanently dropped from the turn.
//! A conversation that ran for fifty messages on a 32k model carried at most
//! 24k tokens of history — and when the orchestrator switched to a model with a
//! smaller window, most of that history was silently abandoned.
//!
//! ## The architecture
//!
//! The chat memory bus separates two concerns:
//!
//! 1. **Retention** — how much history is *kept*. This belongs to the
//!    conversation, not the model. Nothing is ever deleted; see
//!    [`CHAT_RETENTION_LIMIT`] for what the 1,000,000-token figure actually
//!    means and what it does not.
//! 2. **Projection** — how much history is *sent to a model* on any given
//!    turn. This is bounded by the model's physical context window and is
//!    computed freshly for every turn.
//!
//! Projection is non-destructive: it selects a subset, and the full record
//! remains available for the next turn, the next model, or the next session.
//!
//! ## What a turn *is*, here
//!
//! The unit of selection is a question together with the assistant reply that
//! answered it, not an individual message. Selecting messages independently
//! produced transcripts where an answer arrived without its question, which
//! reads to a model as the assistant having volunteered it, and is the shape
//! that makes a follow-up answer the wrong thing confidently. A unit is carried
//! whole or not at all.
//!
//! ## What crosses to the model is what was costed
//!
//! Every message is prepared once, by [`super::turn_context::prepare`] — tool
//! summary appended, this-run-only evidence markers neutralised — *before* it
//! is measured. The previous arrangement measured the raw stored text, selected
//! against that, and then transformed the winners on the way out. The transform
//! expands (`[E12]`, five characters, becomes `[cited earlier]`, fifteen), so
//! the projection could exceed the budget it had been given. Measuring the
//! prepared text is what makes [`Projection::projected_tokens`] a figure about
//! the request that is actually sent.

use serde::{Deserialize, Serialize};

use super::conversations::{Conversation, Message, MessageRole};
use super::pins::{self, PinRef};
use crate::ai_engine::ocr_budget::estimate_tokens;

/// The chat retention figure this product advertises, in tokens.
///
/// ## What it is, stated precisely, because it was previously stated by a
/// constant nobody read
///
/// This value was declared and never referenced: no code enforced it, measured
/// against it, or reported on it. A number like that is worse than no number,
/// because it reads as a guarantee somebody checked.
///
/// The policy it now names is **non-destructive retention with measured
/// reporting**:
///
/// - Nothing in the conversation store is ever deleted or truncated to satisfy
///   this figure. `<appdata>/conversations/{id}.json` holds the whole thread.
/// - Every projection measures the thread's full retained size and reports it
///   ([`RetentionStatus`]), so "1M tokens are retained" is a claim with a
///   measurement behind it rather than an assertion.
/// - A thread that grows past this figure is **reported**, not trimmed. The
///   figure is the point at which the product stops being able to say it is
///   operating inside its advertised envelope, and that is a thing to tell
///   somebody — not a licence to start deleting their conversation.
///
/// Archival, when it is built, belongs behind an explicit operator action that
/// says what is being moved and where. It is deliberately not a side effect of
/// asking a question.
pub const CHAT_RETENTION_LIMIT: u32 = 1_000_000;

/// How a projected unit is classified for the memory bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnPriority {
    /// Recent units: always included first (recency anchor).
    Recent,
    /// Units the user pinned: protected, and never dropped in silence.
    Pinned,
    /// Units selected by keyword relevance to the current question.
    Relevant,
    /// Earlier units included if budget remains.
    Background,
}

/// One message ready for projection into a model's window.
#[derive(Debug, Clone)]
pub struct ProjectedTurn {
    /// Index into the eligible-history slice this projection was built from.
    pub message_index: usize,
    /// `user` or `assistant`.
    pub role: &'static str,
    /// The text content, **as prepared** — this is what the model receives.
    pub content: String,
    /// Estimated tokens this turn occupies, measured on `content`.
    pub tokens: u32,
    /// Why this turn was selected.
    pub priority: TurnPriority,
}

/// Why a pinned turn could not be carried.
///
/// Two reasons, kept apart because they call for different answers from the
/// person reading them: the first says this pin can never fit this model, and
/// the second says it lost a race against other pins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "camelCase")]
pub enum PinOmission {
    /// Larger on its own than the whole history budget. No ordering of pins
    /// would have carried it; a wider window or a shorter turn is the answer.
    ExceedsBudget { budget_tokens: u32, cost_tokens: u32 },
    /// It would have fitted alone, but earlier-selected protected content had
    /// already spent the budget.
    BudgetSpent { remaining_tokens: u32, cost_tokens: u32 },
}

/// One pinned obligation this projection could not honour.
///
/// The existence of this type is the point. The previous code skipped an
/// oversized pinned turn with a bare `continue`, and the only trace was a `+1`
/// on an undifferentiated count of turns that did not fit — indistinguishable
/// from an ordinary old message ageing out. A person who pinned something was
/// told nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmittedPin {
    /// The pin as stored, so it can be matched against the panel's own list.
    pub pin: String,
    /// What kind of thing it names: `message`, `source`, `artifact`, `legacy`.
    pub kind: String,
    /// The message the pin protected and the turn could not carry.
    pub message_id: String,
    pub reason: PinOmission,
}

/// What the thread costs to retain, measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetentionStatus {
    /// Tokens across every eligible message in the conversation. Measured on
    /// prepared text, so it is comparable with the projection figures.
    pub retained_tokens: u32,
    /// [`CHAT_RETENTION_LIMIT`].
    pub limit_tokens: u32,
    /// True when the thread has grown past the advertised envelope. Nothing is
    /// deleted when it does; it is reported.
    pub exceeds_limit: bool,
}

/// The result of projecting the chat memory bus into a model's window.
#[derive(Debug, Clone)]
pub struct Projection {
    /// Turns to send, oldest first.
    pub turns: Vec<ProjectedTurn>,
    /// Total estimated tokens in the projection, measured on prepared content.
    ///
    /// Never greater than the `budget_tokens` the projection was given.
    pub projected_tokens: u32,
    /// How many eligible messages were retained but not projected (still safe
    /// in the conversation store).
    pub retained_not_projected: u32,
    /// Pinned obligations this budget could not honour, with the reason for
    /// each. Empty is the ordinary case.
    pub omitted_pins: Vec<OmittedPin>,
    /// What the whole thread costs to keep, and whether that is inside the
    /// advertised envelope.
    pub retention: RetentionStatus,
}

/// Whether a stored message is eligible for the memory bus.
///
/// Delegates to [`super::turn_context::is_eligible`] rather than restating the
/// rules, because two copies of "which messages count" is how the two
/// projection paths drifted apart in the first place.
fn is_eligible(message: &Message) -> bool {
    super::turn_context::is_eligible(message)
}

/// The wire role for an eligible message.
fn role_of(message: &Message) -> Option<&'static str> {
    match message.role {
        MessageRole::User => Some("user"),
        MessageRole::Assistant => Some("assistant"),
        MessageRole::System => None,
    }
}

/// Simple keyword overlap score between a question and past text.
///
/// Returns a value between 0.0 and 1.0. This is deliberately simple — a full
/// embedding-based retrieval is a later phase, and this module is explicit that
/// what it does is lexical. Even naive keyword overlap rescues turns that
/// mention the same entities as the current question, which is the single most
/// common reason a person asks a follow-up.
fn relevance_score(question_words: &[String], text: &str) -> f64 {
    if question_words.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let hits = question_words
        .iter()
        .filter(|word| lower.contains(word.as_str()))
        .count();
    hits as f64 / question_words.len() as f64
}

/// Extract meaningful keywords from a question.
///
/// ## Why the length test is on `chars` and the floor is two
///
/// It used to be `w.len() >= 4`, which is a count of **UTF-8 bytes**. For
/// English that is a rough proxy for "not a stopword". For Hindi, Japanese or
/// Chinese it is the opposite of one: a two-character CJK word is six bytes and
/// a three-character Devanagari word is nine, so a question in those scripts had
/// its real keywords admitted or rejected by how many bytes the script happens
/// to use rather than by anything about the words.
///
/// Counting characters and dropping the floor to two makes the rule mean "not a
/// one-character particle" in every script. English loses almost nothing: the
/// short English words this now admits are mostly stopwords, and a stopword
/// scores against *every* past turn equally, so it moves no turn's ranking
/// relative to another's.
fn extract_keywords(question: &str) -> Vec<String> {
    question
        .to_lowercase()
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|word| word.chars().count() >= 2)
        .collect()
}

/// A question and the reply that answered it, taken or left as one thing.
///
/// A unit may be a lone user message (the last question of a thread whose
/// answer failed, say) or a lone assistant message (a recovered thread whose
/// first eligible message is a reply). What it never is, is half of a pair that
/// was carried whole — which is the property the model reads.
struct Unit<'a> {
    /// Positions within the eligible-history slice, ascending.
    messages: Vec<(usize, &'a Message)>,
    /// Prepared text per message, parallel to `messages`.
    prepared: Vec<String>,
    /// Sum of the prepared messages' token estimates.
    tokens: u32,
    priority: TurnPriority,
    relevance: f64,
    /// Pins protecting a message in this unit: `(encoded pin, kind, message id)`.
    protected_by: Vec<(String, String, String)>,
}

impl Unit<'_> {
    /// The unit's combined text, for relevance scoring.
    fn text(&self) -> String {
        self.prepared.join("\n")
    }

    /// Where the unit sits in the thread, so the final transcript keeps the
    /// order things were said in.
    fn first_index(&self) -> usize {
        self.messages.first().map(|(index, _)| *index).unwrap_or(0)
    }
}

/// Groups eligible history into question-and-answer units.
///
/// A user message opens a unit. Assistant messages attach to the open unit. An
/// assistant message with no open unit becomes a unit of its own rather than
/// being discarded, because dropping it would lose content the store is holding.
fn units<'a>(history: &[(usize, &'a Message)], pins: &[PinRef]) -> Vec<Unit<'a>> {
    let mut out: Vec<Unit<'a>> = Vec::new();

    for (index, message) in history {
        let prepared = super::turn_context::prepare(message);
        let cost = estimate_tokens(&prepared);
        let protecting: Vec<(String, String, String)> = pins::protecting(pins, message, &prepared)
            .into_iter()
            .map(|pin| (pin.encode(), pin.kind().to_string(), message.id.clone()))
            .collect();

        let opens_unit = message.role == MessageRole::User || out.is_empty();
        if opens_unit {
            out.push(Unit {
                messages: vec![(*index, *message)],
                prepared: vec![prepared],
                tokens: cost,
                priority: TurnPriority::Background,
                relevance: 0.0,
                protected_by: protecting,
            });
        } else {
            let unit = out.last_mut().expect("a unit is open");
            unit.messages.push((*index, *message));
            unit.prepared.push(prepared);
            unit.tokens = unit.tokens.saturating_add(cost);
            unit.protected_by.extend(protecting);
        }
    }

    out
}

/// Projects the full conversation history into a model's physical window.
///
/// `budget_tokens` is the number of tokens available for history after the
/// system prompt, documents, tools, and reply reserve have been charged.
///
/// `cell_message_id` identifies the assistant cell for this turn — everything
/// from that cell onward is this turn and is excluded from history.
///
/// `pinned` is the set of pins the user has explicitly protected, as stored.
/// See [`super::pins`] for what a stored pin can mean.
///
/// `question` is the current turn's prompt, used for relevance scoring.
///
/// ## Strategy
///
/// 1. **Pinned units** are offered first, oldest first. One that does not fit
///    is *recorded* in [`Projection::omitted_pins`], never skipped in silence,
///    and never carried past the budget.
/// 2. **Recent units** are included next, newest first, up to 60% of budget.
/// 3. **Relevant units** from earlier history are scored by keyword overlap
///    with the question and included in descending relevance order.
/// 4. **Background units** fill any remaining budget, newest first.
///
/// The result is always sorted oldest-first for the model's benefit, and
/// `projected_tokens` never exceeds `budget_tokens`.
pub fn project(
    conversation: &Conversation,
    cell_message_id: &str,
    budget_tokens: u32,
    pinned: &[String],
    question: &str,
) -> Projection {
    let pins = PinRef::parse_all(pinned);
    let history = eligible_before_cell(conversation, cell_message_id);
    let retention = retention_status(conversation);
    let eligible_messages = history.len() as u32;

    let mut units = units(&history, &pins);

    if budget_tokens == 0 {
        // Nothing is sent, and every pin is reported as unhonourable rather
        // than the turn quietly carrying none of them.
        let omitted = units
            .iter()
            .flat_map(|unit| {
                let cost = unit.tokens;
                unit.protected_by
                    .iter()
                    .map(move |(pin, kind, message_id)| OmittedPin {
                        pin: pin.clone(),
                        kind: kind.clone(),
                        message_id: message_id.clone(),
                        reason: PinOmission::ExceedsBudget {
                            budget_tokens: 0,
                            cost_tokens: cost,
                        },
                    })
            })
            .collect();
        return Projection {
            turns: Vec::new(),
            projected_tokens: 0,
            retained_not_projected: eligible_messages,
            omitted_pins: omitted,
            retention,
        };
    }

    let keywords = extract_keywords(question);

    // Classify. Protection wins over everything; then recency; then relevance.
    for unit in units.iter_mut() {
        unit.relevance = relevance_score(&keywords, &unit.text());
        if !unit.protected_by.is_empty() {
            unit.priority = TurnPriority::Pinned;
        }
    }

    let recency_budget = (f64::from(budget_tokens) * 0.6) as u32;
    let mut recency_spent: u32 = 0;
    for unit in units.iter_mut().rev() {
        if unit.priority == TurnPriority::Pinned {
            continue;
        }
        if recency_spent.saturating_add(unit.tokens) <= recency_budget {
            unit.priority = TurnPriority::Recent;
            recency_spent = recency_spent.saturating_add(unit.tokens);
        }
    }

    for unit in units.iter_mut() {
        if unit.priority == TurnPriority::Background && unit.relevance > 0.3 {
            unit.priority = TurnPriority::Relevant;
        }
    }

    // Select, in priority order, within the budget.
    let mut selected: Vec<usize> = Vec::new();
    let mut omitted_pins: Vec<OmittedPin> = Vec::new();
    let mut remaining = budget_tokens;

    for priority in [
        TurnPriority::Pinned,
        TurnPriority::Recent,
        TurnPriority::Relevant,
        TurnPriority::Background,
    ] {
        let mut candidates: Vec<usize> = units
            .iter()
            .enumerate()
            .filter(|(_, unit)| unit.priority == priority)
            .map(|(index, _)| index)
            .collect();

        match priority {
            // Most relevant first.
            TurnPriority::Relevant => candidates.sort_by(|a, b| {
                units[*b]
                    .relevance
                    .partial_cmp(&units[*a].relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            // Newest first, so what is carried is a recent stretch rather than
            // a jump back to the dawn of the thread.
            TurnPriority::Background | TurnPriority::Recent => candidates.reverse(),
            // Oldest first: somebody who pins several things and cannot have
            // them all should lose the newest, which is the one still on
            // screen, rather than the instruction they set at the start.
            TurnPriority::Pinned => {}
        }

        for index in candidates {
            let cost = units[index].tokens;
            if cost <= remaining {
                selected.push(index);
                remaining = remaining.saturating_sub(cost);
                continue;
            }

            // It does not fit. For an ordinary unit that is the end of it —
            // `retained_not_projected` counts it. For a protected one, the
            // obligation and the reason are recorded instead of vanishing.
            if priority == TurnPriority::Pinned {
                let reason = if cost > budget_tokens {
                    PinOmission::ExceedsBudget {
                        budget_tokens,
                        cost_tokens: cost,
                    }
                } else {
                    PinOmission::BudgetSpent {
                        remaining_tokens: remaining,
                        cost_tokens: cost,
                    }
                };
                for (pin, kind, message_id) in &units[index].protected_by {
                    omitted_pins.push(OmittedPin {
                        pin: pin.clone(),
                        kind: kind.clone(),
                        message_id: message_id.clone(),
                        reason: reason.clone(),
                    });
                }
            }
        }
    }

    selected.sort_by_key(|index| units[*index].first_index());

    let mut turns: Vec<ProjectedTurn> = Vec::new();
    for index in &selected {
        let unit = &units[*index];
        for (slot, (message_index, message)) in unit.messages.iter().enumerate() {
            let Some(role) = role_of(message) else {
                continue;
            };
            let content = unit.prepared[slot].clone();
            turns.push(ProjectedTurn {
                message_index: *message_index,
                role,
                tokens: estimate_tokens(&content),
                content,
                priority: unit.priority,
            });
        }
    }

    let projected_tokens: u32 = turns
        .iter()
        .map(|turn| turn.tokens)
        .fold(0u32, |total, cost| total.saturating_add(cost));
    let retained_not_projected = eligible_messages.saturating_sub(turns.len() as u32);

    Projection {
        turns,
        projected_tokens,
        retained_not_projected,
        omitted_pins,
        retention,
    }
}

/// Eligible messages before the current turn's cell.
///
/// Uses [`super::turn_context::history_slice`] for the boundary, so both paths
/// agree on where this turn begins.
fn eligible_before_cell<'a>(
    conversation: &'a Conversation,
    cell_message_id: &str,
) -> Vec<(usize, &'a Message)> {
    super::turn_context::history_slice(conversation, cell_message_id)
        .iter()
        .enumerate()
        .filter(|(_, message)| is_eligible(message))
        .collect()
}

/// What the whole thread costs to keep, measured on prepared text.
fn retention_status(conversation: &Conversation) -> RetentionStatus {
    let retained_tokens = conversation
        .messages
        .iter()
        .filter(|message| is_eligible(message))
        .map(|message| estimate_tokens(&super::turn_context::prepare(message)))
        .fold(0u32, |total, cost| total.saturating_add(cost));

    RetentionStatus {
        retained_tokens,
        limit_tokens: CHAT_RETENTION_LIMIT,
        exceeds_limit: retained_tokens > CHAT_RETENTION_LIMIT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::conversations::{
        Conversation, Message, MessageRole, MessageStatus,
    };

    fn make_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            conversation_id: "conv-1".to_string(),
            role,
            content: content.to_string(),
            status: MessageStatus::Done,
            run_id: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            completed_at: None,
            elapsed_ms: None,
            error: None,
            model_name: None,
            model_role: None,
            used_fallback: None,
            tokens_in: None,
            tokens_out: None,
            outcome: None,
            verification: None,
            tool_summary: None,
        }
    }

    fn make_conversation(messages: Vec<Message>) -> Conversation {
        Conversation {
            id: "conv-1".to_string(),
            title: "Test Conversation".to_string(),
            messages,
            runs: vec![],
            owner_user_id: "user-1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            last_activity_at: "2026-01-01T00:00:00Z".to_string(),
            compactions: 0,
            pinned_context: vec![],
            routed_role: None,
            routed_model_id: None,
        }
    }

    #[test]
    fn empty_conversation_produces_empty_projection() {
        let conv = make_conversation(vec![]);
        let proj = project(&conv, "cell-1", 4096, &[], "hello");
        assert!(proj.turns.is_empty());
        assert_eq!(proj.projected_tokens, 0);
        assert_eq!(proj.retained_not_projected, 0);
        assert!(proj.omitted_pins.is_empty());
    }

    #[test]
    fn zero_budget_retains_but_projects_nothing() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "What is the pressure rating?"),
            make_message("a1", MessageRole::Assistant, "The pressure rating is 150 PSI."),
            make_message("u2", MessageRole::User, "Can you explain more?"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let proj = project(&conv, "cell", 0, &[], "explain more");
        assert!(proj.turns.is_empty());
        // The cell and u2 are this turn; u1 and a1 are eligible history.
        assert_eq!(proj.retained_not_projected, 2);
        assert!(proj.retention.retained_tokens > 0);
        assert!(!proj.retention.exceeds_limit);
    }

    /// A zero budget is still a budget a pin cannot be honoured under, and that
    /// has to be said rather than looking like an ordinary empty history.
    #[test]
    fn zero_budget_reports_every_pin_as_unhonourable() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "always use metric units"),
            make_message("a1", MessageRole::Assistant, "Understood."),
            make_message("u2", MessageRole::User, "next"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let proj = project(&conv, "cell", 0, &["msg:u1".to_string()], "next");
        assert_eq!(proj.omitted_pins.len(), 1);
        assert_eq!(proj.omitted_pins[0].pin, "msg:u1");
        assert_eq!(proj.omitted_pins[0].kind, "message");
    }

    #[test]
    fn pinned_turns_always_included() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "Important: always use metric units"),
            make_message("a1", MessageRole::Assistant, "Understood, I will use metric units."),
            make_message("u2", MessageRole::User, "What is the temperature?"),
            make_message("a2", MessageRole::Assistant, "The temperature is 25 C."),
            make_message("u3", MessageRole::User, "New question"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let proj = project(&conv, "cell", 200, &["msg:u1".to_string()], "new question");
        assert!(proj.turns.iter().any(|t| t.content.contains("metric units")));
        assert!(proj.omitted_pins.is_empty());
    }

    #[test]
    fn recent_turns_preferred_over_old() {
        let mut messages = vec![];
        for i in 0..20 {
            messages.push(make_message(
                &format!("u{i}"),
                MessageRole::User,
                &format!("User message number {i} with some padding text to consume tokens"),
            ));
            messages.push(make_message(
                &format!("a{i}"),
                MessageRole::Assistant,
                &format!("Assistant response number {i} with enough text to be meaningful"),
            ));
        }
        messages.push(make_message("uN", MessageRole::User, "final question"));
        messages.push(make_message("cell", MessageRole::Assistant, ""));

        let conv = make_conversation(messages);
        let proj = project(&conv, "cell", 200, &[], "final question");
        assert!(proj
            .turns
            .iter()
            .any(|t| t.content.contains("number 19") || t.content.contains("number 18")));
        assert!(!proj.turns.iter().any(|t| t.content.contains("number 0 ")));
    }

    /// The invariant the whole rewrite is for: what is measured is what is sent.
    #[test]
    fn a_projection_never_exceeds_its_budget() {
        let mut messages = vec![];
        for i in 0..30 {
            messages.push(make_message(
                &format!("u{i}"),
                MessageRole::User,
                &format!("question {i} about the pressure rating of the vessel"),
            ));
            // Every answer carries markers that expand on preparation.
            messages.push(make_message(
                &format!("a{i}"),
                MessageRole::Assistant,
                &format!("answer {i} [E1] [E2] [E3] grounded in the drawing"),
            ));
        }
        messages.push(make_message("cell", MessageRole::Assistant, ""));
        let conv = make_conversation(messages);

        for budget in [1u32, 17, 64, 200, 512, 4096] {
            let proj = project(&conv, "cell", budget, &[], "pressure rating");
            assert!(
                proj.projected_tokens <= budget,
                "budget {budget} produced {} tokens",
                proj.projected_tokens
            );
            let rendered: u32 = proj.turns.iter().map(|t| estimate_tokens(&t.content)).sum();
            assert_eq!(rendered, proj.projected_tokens);
        }
    }

    /// A question and its answer travel together or not at all.
    #[test]
    fn units_are_carried_whole() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "what is the flange rating"),
            make_message("a1", MessageRole::Assistant, "class three hundred"),
            make_message("u2", MessageRole::User, "and the gasket"),
            make_message("a2", MessageRole::Assistant, "spiral wound"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        for budget in [8u32, 12, 20, 40, 100] {
            let proj = project(&conv, "cell", budget, &[], "gasket");
            let carried: Vec<usize> = proj.turns.iter().map(|t| t.message_index).collect();
            for (question, answer) in [(0usize, 1usize), (2, 3)] {
                if carried.contains(&answer) {
                    assert!(
                        carried.contains(&question),
                        "budget {budget}: answer {answer} carried without question {question}"
                    );
                }
            }
        }
    }

    #[test]
    fn several_oversized_pins_are_each_reported() {
        let big = "padding ".repeat(200);
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, &format!("first {big}")),
            make_message("a1", MessageRole::Assistant, "ok"),
            make_message("u2", MessageRole::User, &format!("second {big}")),
            make_message("a2", MessageRole::Assistant, "ok"),
            make_message("u3", MessageRole::User, "now"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let proj = project(
            &conv,
            "cell",
            40,
            &["msg:u1".to_string(), "msg:u2".to_string()],
            "now",
        );
        assert_eq!(proj.omitted_pins.len(), 2);
        assert!(proj
            .omitted_pins
            .iter()
            .all(|o| matches!(o.reason, PinOmission::ExceedsBudget { .. })));
        assert!(proj.projected_tokens <= 40);
    }

    /// The two reasons are distinguishable: one pin fits and takes the budget,
    /// the next would have fitted alone and did not.
    #[test]
    fn a_pin_that_lost_the_budget_reads_differently_from_one_that_never_fitted() {
        let chunk = "word ".repeat(30);
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, &chunk),
            make_message("a1", MessageRole::Assistant, "ok"),
            make_message("u2", MessageRole::User, &chunk),
            make_message("a2", MessageRole::Assistant, "ok"),
            make_message("u3", MessageRole::User, "now"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let unit_cost = estimate_tokens(&super::super::turn_context::prepare(&make_message(
            "u1",
            MessageRole::User,
            &chunk,
        ))) + estimate_tokens("ok");
        // Room for one protected unit, not two.
        let budget = unit_cost + 2;
        let proj = project(
            &conv,
            "cell",
            budget,
            &["msg:u1".to_string(), "msg:u2".to_string()],
            "now",
        );
        assert_eq!(proj.omitted_pins.len(), 1);
        assert!(matches!(
            proj.omitted_pins[0].reason,
            PinOmission::BudgetSpent { .. }
        ));
        // The oldest pin is the one kept.
        assert_eq!(proj.omitted_pins[0].message_id, "u2");
    }

    /// Retention is measured and reported; nothing is deleted to satisfy it.
    #[test]
    fn retention_is_measured_and_never_enforced_by_truncation() {
        let conv = make_conversation(vec![
            make_message("u1", MessageRole::User, "a question worth some tokens"),
            make_message("a1", MessageRole::Assistant, "an answer worth some tokens"),
            make_message("cell", MessageRole::Assistant, ""),
        ]);
        let before = conv.messages.len();
        let proj = project(&conv, "cell", 4, &[], "question");
        assert_eq!(conv.messages.len(), before, "storage must not be touched");
        assert_eq!(proj.retention.limit_tokens, CHAT_RETENTION_LIMIT);
        assert!(!proj.retention.exceeds_limit);
        assert!(proj.retention.retained_tokens > proj.projected_tokens);
    }

    #[test]
    fn keywords_survive_non_latin_scripts() {
        // Two CJK characters is a word, not a particle; the old byte-length
        // test admitted or rejected these by script rather than by meaning.
        assert!(extract_keywords("圧力 は").contains(&"圧力".to_string()));
        assert!(extract_keywords("कोड लिखो").contains(&"कोड".to_string()));
        assert!(extract_keywords("what is the rating").contains(&"rating".to_string()));
        // A single character is still dropped.
        assert!(!extract_keywords("a b c").iter().any(|word| word == "a"));
    }
}

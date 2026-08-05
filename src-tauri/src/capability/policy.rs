//! Capability Switch Policy — hysteresis against adapter thrashing.
//!
//! Neither the original build plan nor the legacy routing code addressed what
//! happens across a *conversation*. Real sessions alternate intent constantly:
//!
//! ```text
//! "write the parser"        -> coding
//! "why did you use a stack" -> reasoning
//! "ok now add error handling" -> coding
//! ```
//!
//! Switching on every turn would rebind the LoRA adapter three times in three
//! turns, and — more damagingly — make model behaviour feel unstable to the
//! user for no benefit.
//!
//! This module implements a Schmitt trigger: entering a capability demands more
//! confidence than remaining in it. Between the two thresholds the active
//! capability simply persists, so borderline turns cost nothing. A capability is
//! released only after several consecutive turns show no support for it at all.

use serde::{Deserialize, Serialize};

use crate::capability::classifier::ClassificationResult;
use crate::model_intelligence::intent::PromptIntent;

/// The capability key meaning "no specialization".
pub const GENERAL: &str = "general";

/// Thresholds governing capability transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwitchPolicy {
    /// Confidence required to switch *into* a new capability.
    pub enter_threshold: f32,
    /// Confidence below which the active capability stops being reinforced.
    pub exit_threshold: f32,
    /// Consecutive unsupported turns tolerated before falling back to general.
    pub max_unsupported_turns: u32,
}

impl Default for SwitchPolicy {
    fn default() -> Self {
        Self {
            // Calibrated against `classifier`'s confidence scale: roughly one
            // CLEAR plus one MODERATE signal, uncontested.
            enter_threshold: 0.55,
            exit_threshold: 0.35,
            max_unsupported_turns: 3,
        }
    }
}

impl SwitchPolicy {
    /// Rejects a nonsensical configuration (exit above enter would invert the
    /// hysteresis band and cause the very oscillation this exists to prevent).
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.enter_threshold) {
            return Err(format!("enterThreshold must be in [0,1], got {}", self.enter_threshold));
        }
        if !(0.0..=1.0).contains(&self.exit_threshold) {
            return Err(format!("exitThreshold must be in [0,1], got {}", self.exit_threshold));
        }
        if self.exit_threshold > self.enter_threshold {
            return Err(format!(
                "exitThreshold ({}) must not exceed enterThreshold ({}) — an inverted band causes thrashing",
                self.exit_threshold, self.enter_threshold
            ));
        }
        Ok(())
    }
}

/// What the policy decided for one turn.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchDecision {
    /// Keep the active capability.
    Hold { active: String, reason: String },
    /// Move to a different capability.
    Switch { from: String, to: String, reason: String },
}

impl SwitchDecision {
    /// The capability in force after this decision.
    pub fn resulting_capability(&self) -> &str {
        match self {
            Self::Hold { active, .. } => active,
            Self::Switch { to, .. } => to,
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Self::Hold { reason, .. } | Self::Switch { reason, .. } => reason,
        }
    }

    pub fn is_switch(&self) -> bool {
        matches!(self, Self::Switch { .. })
    }
}

/// Tracks the active capability across the turns of a conversation.
#[derive(Debug, Clone)]
pub struct CapabilityTracker {
    policy: SwitchPolicy,
    active: String,
    turns_held: u32,
    unsupported_turns: u32,
}

impl CapabilityTracker {
    pub fn new(policy: SwitchPolicy) -> Self {
        Self {
            policy,
            active: GENERAL.to_string(),
            turns_held: 0,
            unsupported_turns: 0,
        }
    }

    pub fn active_capability(&self) -> &str {
        &self.active
    }

    pub fn turns_held(&self) -> u32 {
        self.turns_held
    }

    /// Clears state — call when the model is unloaded or the user starts a new
    /// conversation, so capability stickiness never leaks across sessions.
    pub fn reset(&mut self) {
        self.active = GENERAL.to_string();
        self.turns_held = 0;
        self.unsupported_turns = 0;
    }

    /// Decides the capability for this turn and commits the resulting state.
    ///
    /// `manual_override` short-circuits classification entirely: an explicit
    /// user choice is always honoured and pins the capability until changed.
    pub fn decide(
        &mut self,
        classification: &ClassificationResult,
        manual_override: Option<&str>,
    ) -> SwitchDecision {
        // 1. An explicit user selection outranks anything inferred.
        if let Some(forced) = manual_override.map(str::trim).filter(|s| {
            !s.is_empty() && !s.eq_ignore_ascii_case("auto") && !s.eq_ignore_ascii_case("none")
        }) {
            return self.commit(
                forced,
                format!("Manual override pinned capability to '{}'", forced),
            );
        }

        let candidate = classification.capability_name();
        let confidence = classification.confidence;

        // 2. Track how long the active capability has gone unsupported, so a
        //    conversation that drifts away from coding eventually releases it.
        let is_unsupported = classification.intent == PromptIntent::GeneralChat
            || (candidate != self.active && confidence < self.policy.exit_threshold);

        if is_unsupported {
            self.unsupported_turns = self.unsupported_turns.saturating_add(1);
        } else {
            self.unsupported_turns = 0;
        }

        // 3. Classification agrees with what is already active — reinforce.
        if candidate == self.active {
            return self.commit(
                candidate,
                format!(
                    "Capability '{}' reinforced (confidence {:.2})",
                    candidate, confidence
                ),
            );
        }

        // 4. A confident, different verdict crosses the upper threshold.
        if confidence >= self.policy.enter_threshold {
            return self.commit(
                candidate,
                format!(
                    "Switched to '{}' (confidence {:.2} >= enter {:.2})",
                    candidate, confidence, self.policy.enter_threshold
                ),
            );
        }

        // 5. Sustained absence of support releases the capability.
        if self.active != GENERAL && self.unsupported_turns >= self.policy.max_unsupported_turns {
            return self.commit(
                GENERAL,
                format!(
                    "Released '{}' after {} turns without support",
                    self.active, self.unsupported_turns
                ),
            );
        }

        // 6. Inside the hysteresis band — hold, and say why.
        let reason = if self.active == GENERAL {
            format!(
                "Confidence {:.2} below enter threshold {:.2}; staying on base model",
                confidence, self.policy.enter_threshold
            )
        } else {
            format!(
                "Holding '{}': candidate '{}' at {:.2} did not clear enter threshold {:.2}",
                self.active, candidate, confidence, self.policy.enter_threshold
            )
        };
        self.commit_hold(reason)
    }

    /// Applies a target capability and returns the corresponding decision.
    fn commit(&mut self, target: &str, reason: String) -> SwitchDecision {
        if target == self.active {
            self.turns_held = self.turns_held.saturating_add(1);
            SwitchDecision::Hold {
                active: self.active.clone(),
                reason,
            }
        } else {
            let from = std::mem::replace(&mut self.active, target.to_string());
            self.turns_held = 1;
            self.unsupported_turns = 0;
            SwitchDecision::Switch {
                from,
                to: target.to_string(),
                reason,
            }
        }
    }

    fn commit_hold(&mut self, reason: String) -> SwitchDecision {
        self.turns_held = self.turns_held.saturating_add(1);
        SwitchDecision::Hold {
            active: self.active.clone(),
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::classifier::IntentClassifier;

    fn tracker() -> CapabilityTracker {
        CapabilityTracker::new(SwitchPolicy::default())
    }

    fn turn(t: &mut CapabilityTracker, prompt: &str) -> SwitchDecision {
        let c = IntentClassifier::classify(prompt);
        t.decide(&c, None)
    }

    #[test]
    fn starts_on_the_base_model() {
        assert_eq!(tracker().active_capability(), GENERAL);
    }

    #[test]
    fn a_confident_prompt_switches_capability() {
        let mut t = tracker();
        let d = turn(&mut t, "refactor this python class to fix the compile error");

        assert!(d.is_switch(), "expected a switch, got: {:?}", d);
        assert_eq!(d.resulting_capability(), "coding");
        assert_eq!(t.active_capability(), "coding");
    }

    #[test]
    fn borderline_turns_do_not_thrash_the_adapter() {
        let mut t = tracker();
        turn(&mut t, "refactor this python class to fix the compile error");
        assert_eq!(t.active_capability(), "coding");

        // Ambiguous follow-up: some reasoning flavour, but nothing decisive.
        let d = turn(&mut t, "explain this code");

        assert!(!d.is_switch(), "borderline turn must not switch: {:?}", d);
        assert_eq!(t.active_capability(), "coding");
    }

    #[test]
    fn a_strongly_different_intent_still_switches() {
        let mut t = tracker();
        turn(&mut t, "refactor this python class to fix the compile error");
        assert_eq!(t.active_capability(), "coding");

        let d = turn(&mut t, "solve for x and prove that the integral converges");

        assert!(d.is_switch());
        assert_eq!(t.active_capability(), "mathematics");
    }

    #[test]
    fn capability_is_released_after_sustained_lack_of_support() {
        let mut t = tracker();
        turn(&mut t, "refactor this python class to fix the compile error");
        assert_eq!(t.active_capability(), "coding");

        // Three consecutive turns with no domain signal at all.
        turn(&mut t, "thanks");
        turn(&mut t, "that is helpful");
        let d = turn(&mut t, "how are you");

        assert!(d.is_switch(), "expected release to general: {:?}", d);
        assert_eq!(t.active_capability(), GENERAL);
    }

    #[test]
    fn one_offhand_turn_does_not_release_the_capability() {
        let mut t = tracker();
        turn(&mut t, "refactor this python class to fix the compile error");

        turn(&mut t, "thanks");
        assert_eq!(
            t.active_capability(),
            "coding",
            "a single aside must not drop the active capability"
        );
    }

    #[test]
    fn manual_override_wins_over_classification() {
        let mut t = tracker();
        let math = IntentClassifier::classify("solve for x and prove that the integral converges");

        let d = t.decide(&math, Some("coding"));

        assert!(d.is_switch());
        assert_eq!(t.active_capability(), "coding");
    }

    #[test]
    fn auto_and_none_are_not_treated_as_manual_pins() {
        for sentinel in ["auto", "AUTO", "none", "  ", ""] {
            let mut t = tracker();
            let c = IntentClassifier::classify("refactor this python class to fix the compile error");
            t.decide(&c, Some(sentinel));

            assert_eq!(
                t.active_capability(),
                "coding",
                "sentinel {:?} must fall through to classification",
                sentinel
            );
        }
    }

    #[test]
    fn reset_clears_stickiness() {
        let mut t = tracker();
        turn(&mut t, "refactor this python class to fix the compile error");
        assert_eq!(t.active_capability(), "coding");

        t.reset();

        assert_eq!(t.active_capability(), GENERAL);
        assert_eq!(t.turns_held(), 0);
    }

    #[test]
    fn alternating_conversation_switches_far_less_than_it_classifies() {
        // The scenario this module exists for.
        let mut t = tracker();
        let conversation = [
            "write a python function to parse the config file",
            "why did you use a stack there",
            "ok now add error handling to it",
            "explain this code",
            "what are the tradeoffs",
            "refactor it to use a queue instead",
        ];

        let switches = conversation
            .iter()
            .filter(|p| turn(&mut t, p).is_switch())
            .count();

        assert!(
            switches <= 3,
            "hysteresis should suppress most switches in an alternating conversation, saw {}",
            switches
        );
    }

    #[test]
    fn default_policy_is_a_valid_hysteresis_band() {
        assert!(SwitchPolicy::default().validate().is_ok());
    }

    #[test]
    fn inverted_thresholds_are_rejected() {
        let bad = SwitchPolicy {
            enter_threshold: 0.3,
            exit_threshold: 0.8,
            max_unsupported_turns: 3,
        };
        assert!(bad.validate().is_err());
    }
}

//! What a person asked a turn to keep, and what that asking refers to.
//!
//! ## The failure this exists to remove
//!
//! A pin was a `String`. The context panel draws two kinds of row — a turn in
//! the transcript, and a document that was attached — and pressing the pin on
//! either one sent the row's `id`, which for a turn is a message id and for a
//! document is a sha256. Rust then had to guess which it had been handed, and
//! the two projection paths guessed differently:
//!
//! - [`super::turn_context::fit`] matched case-insensitively by message id
//!   *or* by looking for the string inside the turn's text, which is how
//!   pinning a drawing kept the turn that carried it.
//! - [`super::chat_memory_bus::project`], which is the path production
//!   actually takes, compared `pin == message.id` — exact, case-sensitive, and
//!   with no content match at all. Pinning a document did nothing whatsoever.
//!
//! So the tested behaviour and the shipped behaviour were different behaviours,
//! and the difference was invisible because the tested one had no production
//! caller.
//!
//! ## The encoding, and why it is a prefix rather than a new field
//!
//! A pin is stored in `Conversation.pinned_context: Vec<String>`, written to
//! `<appdata>/conversations/{id}.json`. Changing that to a list of structs
//! would need a migration and would make a file written by this build
//! unreadable by the one before it, on a product that is installed rather than
//! deployed.
//!
//! So the storage shape does not change. What changes is that a new pin is
//! written with a prefix naming what it refers to — `msg:`, `sha256:`,
//! `artifact:` — and anything without one is a [`PinRef::Legacy`], matched
//! exactly the way `fit` matched it before this module existed. An old file
//! loads with every pin still honoured; a new file is still a list of strings.
//!
//! ## Why `Legacy` is a variant and not a fallback to `Message`
//!
//! Because the two are not the same rule, and collapsing them is precisely the
//! regression this module is undoing. A legacy pin may be a message id *or* a
//! document hash *or* something a person pinned from a row that no longer
//! exists, and the only safe reading of it is the wide one it was written
//! under. Naming it keeps that widening confined to pins that predate the
//! prefix, rather than making it the rule for all of them.

use serde::{Deserialize, Serialize};

use super::conversations::Message;

/// The longest a single stored pin may be.
///
/// A pin reaches a durable file and is compared against every message in the
/// thread on every turn. An unbounded one is a way to put a document into a
/// field meant to hold an identifier, and a very long one makes the
/// `contains` below costly in a long thread. A sha256 is 64 characters and a
/// UUID is 36, so this is generous by an order of magnitude.
pub const MAX_PIN_CHARS: usize = 512;

/// What a pin points at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PinRef {
    /// One turn of the transcript, by its `Message` id.
    Message { message_id: String },
    /// A document, by the sha256 the store holds it under. Matched against the
    /// text of a turn, because that is where an attachment's hash appears.
    Source { sha256: String },
    /// A produced file, by its artifact id. Matched the same way.
    Artifact { artifact_id: String },
    /// A pin written before pins were typed.
    ///
    /// Matched the old way — by message id *or* by content — because that is
    /// the rule it was stored under and narrowing it now would silently drop
    /// protections somebody is relying on.
    Legacy { raw: String },
}

impl PinRef {
    /// Reads one stored pin.
    ///
    /// `None` for a pin that is blank once trimmed, or one whose payload is
    /// blank (`"msg:"` with nothing after it). Both would match everything
    /// under the `contains` test below and pin the entire history from one
    /// stray string, which is a window that fills and a turn that fails.
    ///
    /// Over-long pins are truncated rather than rejected, so a pin already on
    /// disk cannot make a conversation unreadable. Truncation cannot narrow a
    /// match in the direction that loses protection: a shorter needle matches a
    /// superset of what the longer one did, which for a pin errs towards
    /// keeping.
    pub fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let bounded: String = trimmed.chars().take(MAX_PIN_CHARS).collect();

        let builders: [(&str, fn(String) -> PinRef); 3] = [
            ("msg:", |v| PinRef::Message { message_id: v }),
            ("sha256:", |v| PinRef::Source { sha256: v }),
            ("artifact:", |v| PinRef::Artifact { artifact_id: v }),
        ];
        for (prefix, build) in builders {
            if let Some(rest) = bounded.strip_prefix(prefix) {
                let value = rest.trim();
                if value.is_empty() {
                    return None;
                }
                return Some(build(value.to_string()));
            }
        }

        Some(PinRef::Legacy { raw: bounded })
    }

    /// Reads a whole stored list, dropping the entries that cannot mean
    /// anything.
    pub fn parse_all(raw: &[String]) -> Vec<Self> {
        raw.iter().filter_map(|pin| PinRef::parse(pin)).collect()
    }

    /// How this pin is written back to the conversation file.
    ///
    /// A [`PinRef::Legacy`] round-trips to exactly the string it came from, so
    /// normalising a stored list never rewrites a pin into a narrower one.
    pub fn encode(&self) -> String {
        match self {
            PinRef::Message { message_id } => format!("msg:{message_id}"),
            PinRef::Source { sha256 } => format!("sha256:{sha256}"),
            PinRef::Artifact { artifact_id } => format!("artifact:{artifact_id}"),
            PinRef::Legacy { raw } => raw.clone(),
        }
    }

    /// Whether this pin protects `message`.
    ///
    /// `prepared_upper` is the turn's text *as it will be sent* — after the
    /// tool summary is appended and the evidence markers are neutralised —
    /// upper-cased once by the caller, because this is called once per pin per
    /// message and upper-casing inside would repeat that work for every pin.
    ///
    /// Case-insensitive throughout, matching `pruneStaleToolResults` in the
    /// runtime, so a pin cannot be honoured by one side and dropped by the
    /// other over how an id happened to be spelled.
    pub fn protects(&self, message: &Message, prepared_upper: &str) -> bool {
        match self {
            PinRef::Message { message_id } => message.id.eq_ignore_ascii_case(message_id),
            PinRef::Source { sha256 } => prepared_upper.contains(&sha256.to_uppercase()),
            PinRef::Artifact { artifact_id } => {
                prepared_upper.contains(&artifact_id.to_uppercase())
            }
            PinRef::Legacy { raw } => {
                message.id.eq_ignore_ascii_case(raw) || prepared_upper.contains(&raw.to_uppercase())
            }
        }
    }

    /// The identifier this pin names, for a record a person will read.
    pub fn target(&self) -> &str {
        match self {
            PinRef::Message { message_id } => message_id,
            PinRef::Source { sha256 } => sha256,
            PinRef::Artifact { artifact_id } => artifact_id,
            PinRef::Legacy { raw } => raw,
        }
    }

    /// One word for what kind of thing this is, for the same record.
    pub fn kind(&self) -> &'static str {
        match self {
            PinRef::Message { .. } => "message",
            PinRef::Source { .. } => "source",
            PinRef::Artifact { .. } => "artifact",
            PinRef::Legacy { .. } => "legacy",
        }
    }
}

/// Whether any pin in `pins` protects `message`.
pub fn any_protects(pins: &[PinRef], message: &Message, prepared: &str) -> bool {
    if pins.is_empty() {
        return false;
    }
    let upper = prepared.to_uppercase();
    pins.iter().any(|pin| pin.protects(message, &upper))
}

/// Which pins protect `message`, for reporting what could not be carried.
pub fn protecting<'a>(pins: &'a [PinRef], message: &Message, prepared: &str) -> Vec<&'a PinRef> {
    if pins.is_empty() {
        return Vec::new();
    }
    let upper = prepared.to_uppercase();
    pins.iter()
        .filter(|pin| pin.protects(message, &upper))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::conversations::{MessageRole, MessageStatus};

    fn message(id: &str, content: &str) -> Message {
        Message {
            id: id.to_string(),
            conversation_id: "c1".to_string(),
            role: MessageRole::User,
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

    const SHA: &str = "ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34";

    #[test]
    fn a_prefixed_pin_reads_as_its_kind() {
        assert_eq!(
            PinRef::parse("msg:u1"),
            Some(PinRef::Message {
                message_id: "u1".into()
            })
        );
        assert_eq!(
            PinRef::parse(&format!("sha256:{SHA}")),
            Some(PinRef::Source { sha256: SHA.into() })
        );
        assert_eq!(
            PinRef::parse("artifact:art-9"),
            Some(PinRef::Artifact {
                artifact_id: "art-9".into()
            })
        );
    }

    #[test]
    fn an_unprefixed_pin_is_legacy_and_keeps_its_exact_text() {
        let pin = PinRef::parse("u1").expect("parses");
        assert_eq!(pin, PinRef::Legacy { raw: "u1".into() });
        assert_eq!(pin.encode(), "u1");
    }

    /// The property the whole prefix scheme rests on: normalising a stored list
    /// never rewrites an old pin into a narrower one.
    #[test]
    fn legacy_pins_round_trip_unchanged() {
        for raw in ["u1", SHA, "Some Document.pdf", "MiXeD-CaSe-Id"] {
            let pin = PinRef::parse(raw).expect("parses");
            assert_eq!(pin.encode(), raw, "{raw} did not round-trip");
        }
    }

    #[test]
    fn blank_and_payload_less_pins_are_dropped() {
        for raw in ["", "   ", "\t\n", "msg:", "sha256:   ", "artifact:"] {
            assert_eq!(PinRef::parse(raw), None, "{raw:?} should not parse");
        }
        assert!(PinRef::parse_all(&["".into(), "  ".into(), "msg:".into()]).is_empty());
    }

    #[test]
    fn a_message_pin_matches_its_id_whatever_the_case() {
        let pin = PinRef::parse("msg:U1").expect("parses");
        assert!(pin.protects(&message("u1", "anything"), "ANYTHING"));
        assert!(!pin.protects(&message("u2", "anything"), "ANYTHING"));
    }

    /// A message pin must *not* match on content, or pinning a turn whose id
    /// happens to appear in another turn's text would pin both.
    #[test]
    fn a_message_pin_does_not_match_on_content() {
        let pin = PinRef::parse("msg:u1").expect("parses");
        let other = message("u2", "see u1 for the rating");
        assert!(!pin.protects(&other, "SEE U1 FOR THE RATING"));
    }

    #[test]
    fn a_source_pin_matches_the_hash_inside_a_turn() {
        let pin = PinRef::parse(&format!("sha256:{SHA}")).expect("parses");
        let carrying = message("u9", &format!("Attached drawing {SHA}"));
        let prepared = format!("ATTACHED DRAWING {}", SHA.to_uppercase());
        assert!(pin.protects(&carrying, &prepared));
        assert!(!pin.protects(&message("u8", "no drawing here"), "NO DRAWING HERE"));
    }

    /// The behaviour the production path lost: a pin pressed on a document row
    /// keeps the turn that carries it.
    #[test]
    fn a_legacy_pin_still_matches_by_id_or_by_content() {
        let by_id = PinRef::parse("u1").expect("parses");
        assert!(by_id.protects(&message("U1", "unrelated"), "UNRELATED"));

        let by_content = PinRef::parse(SHA).expect("parses");
        let carrying = message("u9", &format!("Attached drawing {SHA}"));
        assert!(by_content.protects(
            &carrying,
            &format!("ATTACHED DRAWING {}", SHA.to_uppercase())
        ));
    }

    #[test]
    fn an_over_long_pin_is_bounded_rather_than_refused() {
        let long = "x".repeat(MAX_PIN_CHARS * 2);
        let pin = PinRef::parse(&long).expect("parses");
        assert_eq!(pin.target().chars().count(), MAX_PIN_CHARS);
    }

    #[test]
    fn protecting_names_every_pin_that_applies() {
        let pins = PinRef::parse_all(&[
            "msg:u1".into(),
            format!("sha256:{SHA}"),
            "msg:u2".into(),
        ]);
        let msg = message("u1", &format!("drawing {SHA}"));
        let hits = protecting(&pins, &msg, &msg.content);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|p| p.kind() == "message"));
        assert!(hits.iter().any(|p| p.kind() == "source"));
    }
}

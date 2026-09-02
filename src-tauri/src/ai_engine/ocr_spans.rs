//! Incremental parsing of Unlimited-OCR grounded output.
//!
//! ## The format, as the model actually emits it
//!
//! The model card documents `<|det|>label [x, y, x, y]<|/det|>text`. The
//! build we run does not produce that. Measured against a page with known
//! ink positions, `Free OCR.` returns one region per line, like this:
//!
//! ```text
//! title [77, 51, 723, 86]QUARTERLY FIELD REPORT
//! text [76, 106, 573, 126]Substation 14 | Issued 2026-09-01
//! table [78, 304, 785, 460]<table>PhaseReadingLimit...</table>
//! footer [615, 923, 866, 956]AUTH-7731
//! ```
//!
//! No markers, no closing tag — a bare label, a bracketed box, then the text
//! to end of line. Parsing the documented format instead yields a page with
//! no regions at all, which is why this module follows the observed output
//! and the tests below are built from a real captured response.
//!
//! Lines that carry no box (the model sometimes emits a stray header line)
//! are still surfaced as text rather than dropped.
//!
//! ## Why it is incremental
//!
//! Each closing `]` is the moment the model commits to a region, and the text
//! after it is what it read there. Emitting the region before its text is
//! what lets the scan view draw the box first and fill it in — the user
//! watches the model move down the page.
//!
//! Tokens arrive in chunks that respect nothing, so a header can be split at
//! any byte. The parser holds a bounded tail and the tests feed it the same
//! response split at every offset to prove no boundary is special.
//!
//! ## What it deliberately does not do
//!
//! It does not map coordinates. The numbers are reported exactly as emitted;
//! turning them into page pixels needs the image geometry and the measured
//! coordinate convention, which live in [`super::ocr_profile`].

use serde::{Deserialize, Serialize};

/// Longest plausible label (`title`, `text`, `table`, `figure`, `footer`).
/// If no `[` appears within this many bytes of a line start, the line is
/// prose, not a region header.
const MAX_LABEL: usize = 24;

/// Longest plausible `[x1, y1, x2, y2]` body. Bounds the wait for a `]` that
/// may never come, so a malformed stream degrades to text instead of stalling.
const MAX_BOX: usize = 64;

/// A detection box exactly as the model wrote it. Untransformed: see the
/// module note on why no coordinate space is assumed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// What the parser hands the UI as the stream arrives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "event")]
pub enum OcrEvent {
    /// A region is committed. Emitted before any of its text.
    Region {
        index: usize,
        /// `title`, `text`, `table`, `figure`, `footer` — the model's label.
        label: String,
        bbox: Option<RawBox>,
    },
    /// Text belonging to `index`, or to no region for lines with no header.
    Text { index: Option<usize>, delta: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// At the start of a line, deciding whether it opens a region.
    LineStart,
    /// Inside `[...]`, waiting for the closing bracket.
    Box,
    /// Streaming a region's text to end of line.
    Text,
}

/// Feed it chunks, get events. One per OCR run.
#[derive(Debug)]
pub struct SpanParser {
    buffer: String,
    state: State,
    label: String,
    current: Option<usize>,
    next_index: usize,
}

impl Default for SpanParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            state: State::LineStart,
            label: String::new(),
            current: None,
            next_index: 0,
        }
    }

    /// Consume a chunk. Returns every event now complete; a chunk ending
    /// mid-header produces none and is carried forward.
    pub fn feed(&mut self, chunk: &str) -> Vec<OcrEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        loop {
            match self.state {
                State::LineStart => {
                    if !self.step_line_start(&mut events) {
                        break;
                    }
                }
                State::Box => {
                    if !self.step_box(&mut events) {
                        break;
                    }
                }
                State::Text => {
                    if !self.step_text(&mut events) {
                        break;
                    }
                }
            }
        }
        events
    }

    /// End of stream: release whatever is still held.
    pub fn finish(&mut self) -> Vec<OcrEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let rest: String = self.buffer.drain(..).collect();
            let rest = if self.state == State::Box {
                // An unterminated header is not a region.
                format!("{} [{}", self.label, rest)
            } else {
                rest
            };
            self.push_text(&mut events, rest);
        }
        self.state = State::LineStart;
        events
    }

    /// Returns false when it needs more input.
    fn step_line_start(&mut self, events: &mut Vec<OcrEvent>) -> bool {
        let newline = self.buffer.find('\n');
        let bracket = self.buffer.find('[');

        // A header must open its bracket on this line, close to the start.
        if let Some(at) = bracket {
            let before_newline = newline.map_or(true, |nl| at < nl);
            if before_newline && at <= MAX_LABEL {
                let head: String = self.buffer[..at].to_string();
                if is_label(&head) {
                    self.label = head.trim().to_string();
                    self.buffer.drain(..=at);
                    self.state = State::Box;
                    return true;
                }
            }
        }

        if let Some(nl) = newline {
            // A whole line with no header: prose, emitted as-is.
            let line: String = self.buffer.drain(..=nl).collect();
            self.push_text(events, line);
            return true;
        }

        // No newline yet. Once past the label window there can be no header
        // on this line, so the text can start flowing instead of stalling.
        if self.buffer.len() > MAX_LABEL {
            let text: String = self.buffer.drain(..).collect();
            self.push_text(events, text);
            self.state = State::Text;
            return true;
        }
        false
    }

    fn step_box(&mut self, events: &mut Vec<OcrEvent>) -> bool {
        if let Some(at) = self.buffer.find(']') {
            let inner: String = self.buffer.drain(..at).collect();
            self.buffer.drain(..1);
            match parse_box(&inner) {
                Some(bbox) => {
                    let index = self.next_index;
                    self.next_index += 1;
                    self.current = Some(index);
                    events.push(OcrEvent::Region {
                        index,
                        label: std::mem::take(&mut self.label),
                        bbox: Some(bbox),
                    });
                }
                None => {
                    // Not coordinates after all — a bracket in ordinary prose.
                    // Put it back as text rather than inventing a region.
                    let label = std::mem::take(&mut self.label);
                    self.push_text(events, format!("{label} [{inner}]"));
                }
            }
            self.state = State::Text;
            return true;
        }
        if self.buffer.len() > MAX_BOX {
            let stray: String = self.buffer.drain(..).collect();
            let label = std::mem::take(&mut self.label);
            self.push_text(events, format!("{label} [{stray}"));
            self.state = State::Text;
            return true;
        }
        false
    }

    fn step_text(&mut self, events: &mut Vec<OcrEvent>) -> bool {
        if let Some(nl) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=nl).collect();
            self.push_text(events, line);
            self.state = State::LineStart;
            return true;
        }
        if self.buffer.is_empty() {
            return false;
        }
        // Stream what is here; the rest of the line follows.
        let text: String = self.buffer.drain(..).collect();
        self.push_text(events, text);
        false
    }

    fn push_text(&mut self, events: &mut Vec<OcrEvent>, delta: String) {
        if delta.is_empty() {
            return;
        }
        events.push(OcrEvent::Text {
            index: self.current,
            delta,
        });
    }
}

/// Whether `head` looks like a region label rather than the start of prose.
fn is_label(head: &str) -> bool {
    let t = head.trim();
    !t.is_empty()
        && t.len() <= MAX_LABEL
        && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `77, 51, 723, 86` into a box. Anything else is not a box.
fn parse_box(inner: &str) -> Option<RawBox> {
    let nums: Vec<i32> = inner
        .split(',')
        .map(|p| p.trim().parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    match nums.as_slice() {
        [x1, y1, x2, y2] => Some(RawBox {
            x1: *x1,
            y1: *y1,
            x2: *x2,
            y2: *y2,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured verbatim from a real run: Q6_K + the patched projector,
    /// `Free OCR.` on a page whose ink positions are known.
    const SAMPLE: &str = concat!(
        " \n",
        "www.free.com\n",
        "title [77, 51, 723, 86]QUARTERLY FIELD REPORT\n",
        "text [76, 106, 573, 126]Substation 14 | Issued 2026-09-01 | Sheet 1 of 1\n",
        "table [78, 304, 785, 460]<table>PhaseReadingLimit</table>\n",
        "footer [615, 923, 866, 956]AUTH-7731\n"
    );

    fn drain(chunks: &[&str]) -> Vec<OcrEvent> {
        let mut parser = SpanParser::new();
        let mut events: Vec<OcrEvent> = chunks.iter().flat_map(|c| parser.feed(c)).collect();
        events.extend(parser.finish());
        events
    }

    fn regions(events: &[OcrEvent]) -> Vec<(usize, String, Option<RawBox>, String)> {
        let mut out: Vec<(usize, String, Option<RawBox>, String)> = Vec::new();
        for event in events {
            match event {
                OcrEvent::Region { index, label, bbox } => {
                    out.push((*index, label.clone(), *bbox, String::new()))
                }
                OcrEvent::Text { index, delta } => {
                    if let Some(i) = index {
                        if let Some(slot) = out.iter_mut().find(|r| r.0 == *i) {
                            slot.3.push_str(delta);
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn a_real_response_yields_every_region_in_reading_order() {
        let rows = regions(&drain(&[SAMPLE]));
        assert_eq!(rows.len(), 4, "got {rows:#?}");
        assert_eq!(rows[0].1, "title");
        assert_eq!(
            rows[0].2,
            Some(RawBox {
                x1: 77,
                y1: 51,
                x2: 723,
                y2: 86
            })
        );
        assert_eq!(rows[0].3.trim(), "QUARTERLY FIELD REPORT");
        assert_eq!(rows[3].1, "footer");
        assert_eq!(rows[3].3.trim(), "AUTH-7731");
    }

    #[test]
    fn the_table_markup_survives_intact() {
        let rows = regions(&drain(&[SAMPLE]));
        let table = rows.iter().find(|r| r.1 == "table").expect("table region");
        assert!(table.3.contains("<table>"), "got {:?}", table.3);
        assert!(table.3.contains("PhaseReadingLimit"));
    }

    /// The failure this module exists to prevent: a header split across
    /// chunks silently losing its region, leaving a page that looks merely
    /// sparse rather than broken.
    #[test]
    fn no_chunk_boundary_can_change_the_result() {
        let expected = regions(&drain(&[SAMPLE]));
        for split in 1..SAMPLE.len() {
            if !SAMPLE.is_char_boundary(split) {
                continue;
            }
            let (head, tail) = SAMPLE.split_at(split);
            assert_eq!(
                regions(&drain(&[head, tail])),
                expected,
                "split at byte {split} changed the parse"
            );
        }
    }

    #[test]
    fn one_char_at_a_time_parses_the_same_as_one_chunk() {
        let expected = regions(&drain(&[SAMPLE]));
        let singles: Vec<String> = SAMPLE.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = singles.iter().map(|s| s.as_str()).collect();
        assert_eq!(regions(&drain(&refs)), expected);
    }

    #[test]
    fn a_region_is_announced_before_its_text_so_the_box_can_be_drawn_first() {
        let events = drain(&[SAMPLE]);
        let region = events
            .iter()
            .position(|e| matches!(e, OcrEvent::Region { index: 0, .. }));
        let text = events
            .iter()
            .position(|e| matches!(e, OcrEvent::Text { index: Some(0), .. }));
        assert!(region < text);
    }

    #[test]
    fn preamble_lines_with_no_box_are_kept_as_text() {
        // The model emits a stray header line before the first region;
        // dropping it would silently lose page content in other documents.
        let events = drain(&[SAMPLE]);
        let loose: String = events
            .iter()
            .filter_map(|e| match e {
                OcrEvent::Text { index: None, delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(loose.contains("www.free.com"), "got {loose:?}");
    }

    #[test]
    fn prose_containing_a_bracket_does_not_become_a_region() {
        let events = drain(&["see figure [1] on page two\n"]);
        assert!(!events.iter().any(|e| matches!(e, OcrEvent::Region { .. })));
    }

    #[test]
    fn a_three_number_bracket_is_not_a_box() {
        let events = drain(&["text [80, 78, 717]still prose\n"]);
        assert!(
            !events.iter().any(|e| matches!(e, OcrEvent::Region { .. })),
            "a three-number bracket must not become a rectangle"
        );
    }

    #[test]
    fn an_unterminated_header_is_surfaced_as_text_not_swallowed() {
        let mut parser = SpanParser::new();
        let mut events = parser.feed("title [77, 51");
        events.extend(parser.finish());
        let text: String = events
            .iter()
            .map(|e| match e {
                OcrEvent::Text { delta, .. } => delta.as_str(),
                _ => "",
            })
            .collect();
        assert!(text.contains("title"), "got {text:?}");
        assert!(!events.iter().any(|e| matches!(e, OcrEvent::Region { .. })));
    }

    #[test]
    fn a_runaway_bracket_does_not_buffer_without_bound() {
        let mut parser = SpanParser::new();
        parser.feed("text [");
        let events = parser.feed(&"9".repeat(MAX_BOX + 8));
        assert!(events.iter().any(|e| matches!(e, OcrEvent::Text { .. })));
        assert!(parser.buffer.len() <= MAX_BOX);
    }

    #[test]
    fn multibyte_text_survives_the_buffering() {
        let devanagari = "\u{0909}\u{092A}\u{0915}\u{0947}\u{0902}\u{0926}\u{094D}\u{0930}";
        let src = format!("text [1, 2, 3, 4]{devanagari}\n");
        let rows = regions(&drain(&[&src]));
        assert_eq!(rows[0].3.trim(), devanagari);
    }

    #[test]
    fn an_ungrounded_reply_is_all_text_and_no_regions() {
        let events = drain(&["Insulation resistance ", "was measured.\n"]);
        assert!(!events.iter().any(|e| matches!(e, OcrEvent::Region { .. })));
        let joined: String = events
            .iter()
            .map(|e| match e {
                OcrEvent::Text { delta, .. } => delta.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(joined.trim(), "Insulation resistance was measured.");
    }
}

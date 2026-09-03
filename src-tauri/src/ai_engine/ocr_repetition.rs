//! Stopping a read that has stopped reading.
//!
//! ## The failure this exists for
//!
//! Handed a page it cannot resolve, Unlimited-OCR does not stop and it does
//! not say so. It emits region lines forever:
//!
//! ```text
//! image_caption [301, 55, 420, 76]OVO
//! image_caption [304, 58, 423, 79]AUDIO
//! image_caption [307, 61, 426, 82]AVO
//! image_caption [310, 64, 429, 85]AUDIO
//! ```
//!
//! — until the decode cap, which at 16,384 tokens is minutes of GPU time for
//! a page it never read. The same shape appears as a bare `*` and newline
//! ladder on some documents.
//!
//! ## Why the sampler does not catch it
//!
//! [`super::ocr_profile`] attaches DRY to every request as a stand-in for
//! Baidu's `no_repeat_ngram_size=35`. DRY penalises a repeated **token
//! sequence**, and the sequence above never repeats: the bracketed
//! coordinates advance a few pixels each line, so no literal 35-token n-gram
//! ever occurs twice and the penalty is never applied at all. Raising the
//! multiplier does nothing. Lowering `dry_allowed_length` far enough to bite
//! would start penalising the repeated units and headers a real table is made
//! of, which is a worse failure and a quieter one.
//!
//! So the loop is not detectable as token repetition. It is detectable as
//! what it is: a long run of short lines drawn from a handful of distinct
//! values, once the coordinates are discounted. That is what this module
//! measures, and it needs no cooperation from the server.
//!
//! ## What it refuses to do
//!
//! It does not repair the text and it does not quietly drop the tail. It
//! reports the offset where the run began so the caller can cut there **and
//! tell the reader the page was cut short**. A looped read that arrives
//! looking like an ordinary one is the failure this area has already been
//! bitten by once — half of that fix was surfacing `hit_decode_cap` instead
//! of dropping it on the floor.

use std::collections::{HashSet, VecDeque};

/// Consecutive short lines examined together.
///
/// Set by the false positive, not by how fast a loop can be caught. The
/// documents this product reads are field records, and a measurement register
/// is a long column of short, near-identical rows — exactly the shape of a
/// loop. Forty-eight lines was not enough separation: a sixty-row status
/// column would have been cut as degenerate.
///
/// Ninety-six consecutive short lines drawn from [`MAX_DISTINCT`] distinct
/// values is a register no page carries and a loop reaches in about a second
/// of decoding. The higher bar costs a hundred wasted tokens; the lower one
/// cost a real register.
const WINDOW_LINES: usize = 96;

/// Longest normalised line that counts toward the window.
///
/// A line longer than this is content — a sentence, a table row — and it
/// clears the window rather than counting toward it. The loop only ever emits
/// short lines, so nothing is lost and a dense page can never trip the guard.
const MAX_LINE_CHARS: usize = 96;

/// Distinct normalised lines a full window may contain before it is a loop.
///
/// Eight rather than one: the observed loop cycles between several short
/// fragments rather than repeating a single one, and a guard that only caught
/// exact repetition would have missed the case it was written for.
const MAX_DISTINCT: usize = 8;

/// Watches decoded OCR text for the point at which it stops being a page.
///
/// Fed the raw model output in whatever chunks arrive. Line-buffered
/// internally, so a chunk boundary in the middle of a line cannot hide a
/// repeat or invent one.
#[derive(Debug, Default)]
pub struct RepetitionGuard {
    /// The last [`WINDOW_LINES`] short lines, normalised, each with the
    /// character offset it began at.
    window: VecDeque<(usize, String)>,
    /// A line that has arrived without its newline yet.
    pending: String,
    /// Where `pending` starts, in characters of decoded text.
    pending_start: usize,
    /// Characters handed to `feed` so far.
    consumed: usize,
    tripped_at: Option<usize>,
}

impl RepetitionGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes a chunk of decoded text.
    ///
    /// Returns the character offset at which a degenerate run began, once,
    /// the first time one is recognised. Later calls return `None` — the
    /// caller has already been told, and the answer does not change.
    pub fn feed(&mut self, delta: &str) -> Option<usize> {
        if self.tripped_at.is_some() {
            self.consumed += delta.chars().count();
            return None;
        }
        for character in delta.chars() {
            self.consumed += 1;
            if character == '\n' {
                let line = std::mem::take(&mut self.pending);
                let start = self.pending_start;
                self.pending_start = self.consumed;
                if let Some(at) = self.push_line(start, &line) {
                    return Some(at);
                }
            } else {
                self.pending.push(character);
            }
        }
        None
    }

    /// The offset a degenerate run began at, or `None` while the read still
    /// looks like a page.
    pub fn tripped_at(&self) -> Option<usize> {
        self.tripped_at
    }

    /// Adds one completed line and re-tests the window.
    fn push_line(&mut self, start: usize, line: &str) -> Option<usize> {
        let normalised = normalise(line);
        if normalised.is_empty() {
            // Blank lines separate content, and they also separate the rungs
            // of a ladder. Neither counted nor treated as content.
            return None;
        }
        if normalised.chars().count() > MAX_LINE_CHARS {
            // Real content. Whatever came before it was not a loop.
            self.window.clear();
            return None;
        }
        if self.window.len() == WINDOW_LINES {
            self.window.pop_front();
        }
        self.window.push_back((start, normalised));

        if self.window.len() < WINDOW_LINES {
            return None;
        }
        let distinct: HashSet<&str> = self.window.iter().map(|(_, line)| line.as_str()).collect();
        if distinct.len() > MAX_DISTINCT {
            return None;
        }
        let at = self.run_start();
        self.tripped_at = Some(at);
        Some(at)
    }

    /// Where the repeating run actually began, rather than where the window
    /// happens to start.
    ///
    /// The two are not the same, and the difference is real content. The
    /// window fills gradually: at the moment it trips, its oldest lines can
    /// still be the last of the page that read correctly, because a handful of
    /// genuine lines fits inside [`MAX_DISTINCT`] alongside the loop's few
    /// values. Cutting at the window's front therefore threw away the tail of
    /// the real page — measured at four lines of a six-line document.
    ///
    /// So the loop's own vocabulary is taken from the newest half of the
    /// window, which is unambiguously the loop, and the cut is walked back to
    /// the first line that belongs to it. A line that does not belong is where
    /// the page stopped and the repetition started.
    ///
    /// When the run began before the window, every line belongs and the cut is
    /// the window's front — the earliest point still in view. Some of the loop
    /// then survives the cut, which is the right way round: it is visible in
    /// the output, and the reader has already been told the page was cut
    /// short.
    fn run_start(&self) -> usize {
        let half = self.window.len() / 2;
        let vocabulary: HashSet<&str> = self
            .window
            .iter()
            .skip(half)
            .map(|(_, line)| line.as_str())
            .collect();

        let mut at = self.window.back().map(|(start, _)| *start).unwrap_or(0);
        for (start, line) in self.window.iter().rev() {
            if !vocabulary.contains(line.as_str()) {
                break;
            }
            at = *start;
        }
        at
    }
}

/// Where a degenerate run begins in a finished transcription, if one does.
///
/// The same measurement the streaming guard makes, over text that has already
/// arrived. Used to cut the tail off a page that reached the decode cap, and
/// to test the two against each other — a run the stream aborted and a run
/// found afterwards have to be found in the same place.
pub fn degenerate_tail_start(text: &str) -> Option<usize> {
    let mut guard = RepetitionGuard::new();
    if let Some(at) = guard.feed(text) {
        return Some(at);
    }
    // A final line with no trailing newline still counts.
    guard.feed("\n");
    guard.tripped_at()
}

/// A line reduced to what it says, discarding what merely moves.
///
/// One transformation, and it is the load-bearing one: **the coordinate block
/// is removed**. `[301, 55, 420, 76]` is the model's detection box, it
/// advances every line of the loop, and leaving it in is precisely what makes
/// the repetition invisible to a token-level penalty. Case and internal
/// whitespace are folded for the same reason.
///
/// Digits are **kept**, and that is a decision rather than an omission. An
/// earlier version collapsed every digit run to `#` so a counting loop
/// (`Item 1`, `Item 2`) would read as one repeated line. It also collapsed
/// every row of a measurement register — `Phase A 412 MOhm` and
/// `Phase B 388 MOhm` differ mostly in their numbers — and a register is the
/// thing this product exists to read. No observed loop counts; every observed
/// one cycles a handful of fixed fragments. Catching the hypothetical failure
/// was not worth cutting the real document.
fn normalise(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut last_was_space = true;

    while let Some(character) = chars.next() {
        if character == '[' {
            // A bracketed run of coordinates is dropped whole. Anything else
            // between brackets is prose and is kept, brackets included.
            let mut inner = String::new();
            let mut closed = false;
            for next in chars.by_ref() {
                if next == ']' {
                    closed = true;
                    break;
                }
                inner.push(next);
            }
            let is_box = closed
                && !inner.is_empty()
                && inner
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == ',' || c == ' ' || c == '-' || c == '.');
            if is_box {
                continue;
            }
            out.push('[');
            out.push_str(&inner.to_lowercase());
            if closed {
                out.push(']');
            }
            last_was_space = false;
            continue;
        }

        if character.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }

        last_was_space = false;
        out.extend(character.to_lowercase());
    }

    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop as it was captured: region lines whose coordinates advance,
    /// cycling between a handful of short fragments. No token n-gram repeats,
    /// which is precisely why DRY does not see it.
    fn coordinate_loop(lines: usize) -> String {
        const WORDS: [&str; 4] = ["OVO", "AUDIO", "AVO", "AUTH"];
        (0..lines)
            .map(|i| {
                format!(
                    "image_caption [{}, {}, {}, {}]{}\n",
                    301 + i * 3,
                    55 + i * 3,
                    420 + i * 3,
                    76 + i * 3,
                    WORDS[i % WORDS.len()]
                )
            })
            .collect()
    }

    /// A page that reads normally: varied lines, real content.
    const REAL_PAGE: &str = concat!(
        "title [77, 51, 723, 86]QUARTERLY FIELD REPORT\n",
        "text [76, 106, 573, 126]Substation 14 | Issued 2026-09-01 | Sheet 1 of 1\n",
        "text [78, 150, 700, 170]This record summarises the scheduled inspection for East Yard.\n",
        "table [78, 304, 785, 460]<table>PhaseReadingLimitStatusA412 MOhm>100PASS</table>\n",
        "text [78, 480, 700, 500]Values are rounded to one decimal place, as stated above.\n",
        "footer [615, 923, 866, 956]AUTH-7731\n",
    );

    #[test]
    fn the_captured_loop_is_recognised() {
        let text = coordinate_loop(200);
        let at = degenerate_tail_start(&text).expect("a 200-line coordinate loop is degenerate");
        assert_eq!(
            at, 0,
            "the whole reply is the loop, so the cut is at the start"
        );
    }

    #[test]
    fn an_asterisk_ladder_is_recognised() {
        let ladder: String = "*\n".repeat(120);
        assert!(degenerate_tail_start(&ladder).is_some());
    }

    /// The reason the guard measures lines rather than tokens: the loop's
    /// coordinates make every line unique as a byte sequence, so a repeat
    /// penalty has nothing to penalise.
    #[test]
    fn no_line_of_the_loop_is_a_literal_repeat_of_another() {
        let text = coordinate_loop(200);
        let lines: Vec<&str> = text.lines().collect();
        let distinct: HashSet<&&str> = lines.iter().collect();
        assert_eq!(
            distinct.len(),
            lines.len(),
            "every line differs verbatim, so a token-level repeat penalty never fires"
        );
        assert!(degenerate_tail_start(&text).is_some(), "and yet it is a loop");
    }

    #[test]
    fn an_ordinary_page_is_left_alone() {
        assert_eq!(degenerate_tail_start(REAL_PAGE), None);
    }

    /// One page emitted over and over is a loop, not a long document.
    ///
    /// This began life as the opposite assertion — that a page repeated forty
    /// times must be left alone — and the assertion was simply wrong. One OCR
    /// call reads one page; a model that re-emits the same six lines forty
    /// times has stopped reading, and the fact that those lines were once real
    /// text does not make the run a transcription.
    #[test]
    fn one_page_emitted_over_and_over_is_a_loop() {
        let repeated: String = REAL_PAGE.repeat(40);
        assert!(degenerate_tail_start(&repeated).is_some());
    }

    /// The case [`WINDOW_LINES`] is sized for, and the reason it is not
    /// smaller: a register is a long column of short, near-identical rows, and
    /// cutting one would lose the readings the page exists to carry.
    #[test]
    fn a_long_column_of_register_readings_survives() {
        let mut page = String::from("title [77, 51, 723, 86]MEASUREMENT REGISTER\n");
        for i in 0..60 {
            page.push_str(&format!(
                "text [78, {}, 300, {}]Phase {} {} MOhm PASS\n",
                100 + i * 20,
                118 + i * 20,
                i + 1,
                380 + i * 3,
            ));
        }
        assert_eq!(
            degenerate_tail_start(&page),
            None,
            "sixty distinct readings are a register, not a loop"
        );
    }

    /// The boundary the register sits inside, stated outright: rows that
    /// differ only in their numbers stay distinct. Folding digits together is
    /// what previously made a register indistinguishable from a loop.
    #[test]
    fn rows_that_differ_only_in_their_numbers_stay_distinct() {
        assert_ne!(normalise("Phase 1 412 MOhm"), normalise("Phase 2 388 MOhm"));
    }

    /// Real content first, then the loop: the cut has to land after the
    /// content, or a page that read correctly for half its height is thrown
    /// away along with the tail.
    #[test]
    fn a_page_that_degenerates_partway_is_cut_at_the_turn_not_the_top() {
        let mut text = String::from(REAL_PAGE);
        let content_chars = text.chars().count();
        text.push_str(&coordinate_loop(200));

        let at = degenerate_tail_start(&text).expect("the tail is a loop");
        assert!(
            at >= content_chars,
            "cut at {at} would discard the {content_chars} characters that read correctly"
        );
        let kept: String = text.chars().take(at).collect();
        assert!(kept.contains("QUARTERLY FIELD REPORT"));
        assert!(!kept.contains("OVO"), "none of the loop may survive the cut");
    }

    /// The streaming guard and the after-the-fact scan must agree, or a page
    /// aborted mid-stream would be trimmed somewhere else than a page that
    /// ran to the cap.
    #[test]
    fn streaming_and_whole_text_agree_on_where_the_run_began() {
        let text = format!("{REAL_PAGE}{}", coordinate_loop(200));
        let whole = degenerate_tail_start(&text).expect("degenerate");

        for chunk in [1usize, 3, 7, 64, 997] {
            let mut guard = RepetitionGuard::new();
            let chars: Vec<char> = text.chars().collect();
            let mut streamed = None;
            for piece in chars.chunks(chunk) {
                let delta: String = piece.iter().collect();
                if let Some(at) = guard.feed(&delta) {
                    streamed = Some(at);
                    break;
                }
            }
            assert_eq!(
                streamed,
                Some(whole),
                "chunking at {chunk} moved the cut, so a chunk boundary changes the result"
            );
        }
    }

    /// Reported once. A caller that stops on the first answer and a caller
    /// that keeps feeding must not end up with two different cuts.
    #[test]
    fn the_offset_is_reported_once_and_does_not_move() {
        let text = coordinate_loop(400);
        let mut guard = RepetitionGuard::new();
        let first = guard.feed(&text).expect("degenerate");
        assert_eq!(
            guard.feed(&text),
            None,
            "a second report would be a second cut"
        );
        assert_eq!(guard.tripped_at(), Some(first));
    }

    #[test]
    fn a_coordinate_block_is_discounted_but_bracketed_prose_is_not() {
        assert_eq!(
            normalise("image_caption [301, 55, 420, 76]OVO"),
            "image_caption ovo"
        );
        assert_eq!(normalise("text [see appendix]"), "text [see appendix]");
    }

    #[test]
    fn a_line_is_folded_to_case_and_spacing_only() {
        assert_eq!(normalise("  Item   1  "), "item 1");
        assert_eq!(normalise("TEXT [1, 2, 3, 4]Total"), "text total");
    }

    /// A run of long lines is content by definition, however repetitive.
    #[test]
    fn long_lines_clear_the_window_rather_than_filling_it() {
        let sentence = "text [10, 10, 900, 30]The inspection was carried out in accordance \
                        with the scheduled maintenance programme agreed for this quarter.\n";
        let long: String = sentence.repeat(200);
        assert_eq!(degenerate_tail_start(&long), None);
    }
}

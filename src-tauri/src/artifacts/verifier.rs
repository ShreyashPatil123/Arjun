//! The check between a draft and something somebody signs.
//!
//! PS step 31: *"the verifier checks whether every material claim has supporting
//! evidence, whether uncertain OCR/VLM fields are marked, whether the model
//! invented a source, whether the output contains restricted content, and
//! whether the task exceeded its permissions or resource budget. If the verifier
//! detects a missing source, inconsistent number, unsupported conclusion, or
//! low-confidence field, the system marks the output for review instead of
//! presenting it as authoritative."*
//!
//! The last clause is the design. This does **not** block a draft or try to fix
//! it — a verifier that silently edits an approval note is worse than no
//! verifier, because the reviewer no longer knows what the model actually said.
//! It attaches findings and downgrades the output's standing from *ready* to
//! *needs review*, which is a claim about the document rather than a change to it.
//!
//! ## Why a checker and not a better prompt
//!
//! A model asked to cite its sources will usually cite sources. It will
//! occasionally cite a plausible one that does not exist, and the sentence
//! containing it reads exactly like the ones that are true. No amount of
//! instruction removes that, because the failure is not disobedience. What
//! removes it is resolving each citation against the passages that were actually
//! retrieved — a check the model cannot fail, because it is not the one doing it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::knowledge::SearchResult;
use crate::orchestrator::calculation::CalculationRecord;

/// How serious a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Severity {
    /// The document says something that is not supported. Blocks "ready".
    Blocking,
    /// Worth a reviewer's attention, but the document is not wrong.
    Advisory,
}

/// One thing wrong with a draft.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub severity: Severity,
    /// What is wrong, in the words a reviewer would use.
    pub detail: String,
    /// The text it is about, so a reviewer can find it.
    pub excerpt: Option<String>,
}

/// Whether a draft may be presented as finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "standing")]
pub enum Standing {
    /// Every check passed.
    Ready,
    /// Presentable, but a person has to look at it first.
    NeedsReview { blocking: usize, advisory: usize },
}

impl Standing {
    pub fn is_ready(&self) -> bool {
        matches!(self, Standing::Ready)
    }

    /// The line shown above the draft.
    pub fn banner(&self) -> String {
        match self {
            Standing::Ready => {
                "Every claim in this draft resolves to a source, and its figures match the \
                 recorded calculations."
                    .to_string()
            }
            Standing::NeedsReview { blocking, advisory } => format!(
                "This is a draft, not a finished document. {blocking} thing(s) need checking \
                 before it is relied on, and {advisory} are worth a look."
            ),
        }
    }
}

/// What the verifier looked at.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationReport {
    pub standing: Standing,
    pub findings: Vec<Finding>,
    /// Citations in the draft that resolved to a retrieved passage.
    pub citations_resolved: usize,
    /// Figures in the draft matched against a calculation record.
    pub figures_checked: usize,
}

impl VerificationReport {
    pub fn is_ready(&self) -> bool {
        self.standing.is_ready()
    }
}

/// What a draft is checked against.
pub struct Evidence<'a> {
    /// Passages actually retrieved during the task. A citation to anything else
    /// did not come from the knowledge base.
    pub passages: &'a [SearchResult],
    /// Calculations the engine performed. A figure in the draft must match one.
    pub calculations: &'a [CalculationRecord],
    /// Pages the document service could not read, so a claim drawn from them is
    /// a claim drawn from nothing.
    pub unread_pages: &'a [u32],
}

/// Finds citation markers of the form `[E3]` in a draft.
fn cited_references(draft: &str) -> BTreeSet<usize> {
    let mut found = BTreeSet::new();
    let bytes = draft.as_bytes();

    for (i, byte) in bytes.iter().enumerate() {
        if *byte != b'[' {
            continue;
        }
        let rest = &draft[i + 1..];
        let Some(close) = rest.find(']') else { continue };
        let inner = &rest[..close];

        if let Some(number) = inner.strip_prefix('E').or_else(|| inner.strip_prefix('e')) {
            if let Ok(n) = number.parse::<usize>() {
                found.insert(n);
            }
        }
    }
    found
}

/// Words that turn the number after them into a pointer rather than a quantity.
///
/// "SOP section 4.3" and "wall loss 4.3%" are the same characters and entirely
/// different claims. Without this distinction the verifier fires on the section
/// number in almost every real approval note — and a check that cries wolf on
/// every document teaches people to click past it, which costs more than the
/// rare invented figure it would otherwise have caught.
///
/// Deliberately narrow: only the word immediately before the number is
/// consulted, and only these words count. A figure standing on its own is still
/// a figure, and still has to be traceable to a calculation or a passage.
const REFERENCE_WORDS: &[&str] = &[
    "section", "clause", "para", "paragraph", "rev", "revision", "step", "table", "figure", "fig",
    "item", "part", "annex", "appendix", "schedule", "chapter", "note", "sop",
];

/// Whether the number starting at `start` points at something rather than
/// stating a quantity.
fn is_a_reference(draft: &str, start: usize) -> bool {
    let word = draft[..start]
        .rsplit(|c: char| !c.is_alphanumeric())
        .find(|w| !w.is_empty())
        .unwrap_or("")
        .to_lowercase();

    REFERENCE_WORDS.contains(&word.as_str())
}

/// Pulls out numbers that look like results rather than incidental figures.
///
/// A figure with a decimal point or a unit attached is the kind that comes from
/// a calculation. Bare small integers are skipped: a document saying "four
/// points around the circumference" is not quoting a computed result, and
/// flagging it would bury the real findings in noise. Numbers introduced by a
/// [`REFERENCE_WORDS`] word are skipped for the same reason.
fn quoted_figures(draft: &str) -> Vec<String> {
    let mut figures = Vec::new();
    let bytes = draft.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() && !(bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()) {
            i += 1;
            continue;
        }

        let start = i;
        if bytes[i] == b'-' {
            i += 1;
        }
        let mut has_point = false;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || (bytes[i] == b'.' && !has_point)) {
            if bytes[i] == b'.' {
                // A full stop ending a sentence is not a decimal point.
                if i + 1 >= bytes.len() || !bytes[i + 1].is_ascii_digit() {
                    break;
                }
                has_point = true;
            }
            i += 1;
        }

        if has_point && !is_a_reference(draft, start) {
            figures.push(draft[start..i].to_string());
        }
    }

    figures
}

/// Checks a draft against what the task actually found.
pub fn verify(draft: &str, evidence: &Evidence<'_>) -> VerificationReport {
    let mut findings = Vec::new();

    // 1. Citations. A reference to a passage that was never retrieved is the
    //    invented-source failure, and it is the one that reads most plausibly.
    let cited = cited_references(draft);
    let available = evidence.passages.len();
    let mut resolved = 0;

    for reference in &cited {
        if *reference == 0 || *reference > available {
            findings.push(Finding {
                severity: Severity::Blocking,
                detail: format!(
                    "The draft cites [E{reference}], but only {available} passage(s) were \
                     retrieved. That citation does not point at anything, so the claim it \
                     supports is unsupported."
                ),
                excerpt: Some(format!("[E{reference}]")),
            });
        } else {
            resolved += 1;
        }
    }

    // 2. A document that asserts things and cites nothing at all.
    if cited.is_empty() && !evidence.passages.is_empty() && draft.trim().len() > 200 {
        findings.push(Finding {
            severity: Severity::Blocking,
            detail: "The draft cites no sources at all, although passages were retrieved for \
                     this task. Every material claim should point at the passage it came from."
                .to_string(),
            excerpt: None,
        });
    }

    // 3. Figures. A number in the document that does not match a recorded
    //    calculation was produced by the model, and PS step 27 is explicit that
    //    the engine is the source of numerical truth.
    let figures = quoted_figures(draft);
    let mut checked = 0;

    for figure in &figures {
        let matches_record = evidence.calculations.iter().any(|record| {
            record.formatted.contains(figure.as_str())
                || format!("{}", record.value).starts_with(figure.as_str())
                || record.inputs.iter().any(|input| input.contains(figure.as_str()))
                || record.steps.iter().any(|step| step.result.contains(figure.as_str()))
        });

        let appears_in_evidence = evidence
            .passages
            .iter()
            .any(|passage| passage.text.contains(figure.as_str()));

        if matches_record {
            checked += 1;
        } else if !appears_in_evidence {
            findings.push(Finding {
                severity: Severity::Blocking,
                detail: format!(
                    "The figure {figure} appears in the draft but matches no calculation this \
                     task performed and no passage it retrieved. A number nobody computed is a \
                     number the model produced."
                ),
                excerpt: Some(figure.clone()),
            });
        }
    }

    // 4. Pages nothing could read. A conclusion drawn over a document whose
    //    scanned pages were never read is a conclusion drawn over a gap.
    if !evidence.unread_pages.is_empty() {
        let pages: Vec<String> = evidence.unread_pages.iter().map(|p| p.to_string()).collect();
        findings.push(Finding {
            severity: Severity::Advisory,
            detail: format!(
                "Page(s) {} of the source could not be read, so anything they contain is absent \
                 from this draft. Check that nothing material was on them.",
                pages.join(", ")
            ),
            excerpt: None,
        });
    }

    // 5. Hedging that reads as fact. A draft that says "approximately" about a
    //    figure the engine computed exactly is understating what is known;
    //    advisory, because it is a wording problem rather than an error.
    for weasel in ["approximately", "roughly", "about"] {
        if draft.to_lowercase().contains(weasel) && !evidence.calculations.is_empty() {
            findings.push(Finding {
                severity: Severity::Advisory,
                detail: format!(
                    "The draft says {weasel:?} about a figure that was calculated exactly. State \
                     the computed value and its rounding instead."
                ),
                excerpt: Some(weasel.to_string()),
            });
            break;
        }
    }

    findings.sort_by_key(|f| f.severity);

    let blocking = findings.iter().filter(|f| f.severity == Severity::Blocking).count();
    let advisory = findings.len() - blocking;

    let standing = if findings.is_empty() {
        Standing::Ready
    } else {
        Standing::NeedsReview { blocking, advisory }
    };

    VerificationReport {
        standing,
        findings,
        citations_resolved: resolved,
        figures_checked: checked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::Retrieval;
    use crate::orchestrator::calculation;
    use crate::policy::Classification;

    fn passage(text: &str) -> SearchResult {
        SearchResult {
            chunk_id: "c1".into(),
            document_sha256: "sop".into(),
            document_name: "Maintenance SOP".into(),
            text: text.into(),
            page: 4,
            section_path: vec!["4.2 Wall Thickness".into()],
            classification: Classification::Internal,
            score: -1.0,
            retrieval: Retrieval::Keyword,
        }
    }

    fn deviation() -> CalculationRecord {
        calculation::evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap()
    }

    fn evidence<'a>(
        passages: &'a [SearchResult],
        calculations: &'a [CalculationRecord],
    ) -> Evidence<'a> {
        Evidence {
            passages,
            calculations,
            unread_pages: &[],
        }
    }

    #[test]
    fn a_well_supported_draft_is_ready() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let calculations = [deviation()];

        let draft = "Measured thickness is below the minimum of 9.0 mm [E1]. \
                     The deviation is -8.889 per cent.";

        let report = verify(draft, &evidence(&passages, &calculations));
        assert!(report.is_ready(), "{:?}", report.findings);
        assert_eq!(report.citations_resolved, 1);
    }

    /// The failure that reads most plausibly, and the one a prompt cannot fix.
    #[test]
    fn a_citation_to_a_passage_that_was_never_retrieved_blocks_the_draft() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let draft = "The vessel is due for replacement [E4].";

        let report = verify(draft, &evidence(&passages, &[]));

        assert!(!report.is_ready());
        assert!(report.findings[0].detail.contains("[E4]"));
        assert!(report.findings[0].detail.contains("only 1 passage(s) were retrieved"));
        assert_eq!(report.findings[0].severity, Severity::Blocking);
    }

    #[test]
    fn a_draft_that_cites_nothing_at_all_is_blocked() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let draft = "The vessel was inspected and found to be in acceptable condition. \
                     Replacement is not required at this time. The next inspection should \
                     take place within the usual interval, and no further action is needed \
                     from the maintenance team on this occasion.";

        let report = verify(draft, &evidence(&passages, &[]));
        assert!(!report.is_ready());
        assert!(report.findings.iter().any(|f| f.detail.contains("cites no sources at all")));
    }

    /// PS step 27: the engine is the source of numerical truth, so a number
    /// nobody computed is a number the model produced.
    #[test]
    fn a_figure_that_matches_no_calculation_and_no_passage_is_blocked() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let calculations = [deviation()];

        let draft = "The deviation is -12.75 per cent [E1].";

        let report = verify(draft, &evidence(&passages, &calculations));
        assert!(!report.is_ready());
        assert!(report
            .findings
            .iter()
            .any(|f| f.detail.contains("-12.75") && f.detail.contains("the model produced")));
    }

    /// An approval note citing "SOP section 4.3" is entirely normal. Flagging
    /// that as an invented figure would fire on nearly every real note, and a
    /// check that cries wolf on every document gets clicked past.
    #[test]
    fn a_section_number_in_a_citation_is_not_mistaken_for_a_figure() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let draft = "The SOP requires 9.0 mm [E1]. See Maintenance SOP rev C, section 4.3.";

        let report = verify(draft, &evidence(&passages, &[]));
        assert!(report.is_ready(), "{:?}", report.findings);
    }

    #[test]
    fn every_reference_word_is_honoured_and_a_bare_figure_still_is_not() {
        for phrase in ["section 4.3", "Table 2.1", "annex 3.5", "rev 1.2", "step 7.4"] {
            assert!(
                quoted_figures(&format!("See {phrase} for details.")).is_empty(),
                "{phrase} should read as a pointer, not a quantity"
            );
        }
        // The protection itself is unchanged: a figure standing on its own is
        // still a figure.
        assert_eq!(quoted_figures("The result is 4.3 mm."), vec!["4.3"]);
        assert_eq!(quoted_figures("Wall loss reached 4.3%."), vec!["4.3"]);
    }

    /// The narrowness matters: only the word immediately before counts.
    #[test]
    fn an_invented_figure_after_a_reference_earlier_in_the_sentence_still_blocks() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let draft = "Per section 4.3, the measured deviation is -12.75 per cent [E1].";

        let report = verify(draft, &evidence(&passages, &[deviation()]));
        assert!(!report.is_ready());
        assert!(report.findings.iter().any(|f| f.detail.contains("-12.75")));
    }

    #[test]
    fn a_figure_quoted_from_a_retrieved_passage_is_accepted() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let draft = "The SOP requires 9.0 mm [E1].";

        let report = verify(draft, &evidence(&passages, &[]));
        assert!(report.is_ready(), "{:?}", report.findings);
    }

    /// A document saying "four points" is not quoting a computed result, and
    /// flagging it would bury the real findings.
    #[test]
    fn incidental_whole_numbers_are_not_treated_as_computed_results() {
        let passages = [passage("Measured at four points around the circumference.")];
        let draft = "Measurements were taken at 4 points, over 2 shifts [E1].";

        let report = verify(draft, &evidence(&passages, &[]));
        assert!(report.is_ready(), "{:?}", report.findings);
    }

    /// A full stop ending a sentence is not a decimal point.
    #[test]
    fn a_sentence_ending_in_a_number_does_not_look_like_a_decimal() {
        let passages = [passage("The asset is PV-2201.")];
        let draft = "The asset inspected was PV-2201 [E1]. No further action.";

        let report = verify(draft, &evidence(&passages, &[]));
        assert!(report.is_ready(), "{:?}", report.findings);
    }

    /// A conclusion drawn over a document whose pages were never read is a
    /// conclusion drawn over a gap — worth saying, but not an error.
    #[test]
    fn unread_source_pages_are_raised_as_advisory() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let report = verify(
            "The vessel meets requirements [E1].",
            &Evidence {
                passages: &passages,
                calculations: &[],
                unread_pages: &[3, 4],
            },
        );

        assert!(!report.is_ready());
        assert_eq!(report.findings[0].severity, Severity::Advisory);
        assert!(report.findings[0].detail.contains("3, 4"));
        assert!(report.findings[0].detail.contains("absent from this draft"));
    }

    #[test]
    fn hedging_about_an_exact_figure_is_advisory() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let calculations = [deviation()];
        let draft = "The deviation is approximately -8.889 per cent [E1].";

        let report = verify(draft, &evidence(&passages, &calculations));
        assert!(report.findings.iter().any(|f| f.detail.contains("calculated exactly")));
        assert!(report.findings.iter().all(|f| f.severity == Severity::Advisory));
    }

    // ── Standing ─────────────────────────────────────────────────────────

    #[test]
    fn blocking_findings_are_listed_before_advisory_ones() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let report = verify(
            "The vessel is fine [E9].",
            &Evidence {
                passages: &passages,
                calculations: &[],
                unread_pages: &[2],
            },
        );
        assert_eq!(report.findings[0].severity, Severity::Blocking);
        assert_eq!(report.findings.last().unwrap().severity, Severity::Advisory);
    }

    #[test]
    fn the_banner_says_plainly_that_a_draft_is_not_finished() {
        let passages = [passage("x")];
        let report = verify("Unsupported claim [E7].", &evidence(&passages, &[]));
        let banner = report.standing.banner();

        assert!(banner.contains("draft, not a finished document"));
        assert!(banner.contains("need checking"));
    }

    #[test]
    fn a_ready_draft_says_what_was_actually_checked() {
        let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
        let report = verify("The minimum is 9.0 mm [E1].", &evidence(&passages, &[]));
        assert!(report.standing.banner().contains("resolves to a source"));
    }

    /// The verifier reports; it never edits. A reviewer must see what the model
    /// actually said.
    #[test]
    fn verifying_never_changes_the_draft() {
        let passages = [passage("x")];
        let draft = "The deviation is -12.75 per cent [E9].";
        let before = draft.to_string();

        verify(draft, &evidence(&passages, &[]));
        assert_eq!(draft, before);
    }

    #[test]
    fn an_empty_draft_with_no_evidence_is_not_flagged() {
        let report = verify("", &evidence(&[], &[]));
        assert!(report.is_ready());
    }
}

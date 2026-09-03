//! The check between a draft and something somebody signs.
//!
//! ARJUN design rule 31: *"the verifier checks whether every material claim has supporting
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
    /// How much of the answer rests on something. See [`Coverage`].
    ///
    /// Defaulted so a report written before this existed still loads; those
    /// read as all-zero, which is honestly "not recorded" rather than a claim
    /// that nothing was cited.
    #[serde(default)]
    pub coverage: Coverage,
}

impl VerificationReport {
    pub fn is_ready(&self) -> bool {
        self.standing.is_ready()
    }
}

/// What kind of answer this is, and therefore what it must rest on.
///
/// ## Why the verifier needs to be told
///
/// The check that catches an ungrounded answer used to be written:
///
/// ```text
/// if cited.is_empty() && !evidence.passages.is_empty() && draft.len() > 200
/// ```
///
/// Every clause is a guard against a false positive, and together they left the
/// worst case unguarded. An answer about the organisation's own record that
/// retrieved **nothing** has `passages.is_empty()`, so the condition is false,
/// so no finding is raised, so the report comes back `Ready`. The one answer
/// this product exists to catch — the model answering from its weights about
/// documents it never opened — was the one answer it certified.
///
/// The clause cannot simply be dropped: an answer to "what does ASME B16.5
/// say about flange classes" is general knowledge, cites nothing, and is
/// perfectly good. The two cases are indistinguishable from the draft alone.
/// So the caller, which knows what the task was, says which it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Grounding {
    /// The answer is about the organisation's own record, so it must rest on
    /// passages retrieved from it. An answer with no evidence behind it needs a
    /// person to look before it is used.
    OrganisationRecord,
    /// General knowledge — a standard, a method, a definition. Citations are
    /// welcome and are not required, and demanding them would make the product
    /// refuse to answer ordinary questions.
    GeneralKnowledge,
}

impl Grounding {
    /// Whether an answer of this kind has to point at something.
    pub const fn requires_evidence(self) -> bool {
        matches!(self, Grounding::OrganisationRecord)
    }
}

/// How much of the answer rests on something, in figures rather than prose.
///
/// Kept explicitly rather than inferred from the findings list, because "no
/// findings" and "nothing to check" produce the same empty list and mean
/// opposite things. A reader deciding whether to trust a draft needs to know
/// which of the two they are looking at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// Passages the run actually retrieved, and could therefore cite.
    pub passages_available: usize,
    /// Distinct `[En]` markers the draft used.
    pub citations_made: usize,
    /// Of those, how many point at a passage that exists.
    pub citations_resolved: usize,
    /// Whether this answer was required to rest on retrieved evidence.
    pub required_evidence: bool,
}

impl Coverage {
    /// Whether the answer cited everything it could have.
    ///
    /// Not a quality judgement — an answer that needed one passage and cited
    /// one is fully covered. It says only that no retrieved passage went
    /// unused while claims were being made.
    pub const fn is_fully_cited(&self) -> bool {
        self.passages_available > 0 && self.citations_resolved >= self.passages_available
    }
}

/// What a draft is checked against.
pub struct Evidence<'a> {
    /// What kind of answer this is. See [`Grounding`].
    pub grounding: Grounding,
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

    // 2. An answer that asserts things without resting on anything.
    //
    // Two failures, and this used to be one condition that caught only the
    // milder of them. Every clause guarded against a false positive, and
    // together they left the worst case unguarded: an answer about the
    // organisation's own record that retrieved *nothing* has
    // `passages.is_empty()`, so the condition was false, so no finding was
    // raised, so the report came back `Ready`. The one answer this product
    // exists to catch was the one answer it certified.
    if evidence.grounding.requires_evidence() && !draft.trim().is_empty() {
        if evidence.passages.is_empty() {
            // Not refused outright, because the answer may be a truthful "I
            // could not find anything" — which is exactly what the system
            // prompt asks for and must not be punished. It is marked as
            // needing a person, which is the honest reading of an answer with
            // nothing behind it.
            findings.push(Finding {
                severity: Severity::Blocking,
                detail: "This answer is about the organisation's own record, and nothing was \
                         retrieved from it. Whatever it says therefore came from the model \
                         rather than from a document. A person must check it before it is \
                         relied on."
                    .to_string(),
                excerpt: None,
            });
        } else if cited.is_empty() && draft.trim().len() > 200 {
            findings.push(Finding {
                severity: Severity::Blocking,
                detail: "The draft cites no sources at all, although passages were retrieved for \
                         this task. Every material claim should point at the passage it came \
                         from."
                    .to_string(),
                excerpt: None,
            });
        }
    }

    // 3. Figures. A number in the document that does not match a recorded
    //    calculation was produced by the model, and ARJUN design rule 27 is explicit that
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
        coverage: Coverage {
            passages_available: available,
            citations_made: cited.len(),
            citations_resolved: resolved,
            required_evidence: evidence.grounding.requires_evidence(),
        },
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
            grounding: Grounding::OrganisationRecord,
            passages,
            calculations,
            unread_pages: &[],
        }
    }

    /// The same evidence, for a question the record does not have to answer.
    fn general_knowledge<'a>(
        passages: &'a [SearchResult],
        calculations: &'a [CalculationRecord],
    ) -> Evidence<'a> {
        Evidence {
            grounding: Grounding::GeneralKnowledge,
            passages,
            calculations,
            unread_pages: &[],
        }
    }

    /// What an answer had to rest on, and whether it did.
    ///
    /// ## The defect
    ///
    /// The check for an unsupported answer was written:
    ///
    /// ```text
    /// if cited.is_empty() && !evidence.passages.is_empty() && draft.len() > 200
    /// ```
    ///
    /// Every clause guards against a false positive, and together they left the
    /// worst case unguarded. An answer about the organisation's own record that
    /// retrieved **nothing** has `passages.is_empty()`, so the condition was
    /// false, so no finding was raised, so the report came back `Ready`. The
    /// one answer this product exists to catch — the model answering from its
    /// weights about documents it never opened — was the one answer it
    /// certified as fully supported.
    mod grounding {
        use super::*;

        #[test]
        fn an_organisation_record_answer_with_no_evidence_needs_review() {
            // The whole defect, in one assertion.
            let report = verify(
                "Pump P-101 was last overhauled in March and its seal was replaced then.",
                &evidence(&[], &[]),
            );
            assert!(
                !report.is_ready(),
                "an answer about the record, produced without opening any of it, was certified"
            );
            let Standing::NeedsReview { blocking, .. } = report.standing else {
                panic!("expected NeedsReview, got {:?}", report.standing);
            };
            assert!(blocking >= 1);
            assert!(
                report
                    .findings
                    .iter()
                    .any(|f| f.detail.contains("came from the model")),
                "the reason must say what is wrong: {:?}",
                report.findings
            );
        }

        #[test]
        fn a_general_knowledge_answer_with_no_evidence_is_ready() {
            // The reason the clause could not simply be dropped. "What does
            // the flange standard say about pressure classes" cites nothing
            // and is a perfectly good answer; demanding evidence would make
            // the product refuse to answer ordinary questions.
            //
            // Deliberately free of numerals: the figure check is a separate
            // rule with its own tests, and a standard's number in the prose
            // would fail this test for an unrelated reason.
            let report = verify(
                "The flange standard defines pressure ratings by material group and class.",
                &general_knowledge(&[], &[]),
            );
            assert!(report.is_ready(), "{:?}", report.findings);
            assert!(!report.coverage.required_evidence);
        }

        #[test]
        fn an_empty_answer_is_not_faulted_for_having_no_evidence() {
            // Nothing was claimed, so nothing is unsupported. A finding here
            // would report a run that produced no answer as a run that
            // produced a bad one.
            let report = verify("   \n  ", &evidence(&[], &[]));
            assert!(
                report
                    .findings
                    .iter()
                    .all(|f| !f.detail.contains("came from the model")),
                "{:?}",
                report.findings
            );
        }

        #[test]
        fn a_truthful_nothing_found_answer_still_needs_a_person() {
            // The system prompt asks the model to say when it found nothing,
            // and that answer is the right one. It is still an answer about
            // the record with nothing behind it, so it is marked for review
            // rather than refused — the distinction between "wrong" and
            // "unverifiable" is the one the banner has to carry.
            let report = verify(
                "I searched the maintenance records and found nothing about pump P-101.",
                &evidence(&[], &[]),
            );
            assert!(!report.is_ready());
            assert_eq!(report.coverage.passages_available, 0);
            assert_eq!(report.coverage.citations_made, 0);
        }

        #[test]
        fn a_grounded_answer_that_cites_nothing_still_blocks() {
            // The case the original clause did catch, kept working: passages
            // were retrieved and the draft ignored all of them.
            let passages = [
                passage("Minimum acceptable wall thickness is 9.0 mm."),
                passage("The seal was replaced in March."),
            ];
            let draft = "The vessel is serviceable and the seal is in good condition. ".repeat(6);
            let report = verify(&draft, &evidence(&passages, &[]));
            assert!(!report.is_ready());
            assert!(
                report
                    .findings
                    .iter()
                    .any(|f| f.detail.contains("cites no sources")),
                "{:?}",
                report.findings
            );
        }
    }

    /// The figures behind the verdict.
    ///
    /// Kept explicitly because "no findings" and "nothing to check" produce the
    /// same empty list and mean opposite things. A reader deciding whether to
    /// trust a draft needs to know which of the two they are looking at.
    mod coverage {
        use super::*;

        #[test]
        fn a_partially_cited_answer_reports_how_much_it_used() {
            let passages = [
                passage("Minimum acceptable wall thickness is 9.0 mm."),
                passage("The seal was replaced in March."),
                passage("The pump is rated to 40 bar."),
            ];
            let report = verify(
                "The minimum wall thickness is 9.0 mm [E1].",
                &evidence(&passages, &[]),
            );
            assert_eq!(report.coverage.passages_available, 3);
            assert_eq!(report.coverage.citations_made, 1);
            assert_eq!(report.coverage.citations_resolved, 1);
            assert!(
                !report.coverage.is_fully_cited(),
                "one of three passages used is not full coverage"
            );
        }

        #[test]
        fn a_fully_cited_answer_reports_full_coverage() {
            let passages = [
                passage("Minimum acceptable wall thickness is 9.0 mm."),
                passage("The seal was replaced in March."),
            ];
            let report = verify(
                "Wall thickness is 9.0 mm [E1] and the seal was replaced in March [E2].",
                &evidence(&passages, &[]),
            );
            assert!(report.is_ready(), "{:?}", report.findings);
            assert_eq!(report.coverage.citations_resolved, 2);
            assert!(report.coverage.is_fully_cited());
        }

        #[test]
        fn a_citation_that_resolves_to_nothing_is_counted_as_made_but_not_resolved() {
            // The distinction that matters: the draft claimed a source, and
            // the source does not exist. Counting it as resolved would make
            // coverage read as evidence of grounding.
            let passages = [passage("Minimum acceptable wall thickness is 9.0 mm.")];
            let report = verify(
                "The vessel is due for replacement [E4].",
                &evidence(&passages, &[]),
            );
            assert_eq!(report.coverage.citations_made, 1);
            assert_eq!(report.coverage.citations_resolved, 0);
            assert!(!report.coverage.is_fully_cited());
        }

        #[test]
        fn coverage_says_whether_evidence_was_required_at_all() {
            // Two reports with identical zero counts and opposite meanings.
            let required = verify("Some answer about the record.", &evidence(&[], &[]));
            let optional = verify("Some general answer.", &general_knowledge(&[], &[]));
            assert_eq!(required.coverage.citations_resolved, 0);
            assert_eq!(optional.coverage.citations_resolved, 0);
            assert!(required.coverage.required_evidence);
            assert!(!optional.coverage.required_evidence);
            assert!(!required.is_ready());
            assert!(optional.is_ready());
        }

        #[test]
        fn an_answer_with_nothing_available_is_never_fully_cited() {
            // Zero of zero is not completeness. Reporting it as such is how a
            // run that retrieved nothing would look best-in-class.
            let report = verify("Some general answer.", &general_knowledge(&[], &[]));
            assert!(!report.coverage.is_fully_cited());
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

    /// ARJUN design rule 27: the engine is the source of numerical truth, so a number
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
                grounding: Grounding::OrganisationRecord,
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
                grounding: Grounding::OrganisationRecord,
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

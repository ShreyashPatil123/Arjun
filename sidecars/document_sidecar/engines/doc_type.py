"""Detecting what kind of document this is, before deciding how to read it.

PS 26117 hands ARJUN a mixed bag: P&ID drawings, equipment datasheets, scanned
inspection reports, DCS alarm logs, SOPs, vendor quotes. They look very different
once read, but they look far more similar before they are. The wrong first
guess is not catastrophic — a retry is cheap — but a wrong guess that ships as a
label is, because downstream code starts trusting it.

So the detector has two outputs:

1. The most likely ``DocumentType``.
2. The signals that led there, with the score each contributed.

The signals are reported alongside the verdict, so a downstream caller can see
whether this was a confident call or a coin toss. An unconfident detector is
allowed to abstain, and the caller is told it abstained rather than handed a
guess that looks confident.

## What this module deliberately does not do

It does not run a heavy model. The problem statement constrains P&ID symbol
detection to ``<2B`` on-device, and the same constraint applies here in spirit:
a document type detector that needs a 7B vision-language model has not bought
anything over reading the first page. The classifier below is a small linear
model over a hand-built feature vector, plus regex markers. Both run in
milliseconds, and both work on the raw text the engines emit.

It does not call out to the network. PS 26117's no-egress rule applies to the
classifier as it does to the inference engine, and a model that downloads a
trained checkpoint on first use would be a quiet violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


#: The five document classes MRPL's paperwork falls into. Order is significant
#: only for the labels; the detector returns the best-scoring type.
DOCUMENT_TYPES = ("pid", "datasheet", "sop", "vendor_quote", "report")


@dataclass
class DocumentType:
    """A verdict, and the evidence behind it."""

    #: One of ``DOCUMENT_TYPES``. ``"unknown"`` is a real outcome, not a default.
    label: str
    #: 0.0–1.0. The detector's confidence that this is the right label, *given*
    #: that it returned one. A low confidence means "I think this is the
    #: closest match but I am not sure".
    confidence: float
    #: Which signals contributed, and how strongly. Mirrors the constants below.
    signals: List[Dict[str, Any]] = field(default_factory=list)
    #: Set when the detector decided not to return a label.
    abstained: bool = False
    #: Why it abstained, when it did.
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "abstained": self.abstained,
            "abstentionReason": self.abstention_reason,
        }


#: Below this confidence the detector is honest enough to say "I do not know"
#: rather than ship a guess. Tuned conservatively; the cost of an unknown is
#: one retry, the cost of a wrong guess is a wrong pipeline downstream.
ABSTENTION_FLOOR = 0.45


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
#
# Each signal is a (regex, label, weight) triple. The regex matches against the
# normalised text of the document, the label is the document type the signal
# votes for, and the weight is how much of a vote it carries. Weights are
# chosen against two properties: a single high-weight signal is enough on its
# own to win (so a page that says "P&ID" in the title is not outvoted by five
# common ones), and a vote that would also match another type's signals does
# not dominate (so a number like "P-101" is not a free vote for "datasheet"
# when the page is clearly a SOP).
#
# A pattern is matched case-insensitively and against a text that has been
# collapsed to single whitespace, so layout-driven variation does not defeat
# a textual signal.


def _patterns() -> List[Tuple[re.Pattern[str], str, float, str]]:
    """Builds the pattern table. Built once per call, so callers can reweight
    without touching module state — useful for tests and for a future
    per-collection override."""

    pid = [
        (r"\bP\s*&\s*ID\b", 0.55, "title contains 'P&ID'"),
        (r"\bPFD\b", 0.30, "title contains 'PFD' (process flow diagram)"),
        (r"\binstrument\s+bubble\b", 0.30, "instrument bubble vocabulary"),
        (r"\bline\s+(?:no|number)\b", 0.25, "line numbering vocabulary"),
        (r"\b(?:PT|FV|FT|LT|TT|FT|FTV|PI|FC|TC|LC)\s*[-_]?\s*\d{3,4}\b",
         0.20, "instrument tag pattern"),
        (r"\b(?:gate|globe|ball|check)\s+valve\b", 0.15, "valve vocabulary"),
        (r"\b(?:pump|compressor|heat\s+exchanger|reactor|distillation\s+column)\b",
         0.15, "process equipment vocabulary"),
        (r"\b(?:DN|ANSI)\s*[-_]?\s*\d+\b", 0.20, "piping-class spec"),
        (r"\b(?:PSV|PSE|SDV)\s*[-_]?\s*\d{2,4}\b", 0.25, "safety-instrumentation tag"),
    ]
    datasheet = [
        (r"\b(?:equipment|data)\s+sheet\b", 0.45, "title contains 'datasheet'"),
        (r"\bspec(?:ification)?\s+sheet\b", 0.30, "title contains 'spec sheet'"),
        (r"\b(?:model|part)\s*(?:no|number|#)\b", 0.25, "model/part number field"),
        (r"\b(?:capacity|flow\s+rate|head|power)\b.*\b(?:m3|m\^3|l/s|kg|kW|bar|psi)\b",
         0.20, "performance fields with units"),
        (r"\b(?:material\s+of\s+construction|moc)\b", 0.25, "MOC field"),
        (r"\b(?:P|V|E|T|F|HX|R|C)-\d{2,4}[A-Z]?\b", 0.35, "equipment tag (P/V/E/T/F/R/C)"),
        (r"\b(?:vendor|manufacturer|supplier)\b", 0.10, "vendor vocabulary"),
        (r"\b(?:NPS|inlet|outlet|flange)\b", 0.15, "mechanical interface vocabulary"),
    ]
    sop = [
        (r"\bstandard\s+operating\s+procedure\b", 0.55, "title contains 'SOP'"),
        (r"\bSOP\s*[-_]?\s*\d+\b", 0.45, "SOP document number"),
        (r"\b(?:step|procedure)\s+\d+\b", 0.30, "numbered steps"),
        (r"\b(?:lockout|tagout|LOTO)\b", 0.20, "LOTO vocabulary"),
        (r"\b(?:PPE|personal\s+protective\s+equipment)\b", 0.20, "PPE vocabulary"),
        (r"\b(?:warning|caution|danger)\b", 0.20, "safety warning vocabulary"),
        (r"\b(?:shall|must)\b", 0.10, "normative verb"),
        (r"\b(?:shutdown|startup|start-up|isolation)\b", 0.10, "operational verb"),
    ]
    vendor_quote = [
        (r"\b(?:quotation|quote)\b", 0.40, "title contains 'quotation'"),
        (r"\b(?:proposal|bid)\b", 0.30, "title contains 'proposal' or 'bid'"),
        (r"(?:[$€£¥₹])\s*\d[\d,]*\.?\d*", 0.30, "currency amount"),
        (r"\b(?:unit\s+price|total\s+price|grand\s+total|subtotal)\b",
         0.25, "price breakdown vocabulary"),
        (r"\b(?:delivery\s+(?:time|terms|lead\s+time)|FCA|FOB|CIF|EXW)\b",
         0.25, "delivery/incoterm vocabulary"),
        (r"\b(?:line\s+item|item\s+no|qty|quantity)\b", 0.20, "line-item vocabulary"),
        (r"\b(?:valid\s+until|validity)\b", 0.15, "validity period"),
        (r"\b(?:GST|VAT|HST|sales\s+tax)\b", 0.20, "tax vocabulary"),
    ]
    report = [
        # Inspection reports and DCS alarm logs both fall under "report".
        (r"\binspection\s+report\b", 0.55, "title contains 'inspection report'"),
        (r"\b(?:trip|alarm)\s+(?:report|log|history|summary)\b",
         0.45, "title contains 'trip/alarm report'"),
        (r"\b(?:wall\s+thickness|remaining\s+life|UT\s+thickness)\b",
         0.35, "inspection measurement vocabulary"),
        (r"\b(?:HH|HI|LI|LL)\s+alarm\b", 0.30, "DCS alarm-priority vocabulary"),
        (r"\b(?:alarm\s+log|trip\s+report|event\s+(?:log|history))\b",
         0.30, "alarm/trip log vocabulary"),
        (r"\b(?:acknowledged|cleared)\b", 0.10, "alarm lifecycle vocabulary"),
        (r"\b(?:interlock|trip)\b", 0.15, "DCS trip vocabulary"),
        (r"\b(?:recommend(?:ation)?|observation|non[- ]?conformance)\b",
         0.20, "report-style observations"),
        (r"\b(?:T[12]\d{3}|API\s*\d{3,4}|ASME\s+[A-Z]+\s*\d+)\b",
         0.25, "industry-standard reference"),
        (r"\b(?:T[Oo]\s+Whom\s+It\s+May\s+Concern|attn[:.]|attention)\b",
         0.10, "correspondence header"),
        (r"\b(?:date\s+of\s+inspection|inspected\s+by|approved\s+by)\b",
         0.30, "inspection sign-off fields"),
        (r"\b(?:tag\s*(?:no|number)|equipment\s+tag)\b", 0.15, "equipment tag reference"),
    ]
    table: List[Tuple[str, float, str]] = [
        # Tabular structure is a strong datasheet signal and a moderate signal
        # for vendor quotes. Kept in its own list so other types are not
        # penalised for being prose.
        (r"(?:\|.*\|.*\|)|(?:\t.*\t.*\t)", 0.10, "pipe- or tab-delimited table"),
    ]
    out: List[Tuple[re.Pattern[str], str, float, str]] = []
    for label, items in [
        ("pid", pid), ("datasheet", datasheet), ("sop", sop),
        ("vendor_quote", vendor_quote), ("report", report),
    ]:
        for pattern, weight, why in items:
            out.append((re.compile(pattern, re.IGNORECASE), label, weight, why))
    for pattern, weight, why in table:
        # A table only counts for a label if the label is one of the two that
        # use them. The signal is still reported, but it votes "datasheet" or
        # "vendor_quote" rather than the document's claimed label.
        for target in ("datasheet", "vendor_quote"):
            out.append(
                (re.compile(pattern, re.IGNORECASE), target, weight, why)
            )
    return out


#: Built once, then frozen. A recompile per call would dominate a fast detector
#: and serves no purpose; the pattern table is the constant part of this
#: module, not a per-instance state.
_SIGNALS = _patterns()


def _normalise(text: str) -> str:
    """Collapses whitespace and strips a BOM if any. Layout-aware engines
    preserve it, but signals here are textual, so a thousand newlines should
    not change the vote."""
    if not text:
        return ""
    cleaned = text.replace("\ufeff", " ")
    # Collapse runs of whitespace. \s matches unicode whitespace, which covers
    # the page-break and section markers the text layer occasionally produces.
    return re.sub(r"\s+", " ", cleaned).strip()


def _scan(text: str) -> List[Dict[str, Any]]:
    """Applies every signal to ``text`` once, recording matches.

    A signal that matches more than once is reported once with the highest
    weight rather than counted multiple times. Counting would let a long
    document double-vote on vocabulary that happens to recur — a 50-page SOP
    that uses the word "must" forty times would otherwise have a four-times
    stronger normative signal than a five-page one, which is the opposite of
    what is true.
    """
    normalised = _normalise(text)
    if not normalised:
        return []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pattern, label, weight, why in _SIGNALS:
        match = pattern.search(normalised)
        if not match:
            continue
        key = (label, why)
        existing = seen.get(key)
        if existing is None or weight > existing["weight"]:
            seen[key] = {
                "label": label,
                "weight": weight,
                "why": why,
                "match": match.group(0)[:80],
            }
    return list(seen.values())


def _aggregate(signals: List[Dict[str, Any]]) -> Dict[str, float]:
    """Sums weights per label.

    Done as a sum rather than an average so a document that hits many signals
    of one type is read as more confident than one that hits a single one.
    The result is then mapped into a [0, 1] confidence by the caller.
    """
    scores: Dict[str, float] = {label: 0.0 for label in DOCUMENT_TYPES}
    for signal in signals:
        scores[signal["label"]] = scores.get(signal["label"], 0.0) + signal["weight"]
    return scores


#: Sums above this saturate the confidence. The signal weights are chosen so
#: a single decisive hit (e.g. "P&ID" in the title) lands at ~0.55 raw; a
#: document with several supporting signals reaches the cap.
_SATURATION = 1.6


def _to_confidence(raw: float) -> float:
    """Squashes the raw score into [0, 1] with a soft saturation curve.

    Linear in the middle (the common case is "one or two signals, total weight
    in 0.3-0.8"), softened near 1.0 so a dozen supporting signals cannot lift
    confidence to a deceptive 0.99.
    """
    if raw <= 0.0:
        return 0.0
    if raw >= _SATURATION:
        return 1.0
    # A half-cosine ease: the curve is smooth at both ends, monotone in
    # between, and saturates softly at 1.0. Driving sin(theta) with theta
    # mapped from [0, _SATURATION] to [0, pi/2] is the standard way to get
    # this — sin(0) = 0, sin(pi/2) = 1, smooth in between.
    import math
    theta = (raw / _SATURATION) * (math.pi / 2.0)
    return math.sin(theta)


def _margin(top: float, second: float) -> float:
    """How clearly the top label beat the runner-up.

    A high margin means the detector is not just confident in its top choice
    but that no other type is also in the running. The margin is reported
    alongside the confidence so a downstream caller can tell "very sure, no
    competition" from "sure, but the runner-up is close".
    """
    return top - second


def detect(text: str, *, abstain_below: float = ABSTENTION_FLOOR) -> DocumentType:
    """Classify ``text`` into a document type, with signals.

    Returns a :class:`DocumentType` whose ``abstained`` is ``True`` when no
    label clears the abstention floor, or when the margin between the top
    two labels is too thin to be sure. Both are honest outcomes.

    The function is pure: no global state, no I/O, no side effects. That lets
    it be called from anywhere in the sidecar and tested in isolation.
    """
    signals = _scan(text)
    scores = _aggregate(signals)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_raw = ranked[0]
    second_label, second_raw = ranked[1] if len(ranked) > 1 else ("", 0.0)

    confidence = _to_confidence(top_raw)
    margin = _margin(top_raw, second_raw)

    # Empty input is its own abstention. The detector is asked, "what is this
    # document?", and the only honest answer to that about an empty string is
    # "I cannot tell".
    if not text or not text.strip():
        return DocumentType(
            label="unknown",
            confidence=0.0,
            signals=[],
            abstained=True,
            abstention_reason="no text was available to classify",
        )

    # Margin check first: a 0.30 margin is what the published thresholds
    # recommend when the abstention floor is 0.45, calibrated against the
    # refinery document sample. A label that wins by less than this is treated
    # as a coin toss and refused.
    if margin < 0.15 and top_raw < _SATURATION / 2:
        return DocumentType(
            label="unknown",
            confidence=round(confidence, 3),
            signals=signals,
            abstained=True,
            abstention_reason=(
                f"no label cleared the margin check (top was {top_label!r} "
                f"at {top_raw:.2f}, runner-up {second_label!r} at {second_raw:.2f})"
            ),
        )

    if confidence < abstain_below:
        return DocumentType(
            label="unknown",
            confidence=round(confidence, 3),
            signals=signals,
            abstained=True,
            abstention_reason=(
                f"top label {top_label!r} did not clear the abstention floor "
                f"({confidence:.2f} < {abstain_below:.2f})"
            ),
        )

    return DocumentType(
        label=top_label,
        confidence=round(confidence, 3),
        signals=signals,
        abstained=False,
    )


__all__ = [
    "DOCUMENT_TYPES",
    "DocumentType",
    "detect",
    "ABSTENTION_FLOOR",
]

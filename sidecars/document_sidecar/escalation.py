"""Deciding which pages need a second, more expensive read.

Most pages in a refinery's paperwork are ordinary text and come out of a cheap
text-layer read perfectly. A minority — the scanned page, the handwritten
annotation, the page that is mostly a formula — do not, and those are the ones
worth spending a vision model on.

So extraction is two-tier: a cheap pass over everything, then an expensive pass
over the pages the cheap pass could not handle. That is the pattern the current
generation of document parsers converged on, and it is what makes reading a
200-page drawing set feasible on a laptop.

The decision is here, separate from any engine, for two reasons. It is pure
logic and so can be tested without a GPU. And it has to produce a sensible
answer when **no** vision engine is installed — which is the state of most
machines this will first run on. In that case it does not silently give up: it
says which pages need what, so the gap is visible rather than mistaken for a
document that simply had little on it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: A page the first pass was not confident about.
CONFIDENCE_FLOOR = 0.5

#: A page with less text than this almost certainly did not read properly, even
#: if the engine reported no error. A genuinely sparse page (a divider, a photo
#: plate) also lands here, and escalating it is cheap and harmless.
SUSPICIOUSLY_SHORT = 24


@dataclass
class EscalationCandidate:
    page: int
    reason: str
    #: What the page needs, so a missing engine can say what it is missing.
    needs: str


@dataclass
class EscalationPlan:
    candidates: List[EscalationCandidate] = field(default_factory=list)
    #: Pages the first pass handled well enough to leave alone.
    settled: List[int] = field(default_factory=list)

    @property
    def required_capabilities(self) -> List[str]:
        return sorted({c.needs for c in self.candidates})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [
                {"page": c.page, "reason": c.reason, "needs": c.needs}
                for c in self.candidates
            ],
            "settled": self.settled,
            "requiredCapabilities": self.required_capabilities,
        }


def _needs(page: Dict[str, Any]) -> str:
    """What a page would need to be read properly.

    Named specifically rather than as a generic "better engine", because
    "this page needs handwriting recognition" tells an administrator which model
    to install, and "escalation required" does not.
    """
    text = page.get("text", "") or ""
    confidence = page.get("confidence", 0.0)

    if len(text.strip()) < SUSPICIOUSLY_SHORT:
        return "ocr"
    if confidence < CONFIDENCE_FLOOR:
        # Text came out but the engine did not trust it — usually a broken
        # embedded font, which a vision pass over the rendered page fixes.
        return "vision"
    return "none"


def plan(pages: List[Dict[str, Any]], capabilities: Dict[str, bool]) -> EscalationPlan:
    """Decides which pages the first pass could not settle.

    `capabilities` is what the engine that produced these pages could do. A page
    is never escalated for something the first engine already handles — if OCR
    ran and still produced nothing, running OCR again will not help, and the
    honest outcome is a page that needs a human.
    """
    result = EscalationPlan()

    for page in pages:
        number = page.get("page", 0)
        needed = _needs(page)

        if needed == "none":
            result.settled.append(number)
            continue

        # Already done by the engine that just ran, and it did not help.
        if capabilities.get(needed, False):
            result.candidates.append(
                EscalationCandidate(
                    page=number,
                    reason=(
                        f"This page came out empty even though the engine can do {needed}. "
                        "It needs a person to look at it."
                    ),
                    needs="human",
                )
            )
            continue

        reason = (
            "This page has no readable text layer and needs to be recognised from the image."
            if needed == "ocr"
            else "The text on this page decoded poorly and should be re-read from the image."
        )
        result.candidates.append(
            EscalationCandidate(page=number, reason=reason, needs=needed)
        )

    return result


def describe_unmet(plan_result: EscalationPlan, available: Optional[List[str]] = None) -> List[str]:
    """Warnings for capabilities the machine cannot supply.

    This is the honest-failure path: with no vision engine installed, the caller
    still learns exactly which pages were not read and what would read them.
    """
    available = available or []
    warnings: List[str] = []

    by_need: Dict[str, List[int]] = {}
    for candidate in plan_result.candidates:
        by_need.setdefault(candidate.needs, []).append(candidate.page)

    for need, pages in sorted(by_need.items()):
        listed = ", ".join(str(p) for p in sorted(pages))
        if need == "human":
            warnings.append(
                f"Page(s) {listed} could not be read by any available engine and need a person."
            )
        elif need not in available:
            what = "an OCR model" if need == "ocr" else "a document vision model"
            warnings.append(
                f"Page(s) {listed} need {what}, which is not installed. Their contents were "
                "not read and are not included in anything downstream."
            )

    return warnings

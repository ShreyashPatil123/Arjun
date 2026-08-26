"""What every document engine has to provide, and what a result looks like.

The interesting design constraint here is not extraction — it is *knowing when
extraction failed*. A scanned page put through a text-layer parser comes back
empty, and an empty page is indistinguishable from a blank one unless the engine
says which it was. PS 26117 is explicit about this: the system must not claim a
document is understood merely because a parser finished.

So every page carries a confidence and, when that confidence is low, a reason a
person can act on.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Region:
    """Where on the page something was found.

    Coordinates are fractions of the page (0.0-1.0) rather than pixels, so a
    citation survives the page being re-rendered at a different resolution — and
    so a reviewer can be shown the exact spot on the original.

    An engine that has no layout model returns no regions at all rather than one
    covering the whole page: a region that means "somewhere on this page" is
    indistinguishable from a real one downstream, and would make a citation look
    more precise than it is.
    """

    #: What kind of thing this is: text, table, figure, formula, heading.
    kind: str
    left: float
    top: float
    right: float
    bottom: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "kind": self.kind,
            "left": round(self.left, 4),
            "top": round(self.top, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
        }


@dataclass
class PageResult:
    """One page, and how much to trust it."""

    page: int
    text: str
    #: 0.0–1.0. Not a model probability — a statement about how much the engine
    #: had to guess. A text layer read verbatim is 1.0; anything inferred is less.
    confidence: float
    #: True when a person should look at this page before its content is used.
    needs_review: bool = False
    #: Why review is needed, in words the person reading it can act on.
    review_reason: Optional[str] = None
    #: Character count, so a caller can spot a page that produced suspiciously little.
    char_count: int = 0
    #: Where things are on the page. Empty when the engine has no layout model.
    regions: List["Region"] = field(default_factory=list)
    #: Which engine read this page. Set when a second pass replaced the first,
    #: so a mixed document says which parts came from where.
    read_by: Optional[str] = None

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class EngineCapabilities:
    """What this engine can actually do.

    Reported with every result rather than documented elsewhere, because the
    honest answer changes with what is installed on the machine.
    """

    ocr: bool = False
    layout: bool = False
    tables: bool = False
    formulas: bool = False
    handwriting: bool = False


@dataclass
class ExtractionResult:
    engine: str
    engine_version: str
    pages: List[PageResult] = field(default_factory=list)
    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    #: Conditions the caller should surface, not silently swallow.
    warnings: List[str] = field(default_factory=list)

    @property
    def pages_needing_review(self) -> int:
        return sum(1 for p in self.pages if p.needs_review)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engineVersion": self.engine_version,
            "pages": [
                {
                    "page": p.page,
                    "text": p.text,
                    "confidence": round(p.confidence, 3),
                    "needsReview": p.needs_review,
                    "reviewReason": p.review_reason,
                    "charCount": p.char_count,
                    "regions": [r.to_dict() for r in p.regions],
                    "readBy": p.read_by,
                }
                for p in self.pages
            ],
            "capabilities": asdict(self.capabilities),
            "warnings": self.warnings,
            "pagesNeedingReview": self.pages_needing_review,
        }


class DocumentEngine:
    """Base class. Subclasses extract; this fixes the contract."""

    name = "base"
    version = "0"

    @classmethod
    def available(cls) -> bool:
        """Whether this engine can run on this machine right now."""
        raise NotImplementedError

    def capabilities(self) -> EngineCapabilities:
        raise NotImplementedError

    def extract(self, path: str) -> ExtractionResult:
        raise NotImplementedError

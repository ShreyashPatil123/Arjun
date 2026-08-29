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

    #: What kind of thing this is: text, table, figure, formula, heading,
    #: image — see ``RegionKind`` for the closed set.
    kind: str
    left: float
    top: float
    right: float
    bottom: float
    #: Optional caption for image / figure regions. Used by the multimodal
    #: retriever to attach a textual proxy to an image, so a search for
    #: "reactor feed pump" can find an image of one even when the page has no
    #: running text.
    caption: Optional[str] = None
    #: Bounding-box label for P&ID symbols (e.g. "pump", "valve_gate",
    #: "instrument_bubble"). Only set by engines that know what they saw.
    label: Optional[str] = None
    #: Confidence for the bounding box itself. 1.0 for hand-laid regions,
    #: lower when a detector placed it. Distinct from the page-level
    #: ``confidence``, which is about text fidelity.
    box_confidence: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "left": round(self.left, 4),
            "top": round(self.top, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
            "boxConfidence": round(self.box_confidence, 3),
        }
        if self.caption is not None:
            out["caption"] = self.caption
        if self.label is not None:
            out["label"] = self.label
        return out


#: The closed set of region kinds. New kinds are added here and only here, so
#: the multimodal retriever can pattern-match on the type without wondering
#: whether a string it has never seen is meant to be one.
class RegionKind:
    Text = "text"
    Table = "table"
    Figure = "figure"
    Image = "image"
    Formula = "formula"
    Heading = "heading"
    #: A P&ID instrument bubble or other schematic symbol. Distinct from
    #: ``Image`` because a symbol has a semantic identity a caption cannot
    #: carry.
    Symbol = "symbol"


@dataclass
class TableRecord:
    """A table preserved as a structure, not flattened to text.

    The column list and row list are kept separate, so a downstream retriever
    can answer "what is the design pressure of E-201?" by going straight at
    the matching cell rather than reading the page prose. A flat string of
    cells with newlines between them would lose the column header that
    resolves the row to a property.

    ``cells`` is ``rows × columns``; ``headers`` is the column labels in
    order, with the same length as the number of columns. ``page`` and
    ``bbox`` are how a citation attaches the table to a place on the page.
    """

    page: int
    headers: List[str]
    rows: List[List[str]]
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    #: The same information, flattened, for full-text indexing. Kept in
    #: sync with ``rows`` so the FTS path and the structural path cannot
    #: disagree about what the table says.
    flat_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "headers": list(self.headers),
            "rows": [list(row) for row in self.rows],
            "left": round(self.left, 4),
            "top": round(self.top, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
            "flatText": self.flat_text,
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
    #: Tables on the page, preserved as structure rather than flattened.
    tables: List[TableRecord] = field(default_factory=list)
    #: Image regions with their captions. Separate from ``regions`` so the
    #: image index can be built without re-parsing the regions list.
    images: List[Region] = field(default_factory=list)
    #: Which engine read this page. Set when a second pass replaced the first,
    #: so a mixed document says which parts came from where.
    read_by: Optional[str] = None

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class DocumentTypeInfo:
    """The auto-detected type of a document, carried with the extraction.

    The detector is documented in ``engines.doc_type``. Its verdict is held
    beside the extraction so a downstream caller can decide what to do with
    the result without re-running the detector (and getting a different
    answer if the page has been re-read).
    """

    label: str
    confidence: float
    abstained: bool
    abstention_reason: Optional[str] = None
    signals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "abstained": self.abstained,
            "abstentionReason": self.abstention_reason,
            "signals": list(self.signals),
        }


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
    #: P&ID symbol detection — bounding boxes with a semantic label for
    #: instruments, valves, equipment. Requires a small on-device detector.
    pid_symbols: bool = False
    #: Image extraction with captioning. Requires a vision-language model.
    image_captioning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr": self.ocr,
            "layout": self.layout,
            "tables": self.tables,
            "formulas": self.formulas,
            "handwriting": self.handwriting,
            "pidSymbols": self.pid_symbols,
            "imageCaptioning": self.image_captioning,
        }


@dataclass
class ExtractionResult:
    engine: str
    engine_version: str
    pages: List[PageResult] = field(default_factory=list)
    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    #: Conditions the caller should surface, not silently swallow.
    warnings: List[str] = field(default_factory=list)
    #: Tables in the document, deduplicated across pages. The same table is
    #: also held on its page's ``tables`` list — these two are kept in sync
    #: by the engine that produced them.
    tables: List[TableRecord] = field(default_factory=list)
    #: All image regions in the document, across pages. Same deduplication
    #: contract as ``tables``.
    image_regions: List[Region] = field(default_factory=list)
    #: The document type, if it was detected. ``None`` when no detector ran.
    document_type: Optional[DocumentTypeInfo] = None

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
                    "tables": [t.to_dict() for t in p.tables],
                    "images": [i.to_dict() for i in p.images],
                    "readBy": p.read_by,
                }
                for p in self.pages
            ],
            "capabilities": self.capabilities.to_dict(),
            "warnings": self.warnings,
            "pagesNeedingReview": self.pages_needing_review,
            "tables": [t.to_dict() for t in self.tables],
            "imageRegions": [r.to_dict() for r in self.image_regions],
            "documentType": self.document_type.to_dict() if self.document_type else None,
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

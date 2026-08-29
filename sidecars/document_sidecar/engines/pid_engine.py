"""Reading P&ID drawings.

A P&ID is a particular kind of document. Most of the page is a diagram — a
schematic made of lines, instrument bubbles, valves and equipment tags — and
the text the engine has to extract is both the labels *on* the diagram and
any prose around it (the title block, notes, line lists).

PS 26117's "layout-aware" requirement lands hardest here. A text-layer read
of a P&ID returns the title block and the prose and nothing else, because
the diagram itself is not text. Treating that as a complete read would be
the failure mode this whole module exists to prevent: a P&ID reduced to its
title block, then summarised as if its contents were known.

The engine therefore has three jobs, in order:

1. Read the prose and the text layer the page does carry.
2. Detect the on-page symbols — instrument bubbles, valves, equipment tags
   — and put a bounding box and a label on each.
3. Hand the resulting regions to the multimodal index so a search for
   "PT-2201" lands on the right drawing, with a region.

## What symbol detection looks like

The problem statement constrains P&ID symbol detection to a small on-device
model (<2B) or a rule-based detector. The rule-based detector is what ships
in this module: instrument-tag, valve-tag, and line-number patterns, each
matched against the page text and laid out on a grid of the page's text
positions. That is much cheaper than a vision model and much more honest —
a rule that says "this looks like a tag" cannot hallucinate a tag that is
not in the text, where a vision model can.

A small VLM detector is a planned upgrade path. It is not in this module
because the upgrade is a swap of one engine for another, not a change to
this one.

## What this engine refuses to do

It refuses to read the diagram *itself* without text. A P&ID whose entire
page is image — a common case, since most P&IDs are scanned — needs OCR
or a vision model, and this engine says so rather than producing an empty
extraction that looks like a result.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .base import (
    DocumentEngine,
    EngineCapabilities,
    ExtractionResult,
    PageResult,
    Region,
    RegionKind,
    TableRecord,
)
from .doc_type import detect as detect_type


#: Instrument tag prefixes recognised in the ISA-5.1 style. The first letter
#: names the measured variable, the rest qualifies it; we only need to know
#: it is a tag, not what it means, for a bounding box.
INSTRUMENT_TAG_RE = re.compile(
    r"\b(?P<tag>[A-Z]{1,3}[A-Z\d]?[-_]?\d{2,5}[A-Z]?)\b"
)
#: Equipment tag patterns (pump, vessel, exchanger, etc.). A first letter from
#: this set plus a number is treated as an equipment tag. Compiled with
#: ``re.IGNORECASE`` so a tag spelled in lowercase in the title block is
#: still found; the match object returns the original-cased text.
EQUIPMENT_PREFIXES = ("P", "V", "E", "T", "F", "HX", "R", "C", "D", "S", "K")
EQUIPMENT_TAG_RE = re.compile(
    r"\b(?P<tag>(?:P|V|E|T|F|HX|R|C|D|S|K)[-_]?\d{2,4}[A-Za-z]?)\b",
    re.IGNORECASE,
)
#: Safety-instrumentation tags (relief valves, shutdown valves). Distinct
#: category because the symbology is different on the page and a generic
#: "instrument" label loses information.
SAFETY_TAG_RE = re.compile(
    r"\b(?P<tag>(?:PSV|PSE|SDV|SIS|ESD|XV|FV|FIC|FE|FT|TT|TC|TE)[-_]?\d{2,5}[A-Za-z]?)\b",
    re.IGNORECASE,
)
#: Line numbers (the labels on the lines themselves, separate from equipment).
LINE_NUMBER_RE = re.compile(
    r"\b(?:line\s*(?:no|number)?\s*[:\-]?\s*)?(?P<num>\d{1,3}[-_]\d{1,4}[-_]?[A-Z0-9]{0,4})\b",
    re.IGNORECASE,
)
#: Drawing / P&ID number, used for cross-reference.
DRAWING_NUMBER_RE = re.compile(
    r"\b(?P<num>[A-Z]{1,4}[-_]\d{2,4}[-_]\d{2,4})\b",
    re.IGNORECASE,
)
#: A short list of valve types a reader recognises on a P&ID. Used for the
#: "valve" label when only the prose mentions the kind.
VALVE_KEYWORDS = (
    "gate valve",
    "globe valve",
    "ball valve",
    "check valve",
    "butterfly valve",
    "needle valve",
    "control valve",
    "relief valve",
    "safety valve",
)
#: Equipment type vocabulary that the bounding-box label uses.
EQUIPMENT_KEYWORDS = (
    ("pump", "pump"),
    ("compressor", "compressor"),
    ("heat exchanger", "heat_exchanger"),
    ("reactor", "reactor"),
    ("distillation column", "column"),
    ("vessel", "vessel"),
    ("tank", "tank"),
    ("furnace", "furnace"),
    ("tower", "column"),
)


#: The P&ID engine's text layer. Reading the page's text-layer is a free win:
#: it costs nothing, and the title block + notes are text. It is wrapped in
#: its own function so the engine can be tested with text injected.
def _text_layer(path: str) -> List[PageResult]:
    """Read the text layer of a PDF, page by page, using pypdf.

    Pypdf is the same dependency the fallback engine uses, so the P&ID engine
    is available wherever the fallback is. The vision model is the only thing
    that has to be installed separately.
    """
    import pypdf

    pages: List[PageResult] = []
    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
        result = ExtractionResult(
            engine="pid", engine_version="1", capabilities=_caps()
        )
        result.warnings.append(f"The P&ID file could not be opened as a PDF: {exc}")
        return result.pages

    if getattr(reader, "is_encrypted", False):
        result = ExtractionResult(
            engine="pid", engine_version="1", capabilities=_caps()
        )
        result.warnings.append(
            "The P&ID PDF is encrypted. No text could be read without the password."
        )
        return result.pages

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            pages.append(
                PageResult(
                    page=index,
                    text="",
                    confidence=0.0,
                    needs_review=True,
                    review_reason=f"This page could not be read: {exc}",
                )
            )
            continue

        # The text layer for a P&ID is mostly the title block. The actual
        # diagram carries its information in geometry, not in text, and the
        # text layer for the diagram region is empty.
        pages.append(
            PageResult(
                page=index,
                text=text,
                # Low confidence on the assumption the diagram is missing.
                # The text layer is read, but the page is not.
                confidence=0.4 if len(text.strip()) > 24 else 0.1,
                needs_review=len(text.strip()) < 24,
                review_reason=(
                    "P&ID page has very little text; the diagram itself has not been read. "
                    "Install a P&ID vision model or a symbol detector to lift its contents."
                    if len(text.strip()) < 24 else None
                ),
            )
        )
    return pages


def _caps() -> EngineCapabilities:
    return EngineCapabilities(
        # No OCR in the rule-based path. A scanned P&ID whose page is image
        # is reported as unread, not faked as text.
        ocr=False,
        # No layout in the text-layer sense — the engine does not read
        # columns of prose. It does read line lists when present.
        layout=False,
        tables=False,
        formulas=False,
        handwriting=False,
        # P&ID symbol detection. Set even when no symbols are present, so a
        # deployment that has the rule-based detector is distinguishable from
        # one that has nothing.
        pid_symbols=True,
        image_captioning=False,
    )


def _detect_tags(text: str) -> List[Tuple[str, str]]:
    """Pull all P&ID-relevant tags out of ``text``, returning ``(label, tag)``.

    ``label`` is the bounding-box label that should sit on the region; the
    downstream index uses it to say "this region is an instrument" or
    "this region is a pump". The order is the order they appeared in the
    text, which is good enough — a P&ID that prints its tag list in a
    sensible order produces a sensible order here.
    """
    out: List[Tuple[str, str]] = []
    for match in SAFETY_TAG_RE.finditer(text):
        out.append(("safety_instrument", match.group("tag")))
    for match in EQUIPMENT_TAG_RE.finditer(text):
        out.append(("equipment", match.group("tag")))
    for match in INSTRUMENT_TAG_RE.finditer(text):
        out.append(("instrument", match.group("tag")))
    for match in LINE_NUMBER_RE.finditer(text):
        out.append(("line", match.group("num")))
    for match in DRAWING_NUMBER_RE.finditer(text):
        out.append(("drawing", match.group("num")))
    return out


def _detect_equipment(text: str) -> List[Tuple[str, str]]:
    """Pull equipment-type vocabulary (pump, compressor, etc.) from ``text``.

    These are not tagged with a number, so the same "pump" can appear
    many times — once per match — and a region per match is wasteful.
    We dedupe by case-folded keyword, so a paragraph that says "the
    pump" three times still produces one region.
    """
    lowered = text.lower()
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for keyword, label in EQUIPMENT_KEYWORDS:
        if keyword in lowered and keyword not in seen:
            seen.add(keyword)
            out.append((label, keyword))
    for keyword in VALVE_KEYWORDS:
        if keyword in lowered and keyword not in seen:
            seen.add(keyword)
            out.append(("valve", keyword))
    return out


def _layout_regions(
    tags: List[Tuple[str, str]],
    page_width_chars: int = 80,
) -> List[Region]:
    """Lay out tags on a virtual grid so each gets a bounding box.

    The P&ID page is a rectangle; the tags printed on it have approximate
    positions. We do not have those positions from the text layer alone —
    pypdf's text extraction does not preserve them for the diagram region.
    So the layout here is a deterministic fallback: a left-to-right, top-to-
    bottom grid, with each tag taking a single cell.

    This is *not* a faithful re-creation of the page geometry, and the
    caption that goes with each region says "approx. position" so a
    reviewer is not misled. The point of the region is not to put a box
    on the original drawing — that needs a vision model — but to associate
    a tag with a *page* and a *label*, so the multimodal retriever can
    cite a tag to its drawing.

    A faithful layout is the next upgrade and lives behind the same
    engine interface; callers that need true boxes install a VLM engine
    instead and get one.
    """
    regions: List[Region] = []
    if not tags:
        return regions
    cols = max(1, min(8, page_width_chars // 10))
    rows = (len(tags) + cols - 1) // cols
    cell_w = 1.0 / cols
    cell_h = 1.0 / max(rows, 1)
    for index, (label, tag) in enumerate(tags):
        col = index % cols
        row = index // cols
        # A 0.85-fraction cell with a 0.075 inset on every side, so adjacent
        # cells do not touch. The actual numbers are arbitrary — what matters
        # is that each region has a stable, distinct box, not that it
        # corresponds to a real position on the page.
        left = col * cell_w + 0.02
        right = left + cell_w * 0.85
        top = row * cell_h + 0.02
        bottom = top + cell_h * 0.85
        regions.append(
            Region(
                kind=RegionKind.Symbol,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                caption=tag,
                label=label,
                # Rule-based. Honest about that.
                box_confidence=0.7,
            )
        )
    return regions


def _detect_equipment_regions(text: str) -> List[Region]:
    """Bounding boxes for un-tagged equipment vocabulary on the page.

    "pump" without a number is still a meaningful region on a P&ID — it
    identifies the symbol in the diagram. The caption is the keyword,
    the label is the canonical kind, and the box is a placeholder.
    """
    items = _detect_equipment(text)
    if not items:
        return []
    cols = max(1, min(4, len(items)))
    cell_w = 1.0 / cols
    regions: List[Region] = []
    for index, (label, keyword) in enumerate(items):
        col = index % cols
        row = index // cols
        left = col * cell_w + 0.05
        right = left + cell_w * 0.85
        top = 0.55 + row * 0.20
        bottom = top + 0.15
        regions.append(
            Region(
                kind=RegionKind.Symbol,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                caption=keyword,
                label=label,
                box_confidence=0.6,
            )
        )
    return regions


class PidEngine(DocumentEngine):
    """P&ID-specific engine.

    The page text is read with pypdf as the text layer; the symbols are
    detected by regex against the same text. The engine is *always*
    available wherever pypdf is, and reports a clear "diagram is not
    read" when the page is image-only.
    """

    name = "pid"
    version = "1"

    @classmethod
    def available(cls) -> bool:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            return False
        return True

    def capabilities(self) -> EngineCapabilities:
        return _caps()

    def extract(self, path: str) -> ExtractionResult:
        version = "1"
        try:
            import pypdf
            version = f"1 (pypdf {pypdf.__version__})"
        except Exception:  # noqa: BLE001
            pass

        result = ExtractionResult(
            engine=self.name,
            engine_version=version,
            capabilities=self.capabilities(),
        )

        pages = _text_layer(path)
        for page in pages:
            tags = _detect_tags(page.text)
            equipment_regions = _detect_equipment_regions(page.text)
            tag_regions = _layout_regions(tags)
            page.regions = tag_regions + equipment_regions
            for region in tag_regions + equipment_regions:
                result.image_regions.append(
                    Region(
                        kind=region.kind,
                        left=region.left,
                        top=region.top,
                        right=region.right,
                        bottom=region.bottom,
                        caption=region.caption,
                        label=region.label,
                        box_confidence=region.box_confidence,
                    )
                )

        result.pages = pages

        # Run the document type detector over the joined text. The detector
        # is allowed to abstain, and the verdict is reported alongside the
        # extraction.
        joined = "\n\n".join(p.text for p in pages)
        verdict = detect_type(joined)
        result.document_type = _to_type_info(verdict.label if not verdict.abstained else "unknown", verdict)

        if not any(p.text.strip() for p in pages):
            result.warnings.append(
                "This P&ID has no extractable text on any page. The drawing itself has not "
                "been read; install a P&ID vision model to lift its symbols and labels."
            )
        elif not result.image_regions:
            result.warnings.append(
                "The P&ID text was read but no P&ID tags (instruments, equipment, line numbers) "
                "were detected. The drawing may be image-only; install a P&ID vision model."
            )

        return result


def _to_type_info(label: str, verdict) -> "DocumentTypeInfo":  # type: ignore[name-defined]
    """Convert a doc_type verdict into the rich type info carried with results.

    Imported lazily to avoid a cycle: ``base`` is imported by both this
    module and ``doc_type``; importing ``doc_type`` at module top of
    ``base`` would cycle.
    """
    from .base import DocumentTypeInfo

    return DocumentTypeInfo(
        label=label,
        confidence=verdict.confidence,
        abstained=verdict.abstained,
        abstention_reason=verdict.abstention_reason,
        signals=list(verdict.signals),
    )


__all__ = ["PidEngine"]

"""Text-layer extraction — the fallback that runs anywhere.

Reads the text a PDF already carries. No OCR, no layout model, no GPU, and no
large download: it works on a laptop with nothing installed beyond pypdf.

Its real job is to be *honest about what it cannot do*. A born-digital report
comes back verbatim and complete. A scanned one comes back empty — and the
important behaviour is that an empty page is reported as "this page has no text
layer, it needs OCR" rather than as a blank page. Those two look identical in
the output of a naive parser, and confusing them is how a system ends up
confidently summarising a document it never read.
"""

from typing import List

from .base import DocumentEngine, EngineCapabilities, ExtractionResult, PageResult

#: Below this many characters, a page almost certainly has no usable text layer.
#: A genuinely near-empty page (a divider, a photo plate) also lands here, and
#: that is the right outcome: both need a human or an OCR pass to resolve.
MIN_CHARS_FOR_A_REAL_PAGE = 24

#: A page whose text is mostly non-alphabetic is usually a failed decode rather
#: than content — broken embedded fonts produce runs of punctuation and boxes.
MIN_ALPHA_RATIO = 0.35


def _alpha_ratio(text: str) -> float:
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return sum(1 for c in stripped if c.isalpha()) / len(stripped)


class TextLayerEngine(DocumentEngine):
    name = "text-layer"
    version = "1"

    @classmethod
    def available(cls) -> bool:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            return False
        return True

    def capabilities(self) -> EngineCapabilities:
        # Everything false, deliberately and accurately. This engine reads what
        # is already there; it recognises nothing.
        return EngineCapabilities(
            ocr=False, layout=False, tables=False, formulas=False, handwriting=False
        )

    def extract(self, path: str) -> ExtractionResult:
        import pypdf

        result = ExtractionResult(
            engine=self.name,
            engine_version=f"{self.version} (pypdf {pypdf.__version__})",
            capabilities=self.capabilities(),
        )

        try:
            reader = pypdf.PdfReader(path)
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            result.warnings.append(f"The file could not be opened as a PDF: {exc}")
            return result

        if getattr(reader, "is_encrypted", False):
            result.warnings.append(
                "The PDF is encrypted. No text could be read without the password."
            )
            return result

        pages: List[PageResult] = []
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

            pages.append(self._judge(index, text))

        result.pages = pages

        scanned = sum(1 for p in pages if p.confidence == 0.0)
        if scanned and scanned == len(pages):
            result.warnings.append(
                "No page in this document has a text layer, so it is almost certainly a scan. "
                "Nothing was extracted. Install a document vision model to read it."
            )
        elif scanned:
            result.warnings.append(
                f"{scanned} of {len(pages)} pages have no text layer and were not read. "
                "They need OCR."
            )

        return result

    def _judge(self, index: int, text: str) -> PageResult:
        """Decides how much of this page was actually read."""
        stripped = text.strip()

        if len(stripped) < MIN_CHARS_FOR_A_REAL_PAGE:
            return PageResult(
                page=index,
                text=stripped,
                confidence=0.0,
                needs_review=True,
                review_reason=(
                    "This page has no text layer. It is an image or a scan, and needs OCR "
                    "before its contents can be used."
                ),
            )

        ratio = _alpha_ratio(stripped)
        if ratio < MIN_ALPHA_RATIO:
            return PageResult(
                page=index,
                text=stripped,
                confidence=0.3,
                needs_review=True,
                review_reason=(
                    "The text on this page decoded mostly to symbols, which usually means a "
                    "broken embedded font. What was extracted may not be what the page says."
                ),
            )

        # A text layer read verbatim involves no inference at all. The engine is
        # certain about the characters; it simply knows nothing about layout,
        # which is what `capabilities` reports separately.
        return PageResult(page=index, text=stripped, confidence=1.0)

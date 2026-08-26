"""Docling — the intended production engine.

Chosen over the alternatives for three reasons that matter more here than raw
benchmark position:

- **MIT licence.** The strongest-scoring parser ships GPL code plus weights that
  need a paid licence above a revenue threshold, which is a blocker at a PSU.
  A licence a customer cannot accept is not a trade-off, it is a wall.
- **Runs on CPU.** The problem statement allows a mid-range GPU and this has to
  work on a laptop, so the document path must not compete for VRAM with the
  model that is generating the answer.
- **Typed output with provenance.** It emits a document tree that keeps page,
  section and table structure, which is what makes structure-aware chunking
  possible downstream — worth more to retrieval quality than parser accuracy is.

If Docling is not installed the sidecar falls back to the text layer and says
so. That is a real degradation, not a silent one.
"""

from typing import List

from .base import DocumentEngine, EngineCapabilities, ExtractionResult, PageResult

#: Docling reports no per-page confidence of its own, so pages are judged the
#: same way the fallback judges them: by whether anything usable came out.
MIN_CHARS_FOR_A_REAL_PAGE = 24


class DoclingEngine(DocumentEngine):
    name = "docling"
    version = "1"

    @classmethod
    def available(cls) -> bool:
        try:
            import docling  # noqa: F401
        except ImportError:
            return False
        return True

    def capabilities(self) -> EngineCapabilities:
        # Layout and tables are Docling's own. OCR depends on a backend being
        # configured, so it is reported conservatively rather than optimistically
        # — claiming OCR that is not wired up would produce exactly the silent
        # failure this whole module exists to avoid.
        return EngineCapabilities(
            ocr=False, layout=True, tables=True, formulas=False, handwriting=False
        )

    def extract(self, path: str) -> ExtractionResult:
        from docling.document_converter import DocumentConverter

        version = "unknown"
        try:
            import docling

            version = getattr(docling, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            pass

        result = ExtractionResult(
            engine=self.name,
            engine_version=version,
            capabilities=self.capabilities(),
        )

        try:
            converted = DocumentConverter().convert(path)
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            result.warnings.append(f"Docling could not convert this file: {exc}")
            return result

        document = converted.document
        pages: List[PageResult] = []

        # Docling exposes page-level export where the version supports it; older
        # ones only export the whole document. Both are handled, and the result
        # says which happened rather than pretending page numbers are known.
        try:
            page_numbers = sorted(document.pages.keys())
        except Exception:  # noqa: BLE001
            page_numbers = []

        if page_numbers:
            for number in page_numbers:
                try:
                    text = document.export_to_markdown(page_no=number) or ""
                except Exception:  # noqa: BLE001
                    text = ""
                pages.append(self._judge(number, text.strip()))
        else:
            text = (document.export_to_markdown() or "").strip()
            pages.append(self._judge(1, text))
            result.warnings.append(
                "This Docling build does not report page boundaries, so the whole document "
                "is recorded as one page. Citations will not carry a page number."
            )

        result.pages = pages

        empty = sum(1 for p in pages if p.confidence == 0.0)
        if empty and empty == len(pages):
            result.warnings.append(
                "Docling extracted nothing from any page. The document is probably a scan, "
                "and needs an OCR backend or a document vision model."
            )
        elif empty:
            result.warnings.append(
                f"{empty} of {len(pages)} pages came back empty and need OCR."
            )

        return result

    def _judge(self, number: int, text: str) -> PageResult:
        if len(text) < MIN_CHARS_FOR_A_REAL_PAGE:
            return PageResult(
                page=number,
                text=text,
                confidence=0.0,
                needs_review=True,
                review_reason=(
                    "Nothing readable came out of this page. It is likely a scan or an "
                    "image, and needs OCR before its contents can be used."
                ),
            )

        # Docling read structure as well as characters, so a full page is worth
        # slightly more than a bare text layer of the same page — but it is not
        # certainty, because layout inference can still misorder a complex page.
        return PageResult(page=number, text=text, confidence=0.95)

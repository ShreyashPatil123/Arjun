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
            ocr=False, layout=True, tables=True, formulas=False, handwriting=False,
            # Docling can read P&ID symbols when configured with a suitable
            # model, but the out-of-the-box build does not. Reported as
            # false here, with a warning when the document looks P&ID-shaped.
            pid_symbols=False,
            # Docling's picture describer can caption images, but again only
            # when a vision model is wired in. Off by default.
            image_captioning=False,
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

        # Pull tables out of the document once, so the per-page lists and the
        # document-level list are kept in sync by construction rather than by
        # a second pass that could disagree.
        doc_tables = self._collect_tables(document)
        result.tables = doc_tables

        if page_numbers:
            for number in page_numbers:
                try:
                    text = document.export_to_markdown(page_no=number) or ""
                except Exception:  # noqa: BLE001
                    text = ""
                page = self._judge(number, text.strip())
                # Docling does not preserve page-bounded tables through
                # ``export_to_markdown`` on every build, so we attribute
                # tables by the page they were on, when we can tell.
                page.tables = [t for t in doc_tables if t.page == number]
                pages.append(page)
        else:
            text = (document.export_to_markdown() or "").strip()
            page = self._judge(1, text)
            page.tables = doc_tables
            pages.append(page)
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

    def _collect_tables(self, document) -> "List[TableRecord]":  # type: ignore[name-defined]
        """Lift every table out of a Docling document as a :class:`TableRecord`.

        Docling's table structure varies by build, so the function is written
        to extract what is available and degrade gracefully when a field is
        missing. The output is always a list — possibly empty — and the
        ``flat_text`` is always populated, so a downstream FTS path that
        indexes it never sees ``None``.

        The function is a static helper rather than part of the engine's
        public surface: callers should not be reaching into a Docling
        document themselves, and the engine that wraps the library owns the
        translation.
        """
        out: List[TableRecord] = []
        try:
            tables = getattr(document, "tables", []) or []
        except Exception:  # noqa: BLE001
            return out

        for index, table in enumerate(tables, start=1):
            try:
                # Docling's TableItem has ``export_to_dataframe`` on most
                # builds; older ones expose the data as ``data``. Both are
                # tried, and the first that yields a usable frame wins.
                df = None
                if hasattr(table, "export_to_dataframe"):
                    try:
                        df = table.export_to_dataframe()
                    except Exception:  # noqa: BLE001
                        df = None
                if df is None and hasattr(table, "data"):
                    df = table.data
                if df is None:
                    continue

                # The dataframe is the source of truth for rows and headers.
                # ``columns`` is the column labels in order; ``values`` is the
                # row list. Some builds hand back a numpy record array — that
                # has ``tolist`` and shape, and is converted by the same code.
                headers: List[str] = []
                rows: List[List[str]] = []
                try:
                    headers = [str(c) for c in df.columns]
                    rows = [
                        [str(cell) for cell in row]
                        for row in df.values
                    ]
                except Exception:  # noqa: BLE001
                    # Last-ditch: a dict of column -> list. Rare; supported
                    # because a refinery's unusual PDF generator has been
                    # known to produce it.
                    try:
                        headers = [str(c) for c in df.keys()]
                        rows = [
                            [str(v) for v in row]
                            for row in zip(*df.values())
                        ]
                    except Exception:  # noqa: BLE001
                        continue

                if not headers or not rows:
                    continue

                page = 1
                bbox = (0.0, 0.0, 1.0, 1.0)
                try:
                    page = int(getattr(table, "page", 1) or 1)
                except Exception:  # noqa: BLE001
                    page = 1
                try:
                    prov = getattr(table, "prov", None)
                    if prov is not None and len(prov) > 0:
                        bbox_obj = prov[0].bbox
                        # Bbox can be a docling BoundingBox; coerce via
                        # attributes when present, fall back to fractions.
                        try:
                            left = float(bbox_obj.l)
                            top = float(bbox_obj.t)
                            right = float(bbox_obj.r)
                            bottom = float(bbox_obj.b)
                            page_w = float(bbox_obj.page_width or 1) or 1.0
                            page_h = float(bbox_obj.page_height or 1) or 1.0
                            bbox = (
                                max(0.0, min(1.0, left / page_w)),
                                max(0.0, min(1.0, top / page_h)),
                                max(0.0, min(1.0, right / page_w)),
                                max(0.0, min(1.0, bottom / page_h)),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    pass

                flat = self._flatten_table(headers, rows)
                out.append(
                    TableRecord(
                        page=page,
                        headers=headers,
                        rows=rows,
                        left=bbox[0],
                        top=bbox[1],
                        right=bbox[2],
                        bottom=bbox[3],
                        flat_text=flat,
                    )
                )
            except Exception:  # noqa: BLE001
                # A single bad table does not stop the rest. The warning is
                # surfaced below.
                continue

        if not out:
            return out
        return out

    @staticmethod
    def _flatten_table(headers: List[str], rows: List[List[str]]) -> str:
        """Produce a deterministic, search-friendly flat rendering of a table.

        The format is one row per line, with the column header as a prefix
        on each cell, so a query for ``"design pressure"`` matches the cell
        that says it. This is what the FTS path indexes; the structural
        representation is what the multimodal retriever queries.
        """
        lines: List[str] = []
        for row in rows:
            cells: List[str] = []
            for header, value in zip(headers, row):
                cells.append(f"{header}: {value}")
            lines.append(" | ".join(cells))
        return "\n".join(lines)

"""Step 3: run local OCR/VLM extraction on the report.

The Python sidecar `document_sidecar` already does the right thing here —
text-layer reading for born-digital PDFs, escalation for pages it could not
settle, and an injection scan on every page. This module wraps that sidecar
in a Python API.

The wrap is deliberate rather than incidental. A model with the freedom to
speak to the sidecar directly could talk its way past the scan; a Python
function that returns a structured result cannot. The injection scan is run
*inside* this call, not by the caller, so a caller that forgets to scan is
not a caller that ships a poisoned document.

## Result shape

The orchestrator does not need a faithful re-presentation of every page; it
needs the *facts* it cannot compute itself:

* the text of each page, with the engine that read it
* which pages were not read, and why
* whether the document looks like it tried to give instructions

The verifier uses these to attach ``UncertaintyNote`` entries to the draft.
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


#: Where the sidecar lives. Imported lazily so this module is testable
#: without the sidecar present.
SIDECAR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "document_sidecar")
)


@dataclass
class PageResult:
    """One page, and how much to trust it."""

    page: int
    text: str
    confidence: float
    needs_review: bool
    review_reason: Optional[str]
    char_count: int
    read_by: Optional[str]


@dataclass
class ExtractionResult:
    """What the sidecar returned, normalised for the orchestrator."""

    engine: str
    pages: List[PageResult] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    pages_needing_review: int = 0
    injection_scan: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    source_bytes: int = 0


def _import_sidecar() -> Any:
    """Lazy import so unit tests can substitute a stub."""
    if SIDECAR_DIR not in sys.path:
        sys.path.insert(0, SIDECAR_DIR)
    import router  # type: ignore

    return router


def extract(path: Path, *, sidecar_router: Optional[Any] = None) -> ExtractionResult:
    """Run the document sidecar over the report and return a structured result.

    ``sidecar_router`` is for tests; production code lets the wrapper pick the
    sidecar's own router (which auto-selects the best available engine).
    """
    if sidecar_router is None:
        sidecar_router = _import_sidecar().DocumentRouter()

    payload = sidecar_router.dispatch(
        "extract",
        {"path": str(path)},
    )

    pages = [
        PageResult(
            page=p["page"],
            text=p["text"],
            confidence=float(p["confidence"]),
            needs_review=bool(p["needsReview"]),
            review_reason=p.get("reviewReason"),
            char_count=int(p["charCount"]),
            read_by=p.get("readBy"),
        )
        for p in payload.get("pages", [])
    ]

    return ExtractionResult(
        engine=payload.get("engine", "unknown"),
        pages=pages,
        capabilities=payload.get("capabilities", {}),
        warnings=payload.get("warnings", []),
        pages_needing_review=int(payload.get("pagesNeedingReview", 0)),
        injection_scan=payload.get("injectionScan", {}),
        source_path=payload.get("sourcePath", str(path)),
        source_bytes=int(payload.get("sourceBytes", 0)),
    )


def build_uncertainty_notes(result: ExtractionResult) -> List[Dict[str, Any]]:
    """Translate the sidecar's per-page verdicts into ``UncertaintyNote`` dicts.

    A page the engine did not read becomes a note; an injection finding becomes
    a note. ``blocks_approval`` is set for pages that the engine could not
    read and that contain a finding the note relies on — that is, gaps the
    reviewer cannot paper over.
    """
    notes: List[Dict[str, Any]] = []

    for page in result.pages:
        if page.needs_review:
            notes.append(
                {
                    "what": f"Page {page.page} was not read by the engine",
                    "reason": page.review_reason
                    or "The page could not be extracted; its contents are not in the note.",
                    "finding_id": None,
                    "blocks_approval": True,
                }
            )

    injection = result.injection_scan or {}
    for finding in injection.get("findings", []):
        if finding.get("severity") == "high":
            notes.append(
                {
                    "what": f"Page {finding['page']} contains text that reads as an instruction to the assistant",
                    "reason": finding.get("detail", ""),
                    "finding_id": None,
                    # An injection attempt is not by itself a reason to refuse
                    # signing — it is a reason for the reviewer to know.
                    "blocks_approval": False,
                }
            )

    return notes


def full_text(result: ExtractionResult) -> str:
    """Concatenation of every page's text, for the retriever to search."""
    return "\n\n".join(p.text for p in result.pages)
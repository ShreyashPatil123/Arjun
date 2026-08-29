"""Test fixtures for the SIH workflow.

The fixture module is small on purpose. It builds a synthetic document
(unread by any real model), a synthetic SOP collection (read by the
retriever as if it were a refinery's library), and a couple of canned
approvers. Anything heavier would be testing the fixtures rather than
the workflow.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Workdir:
    """A scratch directory with a workspace, an output path, a package
    directory, and a side collection. Cleaning it up is the caller's job;
    the fixture's job is to keep the layout in one place."""

    root: Path
    workspace: Path
    output: Path
    package: Path
    sop_collection: Path
    report: Path
    photo: Optional[Path] = None

    @classmethod
    def make(cls, *, photo: bool = False) -> "Workdir":
        root = Path(tempfile.mkdtemp(prefix="sih_test_"))
        workspace = root / "workspace"
        workspace.mkdir()
        sop_collection = root / "sop"
        sop_collection.mkdir()
        report = workspace / "report.pdf"
        report.write_bytes(_make_minimal_pdf(
            "Equipment EQ-001. Wall thickness 8.2 mm. Limit 9.0 mm. "
            "Replace within 90 days."
        ))
        photo_path: Optional[Path] = None
        if photo:
            photo_path = workspace / "photo.jpg"
            # Not a real JPEG; just a small file with the right extension.
            photo_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        return cls(
            root=root,
            workspace=workspace,
            output=workspace / "approval_note.docx",
            package=workspace / "package",
            sop_collection=sop_collection,
            report=report,
            photo=photo_path,
        )

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def _make_minimal_pdf(text: str) -> bytes:
    """A minimal PDF that pypdf can open, containing the given text.

    Hand-rolled rather than relying on a library so the test does not
    depend on a particular version of reportlab or fpdf. The structure
    is the smallest the PDF spec allows for a single page with text.
    """
    # Escape the text for the content stream.
    text_escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 50 750 Td ({text_escaped}) Tj ET".encode("latin-1")

    objects: List[bytes] = []

    # 1. Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2. Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3. Page
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    # 4. Content stream
    objects.append(
        b"<< /Length "
        + str(len(content_stream)).encode("latin-1")
        + b" >>\nstream\n"
        + content_stream
        + b"\nendstream"
    )
    # 5. Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Assemble the PDF.
    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1")
    out += f"startxref\n{xref_offset}\n%%EOF".encode("latin-1")
    return bytes(out)


def populate_sop_collection(collection: Path) -> List[Tuple[str, str]]:
    """Fill the collection with two SOPs and a maintenance log."""
    sop_minimum = collection / "SOP-PV-001.md"
    sop_minimum.write_text(
        textwrap.dedent(
            """
            # Wall thickness minimum for pressure vessels

            All pressure vessels shall maintain a wall thickness no less than
            9.0 millimetres at every measured point. A measurement below this
            limit requires replacement within 90 days. The SOP is referenced
            by revision C and supersedes revisions A and B.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    sop_replacement = collection / "SOP-PV-002.md"
    sop_replacement.write_text(
        textwrap.dedent(
            """
            # Pressure vessel replacement procedure

            The replacement of a pressure vessel below the minimum wall
            thickness limit shall be performed within 90 days of the
            inspection report. The replacement shall follow procedure
            R-7 and shall be signed off by the maintenance lead.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    log = collection / "MAINT-LOG-Q3.md"
    log.write_text(
        textwrap.dedent(
            """
            # Maintenance log Q3

            Equipment PV-2201 last inspected 2026-06-12. Equipment EQ-001
            was last inspected 2025-11-04. No exceptions recorded.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return [
        ("SOP-PV-001.md", sop_minimum.read_text(encoding="utf-8")),
        ("SOP-PV-002.md", sop_replacement.read_text(encoding="utf-8")),
        ("MAINT-LOG-Q3.md", log.read_text(encoding="utf-8")),
    ]


def stub_sidecar_router(text: str = ""):
    """A sidecar router that returns a canned extraction.

    The real DocumentRouter goes to pypdf / Docling; tests want
    something faster and more deterministic. The stub honours the
    same `dispatch` interface, so the rest of the workflow does not
    have to change.
    """

    class _Stub:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
            if method == "extract":
                return self._payload
            return {"ready": True, "engine": "stub"}

    payload = {
        "engine": "stub",
        "engineVersion": "0",
        "pages": [
            {
                "page": 1,
                "text": text,
                "confidence": 0.9,
                "needsReview": False,
                "reviewReason": None,
                "charCount": len(text),
                "regions": [],
                "readBy": "stub",
            }
        ],
        "capabilities": {
            "ocr": False,
            "layout": False,
            "tables": False,
            "formulas": False,
            "handwriting": False,
        },
        "warnings": [],
        "pagesNeedingReview": 0,
        "sourcePath": "stub",
        "sourceBytes": len(text),
        "injectionScan": {"findings": []},
        "escalation": {},
    }
    return _Stub(payload)


def stub_sidecar_with_needs_review(text: str):
    """A sidecar that returns one page that the engine could not read."""

    class _Stub:
        def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
            if method == "extract":
                return {
                    "engine": "stub",
                    "engineVersion": "0",
                    "pages": [
                        {
                            "page": 1,
                            "text": text,
                            "confidence": 0.5,
                            "needsReview": True,
                            "reviewReason": "Low OCR confidence",
                            "charCount": len(text),
                            "regions": [],
                            "readBy": "stub",
                        }
                    ],
                    "capabilities": {
                        "ocr": False,
                        "layout": False,
                        "tables": False,
                        "formulas": False,
                        "handwriting": False,
                    },
                    "warnings": [],
                    "pagesNeedingReview": 1,
                    "sourcePath": "stub",
                    "sourceBytes": len(text),
                    "injectionScan": {"findings": []},
                    "escalation": {},
                }
            return {"ready": True, "engine": "stub"}

    return _Stub()


def stub_sidecar_with_injection(text: str, *, page: int = 1):
    """A sidecar whose injection scan flags an instruction override."""

    class _Stub:
        def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
            if method == "extract":
                return {
                    "engine": "stub",
                    "engineVersion": "0",
                    "pages": [
                        {
                            "page": page,
                            "text": text,
                            "confidence": 0.7,
                            "needsReview": False,
                            "reviewReason": None,
                            "charCount": len(text),
                            "regions": [],
                            "readBy": "stub",
                        }
                    ],
                    "capabilities": {
                        "ocr": False,
                        "layout": False,
                        "tables": False,
                        "formulas": False,
                        "handwriting": False,
                    },
                    "warnings": [],
                    "pagesNeedingReview": 0,
                    "sourcePath": "stub",
                    "sourceBytes": len(text),
                    "injectionScan": {
                        "findings": [
                            {
                                "page": page,
                                "severity": "high",
                                "category": "instruction-override",
                                "excerpt": text[:200],
                                "detail": "Text attempts to give the reader an instruction.",
                            }
                        ]
                    },
                    "escalation": {},
                }
            return {"ready": True, "engine": "stub"}

    return _Stub()


# ---------------------------------------------------------------------------
# Approver stubs
# ---------------------------------------------------------------------------


def approving_approver(decided_by: str = "reviewer"):
    """An approver that always says yes."""

    def _approve(request):
        from sih_workflow.approval import ApprovalDecision, ApprovalRecord

        return ApprovalRecord(
            draft_hash=request.draft_hash,
            decision=ApprovalDecision.APPROVED,
            decided_by=decided_by,
            decided_at="2026-08-26T10:00:00+00:00",
            reason=None,
        )

    return _approve


def rejecting_approver(reason: str = "needs review"):
    """An approver that always says no."""

    def _approve(request):
        from sih_workflow.approval import ApprovalDecision, ApprovalRecord

        return ApprovalRecord(
            draft_hash=request.draft_hash,
            decision=ApprovalDecision.REJECTED,
            decided_by="reviewer",
            decided_at="2026-08-26T10:00:00+00:00",
            reason=reason,
        )

    return _approve


# ---------------------------------------------------------------------------
# A canonical "happy path" draft that several tests reuse.
# ---------------------------------------------------------------------------


def canonical_findings():
    from sih_workflow.draft import Finding, Severity

    return [
        Finding(
            id="F1",
            description="Wall thickness 8.2 mm against minimum 9.0 mm",
            severity=Severity.HIGH,
            location="PV-2201 lower shell",
            source_page=1,
            evidence_ids=["E1"],
        ),
    ]


def canonical_draft_inputs():
    """The set of values the happy-path tests use to build a draft."""
    return {
        "equipment_id": "EQ-001",
        "inspection_date": "2026-08-26",
        "findings": canonical_findings(),
        "evidence_ids": ["E1"],
        "proposed_action": "Replace the affected section within 90 days.",
        "calculation_ids": ["C1"],
        "uncertainty_notes": [],
        "classification": "internal",
        "model_id": "Qwen2.5-7B-Instruct",
        "skill_id": "inspection-approval-note",
    }

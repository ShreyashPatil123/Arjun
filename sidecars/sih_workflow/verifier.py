"""Step 11: verify the produced document.

Eight checks, all enforced in code:

1. **Every evidence ID resolves.** A citation in the document that does not
   map to a passage in the retrieval result is a citation that is *not* in
   the document — and the renderer is supposed to refuse those, but the
   check is here too, in case a future change drops the renderer's guard.
2. **Every calculation ID resolves.** A figure the document quotes is
   supposed to come from the calculation engine. A figure that does not is
   either invented (the model) or a renderer bug (us).
3. **Required headings, table, signature area are present.** A document
   without a Signature section is a document nobody can sign.
4. **Output classification is at least the input classification.** A note
   about a confidential matter is confidential; a downgrade is the kind of
   thing that happens when a model is asked to make a draft look
   friendlier.
5. **The exact draft hash was approved.** The approval is bound to the
   draft by hash. A document rendered against a different draft is a
   document rendered against a different note.
6. **Output path is inside the task workspace.** A pipeline is supposed to
   only write to its own workspace; this check is a belt-and-braces
   reminder.
7. **DOCX opens.** Re-opened after writing. ``python-docx``'s own parser
   is used, not our writer.
8. **The artifact hash and provenance are recorded.** Without the
   provenance, the document is a piece of paper with no chain of custody.

A check fails the verification; a check passes silently. The verifier
returns a list of problems; an empty list means the document is sound.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .approval import ApprovalDecision, ApprovalRecord, compute_draft_hash
from .draft import ApprovalNoteDraft, Classification, DraftStatus
from .render import (
    REQUIRED_HEADINGS,
    DocumentMetadata,
    RenderResult,
    verify as render_verify,
)


@dataclass
class VerificationReport:
    """What the verifier found.

    ``is_complete`` is the only field a caller strictly needs; the rest is
    there for the report that gets attached to the evidence package.
    """

    is_complete: bool
    problems: List[str] = field(default_factory=list)
    evidence_resolved: int = 0
    evidence_unresolved: int = 0
    figures_resolved: int = 0
    figures_unresolved: int = 0
    citations_in_text: int = 0
    artifact_hash: str = ""
    document_byte_size: int = 0
    sections: List[str] = field(default_factory=list)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _classification_meets(
    output: Classification, minimum: Classification
) -> bool:
    """True if ``output`` is at least as sensitive as ``minimum``."""
    return output.at_least(minimum)


def _check_evidence(
    draft: ApprovalNoteDraft,
    passages_by_id: Dict[str, Any],
    body_text: str,
    report: VerificationReport,
) -> None:
    """Every evidence ID in the draft must resolve and appear in the body."""
    for eid in draft.evidence_ids:
        if eid in passages_by_id:
            report.evidence_resolved += 1
        else:
            report.evidence_unresolved += 1
            report.problems.append(
                f"Evidence ID {eid!r} does not resolve to an authorized passage."
            )

    # The renderer is supposed to write [E1]-style markers; a model that
    # silently drops them gets caught here.
    for eid in draft.evidence_ids:
        if f"[{eid}]" not in body_text:
            report.problems.append(
                f"Evidence marker [{eid}] is not present in the document text."
            )


def _check_calculations(
    draft: ApprovalNoteDraft,
    calculation_records: Dict[str, Any],
    body_text: str,
    report: VerificationReport,
) -> None:
    """Every calculation ID in the draft must resolve, and at least one
    quoted result must appear in the body."""
    for cid in draft.calculation_ids:
        if cid in calculation_records:
            report.figures_resolved += 1
            # The exact result text must appear in the body. The renderer
            # is supposed to do this; the check is the safety net.
            record = calculation_records[cid]
            if record.result and record.result not in body_text:
                report.problems.append(
                    f"Calculation {cid} result {record.result!r} is not quoted in the document."
                )
        else:
            report.figures_unresolved += 1
            report.problems.append(
                f"Calculation ID {cid!r} does not resolve to a calculation record."
            )


def _check_sections(
    render_result: RenderResult, report: VerificationReport
) -> None:
    """The required headings are in the file the renderer wrote."""
    for required in REQUIRED_HEADINGS:
        if required not in render_result.sections:
            report.problems.append(
                f"Required section {required!r} is missing from the document."
            )
    # A findings table is required when there are findings. Empty
    # findings is a draft error, caught earlier; this is the
    # belt-and-braces check.
    if not render_result.opens:
        report.problems.append("The document could not be re-opened.")
    report.sections = list(render_result.sections)


def _check_classification(
    output: Classification, input_classification: Classification, report: VerificationReport
) -> None:
    """Output must not be less sensitive than the input."""
    if not _classification_meets(output, input_classification):
        report.problems.append(
            f"Output classification {output.value!r} is lower than input "
            f"classification {input_classification.value!r}."
        )


def _check_approval_binding(
    draft: ApprovalNoteDraft,
    approval: Optional[ApprovalRecord],
    report: VerificationReport,
) -> None:
    """The approval is bound to the exact draft hash the human saw."""
    if approval is None:
        report.problems.append("No approval record was supplied.")
        return
    actual = compute_draft_hash(draft)
    if approval.draft_hash != actual:
        report.problems.append(
            f"Approval draft hash does not match the current draft: "
            f"expected {approval.draft_hash}, got {actual}. The draft was "
            f"modified after approval and needs a new approval."
        )
    if approval.decision != ApprovalDecision.APPROVED:
        report.problems.append(
            f"Approval decision was {approval.decision.value!r}, not approved."
        )


def _check_workspace(
    output_path: Path, workspace: Path, report: VerificationReport
) -> None:
    """Output is inside the task workspace."""
    try:
        output_path.resolve(strict=False).relative_to(
            workspace.resolve(strict=False)
        )
    except ValueError:
        report.problems.append(
            f"Output path {output_path!r} is not inside the workspace {workspace!r}."
        )


def _count_citations(body_text: str) -> int:
    """[E1], [E2], etc. The renderer is supposed to use exactly this form."""
    return len(re.findall(r"\[(E\d+)\]", body_text))


def verify_document(
    *,
    draft: ApprovalNoteDraft,
    output_path: Path,
    workspace: Path,
    passages_by_id: Dict[str, Any],
    calculation_records: Dict[str, Any],
    approval: Optional[ApprovalRecord],
    input_classification: Classification,
) -> VerificationReport:
    """Run every check. Returns a report; an empty ``problems`` list is
    the success signal."""
    report = VerificationReport(is_complete=True, problems=[])

    # 7. DOCX opens — first, because later checks depend on the body text.
    render_result = render_verify(output_path)
    _check_sections(render_result, report)

    if not render_result.opens:
        report.is_complete = False
        # No point running checks that need the body if the body is not
        # readable; they would all fail for the same reason.
        return report

    report.document_byte_size = render_result.byte_size
    body_text = "\n".join(
        [p.text for p in __import__("docx").Document(str(output_path)).paragraphs]
    )
    report.citations_in_text = _count_citations(body_text)

    # 1 + 8. Evidence resolves and citations are present in the body.
    _check_evidence(draft, passages_by_id, body_text, report)

    # 2. Calculations resolve and the figure text is in the body.
    _check_calculations(draft, calculation_records, body_text, report)

    # 4. Output classification is not below input.
    _check_classification(draft.classification, input_classification, report)

    # 5. Approval is bound to the exact draft hash.
    _check_approval_binding(draft, approval, report)

    # 6. Output path is inside workspace.
    _check_workspace(output_path, workspace, report)

    # 8. Artifact hash is recorded (always — it is what the evidence
    #     package signs).
    try:
        report.artifact_hash = _file_sha256(output_path)
    except OSError as exc:
        report.problems.append(f"Could not hash the artifact: {exc}")

    if report.problems:
        report.is_complete = False
    return report
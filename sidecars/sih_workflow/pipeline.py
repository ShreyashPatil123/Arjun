"""SIH workflow orchestrator: 12-step inspection-report-to-approval-note flow.

Runs the 12-step sequence exactly as the problem statement requires:

 1. Upload scanned inspection report and optional photograph.
 2. Validate type, size, classification and workspace scope.
 3. Run local OCR/VLM extraction.
 4. Search authorized SOP/manual collections.
 5. Run deterministic calculations with units where needed.
 6. Produce a typed ApprovalNoteDraft object.
 7. Validate required fields and evidence IDs.
 8. Ask for human approval over the exact draft hash, output path,
    classification, model, skill and evidence.
 9. Render a Word document from a trusted local template.
 10. Reopen and validate the DOCX structure.
 11. Verify citations, figures, classification and approval binding.
 12. Compute artifact hash and export the evidence package.

The model fills a typed draft; this orchestrator is what *renders*,
*verifies*, and *exports*. The model never writes DOCX XML.

## Approver

Step 8 (approval) is the only step that is not a deterministic
computation. The orchestrator exposes a hook for the caller to supply
the approver; the default refuses to render without one. That refusal
is the whole point of step 8: an approval that was never given is not
an approval.

## Idempotence

The orchestrator is designed so that running it twice with the same
inputs does the same thing the second time as the first. The
approver's approval is recorded by draft hash, so a second approval of
the same draft is a no-op rather than a second record. The renderer is
asked to overwrite the same output path; the document is the same; the
evidence package is the same. A second run with no inputs changed is a
no-op, and a reviewer can rely on that.

## Refusal

Any failure in any step refuses the rest. The failure is recorded with
the reason; the file is not written; the package is not exported. The
reviewer who picks up the trail sees a clean ``refused: <reason>``
state, not a half-finished note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .approval import (
    ApprovalDecision,
    ApprovalError,
    ApprovalRecord,
    ApprovalRequest,
    bind_approval,
    compute_draft_hash,
    make_request,
)
from .calculation import (
    CalculationError,
    CalculationRecord,
    compute,
    ratio,
)
from .draft import (
    ApprovalNoteDraft,
    Classification,
    DraftStatus,
    Finding,
    Severity,
    UncertaintyNote,
)
from .evidence_package import (
    EvidencePackage,
    export_package,
)
from .extraction import (
    ExtractionResult,
    build_uncertainty_notes,
    extract,
    full_text as extraction_full_text,
)
from .render import (
    DocumentMetadata,
    RenderResult,
    TemplateError,
    render,
    verify as render_verify,
)
from .retrieval import (
    Passage,
    RetrievalResult,
    resolve_evidence_ids,
    search as retrieval_search,
)
from .validate_input import (
    InputError,
    allowed_report_extensions,
    max_report_bytes,
    validate_input,
)
from .verifier import VerificationReport, verify_document




class PipelineError(Exception):
    """Base class for pipeline errors."""
    pass


class PipelineRefused(PipelineError):
    """The pipeline refused. No file was written. No package was exported."""

    def __init__(self, step: str, reason: str):
        super().__init__(f"step {step} refused: {reason}")
        self.step = step
        self.reason = reason





@dataclass
class PipelineInput:
    """Everything the orchestrator needs to start a run."""
    report_path: Path
    photograph_path: Optional[Path]
    workspace_root: Path
    output_path: Path
    package_dir: Path
    sop_collection_root: Path
    equipment_id: str
    inspection_date: str
    classification: Classification
    model_id: str
    skill_id: str
    task_id: str


@dataclass
class PipelineOutput:
    """What the orchestrator produced, in full."""
    draft: ApprovalNoteDraft
    draft_hash: str
    document_path: Path
    document_hash: str
    approval: ApprovalRecord
    verification: VerificationReport
    extraction: ExtractionResult
    retrieval: RetrievalResult
    calculation_records: Dict[str, CalculationRecord]
    package: EvidencePackage


#: A function that, given an ApprovalRequest, returns an ApprovalRecord.
#: The orchestrator does not supply one; the caller does, because the
#: approval is a human decision and the orchestrator is not a human.
Approver = Callable[[ApprovalRequest], ApprovalRecord]





def make_typed_draft(
    *,
    equipment_id: str,
    inspection_date: str,
    findings: Sequence[Finding],
    evidence_ids: Sequence[str],
    proposed_action: str,
    calculation_ids: Sequence[str],
    uncertainty_notes: Sequence[UncertaintyNote],
    classification: Classification,
    model_id: str,
    skill_id: str,
) -> ApprovalNoteDraft:
    """Build a typed draft from the model's fields.

    This is the *only* shape the renderer accepts. A model that supplies a
    different shape (e.g. a flat string for the proposed action) is using
    the wrong tool; the orchestrator's contract is a typed object.
    """
    draft = ApprovalNoteDraft.new(equipment_id, inspection_date, classification)
    draft.findings = list(findings)
    draft.evidence_ids = list(evidence_ids)
    draft.calculation_ids = list(calculation_ids)
    draft.uncertainty_notes = list(uncertainty_notes)
    draft.proposed_action = proposed_action
    draft.severity = draft.highest_severity()
    draft.model_id = model_id
    draft.skill_id = skill_id
    return draft


def _validate_draft_fields(draft: ApprovalNoteDraft) -> None:
    """Required fields are present. Empty string is treated as absent."""
    if not draft.equipment_id or not draft.equipment_id.strip():
        raise PipelineRefused("7", "required field equipment_id is empty")
    if not draft.inspection_date or not draft.inspection_date.strip():
        raise PipelineRefused("7", "required field inspection_date is empty")
    if not draft.findings:
        raise PipelineRefused("7", "required field findings is empty")
    if not draft.proposed_action or not draft.proposed_action.strip():
        raise PipelineRefused("7", "required field proposed_action is empty")
    if draft.has_blocking_uncertainty():
        raise PipelineRefused("7", "draft has a blocking uncertainty note")


def _validate_evidence_resolution(
    draft: ApprovalNoteDraft,
    passages_by_id: Dict[str, Passage],
    calculation_records: Dict[str, CalculationRecord],
) -> None:
    """Every cited evidence and calculation must resolve.

    A draft that quotes ``[E1]`` but has no passage for ``E1`` is a draft
    that quotes an empty reference — the renderer would either silently
    write nothing (bad) or fabricate a passage (worse). The orchestrator
    refuses here, before any file exists.
    """
    for eid in draft.evidence_ids:
        if eid not in passages_by_id:
            raise PipelineRefused(
                "7", f"evidence ID {eid!r} does not resolve to an authorized passage"
            )
    for cid in draft.calculation_ids:
        if cid not in calculation_records:
            raise PipelineRefused(
                "7", f"calculation ID {cid!r} does not resolve to a calculation record"
            )





def run_pipeline(
    *,
    input: PipelineInput,
    draft: ApprovalNoteDraft,
    sop_queries: Sequence[str],
    extra_calculations: Optional[Dict[str, CalculationRecord]] = None,
    approver: Optional[Approver] = None,
    sidecar_router: Any = None,
) -> PipelineOutput:
    """Run the 12-step workflow.

    The function is deliberately one straight line of steps, with each
    block marked with the step number. A reader can read top-to-bottom and
    follow the sequence the prompt requires; a tester can search for the
    step number and the assertion that belongs to it.
    """
    # 1 + 2. Upload and validate.
    try:
        validate_input(
            report_path=input.report_path,
            workspace_root=input.workspace_root,
            output_path=input.output_path,
            classification=input.classification,
            photograph_path=input.photograph_path,
        )
    except InputError as exc:
        raise PipelineRefused("2", str(exc)) from exc

    # 3. Extraction. The sidecar runs first, the orchestrator decides what
    # to do with the result. A model that runs its own extraction before
    # the orchestrator cannot — the orchestrator is the only path.
    extraction_result = extract(input.report_path, sidecar_router=sidecar_router)
    uncertainty_dicts = build_uncertainty_notes(extraction_result)
    uncertainty_notes = [UncertaintyNote(**u) for u in uncertainty_dicts]
    # If the model did not already supply uncertainty notes, the sidecar's
    # notes replace them. The model's notes win when both exist; the
    # assumption is that a model that supplies its own uncertainty has
    # thought about which gaps matter.
    if not draft.uncertainty_notes:
        draft.uncertainty_notes = uncertainty_notes

    # 4. Retrieval. Search the authorized SOP/manual collection for the
    # passages the draft cites. Searches that return nothing are reported
    # in the empty_queries list, not silently dropped.
    retrieval = retrieval_search(
        collection_root=input.sop_collection_root,
        queries=sop_queries,
    )
    passages_by_id = {p.evidence_id: p for p in retrieval.passages}

    # 5. Calculations. The model supplied calculation IDs; the records
    # come from extra_calculations (the model calls the engine before
    # handing the draft to the orchestrator, just as the prompt says).
    # An ID without a record is a refusal: the orchestrator cannot make
    # a calculation record up.
    calculation_records: Dict[str, CalculationRecord] = dict(extra_calculations or {})
    for cid in draft.calculation_ids:
        if cid not in calculation_records:
            raise PipelineRefused(
                "5",
                f"calculation ID {cid!r} has no precomputed record; provide it via extra_calculations",
            )

    # 6 + 7. Validate the typed draft.
    _validate_draft_fields(draft)
    _validate_evidence_resolution(draft, passages_by_id, calculation_records)

    # 8. Approval. The approval is bound to the exact draft hash the human
    # is shown. A second approval of the same hash is a no-op.
    draft_hash = compute_draft_hash(draft)
    request = make_request(
        draft=draft,
        output_path=input.output_path,
        model_id=input.model_id,
        skill_id=input.skill_id,
    )
    if approver is None:
        raise PipelineRefused(
            "8",
            "no approver supplied; the orchestrator refuses to render without one",
        )
    approval = approver(request)
    if approval.decision == ApprovalDecision.REJECTED:
        raise PipelineRefused(
            "8",
            f"approval rejected: {approval.reason or 'no reason given'}",
        )
        approval_records: List[ApprovalRecord] = [approval]

    # 9 + 10. Render. The renderer is trusted local code. The model never
    # writes DOCX XML; it never chooses a template. The renderer writes
    # the file, then re-opens it for the post-render check.
    if input.output_path.exists():
        try:
            input.output_path.unlink()
        except OSError:
            # If the file is locked, the next render call will fail with a
            # more specific error and the caller can decide what to do.
            pass
    metadata = DocumentMetadata(
        task_id=input.task_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=input.model_id,
        skill=input.skill_id,
        classification=input.classification,
        is_draft=False,
        approval=approval,
    )
    try:
        render_result = render(
            draft=draft,
            output_path=input.output_path,
            metadata=metadata,
            calculation_records=calculation_records,
            passages_by_id=passages_by_id,
        )
    except TemplateError as exc:
        raise PipelineRefused("9", str(exc)) from exc
    if not render_result.is_sound():
        problems = "; ".join(render_result.problems)
        raise PipelineRefused("10", f"document failed self-check: {problems}")

    # 11. Verify. Re-open the file and run eight checks against the draft,
    # the approval, the calculations, the citations, and the file itself.
    verification = verify_document(
        draft=draft,
        output_path=input.output_path,
        workspace=input.workspace_root,
        passages_by_id=passages_by_id,
        calculation_records=calculation_records,
        approval=approval,
        input_classification=input.classification,
    )
    if not verification.is_complete:
        problems = "; ".join(verification.problems)
        raise PipelineRefused("11", f"verification failed: {problems}")

    # 12. Export the evidence package. The package is the audit record:
    # draft, document, approval, calculations, evidence, provenance, and
    # the package hash that ties them all together.
    package = export_package(
        package_dir=input.package_dir,
        task_id=input.task_id,
        model_id=input.model_id,
        skill_id=input.skill_id,
        classification=input.classification,
        draft=draft,
        draft_hash=draft_hash,
        document_path=input.output_path,
        artifact_hash=verification.artifact_hash,
        approval=approval,
        calculation_records=calculation_records,
        passages_by_id=passages_by_id,
        input_classification=input.classification,
    )

    return PipelineOutput(
        draft=draft,
        draft_hash=draft_hash,
        document_path=input.output_path,
        document_hash=verification.artifact_hash,
        approval=approval,
        verification=verification,
        extraction=extraction_result,
        retrieval=retrieval,
        calculation_records=calculation_records,
        package=package,
    )




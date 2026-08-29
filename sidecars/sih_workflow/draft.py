"""Typed ApprovalNoteDraft and related data structures.

These are the *only* structures the model is allowed to fill in. Trusted local
code (this package's renderer, validator, and verifier) reads them and either
turns them into a Word document or refuses to. The model never writes DOCX XML
directly.

Why typed:

* **Required fields are present or the draft is not renderable.** A missing
  equipment_id, an empty findings list, or a blank proposed action is a draft
  error caught before any file is written.
* **Every figure is bound to a calculation record.** A bare number in the
  proposed action cannot slip through; the verifier cross-checks it against the
  calculation log.
* **Every claim is bound to evidence.** An evidence_id that does not resolve
  to a passage in the authorized collection blocks readiness.

The choice of dataclasses with primitive types rather than free-form dicts is
the whole point of the typed intermediate: the model cannot smuggle a new
field through to the document by calling it something creative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


class DraftStatus(str, Enum):
    """Lifecycle of a draft.

    DRAFT is the only state in which a model may hand a draft to the renderer.
    Anything past DRAFT is set by trusted local code as a record of what
    happened next.
    """

    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RENDERED = "RENDERED"


class Severity(str, Enum):
    """Finding severity, ordered low to critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def order(cls) -> List["Severity"]:
        return [cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]

    def rank(self) -> int:
        return self.order().index(self)

    def label(self) -> str:
        return self.value.capitalize()


class Classification(str, Enum):
    """Material sensitivity.

    Ordered so a higher value (more sensitive) can be required to be matched or
    exceeded at every hop. The verifier uses this ordering to refuse a draft
    that would downgrade the output classification below the input.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    PROCESS_DIAGRAM = "processDiagram"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @classmethod
    def order(cls) -> List["Classification"]:
        return [
            cls.PUBLIC,
            cls.INTERNAL,
            cls.PROCESS_DIAGRAM,
            cls.CONFIDENTIAL,
            cls.RESTRICTED,
        ]

    def rank(self) -> int:
        return self.order().index(self)

    def at_least(self, minimum: "Classification") -> bool:
        return self.rank() >= minimum.rank()


@dataclass
class Finding:
    """A single finding the inspection produced."""

    id: str
    description: str
    severity: Severity
    location: Optional[str] = None
    source_page: Optional[int] = None
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class CalculationRef:
    """Pointer from a figure in the draft to the calculation record that
    produced it.

    The model never invents a number: every numerical figure the draft
    mentions is bound, by ID, to a record from the deterministic calculation
    engine. The verifier cross-checks this binding.
    """

    calculation_id: str
    expression: str
    result: str  # result with units, exactly as the engine returned it
    finding_id: Optional[str] = None


@dataclass
class EvidenceRef:
    """Pointer to a passage in the authorized collection.

    An evidence_id is the model-side label (``E1``, ``E2``...). The verifier
    resolves it against the loaded passage index, and refuses if the ID is not
    in the index.
    """

    evidence_id: str
    document_sha256: str
    page: int
    passage_text: str


@dataclass
class UncertaintyNote:
    """Something the inspection could not establish with certainty.

    ``blocks_approval`` distinguishes a missing value the verifier can
    work around from one that means the note cannot be signed until the gap
    is filled.
    """

    what: str
    reason: str
    finding_id: Optional[str] = None
    blocks_approval: bool = False


@dataclass
class ApprovalNoteDraft:
    """The typed intermediate between the model and the renderer.

    The model fills this in. The renderer reads it. No other channel is
    used for the content of the note.

    The required fields are exactly the ones the problem statement lists. A
    field is "filled" if it is present and non-empty; an empty string for a
    string field or an empty list for a list field counts as missing for the
    purposes of validation.
    """

    # --- Required fields, in the order the problem statement names them ---
    equipment_id: str
    inspection_date: str  # ISO 8601, e.g. 2026-08-26
    findings: List[Finding] = field(default_factory=list)
    severity: Severity = Severity.LOW
    evidence_ids: List[str] = field(default_factory=list)
    proposed_action: str = ""
    calculation_ids: List[str] = field(default_factory=list)
    uncertainty_notes: List[UncertaintyNote] = field(default_factory=list)
    classification: Classification = Classification.INTERNAL
    status: DraftStatus = DraftStatus.DRAFT

    # --- Provenance, set by the orchestrator, not the model ---
    model_id: Optional[str] = None
    skill_id: Optional[str] = None

    # ---------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict for JSON serialisation, with enums as their values."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                out[f.name] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Finding):
                out[f.name] = [asdict(x) for x in value]
            elif isinstance(value, list) and value and isinstance(
                value[0], UncertaintyNote
            ):
                out[f.name] = [asdict(x) for x in value]
            else:
                out[f.name] = value
        return out

    def to_json(self) -> str:
        """Stable JSON for hashing. The renderer does not use this directly;
        the verifier hashes it to bind the approval to a specific draft."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def highest_severity(self) -> Severity:
        """Maximum severity across findings, or LOW when there are none."""
        if not self.findings:
            return Severity.LOW
        return max((f.severity for f in self.findings), key=lambda s: s.rank())

    def has_blocking_uncertainty(self) -> bool:
        return any(u.blocks_approval for u in self.uncertainty_notes)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ApprovalNoteDraft":
        """Build a draft from a plain dict (e.g. JSON from a model).

        Unknown keys are dropped, on purpose: a model that invents a field
        cannot smuggle it into the renderer.
        """
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in payload.items() if k in allowed}

        if "findings" in clean:
            clean["findings"] = [
                Finding(**_coerce_finding(f)) for f in clean["findings"]
            ]
        if "uncertainty_notes" in clean:
            clean["uncertainty_notes"] = [
                UncertaintyNote(**u) for u in clean["uncertainty_notes"]
            ]
        if "severity" in clean and isinstance(clean["severity"], str):
            clean["severity"] = Severity(clean["severity"])
        if "classification" in clean and isinstance(clean["classification"], str):
            clean["classification"] = Classification(clean["classification"])
        if "status" in clean and isinstance(clean["status"], str):
            clean["status"] = DraftStatus(clean["status"])
        return cls(**clean)


def _coerce_finding(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce one finding dict's enum string into the enum value."""
    out = dict(payload)
    if isinstance(out.get("severity"), str):
        out["severity"] = Severity(out["severity"])
    return out
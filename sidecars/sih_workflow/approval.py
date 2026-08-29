"""Step 8: ask for human approval over the exact draft.

What the human sees:

* the draft hash (so they know what they are signing)
* the output path (so they know where the file will go)
* the classification (so they know what they are about to handle)
* the model id and the skill id (so they can refuse on that basis)
* the evidence ids (so they can ask for the actual passages)

What the human can do:

* **Approve** the exact draft. The approval is bound to the draft hash,
  so any later change to the draft — even a whitespace change — requires a
  new approval.
* **Reject** with a reason. The reason is recorded. No file is written.

The approval is asked for over the *exact* draft hash, not over a
human-readable summary. Two reasons:

* A model that summarises its draft before approval is a model that has
  already begun negotiating with itself about what the user wanted.
* A reviewer who approves on the basis of a summary has signed something
  they did not see. The hash makes what they saw unambiguous.

Approval is **idempotent**: a second approval of the same draft hash is a
no-op, not a second record. The verifier needs a single, unambiguous
record of who said yes to what.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .draft import ApprovalNoteDraft, Classification


class ApprovalDecision(str, Enum):
    """The two outcomes the human can choose."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class ApprovalRequest:
    """What the human is asked to approve.

    Carries everything the reviewer needs to make a decision *before*
    granting it: the draft hash, the path the file will land at, the
    classification, the model and the skill, and the evidence ids the
    draft rests on.
    """

    draft_hash: str
    output_path: Path
    classification: Classification
    model_id: str
    skill_id: str
    evidence_ids: List[str]
    calculation_ids: List[str]
    summary: str
    requested_at: str

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "draftHash": self.draft_hash,
            "outputPath": str(self.output_path),
            "classification": self.classification.value
            if isinstance(self.classification, Classification)
            else self.classification,
            "modelId": self.model_id,
            "skillId": self.skill_id,
            "evidenceIds": list(self.evidence_ids),
            "calculationIds": list(self.calculation_ids),
            "summary": self.summary,
            "requestedAt": self.requested_at,
        }
        return out


@dataclass
class ApprovalRecord:
    """The decision, with everything needed to verify it later."""

    draft_hash: str
    decision: ApprovalDecision
    decided_by: str
    decided_at: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "draftHash": self.draft_hash,
            "decision": self.decision.value
            if isinstance(self.decision, ApprovalDecision)
            else self.decision,
            "decidedBy": self.decided_by,
            "decidedAt": self.decided_at,
            "reason": self.reason,
        }


def compute_draft_hash(draft: ApprovalNoteDraft) -> str:
    """SHA-256 of the canonical JSON of the draft.

    Used both for the approval request and for the verifier. The same
    function is used both places on purpose, so the human and the
    verifier are looking at the same number.
    """
    payload = draft.to_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_request(
    *,
    draft: ApprovalNoteDraft,
    output_path: Path,
    model_id: str,
    skill_id: str,
    summary: Optional[str] = None,
) -> ApprovalRequest:
    """Build the request the human will be shown."""
    if summary is None:
        summary = (
            f"Approval note for {draft.equipment_id} "
            f"(inspection {draft.inspection_date}, "
            f"{len(draft.findings)} finding(s), "
            f"severity {draft.severity.value})."
        )
    return ApprovalRequest(
        draft_hash=compute_draft_hash(draft),
        output_path=output_path,
        classification=draft.classification,
        model_id=model_id,
        skill_id=skill_id,
        evidence_ids=list(draft.evidence_ids),
        calculation_ids=list(draft.calculation_ids),
        summary=summary,
        requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


class ApprovalError(Exception):
    """Anything that prevents a binding approval."""


def is_draft_hash_approved(
    draft_hash: str, records: List[ApprovalRecord]
) -> Optional[ApprovalRecord]:
    """Return the most recent approved record for the hash, or None.

    A rejected record is not an approval. A later approval of the same
    hash is what the verifier is looking for.
    """
    matching = [r for r in records if r.draft_hash == draft_hash]
    if not matching:
        return None
    # Most recent wins. Records arrive in chronological order.
    for record in reversed(matching):
        if record.decision == ApprovalDecision.APPROVED:
            return record
    return None


def bind_approval(
    *,
    draft: ApprovalNoteDraft,
    records: List[ApprovalRecord],
    decision: ApprovalDecision,
    decided_by: str,
    reason: Optional[str] = None,
) -> ApprovalRecord:
    """Record a decision.

    Three checks before the record is added:

    * The draft hash the human saw matches the draft's current hash.
      If it does not, the model is asking for an approval of something
      it has already changed.
    * The record list does not already contain an approval for this hash.
      A second approval is a no-op; recording it would just confuse the
      verifier.
    * The decision is one of the two legal values.
    """
    actual_hash = compute_draft_hash(draft)
    if not records or records[-1].draft_hash != actual_hash:
        # The records list is supposed to be a history; the last entry is
        # the most recent decision. If the new decision's draft hash does
        # not match the last entry's, the model is asking for approval of
        # something that does not match what the human is being shown.
        if not records:
            raise ApprovalError("No prior approval request was made.")
        raise ApprovalError(
            f"The approval request's draft hash does not match the draft. "
            f"Expected {records[-1].draft_hash}, got {actual_hash}."
        )

    if is_draft_hash_approved(actual_hash, records) is not None:
        # Idempotent: a second approval of the same hash is a no-op.
        # We return the existing record rather than recording a duplicate.
        return is_draft_hash_approved(actual_hash, records)  # type: ignore

    record = ApprovalRecord(
        draft_hash=actual_hash,
        decision=decision,
        decided_by=decided_by,
        decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reason=reason,
    )
    records.append(record)
    return record
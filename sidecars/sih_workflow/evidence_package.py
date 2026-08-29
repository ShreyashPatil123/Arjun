"""Step 12: compute the artifact hash and export the evidence package.

The package is the *whole* record of the task: draft, approval, document,
calculation records, evidence passages, and the provenance that ties them
together. It is the thing a reviewer can read six months from now and ask,
"what was this note, and what was it based on?"

What is in the package, and why:

* **Draft.** A copy of the draft as it was at the moment of approval.
  Without the draft, the document is a piece of paper with no
  chain of custody.
* **Document.** The Word file itself, alongside its hash. A reviewer can
  re-open it and see what was signed.
* **Approval record.** What the human saw, and what they said. A
  document without an approval is a draft; a document with the wrong
  approval hash is a forgery waiting to be questioned.
* **Calculation records.** The exact expressions and results the document
  quotes. A figure that is not in the calculation log is a figure the
  model invented.
* **Evidence passages.** The text behind every [En] citation. A citation
  that does not resolve to a passage is a citation the model invented.
* **Provenance.** Model, skill, task id, classification, all the hashes,
  and the timestamp.
* **Package hash.** SHA-256 over the whole package, so a reviewer can
  tell at a glance whether the package they are looking at is the
  package the system exported.

The package is written as a directory of files plus a manifest, rather
than a single ZIP. A reviewer reading a single file at a time is more
common than a reviewer unpacking an archive, and a directory makes the
parts individually addressable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .approval import ApprovalRecord
from .draft import ApprovalNoteDraft, Classification


@dataclass
class PackageArtifact:
    """One file in the package, with everything needed to verify it."""

    artifact_type: str
    path: str  # path inside the package, relative to the package root
    sha256: str
    size_bytes: int
    created_at: str


@dataclass
class Provenance:
    """The chain of custody for the whole task."""

    task_id: str
    model_id: str
    skill_id: str
    classification: Classification
    evidence_ids: List[str]
    calculation_ids: List[str]
    draft_hash: str
    artifact_hash: str
    exported_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.task_id,
            "modelId": self.model_id,
            "skillId": self.skill_id,
            "classification": self.classification.value
            if isinstance(self.classification, Classification)
            else self.classification,
            "evidenceIds": list(self.evidence_ids),
            "calculationIds": list(self.calculation_ids),
            "draftHash": self.draft_hash,
            "artifactHash": self.artifact_hash,
            "exportedAt": self.exported_at,
        }


@dataclass
class EvidencePackage:
    """The whole package, ready to be written or queried."""

    package_dir: Path
    artifacts: List[PackageArtifact] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    package_hash: str = ""
    manifest_path: Path = field(default_factory=lambda: Path())

    def is_complete(self) -> bool:
        types = {a.artifact_type for a in self.artifacts}
        return {"draft", "document", "approval"}.issubset(types)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_artifact(
    package_dir: Path, name: str, content: bytes
) -> PackageArtifact:
    """Drop a file into the package, hash it, and record the result."""
    path = package_dir / name
    path.write_bytes(content)
    return PackageArtifact(
        artifact_type=name.split(".")[0],
        path=name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def export_package(
    *,
    package_dir: Path,
    task_id: str,
    model_id: str,
    skill_id: str,
    classification: Classification,
    draft: ApprovalNoteDraft,
    draft_hash: str,
    document_path: Path,
    artifact_hash: str,
    approval: ApprovalRecord,
    calculation_records: Dict[str, Any],
    passages_by_id: Dict[str, Any],
    input_classification: Optional[Classification] = None,
) -> EvidencePackage:
    """Export the whole evidence package.

    ``input_classification`` is optional; if supplied, the manifest records
    that the output classification is at least the input. The verifier's
    check is separate; this is a record of what was checked, not the
    check itself.
    """
    package_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[PackageArtifact] = []

    # Draft — as it was at the moment of approval.
    draft_bytes = draft.to_json().encode("utf-8")
    artifacts.append(_write_artifact(package_dir, "draft.json", draft_bytes))

    # Document — a copy of the file. The original is left where it is;
    # the package is the audit record.
    if document_path.exists():
        doc_bytes = document_path.read_bytes()
    else:
        # An empty file is honest: the verifier said the document was not
        # produced, and the package records that.
        doc_bytes = b""
    artifacts.append(
        PackageArtifact(
            artifact_type="document",
            path=str(document_path.name),
            sha256=hashlib.sha256(doc_bytes).hexdigest(),
            size_bytes=len(doc_bytes),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )
    (package_dir / document_path.name).write_bytes(doc_bytes)

    # Approval record.
    approval_bytes = json.dumps(approval.to_dict(), indent=2).encode("utf-8")
    artifacts.append(_write_artifact(package_dir, "approval.json", approval_bytes))

    # Calculation records — a flat file the reviewer can grep.
    calc_bytes = json.dumps(
        {
            cid: {
                "expression": getattr(r, "expression", ""),
                "result": getattr(r, "result", ""),
                "unit": getattr(r, "unit", ""),
                "error": getattr(r, "error", None),
            }
            for cid, r in calculation_records.items()
        },
        indent=2,
    ).encode("utf-8")
    artifacts.append(_write_artifact(package_dir, "calculations.json", calc_bytes))

    # Evidence passages — one file per passage, so a reviewer reading a
    # single citation does not have to open the whole index.
    for eid, passage in passages_by_id.items():
        passage_bytes = json.dumps(
            {
                "evidenceId": eid,
                "documentPath": str(getattr(passage, "document_path", "")),
                "documentSha256": getattr(passage, "document_sha256", ""),
                "page": getattr(passage, "page", 0),
                "passageText": getattr(passage, "passage_text", ""),
            },
            indent=2,
        ).encode("utf-8")
        artifacts.append(
            _write_artifact(
                package_dir, f"evidence-{eid}.json", passage_bytes
            )
        )

    provenance = Provenance(
        task_id=task_id,
        model_id=model_id,
        skill_id=skill_id,
        classification=classification,
        evidence_ids=list(draft.evidence_ids),
        calculation_ids=list(draft.calculation_ids),
        draft_hash=draft_hash,
        artifact_hash=artifact_hash,
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    # Package hash is over the artifact hashes in stable order. The
    # verifier can recompute it from the manifest and check that the
    # package has not been edited since export.
    sorted_hashes = sorted([a.sha256 for a in artifacts])
    package_hash = hashlib.sha256(
        "|".join(sorted_hashes).encode("utf-8")
    ).hexdigest()

    manifest = {
        "packageHash": package_hash,
        "provenance": provenance.to_dict(),
        "inputClassification": input_classification.value
        if input_classification is not None
        else None,
        "artifacts": [
            {
                "artifactType": a.artifact_type,
                "path": a.path,
                "sha256": a.sha256,
                "sizeBytes": a.size_bytes,
                "createdAt": a.created_at,
            }
            for a in artifacts
        ],
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return EvidencePackage(
        package_dir=package_dir,
        artifacts=artifacts,
        provenance=provenance,
        package_hash=package_hash,
        manifest_path=manifest_path,
    )
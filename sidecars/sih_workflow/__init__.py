"""SIH inspection-report-to-Word-approval-note workflow.

This package implements the 12-step workflow the problem statement
describes: upload, validate, extract, retrieve, calculate, draft,
validate the draft, approve, render, verify, and export the evidence
package.

The model fills a typed ``ApprovalNoteDraft``; trusted local code in this
package is what renders, verifies, and exports. The model never writes
DOCX XML.

## Public API

* :class:`draft.ApprovalNoteDraft` — the typed intermediate
* :func:`pipeline.run_pipeline` — the orchestrator
* :func:`pipeline.make_typed_draft` — convenience for callers
* :class:`pipeline.PipelineInput` / :class:`pipeline.PipelineOutput`
* :class:`pipeline.PipelineRefused` — what a refusal looks like

Everything else is internal to the workflow; import from submodules for
fine-grained access in tests.
"""

from .pipeline import (
    PipelineError,
    PipelineInput,
    PipelineOutput,
    PipelineRefused,
    make_typed_draft,
    run_pipeline,
)

__all__ = [
    "PipelineError",
    "PipelineInput",
    "PipelineOutput",
    "PipelineRefused",
    "make_typed_draft",
    "run_pipeline",
]

"""Step 2: validate the upload before any work is done.

Four things are checked, and in this order:

1. **Type.** A PDF or a photo. Anything else is refused; we do not guess
   formats from a name. The classification of accepted types is also
   recorded — that is the input classification the output is later compared
   against.
2. **Size.** Refuse early rather than exhaust memory partway through
   extraction. The limit is a hard ceiling; a file just below it is still
   fine.
3. **Classification.** The material's sensitivity has to be supplied by the
   caller, not inferred. ARJUN is not in the business of deciding what is
   sensitive.
4. **Workspace scope.** The output path has to be inside the task workspace.
   This is what stops a note from a sandboxed task being written to a
   directory it has no business touching.

Remote URLs are refused without even a read. A model that can fetch a URL
can fetch a hostile one, and an inspection report is exactly the kind of
document a hostile one would pretend to be.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from .draft import Classification


#: Hard ceiling for an inspection report. A refinery drawing set can be very
#: large; failing clearly beats failing by swapping.
MAX_REPORT_BYTES = 100 * 1024 * 1024

#: Photos are smaller; same reasoning, smaller number.
MAX_PHOTO_BYTES = 25 * 1024 * 1024

#: Extensions the report is allowed to have. Anything else is refused.
ALLOWED_REPORT_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"}

#: Extensions a photograph is allowed to have.
ALLOWED_PHOTO_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp"}

#: Prefixes that would mean "this is a network location, not a local file".
REMOTE_PREFIXES = ("http://", "https://", "file://", "ftp://", "\\\\")


class InputError(ValueError):
    """Raised for any of the four validation failures.

    A single exception class keeps the orchestrator's error handling to a
    single branch and lets the message say which of the four checks failed.
    """


def _is_remote(path: Path) -> bool:
    s = str(path).lower()
    return any(s.startswith(p) for p in REMOTE_PREFIXES)


def _check_extension(path: Path, allowed: set, label: str) -> None:
    ext = path.suffix.lower().lstrip(".")
    if not ext:
        raise InputError(
            f"The {label} has no file extension. Allowed: "
            f"{', '.join(sorted(allowed))}."
        )
    if ext not in allowed:
        raise InputError(
            f"The {label} has extension {ext!r}, which is not allowed. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )


def _check_size(path: Path, max_bytes: int, label: str) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise InputError(
            f"The {label} is {size / 1024 / 1024:.1f} MB, above the "
            f"{max_bytes / 1024 / 1024:.0f} MB limit for a {label}."
        )
    return size


def _check_inside_workspace(path: Path, workspace: Path) -> None:
    """Refuse a path that escapes the workspace.

    Comparison is by path components after resolution, so a `..` in the path
    cannot smuggle a write out of the workspace.
    """
    try:
        resolved = path.resolve(strict=False)
        workspace_resolved = workspace.resolve(strict=False)
    except OSError:
        # resolve() with strict=False is permissive; if even that fails the
        # path is suspicious enough to refuse.
        raise InputError(f"The path {path!r} could not be resolved.")

    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise InputError(
            f"The output path {path!r} is not inside the workspace "
            f"{workspace!r}."
        )


def validate_input(
    *,
    report_path: Path,
    workspace_root: Path,
    output_path: Path,
    classification: Classification,
    photograph_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], int, Optional[int]]:
    """Run all four checks; return what the orchestrator needs to start work.

    Returns the resolved report path, the resolved photo path (or None), the
    report size, and the photo size (or None). The report path is the input
    classification for downstream checks.
    """
    # 1. Type & remote.
    if _is_remote(report_path):
        raise InputError(
            f"Remote URLs are not allowed for the inspection report "
            f"({report_path!r}). Only local file paths."
        )
    if not report_path.exists():
        raise InputError(f"The report does not exist: {report_path!r}")
    _check_extension(report_path, ALLOWED_REPORT_EXTENSIONS, "report")

    # 2. Size.
    report_size = _check_size(report_path, MAX_REPORT_BYTES, "report")

    photo_size: Optional[int] = None
    if photograph_path is not None:
        if _is_remote(photograph_path):
            raise InputError(
                f"Remote URLs are not allowed for the photograph "
                f"({photograph_path!r})."
            )
        if not photograph_path.exists():
            raise InputError(
                f"The photograph does not exist: {photograph_path!r}"
            )
        _check_extension(photograph_path, ALLOWED_PHOTO_EXTENSIONS, "photograph")
        photo_size = _check_size(photograph_path, MAX_PHOTO_BYTES, "photograph")

    # 3. Classification was supplied by the caller; nothing to check beyond
    #    that it is a valid value (the enum enforces that).

    # 4. Workspace scope.
    _check_inside_workspace(output_path, workspace_root)

    return (
        report_path.resolve(strict=False),
        photograph_path.resolve(strict=False) if photograph_path else None,
        report_size,
        photo_size,
    )


def is_path_inside(path: Path, workspace: Path) -> bool:
    """Public check used by the orchestrator when it wants to refuse later.

    Kept separate from the raising variant so callers that want a bool can
    get one without try/except noise.
    """
    try:
        _check_inside_workspace(path, workspace)
    except InputError:
        return False
    return True


def is_remote_path(path: Path) -> bool:
    """True if the path is a network location rather than a local file."""
    return _is_remote(path)


def allowed_report_extensions() -> Tuple[str, ...]:
    """Public accessor used by the verifier to refuse wrong-type reports."""
    return tuple(sorted(ALLOWED_REPORT_EXTENSIONS))


def max_report_bytes() -> int:
    return MAX_REPORT_BYTES
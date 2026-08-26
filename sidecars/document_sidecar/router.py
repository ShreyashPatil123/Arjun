"""Dispatches document requests to the best engine available on this machine.

Engines are tried in order of preference and the first available one wins. The
choice is reported with every result, so a deployment running on the fallback
never looks like one running on the real thing.
"""

import os
from typing import Any, Dict, List, Optional, Type

import escalation
import injection
from engines.base import DocumentEngine
from engines.docling_engine import DoclingEngine
from engines.text_layer import TextLayerEngine

#: Best first. Adding an engine is adding a class here.
ENGINE_PREFERENCE: List[Type[DocumentEngine]] = [DoclingEngine, TextLayerEngine]

#: Refuse anything larger rather than exhausting memory on a laptop. A refinery
#: drawing set can be very large, and failing clearly beats failing by swapping.
MAX_FILE_BYTES = 512 * 1024 * 1024


class DocumentRouter:
    def __init__(self) -> None:
        self._engine: Optional[DocumentEngine] = None
        self._considered: List[Dict[str, Any]] = []
        self._select()

    def _select(self) -> None:
        for engine_class in ENGINE_PREFERENCE:
            try:
                available = engine_class.available()
            except Exception:  # noqa: BLE001 — a broken engine must not stop the rest
                available = False

            self._considered.append(
                {"engine": engine_class.name, "available": available}
            )

            if available and self._engine is None:
                self._engine = engine_class()

    def status(self) -> Dict[str, Any]:
        """What this sidecar can do, and what it fell back to."""
        if self._engine is None:
            return {
                "ready": False,
                "engine": None,
                "considered": self._considered,
                "detail": (
                    "No document engine is available. Install pypdf for basic text "
                    "extraction, or Docling for layout and tables."
                ),
            }

        degraded = self._engine.name != ENGINE_PREFERENCE[0].name
        return {
            "ready": True,
            "engine": self._engine.name,
            "engineVersion": self._engine.version,
            "capabilities": self._engine.capabilities().__dict__,
            "considered": self._considered,
            "degraded": degraded,
            "detail": (
                "Running on the fallback text-layer engine. Scanned pages cannot be read "
                "until Docling or a document vision model is installed."
                if degraded
                else "Running on the preferred engine."
            ),
        }

    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "health_check":
            return self.status()

        if method == "extract":
            return self._extract(params)

        raise ValueError(f"unknown method {method!r}")

    def _extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path")
        if not path:
            raise ValueError("extract requires a 'path'")

        if not os.path.isfile(path):
            raise ValueError(f"no file at {path!r}")

        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            raise ValueError(
                f"the file is {size / 1024 / 1024:.0f} MB, above the "
                f"{MAX_FILE_BYTES / 1024 / 1024:.0f} MB limit for a single document"
            )

        if self._engine is None:
            raise ValueError(
                "No document engine is available on this machine, so nothing can be read."
            )

        result = self._engine.extract(path)
        payload = result.to_dict()
        payload["sourcePath"] = path
        payload["sourceBytes"] = size

        # Runs on every document, at ingest, before anything downstream sees the
        # text. The scan flags; it never edits — removing an offending line
        # would hide evidence of an attack and alter a record the organisation
        # may need intact.
        payload["injectionScan"] = injection.scan_pages(payload["pages"])

        # Two-tier reading: the cheap pass has run; this decides which pages it
        # could not settle. With no second engine installed the plan is still
        # produced, so the gap is visible rather than looking like a document
        # that simply had little on it.
        plan = escalation.plan(payload["pages"], payload["capabilities"])
        payload["escalation"] = plan.to_dict()
        payload["warnings"].extend(
            escalation.describe_unmet(plan, available=self._available_capabilities())
        )
        return payload

    def _available_capabilities(self) -> List[str]:
        """Capabilities any installed engine could supply for a second pass."""
        supplied: List[str] = []
        for engine_class in ENGINE_PREFERENCE:
            try:
                if not engine_class.available():
                    continue
            except Exception:  # noqa: BLE001
                continue
            caps = engine_class().capabilities()
            if caps.ocr:
                supplied.append("ocr")
            if caps.layout:
                supplied.append("vision")
        return sorted(set(supplied))

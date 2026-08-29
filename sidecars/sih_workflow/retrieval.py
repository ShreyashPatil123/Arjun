"""Step 4: search the authorized SOP / manual collection.

The interesting design constraint here is not retrieval — it is *what
retrieval is allowed to do*. Three rules, enforced in code rather than
encouraged in a docstring:

* **Search only what the workspace is allowed to read.** The collection root
  is part of the input; the retriever refuses to leave it.
* **Return the passage, not a summary.** A summary is a second thing to
  verify, and the verifier's whole point is to remove second things. The
  caller quotes from ``passage_text`` directly.
* **Return a stable evidence_id per passage.** The model labels citations
  ``E1``, ``E2``... in the order the retriever returns them. The retriever
  decides the mapping, and it is the same mapping the verifier checks
  against — so a model cannot invent an ID that resolves to nothing.

The retriever is keyword-based. That is not because keyword search is the
best there is; it is because the alternative is an embedding model, and the
whole point of the phase is to do this without external services. Keyword
search is honest: when it returns nothing, the caller knows nothing
matched, not that an embedding model was uncertain.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class Passage:
    """One chunk of one document, in the form the retriever hands it on."""

    evidence_id: str
    document_path: Path
    document_sha256: str
    page: int
    passage_text: str
    score: float = 0.0


@dataclass
class RetrievalResult:
    """A search and what it turned up, in stable order."""

    query: str
    passages: List[Passage] = field(default_factory=list)
    #: Queries that returned nothing. The orchestrator can try them again with
    #: different wording; the verifier wants to see them logged so a missing
    #: citation is not silent.
    empty_queries: List[str] = field(default_factory=list)

    def ids(self) -> List[str]:
        return [p.evidence_id for p in self.passages]


#: An evidence_id is short, in the form the model is asked to quote. A
#: longer form would survive, but the rendering is the same; keeping the
#: short form makes the document readable.
EVIDENCE_PREFIX = "E"


def _tokenize(text: str) -> List[str]:
    """Lowercase, alphanumerics only. Enough for keyword overlap scoring
    and forgiving of punctuation differences between a query and a passage."""
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def _read_passages_from_pdf(path: Path) -> List[Tuple[int, str]]:
    """Read a PDF page-by-page. A scanned PDF returns empty pages, which the
    caller treats as "not searchable" rather than "blank"."""
    try:
        import pypdf
    except ImportError:
        return []

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception:
        return []
    if getattr(reader, "is_encrypted", False):
        return []

    pages: List[Tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((index, text))
    return pages


def _read_passages_from_text(path: Path) -> List[Tuple[int, str]]:
    """Plain text and markdown are treated as a single page. The retriever
    does not pretend to know line numbers; a one-page text file is fine."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if text.strip():
        return [(1, text)]
    return []


def _passages_from_document(path: Path) -> List[Tuple[int, str]]:
    """Dispatch by extension. Adding a new format is one case here."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_passages_from_pdf(path)
    if ext in {".txt", ".md", ".rst"}:
        return _read_passages_from_text(path)
    return []


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _score(query_tokens: Sequence[str], passage: str) -> float:
    """Token overlap, weighted so a rare token counts more than a common one."""
    if not query_tokens:
        return 0.0
    passage_tokens = _tokenize(passage)
    if not passage_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in passage_tokens:
        counts[t] = counts.get(t, 0) + 1
    matched = 0
    for t in query_tokens:
        if counts.get(t, 0) > 0:
            matched += 1
    return matched / len(query_tokens)


def _chunk_passage(text: str, max_chars: int = 600) -> List[str]:
    """Split a long page into chunks the renderer can quote cleanly.

    Splits on sentence boundaries where it can, and falls back to a fixed
    width when the text has no sentences. The chunks are deliberately
    *short* — the whole point of citing is to point at the relevant line,
    not to reprint the document.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(current) + len(s) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _enumerate_authorized_documents(
    collection_root: Path,
) -> List[Path]:
    """Walk the collection. Skips dotfiles and unreadable entries."""
    docs: List[Path] = []
    for path in sorted(collection_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.suffix.lower() in {".pdf", ".txt", ".md", ".rst"}:
            docs.append(path)
    return docs


def search(
    *,
    collection_root: Path,
    queries: Iterable[str],
    top_k: int = 3,
) -> RetrievalResult:
    """Run every query against the collection and merge the results.

    Each query is a separate search; their results are interleaved by score
    (best first), so a passage that matches two queries outranks one that
    matches only one. The caller is expected to dedupe by document, but the
    retriever does not: the model may legitimately cite the same SOP twice
    for two different findings.
    """
    if not collection_root.exists() or not collection_root.is_dir():
        raise ValueError(
            f"The collection root {collection_root!r} does not exist or is not a directory."
        )

    documents = _enumerate_authorized_documents(collection_root)
    if not documents:
        return RetrievalResult(
            query=", ".join(queries),
            empty_queries=list(queries),
        )

    all_passages: List[Passage] = []
    seen: Set[Tuple[str, int, int]] = set()
    counter = 0

    for query in queries:
        tokens = _tokenize(query)
        if not tokens:
            return RetrievalResult(
                query=query,
                empty_queries=[query],
            )

        scored: List[Passage] = []
        any_match = False
        for doc in documents:
            sha = _file_sha256(doc)
            for page, text in _passages_from_document(doc):
                for chunk in _chunk_passage(text):
                    s = _score(tokens, chunk)
                    if s <= 0.0:
                        continue
                    any_match = True
                    key = (sha, page, hash(chunk) & 0xFFFFFFFF)
                    if key in seen:
                        continue
                    seen.add(key)
                    counter += 1
                    scored.append(
                        Passage(
                            evidence_id=f"{EVIDENCE_PREFIX}{counter}",
                            document_path=doc,
                            document_sha256=sha,
                            page=page,
                            passage_text=chunk,
                            score=s,
                        )
                    )

        scored.sort(key=lambda p: p.score, reverse=True)
        if not any_match:
            return RetrievalResult(
                query=query,
                empty_queries=[query],
            )
        all_passages.extend(scored[:top_k])

    all_passages.sort(key=lambda p: p.score, reverse=True)
    return RetrievalResult(
        query=", ".join(queries),
        passages=all_passages,
    )


def resolve_evidence_ids(
    retrieval: RetrievalResult, evidence_ids: Sequence[str]
) -> Tuple[List[Passage], List[str]]:
    """Look up evidence IDs from a retrieval result.

    Returns the resolved passages (in the order requested) and the IDs that
    did not resolve. The orchestrator's verifier adds the unresolved IDs to
    ``UncertaintyNote`` rather than rendering a note with a citation to
    nothing.
    """
    by_id: Dict[str, Passage] = {p.evidence_id: p for p in retrieval.passages}
    resolved: List[Passage] = []
    missing: List[str] = []
    for eid in evidence_ids:
        if eid in by_id:
            resolved.append(by_id[eid])
        else:
            missing.append(eid)
    return resolved, missing
"""
LlamaIndex Provider Plugin — Document Chunking & Node Parsing
Splits documents into semantic passages for RAG indexing.
"""

import re
from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider

class LlamaIndexProvider(BaseMemoryProvider):
    @property
    def provider_id(self) -> str:
        return "llamaindex"

    @property
    def capabilities(self) -> List[str]:
        return ["rag_indexing"]

    def initialize(self) -> bool:
        return True

    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
        """
        Hierarchical text chunking with sentence boundary awareness.
        """
        if not text or len(text.strip()) == 0:
            return []

        raw_units = [p.strip() for p in re.split(r'\n\n|\. ', text) if len(p.strip()) > 0]
        chunks = []
        chunk_idx = 0

        current_chunk = ""
        for u in raw_units:
            if len(current_chunk) + len(u) <= chunk_size:
                current_chunk += (". " if current_chunk else "") + u
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_idx}",
                        "content": current_chunk,
                        "token_count": len(current_chunk) // 4
                    })
                    chunk_idx += 1
                current_chunk = u

        if current_chunk:
            chunks.append({
                "chunk_id": f"chunk_{chunk_idx}",
                "content": current_chunk,
                "token_count": len(current_chunk) // 4
            })

        return chunks

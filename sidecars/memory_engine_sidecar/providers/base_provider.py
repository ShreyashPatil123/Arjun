"""
Base Memory Provider Interface
Defines the standard abstract plugin contract for all sidecar memory providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseMemoryProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g. 'rule_extractor', 'zep', 'llamaindex')."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of supported capabilities (e.g. ['extraction', 'summarization', 'ranking', 'rag_indexing'])."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initializes the provider. Returns True if ready."""
        pass

    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts facts from input text."""
        return []

    def summarize_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Summarizes conversation history."""
        return {"summary": ""}

    def calculate_rankings(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Ranks candidates by relevance and recency."""
        return candidates

    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
        """Chunks document for RAG indexing."""
        return []

"""
Abstract Base Provider for Memory Engine Sidecar
Frameworks act strictly as pure processors and return structured JSON output.
Frameworks NEVER access storage directly.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseMemoryProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique ID of the provider (e.g. 'mem0', 'letta', 'zep', 'llamaindex')"""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of capabilities provided ('extraction', 'compression', 'summarization', 'rag_indexing')"""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize provider resources. Return True if ready, False if unavailable."""
        pass

    def extract_facts(self, text: str, existing_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract facts and importance scores from text."""
        return []

    def compress_context(self, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        """Compress context window into working memory block."""
        return {"compressed_text": "", "tokens_used": 0}

    def summarize_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate rolling conversation summary."""
        return {"summary": ""}

    def calculate_rankings(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Calculate hybrid relevance, recency decay, and importance score."""
        return candidates

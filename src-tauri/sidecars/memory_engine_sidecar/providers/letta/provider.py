"""
Letta Provider Plugin — Core Working Memory Block Management
Adapts Letta OS-like working memory hierarchy & context budgeting.
"""

from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider

class LettaProvider(BaseMemoryProvider):
    @property
    def provider_id(self) -> str:
        return "letta"

    @property
    def capabilities(self) -> List[str]:
        return ["compression", "working_memory"]

    def initialize(self) -> bool:
        return True

    def compress_context(self, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        """
        Compresses conversation messages into a compact working memory block when context window overflows.
        """
        if not messages:
            return {"compressed_text": "", "tokens_used": 0, "evicted_turns": 0}

        # Keep recent turns, summarize older turns
        total_turns = len(messages)
        keep_turns = min(6, total_turns)
        older_turns = messages[:-keep_turns] if total_turns > keep_turns else []

        summary_parts = []
        for msg in older_turns:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            summary_parts.append(f"{role}: {content}")

        compressed_text = " | ".join(summary_parts)
        approx_tokens = len(compressed_text) // 4

        return {
            "compressed_text": compressed_text,
            "tokens_used": approx_tokens,
            "evicted_turns": len(older_turns),
            "retained_turns": keep_turns
        }

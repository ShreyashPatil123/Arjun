"""
Unified Action Router
Dispatches incoming Stdio NDJSON-RPC requests to registered capability providers.
"""

from typing import Dict, Any
from registry import CapabilityRegistry

class MemoryActionRouter:
    def __init__(self):
        self.registry = CapabilityRegistry()

    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "health_check":
            return self.registry.get_status()

        elif method == "extract_facts":
            text = params.get("text", "")
            context = params.get("context")
            provider = self.registry.get_provider_for_capability("extraction")
            if provider:
                facts = provider.extract_facts(text, context)
                return {"facts": facts, "provider_used": provider.provider_id}
            return {"facts": [], "provider_used": "none"}

        elif method == "compress_context":
            messages = params.get("messages", [])
            max_tokens = params.get("max_tokens", 4096)
            provider = self.registry.get_provider_for_capability("compression")
            if provider:
                res = provider.compress_context(messages, max_tokens)
                res["provider_used"] = provider.provider_id
                return res
            return {"compressed_text": "", "tokens_used": 0, "provider_used": "none"}

        elif method == "summarize_session":
            messages = params.get("messages", [])
            provider = self.registry.get_provider_for_capability("summarization")
            if provider:
                res = provider.summarize_conversation(messages)
                res["provider_used"] = provider.provider_id
                return res
            return {"summary": "", "provider_used": "none"}

        elif method == "calculate_rankings":
            candidates = params.get("candidates", [])
            query = params.get("query", "")
            provider = self.registry.get_provider_for_capability("ranking")
            if provider:
                ranked = provider.calculate_rankings(candidates, query)
                return {"ranked_candidates": ranked, "provider_used": provider.provider_id}
            return {"ranked_candidates": candidates, "provider_used": "none"}

        elif method == "chunk_document":
            text = params.get("text", "")
            chunk_size = params.get("chunk_size", 512)
            overlap = params.get("overlap", 64)
            provider = self.registry.get_provider_for_capability("rag_indexing")
            if provider and hasattr(provider, "chunk_document"):
                chunks = provider.chunk_document(text, chunk_size, overlap)
                return {"chunks": chunks, "provider_used": provider.provider_id}
            return {"chunks": [], "provider_used": "none"}

        else:
            raise ValueError(f"Unknown RPC method: '{method}'")

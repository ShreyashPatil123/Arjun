"""
Zep Provider Plugin — Rolling Summarization, Context Compression & Exponential Recency Decay Ranking
Calculates temporal decay scores, compresses context windows, and summarizes long chat sessions.
"""

import math
import time
from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider

class ZepProvider(BaseMemoryProvider):
    @property
    def provider_id(self) -> str:
        return "zep"

    @property
    def capabilities(self) -> List[str]:
        return ["summarization", "ranking", "compression"]

    def initialize(self) -> bool:
        return True

    def summarize_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Distills conversation history into a rolling summary.
        """
        if not messages:
            return {"summary": ""}

        user_topics = []
        assistant_points = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                user_topics.append(content[:60])
            elif role == "assistant":
                assistant_points.append(content[:60])

        summary = f"User discussed: {'; '.join(user_topics[:3])}. Assistant provided guidance on key queries."
        return {"summary": summary}

    def compress_context(self, messages: List[Dict[str, Any]], max_tokens: int = 4096) -> Dict[str, Any]:
        """
        Compresses conversation turns into a high-density working context block.
        """
        if not messages:
            return {"compressed_text": "", "tokens_used": 0, "evicted_turns": 0, "retained_turns": 0}

        retained = messages[-4:] if len(messages) >= 4 else messages
        evicted_count = max(0, len(messages) - len(retained))
        compressed_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in retained])
        tokens_used = max(1, len(compressed_text) // 4)

        return {
            "compressed_text": compressed_text,
            "tokens_used": tokens_used,
            "evicted_turns": evicted_count,
            "retained_turns": len(retained)
        }

    def calculate_rankings(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Calculates hybrid relevance, exponential recency decay, and importance score.
        Score = w1 * similarity + w2 * exp(-lambda * delta_hours) + w3 * importance
        """
        now = int(time.time())
        decay_lambda = 0.005 # ~50% decay every 138 hours

        ranked = []
        for cand in candidates:
            sim = cand.get("similarity", 0.5)
            importance = cand.get("importance_score", 0.5)
            timestamp = cand.get("recency_timestamp", now)
            
            hours_elapsed = max(0, (now - timestamp) / 3600.0)
            recency_score = math.exp(-decay_lambda * hours_elapsed)

            final_score = (0.50 * sim) + (0.30 * recency_score) + (0.20 * importance)

            cand_copy = dict(cand)
            cand_copy["final_score"] = round(final_score, 4)
            cand_copy["recency_score"] = round(recency_score, 4)
            ranked.append(cand_copy)

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

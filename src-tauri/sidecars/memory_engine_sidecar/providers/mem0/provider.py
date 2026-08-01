"""
Mem0 Provider Plugin — Dynamic Fact & Entity Extraction
Adapts Mem0 extraction algorithms locally without cloud API dependencies.
Returns structured results to Rust MemoryManager.
"""

import math
import re
from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider

class Mem0Provider(BaseMemoryProvider):
    @property
    def provider_id(self) -> str:
        return "mem0"

    @property
    def capabilities(self) -> List[str]:
        return ["extraction", "entity_graph"]

    def initialize(self) -> bool:
        # Check if upstream vendor/mem0 module is available or fallback to local parser
        try:
            import sys
            import os
            vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vendor/mem0"))
            if vendor_path not in sys.path and os.path.exists(vendor_path):
                sys.path.insert(0, vendor_path)
            return True
        except Exception:
            return True

    def extract_facts(self, text: str, existing_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extracts salient facts, user preferences, and entities with importance scoring.
        """
        extracted = []
        if not text or len(text.strip()) == 0:
            return extracted

        # Heuristic fact extraction patterns (Mem0 structured rule fallback)
        patterns = [
            (r"(?:my name is|i am|call me)\s+([A-Z][a-z]+)", "user_fact", "name", 0.95),
            (r"(?:i prefer|i like|i love|my favorite)\s+([^.,;!\n]+)", "preference", "preference", 0.85),
            (r"(?:i work on|i am building|my project is)\s+([^.,;!\n]+)", "project", "project_context", 0.90),
            (r"(?:i use|my stack is|i code in)\s+([^.,;!\n]+)", "tech_stack", "skills", 0.88),
            (r"(?:i live in|i am located in)\s+([^.,;!\n]+)", "user_fact", "location", 0.80),
        ]

        for pattern, memory_type, key, importance in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                value = m.group(1).strip()
                extracted.append({
                    "content": f"{key.replace('_', ' ').capitalize()}: {value}",
                    "memory_type": memory_type,
                    "key": key,
                    "value": value,
                    "importance_score": importance,
                    "confidence": 0.9,
                })

        # General sentence fact extraction if no specific pattern matched
        if len(extracted) == 0 and len(text) > 15:
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 10]
            for s in sentences[:3]:
                if any(kw in s.lower() for kw in ["i ", "my ", "we ", "our ", "prefer", "want", "need", "built", "using"]):
                    extracted.append({
                        "content": s,
                        "memory_type": "user_fact",
                        "key": "general_statement",
                        "value": s,
                        "importance_score": 0.65,
                        "confidence": 0.75,
                    })

        return extracted

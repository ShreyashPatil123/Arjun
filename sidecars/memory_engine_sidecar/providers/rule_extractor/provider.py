"""
Rule Extractor Provider Plugin — High-Precision Fact & Entity Extraction
Extracts user facts, preferences, background, devices, and project goals using regex & entity heuristics.
"""

import re
from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider

class RuleExtractorProvider(BaseMemoryProvider):
    @property
    def provider_id(self) -> str:
        return "rule_extractor"

    @property
    def capabilities(self) -> List[str]:
        return ["extraction"]

    def initialize(self) -> bool:
        return True

    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        facts: List[Dict[str, Any]] = []
        clean_text = text.strip()

        # Pattern 1: Name Extraction ("my name is X", "i am X", "call me X", "myself X")
        name_patterns = [
            r"(?:my name is|i am|call me|myself)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:my name is|i am|call me|myself)\s+([a-zA-Z]+)",
        ]
        for pat in name_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().title()
                # Exclude common non-name words
                if val.lower() not in ["a", "an", "the", "using", "testing", "working", "building", "developer", "student"]:
                    facts.append({
                        "content": f"User's name is {val}",
                        "memory_type": "user_fact",
                        "key": "name",
                        "value": val,
                        "importance_score": 0.98,
                        "confidence": 0.99
                    })
                    break

        # Pattern 2: Birthday Extraction ("my birthday is X", "born on X")
        bday_patterns = [
            r"(?:my birthday is|born on|bday is)\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|[a-z]{3})(?:\s+[0-9]{4})?)",
            r"(?:my birthday is|born on|bday is)\s+([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        ]
        for pat in bday_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                facts.append({
                    "content": f"User's birthday is {val}",
                    "memory_type": "user_fact",
                    "key": "birthday",
                    "value": val,
                    "importance_score": 0.95,
                    "confidence": 0.98
                })
                break

        # Pattern 3: Education / College / School ("i study at X", "student at X", "my college is X", "university X")
        edu_patterns = [
            r"(?:i study at|student at|my college is|my university is|studying at)\s+([A-Za-z0-9\s]+)",
        ]
        for pat in edu_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip(".,!")
                facts.append({
                    "content": f"User studies at {val}",
                    "memory_type": "user_fact",
                    "key": "education",
                    "value": val,
                    "importance_score": 0.92,
                    "confidence": 0.96
                })
                break

        # Pattern 4: Device / Hardware ("my laptop is X", "my pc is X", "i use a X")
        device_patterns = [
            r"(?:my laptop is|my pc is|my computer is|i use a|using a)\s+([A-Za-z0-9\s]+)",
        ]
        for pat in device_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip(".,!")
                if any(w in val.lower() for w in ["lenovo", "macbook", "dell", "hp", "asus", "acer", "laptop", "pc", "rtx"]):
                    facts.append({
                        "content": f"User's device is {val}",
                        "memory_type": "user_fact",
                        "key": "device",
                        "value": val,
                        "importance_score": 0.90,
                        "confidence": 0.95
                    })
                    break

        # Pattern 5: Preferences & Technology ("i like X", "i prefer X", "my favorite X is Y", "i code in X")
        pref_patterns = [
            r"(?:i like|i love|i prefer|i code in|programming in)\s+([A-Za-z0-9\s#\+]+)",
            r"my favorite\s+([a-z]+)\s+is\s+([A-Za-z0-9\s]+)",
        ]
        for pat in pref_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 1:
                    val = groups[0].strip().rstrip(".,!")
                    if val.lower() not in ["it", "this", "that", "them"]:
                        facts.append({
                            "content": f"User prefers {val}",
                            "memory_type": "preference",
                            "key": f"preference_{val.lower().replace(' ', '_')[:15]}",
                            "value": val,
                            "importance_score": 0.88,
                            "confidence": 0.94
                        })
                elif len(groups) == 2:
                    k, v = groups[0].strip().lower(), groups[1].strip().rstrip(".,!")
                    facts.append({
                        "content": f"User's favorite {k} is {v}",
                        "memory_type": "preference",
                        "key": k,
                        "value": v,
                        "importance_score": 0.90,
                        "confidence": 0.95
                    })
                break

        # Pattern 6: Project & Goal Extraction ("my project is X", "my active project is X", "working on X")
        proj_patterns = [
            r"(?:my project is|my active project is|working on|building)\s+([A-Za-z0-9\s]+)",
        ]
        for pat in proj_patterns:
            m = re.search(pat, clean_text, re.IGNORECASE)
            if m and not facts:
                val = m.group(1).strip().rstrip(".,!")
                facts.append({
                    "content": f"User's project is {val}",
                    "memory_type": "project_goal",
                    "key": "project_goal",
                    "value": val,
                    "importance_score": 0.90,
                    "confidence": 0.95
                })
                break

        # Pattern 7: General Fallback Fact ("remember that X", "note that X")
        remember_pattern = r"(?:remember that|note that|keep in mind that)\s+(.+)"
        m_rem = re.search(remember_pattern, clean_text, re.IGNORECASE)
        if m_rem and not facts:
            val = m_rem.group(1).strip().rstrip(".,!")
            facts.append({
                "content": val,
                "memory_type": "user_fact",
                "key": "user_note",
                "value": val,
                "importance_score": 0.85,
                "confidence": 0.90
            })

        return facts

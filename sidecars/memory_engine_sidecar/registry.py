"""
Dynamic Capability Registry
Manages discovery, capability mapping, health monitoring, and fallback execution of memory providers.
"""

from typing import Dict, Any, List, Optional
from providers.base_provider import BaseMemoryProvider
from providers.mem0.provider import Mem0Provider
from providers.letta.provider import LettaProvider
from providers.zep.provider import ZepProvider
from providers.llamaindex.provider import LlamaIndexProvider

class CapabilityRegistry:
    def __init__(self):
        self.providers: Dict[str, BaseMemoryProvider] = {}
        self.capability_map: Dict[str, List[str]] = {}
        self._register_default_providers()

    def _register_default_providers(self):
        """Discovers and registers built-in provider plugins."""
        defaults = [
            Mem0Provider(),
            LettaProvider(),
            ZepProvider(),
            LlamaIndexProvider()
        ]

        for p in defaults:
            try:
                if p.initialize():
                    self.providers[p.provider_id] = p
                    for cap in p.capabilities:
                        if cap not in self.capability_map:
                            self.capability_map[cap] = []
                        self.capability_map[cap].append(p.provider_id)
            except Exception as e:
                print(f"[REGISTRY WARN] Failed to initialize provider {p.provider_id}: {e}")

    def get_provider_for_capability(self, capability: str) -> Optional[BaseMemoryProvider]:
        """Returns the best available provider registered for a capability."""
        provider_ids = self.capability_map.get(capability, [])
        for pid in provider_ids:
            if pid in self.providers:
                return self.providers[pid]
        return None

    def get_status(self) -> Dict[str, Any]:
        """Returns active health status of registry and providers."""
        return {
            "registered_providers": list(self.providers.keys()),
            "capabilities": {k: list(v) for k, v in self.capability_map.items()},
            "status": "healthy" if len(self.providers) > 0 else "degraded"
        }

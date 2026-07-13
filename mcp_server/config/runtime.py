"""Mutable process-wide settings shared by MCP handlers and task skills."""

import os
import threading
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RuntimeConfig:
    """Thread-safe settings that may be changed while the server is running."""

    _vlm_fallback: bool = field(default_factory=lambda: _env_bool("VLM_FALLBACK", True))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def vlm_fallback(self) -> bool:
        with self._lock:
            return self._vlm_fallback

    def set_vlm_fallback(self, enabled: bool) -> None:
        value = bool(enabled)
        with self._lock:
            self._vlm_fallback = value
        # Keep child processes and diagnostics consistent with the live value.
        os.environ["VLM_FALLBACK"] = "1" if value else "0"

    def snapshot(self) -> dict:
        return {"vlm_fallback": self.vlm_fallback}


runtime_config = RuntimeConfig()

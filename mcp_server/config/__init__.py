"""Shared runtime configuration."""

from .runtime import RuntimeConfig, runtime_config
from .paths import PROJECT_ROOT, TMP_DIR, runtime_dir

__all__ = [
    "PROJECT_ROOT",
    "RuntimeConfig",
    "TMP_DIR",
    "runtime_config",
    "runtime_dir",
]

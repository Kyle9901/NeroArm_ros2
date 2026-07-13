"""Filesystem locations shared by runtime components and diagnostics."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = Path(os.environ.get("VISION_GRASP_TMP_DIR", PROJECT_ROOT / "tmp")).resolve()


def runtime_dir(name: str) -> Path:
    """Return a named runtime directory below ``tmp`` and create it."""
    path = TMP_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path

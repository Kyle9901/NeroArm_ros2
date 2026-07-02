"""Shared result types for atomic components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    fatal: bool = False

    @classmethod
    def success(cls, **data: Any) -> "ComponentResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str, *, fatal: bool = False, **data: Any) -> "ComponentResult":
        return cls(ok=False, data=data, error=error, fatal=fatal)


@dataclass
class ImageFrame:
    frame_id: int
    color: Any
    depth: Any
    timestamp_s: float

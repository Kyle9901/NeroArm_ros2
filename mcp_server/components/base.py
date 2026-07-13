"""Shared result types for atomic components."""

from dataclasses import dataclass
from typing import Any

from ..models import OperationResult


@dataclass
class ComponentResult(OperationResult):
    """Result of one atomic hardware or perception operation."""

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

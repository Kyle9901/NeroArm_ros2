"""Common result protocol for component and skill operations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperationResult:
    """Fields shared by every directly executed robot operation."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.ok, "data": self.data, "error": self.error}

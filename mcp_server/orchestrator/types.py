"""Shared orchestration types."""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 1
    recover: str | None = None
    give_up: Literal["report", "skip", "abort"] = "abort"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class Step:
    name: str
    skill: str | None = None
    fn: Callable[..., Any] | None = None
    args_from: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    cond: str | None = None
    on_multiple: Literal["auto", "ask"] = "auto"


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    description: str
    match_patterns: list[str]
    required_params: list[ParamSpec]
    pipeline: list[Step]
    retry_policy: dict[str, RetryConfig] = field(default_factory=dict)
    user_visible: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    status: Literal["completed", "failed", "needs_input"]
    template: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    user_output: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    error: str | None = None
    session_id: str | None = None
    question: str | None = None
    options: list[dict[str, Any]] = field(default_factory=list)

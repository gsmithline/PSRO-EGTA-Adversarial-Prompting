"""Core typed data structures for rainbow teaming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple


@dataclass
class Prompt:
    """Represents a textual prompt sent to a language model."""

    text: str


@dataclass
class ModelResponse:
    """Container for a model response and associated metadata."""

    prompt: Prompt
    raw_text: str
    meta: Dict[str, Any]


@dataclass
class JudgeResult:
    """Result of evaluating a model response with the judge."""

    quality: float
    descriptors: Tuple[float, ...]
    meta: Dict[str, Any]


class LanguageModel(Protocol):
    """Protocol for language model backends."""

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text from the model given a prompt."""


class Judge(Protocol):
    """Protocol for evaluating model responses."""

    def evaluate(self, response: ModelResponse) -> JudgeResult:
        """Return a quality score and descriptor vector for a response."""

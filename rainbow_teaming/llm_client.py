"""Language model client abstractions and OpenAI implementation."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from .types import LanguageModel

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Generic error for LLM interactions."""


@dataclass
class OpenAIChatModel(LanguageModel):
    """Thin wrapper around the OpenAI Chat Completions API.

    The implementation is deliberately minimal and can be swapped out for other
    providers by implementing :class:`~rainbow_teaming.types.LanguageModel`.
    """

    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: int = 60
    max_retries: int = 3
    backoff_seconds: float = 2.0

    def generate(self, prompt: str, *, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate text using the OpenAI chat API.

        Args:
            prompt: User prompt string.
            max_tokens: Optional override for maximum tokens.
            temperature: Optional override for sampling temperature.

        Returns:
            Raw generated string from the model.

        Raises:
            LLMError: If the API call fails after retries.
        """

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMError(f"Missing API key in environment variable {self.api_key_env}")

        try:
            import openai
        except Exception as exc:  # pragma: no cover - depends on openai package
            raise LLMError("openai package is required for OpenAIChatModel") from exc

        openai.api_key = api_key
        chosen_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        chosen_temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=chosen_max_tokens,
                    temperature=chosen_temp,
                    timeout=self.request_timeout,
                )
                content = response["choices"][0]["message"]["content"]
                return content
            except Exception as exc:  # pragma: no cover - network/HTTP errors
                logger.warning("OpenAI API error on attempt %s/%s: %s", attempt + 1, self.max_retries, exc)
                if attempt + 1 >= self.max_retries:
                    logger.error("Exhausted retries for OpenAI API call")
                    raise LLMError("OpenAI API call failed") from exc
                time.sleep(self.backoff_seconds * (attempt + 1))

        raise LLMError("OpenAI API call failed without exception")


__all__ = ["OpenAIChatModel", "LLMError"]

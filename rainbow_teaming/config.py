"""Configuration objects for rainbow teaming search."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class Config:
    """Main configuration for running rainbow teaming search.

    Attributes:
        target_model_name: Name of the target LLM to attack.
        judge_model_name: Name of the model used by the judge.
        mutator_model_name: Name of the model used to generate new prompts.
        descriptor_mins: Minimum values for each descriptor dimension.
        descriptor_maxes: Maximum values for each descriptor dimension.
        descriptor_bins: Number of bins per descriptor dimension.
        max_iterations: Maximum number of search iterations.
        offspring_per_iteration: Number of candidate prompts produced each iteration.
        parent_sample_fraction: Fraction of archive cells to sample parents from.
        max_prompt_length: Maximum allowed prompt length in characters.
        logging_dir: Directory for logs and checkpoints.
        random_seed: Optional random seed for reproducibility.
        max_target_calls: Optional budget for target model calls.
    """

    target_model_name: str
    judge_model_name: str
    mutator_model_name: str

    descriptor_mins: Sequence[float] = field(default_factory=lambda: (0.0, 0.0))
    descriptor_maxes: Sequence[float] = field(default_factory=lambda: (1.0, 1.0))
    descriptor_bins: Sequence[int] = field(default_factory=lambda: (10, 10))

    max_iterations: int = 10
    offspring_per_iteration: int = 5
    parent_sample_fraction: float = 0.5
    max_prompt_length: int = 512
    logging_dir: Path | str = Path("./logs")
    random_seed: int | None = None
    max_target_calls: int | None = None

    def grid_shape(self) -> Tuple[int, ...]:
        """Return the grid shape implied by descriptor_bins."""

        return tuple(int(b) for b in self.descriptor_bins)


__all__ = ["Config"]

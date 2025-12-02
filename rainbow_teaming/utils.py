"""Small helper utilities for rainbow teaming."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str | Path, data: Any) -> None:
    """Save data as JSON to the given path."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON data from the given path."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["set_seed", "save_json", "load_json"]

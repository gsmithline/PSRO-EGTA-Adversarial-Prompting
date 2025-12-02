"""MAP-Elites archive implementation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .types import JudgeResult, Prompt


@dataclass
class ArchiveCell:
    """Container stored in each archive cell."""

    prompt: Prompt
    judge_result: JudgeResult


class MapElitesArchive:
    """Simple grid-based MAP-Elites archive."""

    def __init__(self, descriptor_mins: Sequence[float], descriptor_maxes: Sequence[float], bins: Sequence[int]):
        self.descriptor_mins = tuple(descriptor_mins)
        self.descriptor_maxes = tuple(descriptor_maxes)
        self.bins = tuple(bins)
        self.grid: Dict[Tuple[int, ...], ArchiveCell] = {}

    def _descriptor_to_index(self, descriptors: Tuple[float, ...]) -> Tuple[int, ...]:
        indices = []
        for value, min_val, max_val, num_bins in zip(descriptors, self.descriptor_mins, self.descriptor_maxes, self.bins):
            if max_val == min_val:
                idx = 0
            else:
                normalized = (value - min_val) / (max_val - min_val)
                normalized = max(0.0, min(1.0, normalized))
                idx = int(normalized * num_bins)
                if idx == num_bins:
                    idx = num_bins - 1
            indices.append(idx)
        return tuple(indices)

    def add(self, candidate: ArchiveCell) -> bool:
        """Insert a candidate into the archive if it improves that cell."""

        index = self._descriptor_to_index(candidate.judge_result.descriptors)
        existing = self.grid.get(index)
        if existing is None or candidate.judge_result.quality > existing.judge_result.quality:
            self.grid[index] = candidate
            return True
        return False

    def sample_parents(self, k: int) -> List[ArchiveCell]:
        cells = list(self.grid.values())
        if not cells:
            return []
        k = min(k, len(cells))
        return random.sample(cells, k)

    def all_cells(self) -> List[ArchiveCell]:
        return list(self.grid.values())

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        cells: List[Dict[str, object]] = []
        for index, cell in self.grid.items():
            cells.append(
                {
                    "index": list(index),
                    "prompt": cell.prompt.text,
                    "quality": cell.judge_result.quality,
                    "descriptors": list(cell.judge_result.descriptors),
                    "meta": cell.judge_result.meta,
                }
            )
        return {"bins": list(self.bins), "descriptor_mins": list(self.descriptor_mins), "descriptor_maxes": list(self.descriptor_maxes), "cells": cells}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MapElitesArchive":
        archive = cls(data["descriptor_mins"], data["descriptor_maxes"], data["bins"])
        for cell_data in data.get("cells", []):
            prompt = Prompt(text=cell_data["prompt"])
            judge_result = JudgeResult(
                quality=float(cell_data["quality"]),
                descriptors=tuple(cell_data["descriptors"]),
                meta=dict(cell_data.get("meta", {})),
            )
            archive.grid[tuple(cell_data["index"])] = ArchiveCell(prompt=prompt, judge_result=judge_result)
        return archive


__all__ = ["ArchiveCell", "MapElitesArchive"]

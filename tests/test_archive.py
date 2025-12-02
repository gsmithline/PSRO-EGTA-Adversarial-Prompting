import pytest

from rainbow_teaming.archive import ArchiveCell, MapElitesArchive
from rainbow_teaming.types import JudgeResult, Prompt


def make_cell(prompt_text: str, quality: float, descriptors):
    return ArchiveCell(prompt=Prompt(prompt_text), judge_result=JudgeResult(quality, descriptors, {}))


def test_add_and_replace():
    archive = MapElitesArchive((0, 0), (1, 1), (2, 2))
    cell1 = make_cell("a", 0.5, (0.1, 0.1))
    cell2 = make_cell("b", 0.7, (0.1, 0.1))
    assert archive.add(cell1) is True
    assert archive.add(cell2) is True  # higher quality replaces
    assert archive.all_cells()[0].prompt.text == "b"


def test_clamping_edges():
    archive = MapElitesArchive((0, 0), (1, 1), (2, 2))
    cell = make_cell("edge", 0.5, (1.5, -0.5))
    assert archive.add(cell) is True
    # descriptor 1.5 should clamp to last bin index 1
    index = next(iter(archive.grid.keys()))
    assert index == (1, 0)

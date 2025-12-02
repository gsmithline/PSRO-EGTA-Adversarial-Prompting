from dataclasses import dataclass
from typing import List

from rainbow_teaming.config import Config
from rainbow_teaming.search import run_rainbow_teaming
from rainbow_teaming.types import Judge, JudgeResult, LanguageModel, ModelResponse, Prompt


@dataclass
class MockModel(LanguageModel):
    response_text: str

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:  # type: ignore[override]
        return f"{self.response_text}:{prompt[:10]}"


@dataclass
class MockJudge(Judge):
    qualities: List[float]
    descriptors: List[tuple]
    idx: int = 0

    def evaluate(self, response: ModelResponse) -> JudgeResult:  # type: ignore[override]
        quality = self.qualities[self.idx % len(self.qualities)]
        descriptor = self.descriptors[self.idx % len(self.descriptors)]
        self.idx += 1
        return JudgeResult(quality=quality, descriptors=descriptor, meta={})


def test_run_rainbow_teaming_mock():
    config = Config(
        target_model_name="mock-target",
        judge_model_name="mock-judge",
        mutator_model_name="mock-mutator",
        max_iterations=2,
        offspring_per_iteration=2,
        parent_sample_fraction=1.0,
        max_target_calls=10,
        descriptor_bins=(2, 2),
    )
    target = MockModel("resp")
    mutator = MockModel("mut")
    judge = MockJudge(qualities=[0.1, 0.9], descriptors=[(0.0, 0.0), (1.0, 1.0)])
    initial_prompts = [Prompt("seed1"), Prompt("seed2")]

    archive = run_rainbow_teaming(config, target, mutator, judge, initial_prompts)
    cells = archive.all_cells()
    assert len(cells) > 0
    qualities = {cell.prompt.text: cell.judge_result.quality for cell in cells}
    assert any(q == 0.9 for q in qualities.values())

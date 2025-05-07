from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path

from eval_suite.benchmark import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalStatBase,
)
from eval_suite.benchmark.stat.score import ScoreStat
from eval_suite.client import ClientBase


class EvalOutput(EvalOutputBase):
    content: str


class EvalResult(EvalResultBase):
    score: float


JudgeScoreStat = ScoreStat


class Benchmark[Input: EvalInputBase, Stat: EvalStatBase](
    BenchmarkBase[Input, EvalOutput, EvalResult, Stat, BaseEvalConfig]
):
    eval_config: BaseEvalConfig = BaseEvalConfig()
    judge: ClientBase
    """Judge client to evaluate the model output"""

    # def __init__(
    #     self,
    #     dataset: Sequence[Any],
    #     judge: ClientBase,
    #     *,
    #     name: str = "llm_judge",
    #     config: BaseEvalConfig = BaseEvalConfig(),
    #     base_path: Path | None = None,
    # ):
    #     super().__init__(
    #         dataset=dataset,
    #         name=name,
    #         eval_config=config,
    #         base_path=base_path,
    #     )

    #     self._judge = judge

    @abstractmethod
    def to_judge_result(self, content: str) -> EvalResult:
        """Convert the judge response to a result."""

    async def to_result_batch_async(
        self,
        eval_paths: Sequence[Path],
        inputs: Sequence[Input],
        outputs: Sequence[EvalOutput],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for output in outputs]
        msgs = await self.judge.generate(
            contents,
            system_prompt=self.eval_config.system_prompt,
        )

        results = [
            self.to_judge_result(msg.content)
            if msg
            else ValueError("No content found in the generation")
            for msg in msgs
        ]

        return results

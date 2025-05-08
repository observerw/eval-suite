from abc import abstractmethod
from collections.abc import Sequence

from eval_suite.benchmark import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalStatBase,
    ToResultArgs,
)
from eval_suite.client import ClientBase


class EvalOutput(EvalOutputBase):
    content: str


class EvalResult(EvalResultBase):
    score: float


class Benchmark[Input: EvalInputBase, Stat: EvalStatBase](
    BenchmarkBase[Input, EvalOutput, EvalResult, Stat, BaseEvalConfig]
):
    """
    Benchmark using an LLM to evaluate the model output.

    Args:
        judge: ClientBase
            Judge client to evaluate the model

    Implemented:
        to_result:
            Judge the given content and return a score.
    """

    eval_config: BaseEvalConfig = BaseEvalConfig()
    judge: ClientBase
    """Judge client to evaluate the model output"""

    @abstractmethod
    def to_judge_result(self, content: str) -> EvalResult:
        """Convert the judge response to a result."""

    async def to_result_batch_async(
        self,
        args: Sequence[ToResultArgs[Input, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for _, _, output in args]
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

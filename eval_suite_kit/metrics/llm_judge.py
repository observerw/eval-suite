from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import override

from eval_suite.client import ClientBase, Message
from eval_suite.metric import (
    EvalInputBase,
    EvalOutputBase,
    ToOutput,
    ToResultArgs,
)
from eval_suite_kit.metrics import score


class EvalOutput(EvalOutputBase):
    content: str


EvalResult = score.EvalResult


class Metric[Input: EvalInputBase](
    score.Metric[Input, EvalOutput],
    ToOutput[Input, EvalOutput],
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

    judge: ClientBase
    """Judge client to evaluate the model output"""

    @abstractmethod
    def to_judge_result(self, generation: Message) -> EvalResult:
        """Convert the judge response to a result."""

    @override
    async def to_result_batch_async(
        self,
        args: Iterable[ToResultArgs[EvalInputBase, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for _, _, output in args]
        msgs = await self.judge.generate(contents)

        results = [
            self.to_judge_result(msg)
            if msg
            else ValueError("No content found in the generation")
            for msg in msgs
        ]

        return results

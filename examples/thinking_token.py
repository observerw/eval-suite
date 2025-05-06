from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tokenizers import Tokenizer

from eval_suite.benchmark import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
)
from eval_suite.client import Message


class EvalInput(EvalInputBase):
    pass


class EvalOutput(EvalOutputBase):
    content: str


class EvalResult(EvalResultBase):
    content: str
    length: int


class EvalStat(BaseModel):
    avg_token_length: float
    token_count: dict[str, int]

    @classmethod
    def from_groups(cls, groups: EvalResultGroups[EvalResult]):
        results = [result for group in groups.values() for result in group]

        avg_token_length = (
            sum(result.length for result in results) / len(results) if results else 0
        )

        token_count = {}
        for result in results:
            for token in result.content.split():
                token_count[token] = token_count.setdefault(token, 0) + 1

        return cls(
            avg_token_length=avg_token_length,
            token_count=token_count,
        )


class ThinkingTokenBenchmark(
    BenchmarkBase[
        EvalInput,
        EvalOutput,
        EvalResult,
        EvalStat,
        BaseEvalConfig,
    ]
):
    def __init__(
        self,
        dataset: Sequence[Any],
        tokenizer: Tokenizer,
        *,
        name: str = "thinking-token",
        config: BaseEvalConfig = BaseEvalConfig(),
        base_path: Path | None = None,
    ):
        super().__init__(
            dataset=dataset,
            name=name,
            config=config,
            base_path=base_path,
        )

        self._tokenizer = tokenizer

    def to_output(self, generation: Message, input: EvalInput) -> EvalOutput:
        if not (thinking := generation.reasoning_content):
            raise ValueError("No reasoning content found in the generation")

        return EvalOutput(content=thinking)

    def to_result_batch(
        self,
        eval_paths: Sequence[Path],
        inputs: Sequence[EvalInput],
        outputs: Sequence[EvalOutput],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for output in outputs]
        tokenized = self._tokenizer.encode_batch(contents)

        return [
            EvalResult(
                content=content,
                length=len(tokenized.input_ids),
            )
            for content, tokenized in zip(contents, tokenized)
        ]


async def main():
    pass

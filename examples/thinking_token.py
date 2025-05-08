from collections.abc import Sequence

from tokenizers import Tokenizer

from eval_suite.benchmark import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    ToResultArgs,
)
from eval_suite.benchmark.stat._base import BaseStat
from eval_suite.client import Message


class EvalInput(EvalInputBase):
    pass


class EvalOutput(EvalOutputBase):
    content: str


class EvalResult(EvalResultBase):
    content: str
    length: int


class EvalStat(EvalStatBase):
    base: BaseStat
    avg_token_length: float
    token_count: dict[str, int]


class EvalConfig(BaseEvalConfig):
    pass


class ThinkingTokenBenchmark(
    BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat, EvalConfig]
):
    tokenizer: Tokenizer

    def to_output(self, generation: Message, input: EvalInput) -> EvalOutput:
        if not (thinking := generation.reasoning_content):
            raise ValueError("No reasoning content found in the generation")

        return EvalOutput(content=thinking)

    def to_result_batch(
        self,
        args: Sequence[ToResultArgs[EvalInput, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for _, _, output in args]
        tokenized = self.tokenizer.encode_batch(contents)

        return [
            EvalResult(
                content=content,
                length=len(tokenized.input_ids),
            )
            for content, tokenized in zip(contents, tokenized)
        ]

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        results = [result for group in groups.values() for result in group]

        avg_token_length = (
            sum(result.length for result in results) / len(results) if results else 0
        )

        token_count = {
            token: sum(result.content.count(token) for result in results)
            for result in results
            for token in result.content.split()
        }

        return EvalStat(
            base=base,
            avg_token_length=avg_token_length,
            token_count=token_count,
        )


async def main():
    pass

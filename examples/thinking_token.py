import asyncio as aio
from collections.abc import Iterable, Sequence
from functools import cached_property
from pathlib import Path
from typing import override

from transformers.models.auto.tokenization_auto import AutoTokenizer

from eval_suite.benchmark import BenchmarkBase
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.metric import (
    BaseStat,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    MetricBase,
    ToResult,
    ToResultArgs,
    ToResultList,
    ToStat,
)
from eval_suite.utils.dataset import load_repo_dataset


class EvalInput(EvalInputBase):
    pass


class EvalOutput(EvalOutputBase):
    content: str


class EvalResult(EvalResultBase):
    content: str
    length: int


class ThinkingTokenStat(EvalStatBase):
    avg_token_length: float
    token_count: dict[str, int]


class EvalStat(EvalStatBase):
    base: BaseStat
    thinking_token: ThinkingTokenStat


class ThinkingTokenMetric(
    MetricBase,
    ToResult[EvalInput, EvalOutput, EvalResult],
    ToStat[EvalResult, ThinkingTokenStat],
):
    tokenizer_path: Path

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    @override
    def to_result_batch_sync(
        self,
        args: Iterable[ToResultArgs[EvalInput, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for _, _, output in args]
        tokenized = self.tokenizer(contents)

        return [
            EvalResult(content=content, length=len(tokenized.input_ids))
            for content, tokenized in zip(contents, tokenized)
        ]

    @override
    def to_stat(self, groups: EvalResultGroups[EvalResult]) -> ThinkingTokenStat:
        results = [result for group in groups.values() for result in group]

        avg_token_length = (
            sum(result.length for result in results) / len(results) if results else 0
        )

        token_count = {
            token: sum(result.content.count(token) for result in results)
            for result in results
            for token in result.content.split()
        }

        return ThinkingTokenStat(
            avg_token_length=avg_token_length,
            token_count=token_count,
        )


class ThinkingTokenBenchmark(
    BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]
):
    metric: ThinkingTokenMetric

    @override
    def to_output(self, generation: Message, input: EvalInput) -> EvalOutput:
        if not (thinking := generation.reasoning_content):
            raise ValueError("No reasoning content found in the generation")

        return EvalOutput(content=thinking)

    @override
    async def to_result(
        self, args: Iterable[ToResultArgs[EvalInput, EvalOutput]]
    ) -> ToResultList[EvalResult]:
        return await self.metric.to_result(args)

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        return EvalStat(
            base=base,
            thinking_token=self.metric.to_stat(groups),
        )


async def main():
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")
    model_path = Path("Qwen/Qwen3-4B")
    output_path = Path("output")

    with (
        ThinkingTokenBenchmark(
            dataset=dataset.to_list(),
            base_path=output_path,
            metric=ThinkingTokenMetric(tokenizer_path=model_path),
        ) as benchmark,
        SGLangClient(
            server_args=EvalServerArgs(
                model_path=str(model_path),
                dp_size=4,
                tp_size=2,
            ),
            sampling_params=SGLangSamplingParams(
                temperature=0.7,
                top_p=0.8,
                max_new_tokens=8192,
                repetition_penalty=1.05,
                top_k=20,
            ),
            config=BaseClientConfig(
                batch_size=2048,
            ),
        ) as client,
    ):
        stat = await benchmark.run(client)
        print(stat)


if __name__ == "__main__":
    aio.run(main())

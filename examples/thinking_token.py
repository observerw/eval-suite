import asyncio as aio
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from eval_suite.benchmark import (
    BaseEvalConfig,
    BaseStat,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    ToResultArgs,
)
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.utils.dataset import load_repo_dataset


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
    eval_config: EvalConfig = EvalConfig()

    tokenizer: Tokenizer

    def to_output(self, generation: Message, input: EvalInput) -> EvalOutput:
        if not (thinking := generation.reasoning_content):
            raise ValueError("No reasoning content found in the generation")

        return EvalOutput(content=thinking)

    def to_result_batch_sync(
        self,
        args: Sequence[ToResultArgs[EvalInput, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        contents = [output.content for _, _, output in args]
        tokenized = self.tokenizer.encode_batch(contents)

        return [
            EvalResult(content=content, length=len(tokenized.input_ids))
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
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    with (
        ThinkingTokenBenchmark(
            dataset=dataset.to_list(),
            tokenizer=tokenizer,
            base_path=Path("output"),
        ) as benchmark,
        SGLangClient(
            server_args=EvalServerArgs(
                model_path="Qwen/Qwen3-4B",
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

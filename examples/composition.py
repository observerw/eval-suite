import asyncio as aio
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

from eval_suite.benchmark import BenchmarkBase
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.dummy import DummyClient
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.metric import (
    BaseStat,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
)
from eval_suite.metric.result import ToResultArgs, ToResultList
from eval_suite.utils.dataset import load_repo_dataset
from eval_suite.utils.extract import extract_code
from eval_suite_kit.metrics import llm_judge, pass_k, score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="humaneval.log",
    filemode="w",
)


class EvalInput(EvalInputBase):
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @property
    def input_id(self) -> str:
        return self.task_id

    def __str__(self) -> str:
        return self.prompt


class EvalOutput(EvalOutputBase):
    judge: llm_judge.EvalOutput


class EvalResult(EvalResultBase):
    pass_k: pass_k.EvalResult
    judge: llm_judge.EvalResult


class EvalStat(EvalStatBase):
    base: BaseStat
    passk: pass_k.Stat
    judge: score.Stat


class PassKMetric(pass_k.Metric[EvalInput, EvalOutput]):
    @override
    async def to_result_async(
        self,
        eval_path: Path,
        input: EvalInput,
        output: EvalOutput,
    ) -> pass_k.EvalResult:
        return pass_k.EvalResult(passed=True)


class LLMJudgeMetric(llm_judge.Metric):
    @override
    def to_output(self, generation: Message, input: Any) -> llm_judge.EvalOutput:
        return llm_judge.EvalOutput(content=generation.content)

    @override
    def to_judge_result(self, generation: Message) -> llm_judge.EvalResult:
        return llm_judge.EvalResult(score=1.0)


class CompBenchmark(BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]):
    judge_metric: llm_judge.Metric
    pass_k_metric: pass_k.Metric

    @override
    def to_output(
        self,
        generation: Message[dict],
        input: EvalInput,
    ) -> EvalOutput:
        if not (result := extract_code(generation.content).get("python", id="result")):
            raise ValueError("No code found in the generation")

        return EvalOutput(
            judge=self.judge_metric.to_output(generation, input),
        )

    @override
    async def to_result(
        self, args: Iterable[ToResultArgs[EvalInput, EvalOutput]]
    ) -> ToResultList[EvalResult]:
        judge_results = await self.judge_metric.to_result(
            ToResultArgs(eval_path, input, output.judge)
            for eval_path, input, output in args
        )
        pass_k_results = await self.pass_k_metric.to_result(args)

        return EvalResult.merge(
            pass_k=pass_k_results,
            judge=judge_results,
        )

    @override
    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        return EvalStat(
            base=base,
            passk=self.pass_k_metric.to_stat(groups.map(lambda r: r.pass_k)),
            judge=self.judge_metric.to_stat(groups.map(lambda r: r.judge)),
        )


async def main():
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")

    with (
        SGLangClient(
            server_args=EvalServerArgs(
                model_path="Qwen/Qwen2.5-32B-Instruct",
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
        DummyClient() as judge,
        CompBenchmark(
            dataset=dataset.to_list(),
            base_path=Path("output"),
            judge_metric=LLMJudgeMetric(judge=judge),
            pass_k_metric=PassKMetric(k=5),
        ) as benchmark,
    ):
        stat = await benchmark.run(client)
        print(stat)


if __name__ == "__main__":
    aio.run(main())

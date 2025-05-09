import os
from collections.abc import Iterable
from typing import Self, override

from pydantic import BaseModel, SecretStr

from eval_suite.benchmark import BenchmarkBase
from eval_suite.client.base import Message
from eval_suite.client.openai import OpenAIClient
from eval_suite.metric import (
    BaseStat,
    EvalInputBase,
    EvalOutputBase,
    EvalResultGroups,
    EvalStatBase,
    ToResultList,
)
from eval_suite.metric.result import ToResultArgs
from eval_suite.utils.dataset import load_dataset
from eval_suite_kit.metrics import pass_k


class EvalInput(EvalInputBase):
    question_title: str
    question_content: str
    platform: str
    question_id: str
    contest_id: str
    contest_date: str
    starter_code: str
    difficulty: str
    public_test_cases: str
    private_test_cases: str
    metadata: str

    @property
    @override
    def input_id(self) -> str:
        return self.question_id

    @override
    def __str__(self) -> str:
        raise NotImplementedError


class EvalOutput(EvalOutputBase):
    code: str


EvalResult = pass_k.EvalResult


class EvalStat(EvalStatBase):
    base: BaseStat
    passk: pass_k.Stat


class LiveCodeBenchmark(BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]):
    @override
    def to_output(self, generation: Message, input: EvalInput) -> EvalOutput:
        raise NotImplementedError

    @override
    async def to_result(
        self, args: Iterable[ToResultArgs[EvalInput, EvalOutput]]
    ) -> ToResultList[EvalResult]:
        raise NotImplementedError

    @override
    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        raise NotImplementedError


class Env(BaseModel):
    api_key: SecretStr | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls, prefix: str) -> Self:
        data = {}
        prefix = prefix.upper()

        if api_key := os.getenv(f"{prefix}_API_KEY"):
            data["api_key"] = SecretStr(api_key)

        if base_url := os.getenv(f"{prefix}_BASE_URL"):
            data["base_url"] = base_url

        return cls.model_validate(data)


openai_env = Env.from_env("openai")
dpsk_env = Env.from_env("dpsk")


async def main():
    dataset = load_dataset("livecodebench/code_generation_lite", split="test")
    clients = [
        OpenAIClient(
            model="gpt-4o",
            api_key=openai_env.api_key,
            base_url=openai_env.base_url,
        ),
        OpenAIClient(
            model="deepseek-chat",
            api_key=dpsk_env.api_key,
            base_url=dpsk_env.base_url,
        ),
    ]
    benchmark = LiveCodeBenchmark(
        dataset=dataset.to_list(),
    )

    for client in clients:
        with client, benchmark:
            stat = await benchmark.run(client=client)
            print(stat)

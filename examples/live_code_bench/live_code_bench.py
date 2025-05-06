from pydantic import BaseModel

from eval_suite.benchmark import BaseStat
from eval_suite.benchmark.metric import passk
from eval_suite.benchmark.schema import EvalInputBase

dataset_id = "livecodebench/code_generation_lite"


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


EvalResult = passk.EvalResult


class EvalStat(BaseModel):
    base: BaseStat
    passk: passk.PassKStat


class LiveCodeBenchmark(passk.Benchmark[EvalInput, EvalStat]):
    pass


async def main():
    pass

import asyncio as aio
import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import final, override

from eval_suite.benchmark import BenchmarkBase, BenchmarkConfig
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.command import CommandBase, Process
from eval_suite.exception import EvalException
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
from eval_suite_kit.metrics import pass_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="verilogeval.log",
    filemode="w",
)


class Command(CommandBase):
    @classmethod
    def verilog_compile(
        cls,
        *code_paths: Path,
        cwd: Path | None = None,
        output_path: Path = Path("simv"),
        **kwargs,
    ) -> Process:
        return cls.docker_run(
            "iverilog",
            "-o",
            str(output_path),
            *(str(path) for path in code_paths),
            container="icarus-verilog",
            cwd=cwd,
            **kwargs,
        )

    @classmethod
    def verilog_simulate(
        cls,
        output_path: Path = Path("simv"),
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        return cls.docker_run(
            "vvp",
            str(output_path),
            container="icarus-verilog",
            cwd=cwd,
            **kwargs,
        )


class ResultType(str, Enum):
    compile_error = "compile-error"
    simulation_error = "simulation-error"


class EvalInput(EvalInputBase):
    problem_id: str
    prompt: str
    ref: str
    test: str

    @property
    @override
    def input_id(self) -> str:
        return self.problem_id

    @override
    def __str__(self) -> str:
        return self.prompt


class EvalOutput(EvalOutputBase):
    code: str


class EvalResult(EvalResultBase):
    pass_k: pass_k.EvalResult


class EvalStat(EvalStatBase):
    base: BaseStat
    passk: pass_k.Stat


@final
class PassKMetric(pass_k.Metric[EvalInput, EvalOutput]):
    @override
    async def to_result_async(
        self,
        eval_path: Path,
        input: EvalInput,
        output: EvalOutput,
    ) -> pass_k.EvalResult:
        (design_path := eval_path / "design.v").write_text(output.code)
        (testbench_path := eval_path / "testbench.v").write_text(input.test)

        output_path = eval_path / "simv"

        await Command.verilog_compile(
            design_path,
            testbench_path,
            cwd=eval_path,
            output_path=output_path,
        ).run(exc=EvalException(type=ResultType.compile_error))

        await Command.verilog_simulate(
            output_path=output_path,
            cwd=eval_path,
        ).run(exc=EvalException(type=ResultType.simulation_error))

        return pass_k.EvalResult(passed=True)


class VerilogEvalBenchmark(BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]):
    pass_k_metric: pass_k.Metric

    @override
    def to_output(self, generation: Message[dict], input: EvalInput) -> EvalOutput:
        result = extract_code(generation.content).get("verilog", id="result")

        if not result:
            raise ValueError("No code found in the generation")

        return EvalOutput(code=result.code)

    @override
    async def to_result(
        self,
        args: Iterable[ToResultArgs[EvalInput, EvalOutput]],
    ) -> ToResultList[EvalResult]:
        pass_k_results = await self.pass_k_metric.to_result(args)
        return EvalResult.merge(pass_k=pass_k_results)

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        return EvalStat(
            base=base,
            passk=self.pass_k_metric.to_stat(groups.map(lambda r: r.pass_k)),
        )


async def main():
    dataset = load_repo_dataset(
        "dakies/nvlabs-verilogeval-v2-spec-to-rtl",
        split="test",
    )

    with (
        VerilogEvalBenchmark(
            name="verilog-eval",
            dataset=dataset.to_list(),
            base_path=Path("output"),
            config=BenchmarkConfig(n_samples=10, max_n_samples=20),
            pass_k_metric=PassKMetric(k=5),
        ) as benchmark,
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
    ):
        stat = await benchmark.run(client)
        print(stat)


if __name__ == "__main__":
    aio.run(main())

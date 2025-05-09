import asyncio as aio
import logging
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, override

from eval_suite.benchmark import BenchmarkBase
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.command import CommandBase
from eval_suite.exception import EvalException
from eval_suite.metric import (
    BaseStat,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
)
from eval_suite.metric.result import ToResultArgs
from eval_suite.utils.dataset import load_repo_dataset
from eval_suite.utils.extract import extract_code
from eval_suite_kit.metrics import pass_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="humaneval.log",
    filemode="w",
)


class Command(CommandBase):
    @classmethod
    def python(
        cls,
        code: str,
        cwd: Path | None = None,
        **kwargs,
    ):
        return cls.docker_run(
            "python",
            "-c",
            code,
            container="python:3.12.10",
            # `docker_run` will mount the `cwd` into the container, then `cd` into it to run the command.
            cwd=cwd,
            **kwargs,
        )


class ResultType(str, Enum):
    functional_error = "functional-error"


class EvalInput(EvalInputBase):
    """
    Schema of the dataset specified in <https://huggingface.co/datasets/openai/openai_humaneval>.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @property
    def input_id(self) -> str:
        # Provide a unique identifier for each input.
        return self.task_id

        # for dataset that not contains an id column, maybe apply `hash` to the data column:
        # return str(hash(self.prompt))

    def __str__(self) -> str:
        return self.prompt

        # It is recommended to use `jinja2` template to manage the prompt.
        # But for simple cases, you can just use f-strings:
        # return f"### Problem\n{self.prompt}\n\n### Solution\n{self.canonical_solution}"


# although we will use `EvalOutput` and `EvalResult` provided by `passk`,
# it is still a good practice to define a alias for them for clarity.
class EvalOutput(EvalOutputBase):
    code: str


class EvalResult(EvalResultBase):
    pass_k: pass_k.EvalResult


class EvalStat(EvalStatBase):
    base: BaseStat
    passk: pass_k.Stat


class PassKMetric(pass_k.Metric[EvalInput, EvalOutput]):
    k: int = 5

    @override
    async def to_result_async(
        self,
        eval_path: Path,
        input: EvalInput,
        output: EvalOutput,
    ) -> pass_k.EvalResult:
        _ = await Command.python(code=output.code, cwd=eval_path).run(
            exc=EvalException(type=ResultType.functional_error),
        )

        return pass_k.EvalResult(passed=True)


class HumanEvalBenchmark(BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]):
    pass_k_metric: pass_k.Metric

    def to_input(self, data: Any) -> EvalInput:
        # Convert the raw data into the EvalInput schema.
        # Since `EvalInput` is a `pydantic` model, you can simply use `model_validate` to validate and convert the data.

        return EvalInput.model_validate(data)

    def to_output(
        self,
        generation: Message[dict],
        input: EvalInput,
    ) -> EvalOutput:
        # If the generated content is exactly what you want to evaluate, you can simply return it:
        # return passk.EvalOutput(code=ctx.generation.content)

        # However, it is safer to instruct the model to generate specific format so we can extract the real result from the generation:
        result = extract_code(generation.content).get("python", id="result")

        if not result:
            raise ValueError("No code found in the generation")

        return EvalOutput(code=result.code)

    @override
    async def to_result(
        self,
        args: Iterable[ToResultArgs[EvalInput, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        pass_k_results = await self.pass_k_metric.to_result(args)

        return EvalResult.merge(pass_k=pass_k_results)

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        return EvalStat(
            base=base,
            passk=self.pass_k_metric.to_stat(groups.map(lambda r: r.pass_k)),
        )


async def main():
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")

    with (
        HumanEvalBenchmark(
            name="human-eval",
            dataset=dataset.to_list(),
            base_path=Path("output"),
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

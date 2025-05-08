import asyncio as aio
import logging
from enum import Enum
from pathlib import Path

from eval_suite.benchmark import (
    BaseEvalConfig,
    BaseStat,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
)
from eval_suite.benchmark.metric import pass_k, score
from eval_suite.client import BaseClientConfig, Message
from eval_suite.client.sglang import EvalServerArgs, SGLangClient, SGLangSamplingParams
from eval_suite.command import CommandBase, Process
from eval_suite.exception import EvalException
from eval_suite.utils.dataset import load_repo_dataset
from eval_suite.utils.extract import extract_code

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
    ) -> Process:
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


class EvalOutput(EvalOutputBase):
    code: str


class EvalResult(EvalResultBase):
    pass_k: pass_k.EvalResult
    score: score.EvalResult


class EvalStat(EvalStatBase):
    base: BaseStat
    passk: pass_k.Stat
    score: score.Stat


class EvalConfig(BaseEvalConfig):
    pass_k: pass_k.EvalConfig


class CompBenchmark(
    BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat, EvalConfig]
):
    def to_output(
        self,
        generation: Message[dict],
        input: EvalInput,
    ) -> EvalOutput:
        result = extract_code(generation.content).get("python", id="result")

        if not result:
            raise ValueError("No code found in the generation")

        return EvalOutput(code=result.code)

    async def to_result_async(
        self,
        eval_path: Path,
        input: EvalInput,
        output: EvalOutput,
    ) -> EvalResult:
        _ = await Command.python(code=output.code, cwd=eval_path).run(
            exc=EvalException(type=ResultType.functional_error),
        )

        return EvalResult(
            pass_k=pass_k.EvalResult(passed=True),
            score=score.EvalResult(score=len(output.code)),
        )

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        return EvalStat(
            base=base,
            passk=pass_k.Stat.from_groups(
                groups=groups.map(lambda r: r.pass_k),
                k=self.eval_config.pass_k.k,
            ),
            score=score.Stat.from_groups(
                groups=groups.map(lambda r: r.score),
            ),
        )


async def main():
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")

    with (
        CompBenchmark(
            name="human-eval",
            # remember to turn dataset into a list
            dataset=dataset.to_list(),
            eval_config=EvalConfig(pass_k=pass_k.EvalConfig(k=5)),
            base_path=Path("output"),
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

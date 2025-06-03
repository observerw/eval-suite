import logging
from enum import Enum
from pathlib import Path
from typing import override

from eval_suite_core.benchmark import BenchmarkBase, EvalConfig
from eval_suite_core.client import Message
from eval_suite_core.command import CommandBase
from eval_suite_core.exception import EvalException
from eval_suite_core.metric import ItemID, ResultMap
from eval_suite_core.metric.item import ItemBase
from eval_suite_kit.client.dummy import OfflineDummyClient
from eval_suite_kit.metric import pass_k
from eval_suite_kit.utils.dataset import load_repo_dataset
from eval_suite_kit.utils.extract import extract_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="humaneval.log",
    filemode="w",
)

# ----------------------- ✅ Step 1: Define the Command ----------------------- #


class Command(CommandBase):
    """Command for running the python code in a docker container."""

    @classmethod
    def python(cls, code: str, cwd: Path | None = None, **kwargs):
        return cls.docker_run(
            "python",
            "-c",
            code,
            # Here we use `python:3.12.10` as the base image.
            container="python:3.12.10",
            # `docker_run` will mount the `cwd` into the container,
            # then `cd` into it to run the command.
            cwd=cwd,
            **kwargs,
        )


# Optional, but a good practice to define all result types in a enum.
class ResultType(str, Enum):
    functional_error = "functional-error"


# ------------------------ ✅ Step 2: Define the schema ----------------------- #


class EvalItem(ItemBase):
    """
    Schema of the dataset specified in <https://huggingface.co/datasets/openai/openai_humaneval>.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @property
    @override
    def item_id(self) -> ItemID:
        # Provide a unique identifier for each input.
        # Here we simply use the existing `task_id` field.
        return ItemID(self.task_id)

        # for dataset that not contains an id column,
        # you can apply `hash` to the data column:
        # return str(hash(self.prompt))

    @override
    def format_instruction(self) -> str:
        return self.prompt

        # It is recommended to use `jinja2` template to manage the prompt:
        # prompt_template.render(prompt=prompt, few_shots=[])

        # For simple cases, you can just use f-strings:
        # return f"### Problem\n{self.prompt}\n\n### Solution\n{self.canonical_solution}"


# ------------------- ✅ Step 3: Implement the Pass@k metric ------------------ #


class PassKMetric(
    # We inherit from `pass_k.Metric`,
    # and (optionally) provide the schema here for better hinting.
    pass_k.Metric[EvalItem]
):
    # According to the document, all we need to implement
    # is one of the `to_result*` series of methods.
    # We choose `to_result_async`,
    # because we only need to run the python docker container here.
    @override
    async def to_result(
        self, eval_path: Path, item: EvalItem, generation: Message, prec: ResultMap
    ) -> pass_k.EvalResult:
        if not (result := extract_code(generation.content).get("python", id="result")):
            raise ValueError("No code found in the generation")
        code = result.code

        _ = (
            await Command.python(code=code, cwd=eval_path)
            # If command failed, an EvalException will be raised
            # to indicate this sample has functional error.
            # We don't specify the `message` field here,
            # so it will be set automatically to the error message.
            .map_exception(EvalException(type=ResultType.functional_error))
            .run(timeout=60)
        )

        # If the command succeeded, we should return a successful result.
        return pass_k.EvalResult(passed=True)


# ------------------ ✅ Step 4: Define and run the benchmark ------------------ #


class Benchmark(BenchmarkBase[EvalItem]):
    name = "pass@k"
    config = EvalConfig(
        # To get a reasonable result,
        # we need to set the `n_samples` larger than `k`.
        n_samples=10,
        max_n_samples=20,
    )

    pass_k: PassKMetric


async def main():
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")

    pass_k = PassKMetric(k=5)
    benchmark = Benchmark(
        dataset=dataset,
        base_path=Path("output"),
        pass_k=pass_k,
    )
    client = OfflineDummyClient()

    stat = await benchmark.run(client)
    print(stat)
    # the `stat` will look like this:
    # {
    #     "base": {
    #         "n_samples": 10,
    #         "n_items": 20,
    #         ...
    #     },
    #     "pass_k": {
    #         "k": 5,
    #         "pass_n": {
    #             "pass@1": 0.5,
    #             "pass@2": 0.5,
    #             ...
    #             "pass@5": 0.5,
    #         }
    #     }
    # }

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

# --------------- ✅ Step 1: Define the Command to run the code --------------- #


class Command(CommandBase):
    """Command for running the python code in a docker container."""

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
        # Here we simply use the existing `task_id` field.
        return self.task_id

        # for dataset that not contains an id column,
        # you can apply `hash` to the data column:
        # return str(hash(self.prompt))

    def __str__(self) -> str:
        return self.prompt

        # It is recommended to use `jinja2` template to manage the prompt:
        # prompt_template.render(prompt=prompt, few_shots=[])

        # For simple cases, you can just use f-strings:
        # return f"### Problem\n{self.prompt}\n\n### Solution\n{self.canonical_solution}"


class EvalOutput(EvalOutputBase):
    # We'll extract the code from the generation as the output.
    code: str


class EvalResult(EvalResultBase):
    # To get the final Pass@k statistics,
    # we need to include the pass@k result here.
    pass_k: pass_k.EvalResult


class EvalStat(EvalStatBase):
    # We would like to get the basic statistics of the benchmark,
    # and the pass@k statistics.
    base: BaseStat
    passk: pass_k.Stat


# ------------------- ✅ Step 3: Implement the Pass@k metric ------------------ #


class PassKMetric(
    # We inherit from `pass_k.Metric`,
    # and (optionally) provide some type here for better type hinting.
    pass_k.Metric[EvalInput, EvalOutput]
):
    # According to the document, all we need to implement by ourself
    # is one of the `to_result*` series of methods.
    # We choose `to_result_async`,
    # because we only need to run the python docker container here.
    @override
    async def to_result_async(
        self,
        eval_path: Path,
        input: EvalInput,
        output: EvalOutput,
    ) -> pass_k.EvalResult:
        _ = await Command.python(code=output.code, cwd=eval_path).run(
            # If command failed, an EvalException will be raised
            # to indicate this sample has functional error.
            # We don't specify the `message` field here,
            # so it will be set automatically to the error message.
            exc=EvalException(type=ResultType.functional_error),
        )

        return pass_k.EvalResult(passed=True)


# --------------------- ✅ Step 4: Implement the Benchmark -------------------- #


class HumanEvalBenchmark(
    # We inherit from `BenchmarkBase` and specify the input/output/result/stat types.
    # The types here are mandatory so the benchmark can restore from the previous run.
    BenchmarkBase[EvalInput, EvalOutput, EvalResult, EvalStat]
):
    # We'll use the Pass@k metric in our benchmark.
    pass_k_metric: pass_k.Metric

    @override
    def to_input(self, data: Any) -> EvalInput:
        # Convert the raw data into the EvalInput schema.
        # Since `EvalInput` is a `pydantic` model, you can simply use `model_validate` to validate and convert the data.
        # You can omit this implementation if you don't need further processing.

        return EvalInput.model_validate(data)

    @override
    def to_output(self, generation: Message[dict], input: EvalInput) -> EvalOutput:
        # We'll extract the code from the generation as the output.
        if not (result := extract_code(generation.content).get("python", id="result")):
            raise ValueError("No code found in the generation")

        return EvalOutput(code=result.code)

    @override
    async def to_result(
        self,
        args: Iterable[ToResultArgs[EvalInput, EvalOutput]],
    ) -> Sequence[EvalResult | BaseException]:
        # Process all model outputs generated by the client and convert them to evaluation results
        # This method is the core of the evaluation process, executing generated code and checking its correctness

        # Step 1: Call pass_k_metric.to_result to process all outputs
        # It sends input/output pairs to the to_result_async method we implemented in PassKMetric
        # to_result_async runs the code in a Docker container and checks execution results
        pass_k_results = await self.pass_k_metric.to_result(args)

        # Step 2: Merge the Pass@k evaluation results into our custom EvalResult object
        # The merge method provides a powerful composition mechanism for combining different result types
        # It takes keyword arguments where each value is a sequence of evaluation results from different metrics
        # The method creates a new list of EvalResult objects with fields populated from the corresponding results
        # This composition pattern enables flexible integration of multiple evaluation metrics in a single benchmark
        return EvalResult.merge(pass_k=pass_k_results)

    def to_stat(self, groups: EvalResultGroups[EvalResult], base: BaseStat) -> EvalStat:
        # Aggregate the evaluation results into statistical information
        # This method is called after all evaluations are complete to generate the final statistics

        # The 'groups' parameter contains all evaluation results grouped by input_id
        # The 'base' parameter provides basic statistics like total samples, success/failure counts, etc.

        # We construct our EvalStat object with two components:
        # 1. The base statistics that are common to all benchmarks
        # 2. Pass@k statistics that are specific to this benchmark

        # The 'map' method transforms each EvalResult into its pass_k component
        # This allows the pass_k_metric to calculate statistics on just the relevant part of our results
        return EvalStat(
            base=base,
            passk=self.pass_k_metric.to_stat(groups.map(lambda r: r.pass_k)),
        )


# ------------------------ ✅ Step 5: Run the benchmark ----------------------- #


async def main():
    # Load the HumanEval dataset from HuggingFace
    dataset = load_repo_dataset("openai/openai_humaneval", split="test")

    with (
        # Initialize the benchmark with our custom implementation
        # - name: A descriptive name for the benchmark
        # - dataset: The HumanEval problems to evaluate
        # - base_path: Location to store evaluation artifacts and results
        # - pass_k_metric: Our Pass@k metric implementation with k=5
        HumanEvalBenchmark(
            name="human-eval",
            dataset=dataset.to_list(),
            base_path=Path("output"),
            pass_k_metric=PassKMetric(k=5),
        ) as benchmark,
        # Set up the SGLangClient for generating code solutions
        # - server_args: Configuration for the model server (model path and parallelism settings)
        # - sampling_params: Parameters to control the generation quality and diversity
        # - config: General client configuration like batch size
        SGLangClient(
            # Check <https://docs.sglang.ai/backend/server_arguments.html> for details
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
                batch_size=2048,  # set the generation batch size to 2048
            ),
        ) as client,
    ):
        # Run the benchmark evaluation using the configured client
        # This will:
        # 1. Generate code solutions for each problem using the LLM
        # 2. Execute the generated code to test its functionality
        # 3. Collect results and calculate Pass@k statistics
        stat = await benchmark.run(client)

        print(stat)


if __name__ == "__main__":
    aio.run(main())

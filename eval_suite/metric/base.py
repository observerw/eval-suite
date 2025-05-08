from typing import Any

from pydantic import BaseModel

from eval_suite.client import Message
from eval_suite.metric.config import BaseEvalConfig, BenchmarkConfig
from eval_suite.metric.result import EvalResultBase, ToResult
from eval_suite.metric.schema import EvalInputBase, EvalOutputBase


class MetricBase[
    Input: EvalInputBase,
    Output: EvalOutputBase,
    Result: EvalResultBase,
    Config: BaseEvalConfig,
](BaseModel, ToResult[Input, Output, Result]):
    """Base class for all metrics."""

    eval_config: Config
    """The evaluation configuration."""

    benchmark_config: BenchmarkConfig = BenchmarkConfig()
    """The benchmark configuration."""

    def to_input(self, data: Any) -> Input:
        """Convert the dataset item to a task-specific input.

        Note that this method is not supposed to raise any exceptions; and if it does, the exception will not be considered as a result.

        Args:
            data (Any): The raw data item from the dataset.

        Returns:
            Input: A validated input object for the evaluation task.
        """

        raise NotImplementedError("`to_input` not provided. ")

    def to_output(self, generation: Message, input: Input) -> Output:
        raise NotImplementedError("`to_output` not provided. ")

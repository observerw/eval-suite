from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from eval_suite.client import Message
from eval_suite.metric.result import EvalResultBase, EvalResultGroups
from eval_suite.metric.schema import EvalInputBase, EvalOutputBase
from eval_suite.metric.stat import EvalStatBase


class ToInput[Input: EvalInputBase](ABC):
    @abstractmethod
    def to_input(self, data: Any) -> Input:
        """Convert the dataset item to a task-specific input.

        Note that this method is not supposed to raise any exceptions; and if it does, the exception will not be considered as a result.

        Args:
            data (Any): The raw data item from the dataset.

        Returns:
            Input: A validated input object for the evaluation task.
        """


class ToOutput[Input: EvalInputBase, Output: EvalOutputBase](ABC):
    @abstractmethod
    def to_output(self, generation: Message, input: Input) -> Output:
        """Convert the generated message to a task-specific output.

        Args:
            generation (Message): The generated message from the model.
            input (Input): The input object for the evaluation task.

        Returns:
            Output: A validated output object for the evaluation task.
        """


class ToStat[Result: EvalResultBase, Stat: EvalStatBase](ABC):
    @abstractmethod
    def to_stat(self, groups: EvalResultGroups[Result]) -> Stat:
        """Generate a statistic from all the results.

        Args:
            groups (EvalResultGroups[Result]): Grouped evaluation results for statistical analysis.

        Returns:
            Stat: A statistics object with computed metrics and analysis results.
        """


class MetricBase(ABC, BaseModel):
    pass

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self, final, overload

from pydantic import BaseModel, PrivateAttr, RootModel

from eval_suite_core.client.schema import Message
from eval_suite_core.exception import BaseEvalResultType, EvalException
from eval_suite_core.metric.base import AnyMetric, MetricID
from eval_suite_core.metric.id import MetricEvalID
from eval_suite_core.metric.item import ChatItemBase, EvalID, ItemID


class ResultBase(BaseModel):
    model_config = {"frozen": True}

    result_type: Literal["regular"] = "regular"
    type: str = BaseEvalResultType.success

    _id: MetricEvalID = PrivateAttr()

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        """Retrieve a result from an exception.

        It is required to re-raise the exception if it cannot be converted to a result.

        Args:
            exc (EvalException): The exception to retrieve the result from.

        Raises:
            BaseException: If the exception cannot be converted to a result.

        Returns:
            Self: The retrieved result.
        """

        raise exc


@final
class ExceptionResult(BaseModel):
    model_config = {"frozen": True}

    result_type: Literal["exception"] = "exception"

    type: str = BaseEvalResultType.fail
    message: str | None = None

    # ExceptionResult represents the first exception that occurred during metric graph execution,
    # so `_id` do not need to be specific to a metric.
    _id: EvalID = PrivateAttr()

    @classmethod
    def from_exception(cls, exc: BaseException) -> Self:
        exc = EvalException.from_exception(exc)
        return cls(message=exc.message, type=exc.type)


# item -> *result
class ResultGroups[Result: ResultBase](dict[ItemID, list[Result]]):
    def flatten(self) -> Iterable[Result]:
        return (result for group in self.values() for result in group)


# item -> *(result | exception)
class RawResultGroups(dict[ItemID, list[ResultBase | ExceptionResult]]):
    def filter(self, n_samples: int) -> ResultGroups:
        return ResultGroups(
            {
                item_id: filtered_results
                for item_id, results in self.items()
                # sufficient
                if len(
                    filtered_results := [
                        result
                        for result in results
                        # not exception
                        if not isinstance(result, ExceptionResult)
                    ]
                )
                >= n_samples
            }
        )


# metric -> result
class ResultMap(RootModel[dict[MetricID, ResultBase]]):
    root: dict[MetricID, ResultBase]

    _eval_id: EvalID = PrivateAttr()

    @classmethod
    def create(cls, results: Iterable[ResultBase]) -> Self:
        return cls(root={result._id.metric: result for result in results})

    @overload
    def __getitem__[Result: ResultBase](
        self, key: AnyMetric[Any, Result, Any]
    ) -> Result: ...

    @overload
    def __getitem__(self, key: MetricID) -> ResultBase: ...

    def __getitem__(self, key: AnyMetric | MetricID) -> ResultBase | ResultBase:
        match key:
            case str():
                return self.root[key]
            case AnyMetric(id=id):
                return self.root[id]

    def validate_metrics(self, metrics: Iterable[AnyMetric]):
        """Validate that all metrics are present in the result map."""

        if any(metric.id not in self.root for metric in metrics):
            raise ValueError("Not all metrics are present in the result map.")


class ToResultArgsBase(NamedTuple):
    eval_path: Path
    item: ChatItemBase
    generation: Message

    # load by cache
    result: ResultBase | None = None

    @property
    def eval_id(self) -> EvalID:
        return self.item._eval_id


class ToResultArgs[Item: ChatItemBase](NamedTuple):
    eval_path: Path
    """The output path of the evaluation. Can be used to store temporary files."""

    item: Item
    """The item to evaluate."""

    generation: Message
    """The generated content."""

    prec: ResultMap
    """The previous result map."""


class ToResultBase(ABC):
    """Base class for the `to_result` method.

    Metrics should implement exactly one of the `to_result` methods.
    """


class ToResult[Item: ChatItemBase, Result: ResultBase](ToResultBase):
    @abstractmethod
    def to_result(
        self,
        eval_path: Path,
        item: Item,
        generation: Message,
        prec: ResultMap,
    ) -> Result:
        """Evaluate the generation and return a result.

        Args:
            eval_path (Path): The output path of the evaluation. Can be used to store temporary files.
            item (Item): The item to evaluate.
            generation (Message): The generated content.
            prec (EvalResultMap): The previous results.
        """


class ToResultAsync[Item: ChatItemBase, Result: ResultBase](ToResultBase):
    @abstractmethod
    async def to_result(
        self,
        eval_path: Path,
        item: Item,
        generation: Message,
        prec: ResultMap,
    ) -> Result:
        """Evaluate the generation and return a result.

        Args:
            eval_path (Path): The output path of the evaluation. Can be used to store temporary files.
            item (Item): The item to evaluate.
            generation (Message): The generated content.
            prec (EvalResultMap): The previous results.
        """


class ToResultBatch[Item: ChatItemBase, Result: ResultBase](ToResultBase):
    @abstractmethod
    def to_result(
        self,
        args: Sequence[ToResultArgs[Item]],
    ) -> Iterable[Result | BaseException]:
        """Evaluate the generation and return a result.

        Args:
            args (Sequence[ToResultArgs[Item]]): The arguments to evaluate.

        Returns:
            Sequence[Result | BaseException]: The results of the evaluation.
        """


class ToResultBatchAsync[Item: ChatItemBase, Result: ResultBase](ToResultBase):
    @abstractmethod
    async def to_result(
        self,
        args: Sequence[ToResultArgs[Item]],
    ) -> Iterable[Result | BaseException]:
        """Evaluate the generation and return a result.

        Args:
            args (Sequence[ToResultArgs[Item]]): The arguments to evaluate.

        Returns:
            Sequence[Result | BaseException]: The results of the evaluation.
        """

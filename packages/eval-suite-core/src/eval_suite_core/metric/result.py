from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, cast, final, overload

from pydantic import BaseModel, PrivateAttr

from eval_suite_core.client.schema import Message
from eval_suite_core.exception import BaseEvalResultType, EvalException
from eval_suite_core.metric.base import MetricID
from eval_suite_core.metric.item import EvalID, EvalItemBase, ItemID
from eval_suite_core.metric.stat import EvalStatBase

if TYPE_CHECKING:
    from eval_suite_core.metric.base import _MetricBase


class EvalResultBase(BaseModel):
    model_config = {"frozen": True}

    result_type: Literal["regular"] = "regular"
    type: str = BaseEvalResultType.success

    _eval_id: EvalID = PrivateAttr()
    _metric: str = PrivateAttr()

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        """Retrieve a result from an exception.

        If the exception cannot be converted to a result, re-raise it.

        Args:
            exc (EvalException): The exception to retrieve the result from.

        Raises:
            BaseException: If the exception cannot be converted to a result.

        Returns:
            Self: The retrieved result.
        """

        raise exc


@final
class ExceptionEvalResult(BaseModel):
    model_config = {"frozen": True}

    result_type: Literal["exception"] = "exception"

    type: str = BaseEvalResultType.fail
    message: str | None = None

    _eval_id: EvalID = PrivateAttr()
    _metric: str = PrivateAttr()

    @classmethod
    def from_exception(cls, exc: BaseException) -> Self:
        exc = EvalException.from_exception(exc)
        return cls(message=exc.message, type=exc.type)


class EvalResultGroups[Result: EvalResultBase](dict[ItemID, list[Result]]):
    def flatten(self) -> list[Result]:
        return [result for group in self.values() for result in group]


class RawEvalResultGroups(dict[ItemID, list[EvalResultBase | ExceptionEvalResult]]):
    def filter(self, n_samples: int) -> EvalResultGroups:
        return EvalResultGroups(
            {
                item_id: filtered_results
                for item_id, results in self.items()
                # sufficient
                if len(
                    filtered_results := [
                        result
                        for result in results
                        # not exception
                        if not isinstance(result, ExceptionEvalResult)
                    ]
                )
                >= n_samples
            }
        )


class MetricResultGroups(dict[MetricID, RawEvalResultGroups]):
    def merge(self, update: Mapping[MetricID, EvalResultBase | ExceptionEvalResult]):
        for metric_id, result in update.items():
            (
                self.setdefault(metric_id, RawEvalResultGroups())
                .setdefault(result._eval_id.item_id, [])
                .append(result)
            )


# metric-result 1-to-1 mapping
class EvalResultMap(dict[MetricID, EvalResultBase]):
    @overload
    def __getitem__[Result: EvalResultBase](  # little type trick to extract the type
        self, key: _MetricBase[EvalItemBase, Result, EvalStatBase]
    ) -> Result: ...

    @overload
    def __getitem__(self, key: MetricID) -> EvalResultBase: ...

    def __getitem__[Result: EvalResultBase](
        self, key: _MetricBase[EvalItemBase, Result, EvalStatBase] | MetricID
    ) -> Result | EvalResultBase:
        match key:
            case str():
                return super().__getitem__(key)
            case _MetricBase(id=id):
                return super().__getitem__(id)


# metric-stat 1-to-1 mapping
class EvalStatMap(dict[_MetricBase, EvalStatBase]):
    def __getitem__[Stat: EvalStatBase](
        self, metric: _MetricBase[EvalItemBase, EvalResultBase, Stat]
    ) -> Stat:
        return cast(Stat, super().__getitem__(metric))


class ToResultArgsBase[Item: EvalItemBase](NamedTuple):
    eval_path: Path
    item: Item
    generation: Message

    @property
    def _eval_id(self) -> EvalID:
        return self.item._eval_id


# for some reason, deriving `ToResultArgsBase` doesn't work as expected
class ToResultArgs[Item: EvalItemBase](NamedTuple):
    eval_path: Path
    """The output path of the evaluation. Can be used to store temporary files."""

    item: Item
    """The item to evaluate."""

    generation: Message
    """The generated content."""

    prec: EvalResultMap
    """The previous result map."""


class ToResultBase(ABC):
    """Base class for the `to_result` method.

    Metrics should implement exactly one of the `to_result` methods.
    """


class ToResult[Item: EvalItemBase, Result: EvalResultBase](ToResultBase):
    @abstractmethod
    def to_result(
        self,
        eval_path: Path,
        item: Item,
        generation: Message,
        prec: EvalResultMap,
    ) -> Result:
        """Evaluate the generation and return a result.

        Args:
            eval_path (Path): The output path of the evaluation. Can be used to store temporary files.
            item (Item): The item to evaluate.
            generation (Message): The generated content.
            prec (EvalResultMap): The previous result map.
        """


class ToResultAsync[Item: EvalItemBase, Result: EvalResultBase](ToResultBase):
    @abstractmethod
    async def to_result(
        self,
        eval_path: Path,
        item: Item,
        generation: Message,
        prec: EvalResultMap,
    ) -> Result:
        """Evaluate the generation and return a result.

        Args:
            eval_path (Path): The output path of the evaluation. Can be used to store temporary files.
            item (Item): The item to evaluate.
            generation (Message): The generated content.
            prec (EvalResultMap): The previous result map.
        """


class ToResultBatch[Item: EvalItemBase, Result: EvalResultBase](ToResultBase):
    @abstractmethod
    def to_result(
        self,
        args: Sequence[ToResultArgs[Item]],
    ) -> Sequence[Result | BaseException]:
        """Evaluate the generation and return a result.

        Args:
            args (Sequence[ToResultArgs[Item]]): The arguments to evaluate.

        Returns:
            Sequence[Result | BaseException]: The results of the evaluation.
        """


class ToResultBatchAsync[Item: EvalItemBase, Result: EvalResultBase](ToResultBase):
    @abstractmethod
    async def to_result(
        self,
        args: Sequence[ToResultArgs[Item]],
    ) -> Sequence[Result | BaseException]:
        """Evaluate the generation and return a result.

        Args:
            args (Sequence[ToResultArgs[Item]]): The arguments to evaluate.

        Returns:
            Sequence[Result | BaseException]: The results of the evaluation.
        """

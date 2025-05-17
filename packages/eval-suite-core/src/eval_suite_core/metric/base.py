import contextlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self, cast, override
from uuid import UUID, uuid4

from pydantic import BaseModel, PrivateAttr

from eval_suite_core.metric.config import (
    AsyncMetricConfig,
    MetricConfig,
    SyncMetricConfig,
)
from eval_suite_core.metric.item import EvalItemBase
from eval_suite_core.metric.result import (
    EvalResultBase,
    EvalResultGroups,
    ToResult,
    ToResultAsync,
    ToResultBase,
    ToResultBatch,
    ToResultBatchAsync,
)
from eval_suite_core.metric.stat import BaseEvalStat, EvalStatBase
from eval_suite_core.metric.typevar import TypeVarMixin

type MetricID = str


class _MetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    BaseModel,
    contextlib.AbstractContextManager,
    TypeVarMixin,
    ABC,
):
    model_config = {"frozen": True}

    name: ClassVar[str]
    """Name of the metric"""

    config: MetricConfig
    """Configuration of the metric"""

    _id: UUID = PrivateAttr(default_factory=uuid4)

    def __hash__(self) -> int:
        return hash(self.name)

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *exc_details) -> bool | None:
        return

    @property
    def _Item(self) -> type[Item]:
        return self._resolve_typevar(Item)

    @property
    def _Result(self) -> type[Result]:
        return self._resolve_typevar(Result)

    @property
    def _Stat(self) -> type[Stat]:
        return self._resolve_typevar(Stat)

    @property
    def id(self) -> MetricID:
        """Unique ID of the metric."""

        return self._id.hex

    @property
    def prec(self) -> "set[_MetricBase]":
        """All precursors of the metric."""

        return {
            metric  #
            for metric in dict(self).values()
            if isinstance(metric, _MetricBase)
        }

    @override
    def model_post_init(self, context: Any) -> None:
        assert isinstance(self, ToResultBase), (
            "Metric must implement one of the `to_result` method."
        )

    @abstractmethod
    def to_stat(
        self,
        groups: EvalResultGroups[Result],
        base: BaseEvalStat,
        prec: "EvalStatMap",
    ) -> Stat:
        """Statistics of the metric."""


class MetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResult[Item, Result],
):
    config: MetricConfig = SyncMetricConfig()


type MetricDefault = MetricBase[EvalItemBase, EvalResultBase, EvalStatBase]


class AsyncMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultAsync[Item, Result],
):
    config: MetricConfig = AsyncMetricConfig()


type AsyncMetricDefault = AsyncMetricBase[EvalItemBase, EvalResultBase, EvalStatBase]


class BatchMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultBatch[Item, Result],
):
    config: MetricConfig = SyncMetricConfig()


type BatchMetricDefault = BatchMetricBase[EvalItemBase, EvalResultBase, EvalStatBase]


class AsyncBatchMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultBatchAsync[Item, Result],
):
    config: MetricConfig = AsyncMetricConfig()


type AsyncBatchMetricDefault = AsyncBatchMetricBase[
    EvalItemBase, EvalResultBase, EvalStatBase
]


# metric-result 1-to-1 mapping
class EvalResultMap(dict[_MetricBase, EvalResultBase]):
    def __getitem__[Result: EvalResultBase](  # little type trick to extract the type
        self, metric: _MetricBase[EvalItemBase, Result, EvalStatBase]
    ) -> Result:
        return cast(Result, super().__getitem__(metric))


# metric-stat 1-to-1 mapping
class EvalStatMap(dict[_MetricBase, EvalStatBase]):
    def __getitem__[Stat: EvalStatBase](
        self, metric: _MetricBase[EvalItemBase, EvalResultBase, Stat]
    ) -> Stat:
        return cast(Stat, super().__getitem__(metric))

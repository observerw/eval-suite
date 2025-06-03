import os
from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, PrivateAttr

from eval_suite_core.metric.config import MetricConfig
from eval_suite_core.metric.id import MetricID
from eval_suite_core.metric.item import ChatItemBase
from eval_suite_core.metric.result import (
    ResultBase,
    ResultGroups,
    ToResult,
    ToResultAsync,
    ToResultBatch,
    ToResultBatchAsync,
)
from eval_suite_core.metric.stat import BaseStat, StatBase, StatMap
from eval_suite_core.metric.typevar import TypeVarMixin


class AnyMetric[
    Item: ChatItemBase,
    Result: ResultBase,
    Stat: StatBase,
](BaseModel, TypeVarMixin, ABC):
    """Base class for all types of metrics."""

    model_config = {"frozen": True}

    name: ClassVar[str]
    """Name of the metric"""

    config: ClassVar[MetricConfig]
    """Configuration of the metric"""

    _id: UUID = PrivateAttr(default_factory=uuid4)

    def __hash__(self) -> int:
        return self._id.int

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

        return MetricID(self._id.hex)

    @property
    def prec(self) -> "set[AnyMetric]":
        """All precursors of the metric."""

        return {
            metric  #
            for metric in dict(self).values()
            if isinstance(metric, AnyMetric)
        }

    @abstractmethod
    def to_stat(
        self, groups: ResultGroups[Result], base: BaseStat, prec: StatMap
    ) -> Stat:
        """Statistics of the metric."""


class MetricBase[Item: ChatItemBase, Result: ResultBase, Stat: StatBase](
    AnyMetric[Item, Result, Stat],
    ToResult[Item, Result],
):
    """Default metric base class. Expect to perform light-weight sync operations."""

    class DefaultConfig(MetricConfig): ...

    config = DefaultConfig()


type MetricDefault = MetricBase[ChatItemBase, ResultBase, StatBase]


class ComputeMetricBase[Item: ChatItemBase, Result: ResultBase, Stat: StatBase](
    AnyMetric[Item, Result, Stat],
    ToResult[Item, Result],
):
    """Compute-intensive metric base class. Expect to perform sync computations."""

    class DefaultConfig(MetricConfig):
        num_cpus: int = 1
        batch_size: int = os.cpu_count() or 1

    config = DefaultConfig()


type ComputeMetricDefault = ComputeMetricBase[ChatItemBase, ResultBase, StatBase]


class IOMetricBase[Item: ChatItemBase, Result: ResultBase, Stat: StatBase](
    AnyMetric[Item, Result, Stat],
    ToResultAsync[Item, Result],
):
    """IO-intensive metric base class. Expect to perform async IO operations."""

    class DefaultConfig(MetricConfig): ...

    config = DefaultConfig()


type IOMetricDefault = IOMetricBase[ChatItemBase, ResultBase, StatBase]


class BatchComputeMetricBase[Item: ChatItemBase, Result: ResultBase, Stat: StatBase](
    AnyMetric[Item, Result, Stat],
    ToResultBatch[Item, Result],
):
    """Batch compute-intensive metric base class. Expect to perform batch sync computations."""

    class DefaultConfig(MetricConfig):
        num_cpus: int = 1
        batch_size: int = os.cpu_count() or 1

    config = DefaultConfig()


type BatchComputeMetricDefault = BatchComputeMetricBase[
    ChatItemBase, ResultBase, StatBase
]


class BatchIOMetricBase[Item: ChatItemBase, Result: ResultBase, Stat: StatBase](
    AnyMetric[Item, Result, Stat],
    ToResultBatchAsync[Item, Result],
):
    """Batch IO-intensive metric base class. Expect to perform batch async IO operations."""

    class DefaultConfig(MetricConfig): ...

    config = DefaultConfig()


type BatchIOMetricDefault = BatchIOMetricBase[ChatItemBase, ResultBase, StatBase]

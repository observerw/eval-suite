import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, ClassVar, NewType, Self, override
from uuid import UUID, uuid4

from pydantic import BaseModel, PrivateAttr

from eval_suite_core.metric.config import MetricConfig
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
from eval_suite_core.metric.stat import BaseEvalStat, EvalStatBase, EvalStatMap
from eval_suite_core.metric.typevar import TypeVarMixin

MetricID = NewType("MetricID", str)


class _MetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](BaseModel, TypeVarMixin, ABC):
    model_config = {"frozen": True}

    name: ClassVar[str]
    """Name of the metric"""

    config: MetricConfig
    """Configuration of the metric"""

    _id: UUID = PrivateAttr(default_factory=uuid4)

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    @contextmanager
    def create(cls, *, config: MetricConfig) -> Generator[Self, None, None]:
        yield cls(config=config)

    @classmethod
    @asynccontextmanager
    async def create_async(cls, *, config: MetricConfig):
        yield cls(config=config)

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
        prec: EvalStatMap,
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
    """Default metric base class. Expect to perform light-weight sync operations."""

    class DefaultConfig(MetricConfig): ...

    config: MetricConfig = DefaultConfig()


type MetricDefault = MetricBase[EvalItemBase, EvalResultBase, EvalStatBase]


class ComputeMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResult[Item, Result],
):
    """Compute-intensive metric base class. Expect to perform sync computations."""

    class DefaultConfig(MetricConfig):
        num_cpus: int = 1
        batch_size: int = os.cpu_count() or 1

    config: MetricConfig = DefaultConfig()


class IOMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultAsync[Item, Result],
):
    """IO-intensive metric base class. Expect to perform async IO operations."""

    class DefaultConfig(MetricConfig): ...

    config: MetricConfig = DefaultConfig()


type IOMetricDefault = IOMetricBase[EvalItemBase, EvalResultBase, EvalStatBase]


class BatchComputeMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultBatch[Item, Result],
):
    """Batch compute-intensive metric base class. Expect to perform batch sync computations."""

    class DefaultConfig(MetricConfig):
        num_cpus: int = 1
        batch_size: int = os.cpu_count() or 1

    config: MetricConfig = DefaultConfig()


type BatchMetricDefault = BatchComputeMetricBase[
    EvalItemBase, EvalResultBase, EvalStatBase
]


class BatchIOMetricBase[
    Item: EvalItemBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](
    _MetricBase[Item, Result, Stat],
    ToResultBatchAsync[Item, Result],
):
    """Batch IO-intensive metric base class. Expect to perform batch async IO operations."""

    class DefaultConfig(MetricConfig): ...

    config: MetricConfig = DefaultConfig()


type AsyncBatchMetricDefault = BatchIOMetricBase[
    EvalItemBase, EvalResultBase, EvalStatBase
]

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic._internal._generics import get_model_typevars_map

from eval_suite_core.benchmark.config import EvalConfig
from eval_suite_core.benchmark.executor import BenchmarkExecutor
from eval_suite_core.client.base import AnyClient
from eval_suite_core.metric.base import AnyMetric
from eval_suite_core.metric.item import ChatItemBase
from eval_suite_core.metric.result import ResultMap
from eval_suite_core.metric.stat import StatMap
from eval_suite_core.utils.collections import OrderedSet


@dataclass
class BenchmarkResult:
    results: ResultMap
    stats: StatMap


class BenchmarkBase[Item: ChatItemBase](BaseModel):
    name: ClassVar[str]
    """The name of the benchmark."""

    config: ClassVar[EvalConfig] = EvalConfig()
    """The configuration of the benchmark."""

    dataset: Iterable[Any]
    """The dataset to evaluate."""

    base_path: Path | None = None
    """The base path to store the evaluation results."""

    @cached_property
    def _Item(self) -> type[ChatItemBase]:
        return get_model_typevars_map(self.__class__)[Item]

    @cached_property
    def sink_metrics(self) -> Sequence[AnyMetric]:
        return [metric for _, metric in dict(self) if isinstance(metric, AnyMetric)]

    @cached_property
    def ordered_metrics(self) -> list[AnyMetric]:
        """Topologically sort metrics."""

        metrics = OrderedSet()
        stack = [*self.sink_metrics]

        while stack:
            metric = stack.pop()
            metrics.add(metric)
            stack.extend(metric.prec)

        return list(reversed(metrics))  # reverse to get the real topological order

    @contextlib.contextmanager
    def init(self):
        yield self

    async def run(self, client: AnyClient) -> BenchmarkResult:
        async with BenchmarkExecutor.create(benchmark=self, client=client) as executor:
            return await executor.run()

import contextlib
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Self, override

from pydantic import BaseModel
from pydantic._internal._generics import get_model_typevars_map

from eval_suite_core.benchmark.config import EvalConfig
from eval_suite_core.client.base import _ClientBase
from eval_suite_core.metric.base import _MetricBase
from eval_suite_core.metric.item import EvalItemBase
from eval_suite_core.utils.collections import OrderedSet


class BenchmarkBase[Item: EvalItemBase](BaseModel, contextlib.AbstractContextManager):
    name: ClassVar[str]
    """The name of the benchmark."""

    dataset: Iterable[Any]
    """The dataset to evaluate."""

    config: EvalConfig = EvalConfig()
    """The configuration of the benchmark."""

    base_path: Path | None = None
    """The base path to store the evaluation results."""

    @cached_property
    def _Item(self) -> type[EvalItemBase]:
        return get_model_typevars_map(self.__class__)[Item]

    @cached_property
    def metrics(self) -> list[_MetricBase]:
        return [metric for _, metric in dict(self) if isinstance(metric, _MetricBase)]

    @cached_property
    def ordered_metrics(self) -> list[_MetricBase]:
        """Topologically sort metrics."""

        metrics: OrderedSet[_MetricBase] = OrderedSet()
        stack: list[_MetricBase] = [*self.metrics]

        while stack:
            metric = stack.pop()
            metrics.add(metric)
            stack.extend(metric.prec)

        return list(reversed(metrics))  # reverse to get the real topological order

    @override
    def __enter__(self) -> Self:
        self._entered: None = None
        return self

    @override
    def __exit__(self, *exc_details):
        del self._entered

    async def run(self, client: _ClientBase):
        if not hasattr(self, "_entered"):
            raise RuntimeError("Benchmark should be used with `with` statement")

        raise NotImplementedError

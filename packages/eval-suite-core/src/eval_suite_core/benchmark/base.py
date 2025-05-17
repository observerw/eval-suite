import contextlib
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Self, override

from pydantic import BaseModel
from pydantic._internal._generics import get_model_typevars_map

from eval_suite_core.benchmark.cache import EvalCache
from eval_suite_core.benchmark.config import EvalConfig
from eval_suite_core.metric.base import _MetricBase
from eval_suite_core.metric.item import EvalItemBase


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
    def _Cache(self) -> type[EvalCache]:
        return EvalCache.create_schema(
            *(
                (metric_name, metric._Result)
                for metric_name, metric in self.metrics.items()
            )
        )

    @property
    def metrics(self) -> dict[str, _MetricBase]:
        return {
            metric_name: metric
            for metric_name, metric in dict(self)
            if isinstance(metric, _MetricBase)
        }

    @override
    def __enter__(self) -> Self:
        self._entered: None = None
        return self

    @override
    def __exit__(self, *exc_details):
        del self._entered

    async def run(self):
        if not hasattr(self, "_entered"):
            raise RuntimeError("Benchmark should be used with `with` statement")

        raise NotImplementedError

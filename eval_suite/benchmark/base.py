import tempfile
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar, cast

from pydantic import BaseModel
from pydantic._internal._generics import (
    get_model_typevars_map,  # type: ignore[reportPrivateImportUsage]
)

from eval_suite.benchmark.cache import EvalCache
from eval_suite.benchmark.config import BenchmarkConfig
from eval_suite.benchmark.executor import BenchmarkExcutor
from eval_suite.benchmark.result import EvalResultGroups
from eval_suite.client import ClientBase, Message
from eval_suite.metric.base import MetricBase, ToInput, ToOutput
from eval_suite.metric.result import (
    EvalResultBase,
    ToResultArgs,
    ToResultList,
)
from eval_suite.metric.schema import EvalID, EvalInputBase, EvalOutputBase
from eval_suite.metric.stat import BaseStat, EvalStatBase


@dataclass
class _InputContext:
    data: Any


@dataclass
class _OutputContext:
    gen: Message | None
    input: EvalInputBase

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _ResultContext:
    eval_path: Path
    input: EvalInputBase
    output: EvalOutputBase

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _StatContext:
    eval_path: Path
    base: BaseStat
    groups: EvalResultGroups


class BenchmarkBase[
    Input: EvalInputBase,
    Output: EvalOutputBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
](ToInput[Input], ToOutput[Input, Output], BaseModel):
    """
    Base class for all benchmarks.
    """

    dataset: Sequence[Any]
    """The dataset to evaluate."""

    name: str = "benchmark"
    """The name of the benchmark."""

    config: BenchmarkConfig = BenchmarkConfig()
    """The benchmark configuration."""

    base_path: Path | None = None
    """The base path to store the evaluation results."""

    _typevar_map: ClassVar[dict[TypeVar, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        """Some type magic to resolve the generic type parameters into concrete types"""

        super().__init_subclass__(**kwargs)

        if not issubclass(cls, BaseModel) or cls is BaseModel:
            return

        if not (curr_map := get_model_typevars_map(cls)):
            return

        if all_map := cls._typevar_map:
            for typevar, typ in all_map.items():
                if realtype := curr_map.get(typ):
                    all_map[typevar] = realtype
        else:
            cls._typevar_map = curr_map

    @property
    def _Input(self) -> type[Input]:
        return self._typevar_map[Input]

    @property
    def _Stat(self) -> type[BaseModel]:
        return self._typevar_map[Stat]

    @property
    def _Cache(self) -> type[EvalCache[Output, Result]]:
        return EvalCache[Output, Result]

    def __enter__(self) -> Self:
        if not self.base_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.base_path = Path(self._tmpdir.name)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool | None:
        if tmpdir := getattr(self, "_tmpdir", None):
            tmpdir.cleanup()
            self._base_path = None

    @property
    def doc(self) -> str | None:
        """Documentation string for the benchmark. Can be useful for `stat.AutoStat`."""

        return self.__doc__

    @classmethod
    def metrics(cls) -> dict[str, type[MetricBase]]:
        return {
            name: anno
            for name, field in cls.model_fields.items()
            if (anno := field.annotation) is not None  #
            and issubclass(anno, MetricBase)
        }

    def to_input(self, data: Any) -> Input:
        return self._Input.model_validate(data)

    @abstractmethod
    async def to_result(
        self,
        args: Iterable[ToResultArgs[Input, Output]],
    ) -> ToResultList[Result]: ...

    @abstractmethod
    def to_stat(self, groups: EvalResultGroups[Result], base: BaseStat) -> Stat:
        """Convert the result groups to a statistic object."""

    async def run(self, client: ClientBase) -> Stat | None:
        with BenchmarkExcutor(cast(BenchmarkBase, self), client) as executor:
            if stat := await executor.run():
                return cast(Stat, stat)

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
from pydantic import Field

from eval_suite.exception import BaseEvalResultType, EvalException
from eval_suite.metric import (
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    ToResult,
    ToStat,
)
from eval_suite.metric.base import MetricBase


class EvalResult(EvalResultBase):
    passed: bool = True
    """Whether the code passed the unit test"""

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        if exc.type == BaseEvalResultType.fail:
            raise exc

        # consider all known exceptions as completed but not passed
        return cls(passed=False, type=exc.type)


def pass_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0

    value = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    return float(value)


@dataclass
class _EvalResultGroup:
    """Util class to calcuate pass@k for a group of results"""

    root: list[EvalResult]

    @property
    def n(self) -> int:
        return len(self.root)

    @property
    def c(self) -> int:
        return sum(1 for r in self.root if r.passed)

    def pass_k(self, k: int) -> float:
        return pass_k(n=self.n, c=self.c, k=k)


class Stat(EvalStatBase):
    k: int
    pass_n: dict[str, float]


class Metric[Input: EvalInputBase, Output: EvalOutputBase](
    MetricBase,
    ToResult[Input, Output, EvalResult],
    ToStat[EvalResult, Stat],
):
    """Metric to calculate pass@k for a group of results"""

    k: int = Field(default=5, ge=1)
    """Number of top results to consider for pass@k"""

    @abstractmethod
    async def to_result_async(
        self, eval_path: Path, input: Input, output: Output
    ) -> EvalResult: ...

    @override
    def to_stat(self, groups: EvalResultGroups[EvalResult]) -> Stat:
        all_groups = [_EvalResultGroup(root=list(group)) for group in groups.values()]
        pass_n = {
            f"pass@{n}": float(np.mean([group.pass_k(k=n) for group in all_groups]))
            for n in range(1, self.k + 1)
        }

        return Stat(k=self.k, pass_n=pass_n)

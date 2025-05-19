from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
from eval_suite_core.client import Message
from eval_suite_core.exception import BaseEvalResultType, EvalException
from eval_suite_core.metric import (
    IOMetricBase,
    BaseEvalStat,
    EvalItemBase,
    EvalResultBase,
    EvalResultGroups,
    EvalResultMap,
    EvalStatBase,
    EvalStatMap,
)
from pydantic import Field


class EvalResult(EvalResultBase):
    passed: bool = True
    """Whether the code passed the unit test"""

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        if exc.type == BaseEvalResultType.fail:
            raise exc

        # consider all known exceptions as completed but not passed
        return cls(passed=False)


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


class EvalStat(EvalStatBase):
    k: int
    pass_n: dict[str, float]


class Metric[Item: EvalItemBase](IOMetricBase[Item, EvalResult, EvalStat]):
    """Metric to calculate pass@k for a group of results"""

    k: int = Field(default=5, ge=1)
    """Number of top results to consider for pass@k"""

    @abstractmethod
    @override
    async def to_result(
        self, eval_path: Path, item: Item, generation: Message, prec: EvalResultMap
    ) -> EvalResult:
        return await super().to_result(eval_path, item, generation, prec)

    @override
    def to_stat(
        self,
        groups: EvalResultGroups[EvalResult],
        base: BaseEvalStat,
        prec: EvalStatMap,
    ) -> EvalStat:
        return super().to_stat(groups, base, prec)

from collections.abc import Sequence
from typing import Self, override

import numpy as np

from eval_suite.metric import (
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    MetricBase,
    ToResult,
    ToStat,
)


class EvalResult(EvalResultBase):
    score: float


class Stat(EvalStatBase):
    scores: list[float]

    avg: float
    min: float
    max: float
    std: float
    sum: float
    total: int

    @classmethod
    def from_group(cls, group: Sequence[EvalResult]) -> Self:
        scores = [result.score for result in group]

        return cls(
            scores=[float(score) for score in scores],
            avg=float(np.mean(scores)),
            min=np.min(scores),
            max=np.max(scores),
            std=float(np.std(scores)),
            sum=np.sum(scores),
            total=len(scores),
        )


class Metric[Input: EvalInputBase, Output: EvalOutputBase](
    MetricBase,
    ToResult[Input, Output, EvalResult],
    ToStat[EvalResult, Stat],
):
    @override
    def to_stat(self, groups: EvalResultGroups[EvalResult]) -> Stat:
        return Stat.from_group(
            group=[result for group in groups.values() for result in group]
        )

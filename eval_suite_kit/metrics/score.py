from collections.abc import Sequence
from typing import Self

import numpy as np

from eval_suite import EvalResultBase, EvalResultGroups, EvalStatBase


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

    @classmethod
    def from_groups(cls, groups: EvalResultGroups[EvalResult]) -> Self:
        return cls.from_group(
            group=[result for group in groups.values() for result in group]
        )


class GroupStat(EvalStatBase):
    groups: dict[str, Stat]

    @classmethod
    def from_groups(cls, groups: EvalResultGroups[EvalResult]) -> Self:
        return cls(
            groups={
                str(id): Stat.from_group(group)  #
                for id, group in groups.items()
            }
        )

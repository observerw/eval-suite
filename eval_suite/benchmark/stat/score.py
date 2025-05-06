from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, RootModel

from eval_suite.benchmark import EvalResultBase, EvalResultGroups


class ScoreStat(BaseModel):
    scores: list[float]

    avg: float
    min: float
    max: float
    std: float
    sum: float
    total: int

    @classmethod
    def from_scores(cls, scores: Sequence[float]) -> Self:
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
    def from_groups[T: EvalResultBase](
        cls,
        groups: EvalResultGroups[T],
        *,
        score_field: str = "score",
    ) -> Self:
        scores = [
            float(getattr(result, score_field))
            for group in groups.values()
            for result in group
        ]

        return cls.from_scores(scores)


class GroupScoreStat(RootModel):
    root: dict[str, ScoreStat]

    @classmethod
    def from_groups[T: EvalResultBase](
        cls,
        groups: EvalResultGroups[T],
        *,
        score_field: str = "score",
    ) -> Self:
        return cls(
            root={
                str(id): ScoreStat.from_groups(groups=groups, score_field=score_field)
                for id, group in groups.items()
            }
        )

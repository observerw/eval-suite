from typing import override

import numpy as np
from eval_suite_core.metric import (
    BaseStat,
    ChatItemBase,
    ResultBase,
    ResultGroups,
    StatBase,
    StatMap,
    MetricBase,
)


class EvalResult(ResultBase):
    score: int


class EvalStat(StatBase):
    avg: float
    std: float
    var: float
    min: float
    max: float


class ScoreMetric[Item: ChatItemBase](MetricBase[Item, EvalResult, EvalStat]):
    @override
    def to_stat(
        self,
        groups: ResultGroups[EvalResult],
        base: BaseStat,
        prec: StatMap,
    ) -> EvalStat:
        results = [*groups.flatten()]
        avg = float(np.mean([result.score for result in results]))
        std = float(np.std([result.score for result in results]))
        var = float(np.var([result.score for result in results]))
        min_ = float(np.min([result.score for result in results]))
        max_ = float(np.max([result.score for result in results]))

        return EvalStat(avg=avg, std=std, var=var, min=min_, max=max_)

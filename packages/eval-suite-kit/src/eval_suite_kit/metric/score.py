from typing import override

import numpy as np
from eval_suite_core.metric import (
    BaseEvalStat,
    EvalItemBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
    EvalStatMap,
    MetricBase,
)


class EvalResult(EvalResultBase):
    score: int


class EvalStat(EvalStatBase):
    avg: float
    std: float
    var: float
    min: float
    max: float
    count: int


class ScoreMetric[Item: EvalItemBase](MetricBase[Item, EvalResult, EvalStat]):
    @override
    def to_stat(
        self,
        groups: EvalResultGroups[EvalResult],
        base: BaseEvalStat,
        prec: EvalStatMap,
    ) -> EvalStat:
        results = groups.flatten()
        avg = float(np.mean([result.score for result in results]))
        std = float(np.std([result.score for result in results]))
        var = float(np.var([result.score for result in results]))
        min_ = float(np.min([result.score for result in results]))
        max_ = float(np.max([result.score for result in results]))
        count = len(results)

        return EvalStat(avg=avg, std=std, var=var, min=min_, max=max_, count=count)

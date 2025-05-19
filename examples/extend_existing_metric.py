import random
from pathlib import Path
from typing import override

from eval_suite_core.client.schema import Message
from eval_suite_core.metric.base import MetricBase
from eval_suite_core.metric.config import MetricConfig
from eval_suite_core.metric.item import EvalItemBase, ItemID
from eval_suite_core.metric.result import (
    EvalResultBase,
    EvalResultGroups,
    EvalResultMap,
)
from eval_suite_core.metric.stat import BaseEvalStat, EvalStatBase, EvalStatMap
from eval_suite_core.prompt.schema import ChatSequence


class EvalItem(EvalItemBase):
    id: str

    @property
    @override
    def item_id(self) -> ItemID:
        return ItemID(self.id)

    @override
    def format(self, history: ChatSequence) -> ChatSequence:
        return history


class EvalResultA(EvalResultBase):
    a: int


class EvalStatA(EvalStatBase):
    avg_a: float


class AMetric(MetricBase[EvalItem, EvalResultA, EvalStatA]):
    @override
    def to_result(
        self, eval_path: Path, item: EvalItem, generation: Message, prec: EvalResultMap
    ) -> EvalResultA:
        return EvalResultA(a=random.randint(0, 10))

    @override
    def to_stat(
        self,
        groups: EvalResultGroups[EvalResultA],
        base: BaseEvalStat,
        prec: EvalStatMap,
    ) -> EvalStatA:
        results = groups.flatten()
        avg_a = sum(result.a for result in results) / len(results)
        return EvalStatA(avg_a=avg_a)


class EvalResultB(EvalResultBase):
    b: int


class EvalStatB(EvalStatBase):
    avg_b: float


class BMetric(MetricBase[EvalItem, EvalResultB, EvalStatB]):
    @override
    def to_result(
        self, eval_path: Path, item: EvalItem, generation: Message, prec: EvalResultMap
    ) -> EvalResultB:
        return EvalResultB(b=random.randint(0, 10))

    @override
    def to_stat(
        self,
        groups: EvalResultGroups[EvalResultB],
        base: BaseEvalStat,
        prec: EvalStatMap,
    ) -> EvalStatB:
        results = groups.flatten()
        avg_b = sum(result.b for result in results) / len(results)
        return EvalStatB(avg_b=avg_b)


class EvalResult(EvalResultBase):
    sum: int


class EvalStat(EvalStatBase):
    total_avg: float


class SumMetric(MetricBase[EvalItem, EvalResult, EvalStat]):
    a_metric: AMetric
    b_metric: BMetric

    config = MetricConfig()

    @override
    def to_result(
        self, eval_path: Path, item: EvalItem, generation: Message, prec: EvalResultMap
    ) -> EvalResult:
        a = prec[self.a_metric].a
        b = prec[self.b_metric].b

        return EvalResult(sum=a + b)

    @override
    def to_stat(
        self,
        groups: EvalResultGroups[EvalResult],
        base: BaseEvalStat,
        prec: EvalStatMap,
    ) -> EvalStat:
        stat_a = prec[self.a_metric]
        stat_b = prec[self.b_metric]

        total_avg = (stat_a.avg_a + stat_b.avg_b) / 2

        return EvalStat(total_avg=total_avg)

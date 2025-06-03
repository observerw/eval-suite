from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, overload

from pydantic import AfterValidator, BaseModel, Field

from eval_suite_core.metric.base import MetricID
from eval_suite_core.metric.result import RawResultGroups

if TYPE_CHECKING:
    from eval_suite_core.metric.base import AnyMetric


class StatFile(BaseModel):
    """Extra file to be included as a part of the evaluation result"""

    path: Annotated[Path, AfterValidator(lambda p: p.is_absolute())]
    """Path to the file"""

    name: str = Field(default_factory=lambda d: d["path"].name)
    """File name"""

    desc: str | None = None
    """Description of the file"""


class StatBase(BaseModel):
    model_config = {"frozen": True}

    extra_files: ClassVar[list[StatFile]] = []
    """Extra files to be included as a part of the evaluation result"""


class BaseStat(StatBase):
    """Basic info about the evaluation result"""

    class ResultItem(BaseModel):
        count: int
        percentage: float

    results: dict[str, ResultItem] = {}
    """Statistics of result types"""

    total_samples: int = 0
    """Total number of samples being evaluated"""

    total_inputs: int = 0
    """Total number of items in the evaluation set"""

    @classmethod
    def from_groups(cls, groups: RawResultGroups) -> Self:
        stats = Counter(
            result.type  #
            for results in groups.values()
            for result in results
        )

        total = sum(stats.values())
        result_items = {
            key: cls.ResultItem(
                count=count,
                percentage=count / total * 100,
            )
            for key, count in stats.items()
        }

        total_samples = sum(len(results) for results in groups.values())
        total_inputs = len(groups)

        return cls(
            results=result_items,
            total_samples=total_samples,
            total_inputs=total_inputs,
        )


# metric-stat 1-to-1 mapping
class StatMap(dict[MetricID, StatBase]):
    @overload
    def __getitem__[Stat: StatBase](self, key: AnyMetric[Any, Any, Stat]) -> Stat: ...

    @overload
    def __getitem__(self, key: MetricID) -> StatBase: ...

    def __getitem__(self, key: AnyMetric | MetricID) -> StatBase | StatBase:
        match key:
            case str():
                return super().__getitem__(key)
            case AnyMetric(id=id):
                return super().__getitem__(id)

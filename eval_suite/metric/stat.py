from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

from eval_suite.benchmark.result import _EvalResultGroups


class EvalStatFile(BaseModel):
    id: str
    desc: str
    path: Path


class EvalStatBase(BaseModel):
    model_config = {"frozen": True}

    files: list[EvalStatFile] | None = None
    """Related files for the statistics"""


class BaseStat(EvalStatBase):
    """Some basic statistics of the evaluation results."""

    class ResultItem(BaseModel):
        count: int
        percentage: float

    results: dict[str, ResultItem] = Field(default_factory=dict)
    """Statistics of result types"""

    total_samples: int = 0
    """Total number of samples being evaluated"""

    total_inputs: int = 0
    """Total number of items in the evaluation set"""

    @classmethod
    def from_groups(cls, groups: _EvalResultGroups) -> Self:
        stats = Counter(
            result.type  #
            for results in groups.root.values()
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

        total_samples = sum(len(results) for results in groups.root.values())
        total_inputs = len(groups.root)

        return cls(
            results=result_items,
            total_samples=total_samples,
            total_inputs=total_inputs,
        )

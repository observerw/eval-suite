from dataclasses import dataclass
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from eval_suite.benchmark import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalResultGroups,
    EvalStatBase,
)
from eval_suite.client import Message
from eval_suite.exception import BaseEvalResultType, EvalException


class EvalConfig(BaseEvalConfig):
    k: int = Field(default=5, ge=1)
    """Number of top results to consider for pass@k"""

    n_samples: int = Field(default=10)

    @model_validator(mode="after")
    def _validate_k(self) -> Self:
        if not self.k < self.n_samples:
            raise ValueError(
                f"To calculate a meaningful pass@k, `k` must be less than `n_samples`. "
                f"Got k={self.k} and n_samples={self.n_samples}"
            )

        return self


class EvalOutput(EvalOutputBase):
    code: str
    """The code to execute"""


class EvalResult(EvalResultBase):
    passed: bool = True
    """Whether the code passed the unit test"""

    type: str = BaseEvalResultType.success

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


class PassKStat(BaseModel):
    k: int
    pass_n: dict[str, float]

    @classmethod
    def from_groups(cls, groups: EvalResultGroups[EvalResult], k: int) -> Self:
        all_groups = [_EvalResultGroup(root=list(group)) for group in groups.values()]
        pass_n = {
            f"pass@{n}": float(np.mean([group.pass_k(k=n) for group in all_groups]))
            for n in range(1, k + 1)
        }

        return cls(k=k, pass_n=pass_n)


class Benchmark[Input: EvalInputBase, Stat: EvalStatBase](
    BenchmarkBase[Input, EvalOutput, EvalResult, Stat, EvalConfig]
):
    eval_config: EvalConfig = EvalConfig()

    def to_output(self, generation: Message, input: Input) -> EvalOutput:
        return EvalOutput(code=generation.content)

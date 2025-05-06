from typing import Any, Self

from pydantic import BaseModel, SecretStr

from eval_suite.benchmark.result import EvalResultGroups


class AutoStat(BaseModel):
    """Using an LLM to automatically generate statistics from the evaluation results."""

    stat: Any

    @classmethod
    def from_groups(
        cls,
        model: str,
        groups: EvalResultGroups,
        *,
        requirement: str = "Analze the given data and return the result.",
        api_key: SecretStr | None = None,
        base_url: str | None = None,
    ) -> Self:
        raise NotImplementedError

from typing import Any, Self

from pydantic import Field

from eval_suite.metric import EvalResultBase, EvalResultGroups, EvalStatBase


class AutoStat(EvalStatBase):
    """Using LLM to automatically generate statistics from the evaluation results."""

    code: str = Field(description="The code to generate the statistics.")
    stat: Any = Field(description="The generated statistics.")

    @classmethod
    def from_groups[Result: EvalResultBase](
        cls,
        model: str,
        groups: EvalResultGroups[Result],
        *,
        requirement: str = "Analyze the given data and return the result.",
        doc: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Self:
        raise NotImplementedError("AutoStat.from_groups is not implemented yet.")

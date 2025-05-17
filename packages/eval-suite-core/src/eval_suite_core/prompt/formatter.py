from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from eval_suite_core.metric.item import EvalItemBase
from eval_suite_core.prompt.schema import ChatSequence


class FormatterBase[Item: EvalItemBase](BaseModel, ABC):
    """Provides variables for the prompt template, based on the dataset item and the chat history."""

    @abstractmethod
    def provide(
        self,
        item: Item,
        *,
        history: ChatSequence | None = None,
    ) -> dict[str, Any]:
        """
        Provide the variables to be used for rendering the prompt template.
        """

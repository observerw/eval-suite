from abc import ABC, abstractmethod
from typing import final, override

from pydantic import BaseModel, PrivateAttr

from eval_suite_core import prompt
from eval_suite_core.metric.id import EvalID, ItemID, SampleID
from eval_suite_core.prompt.schema import ChatSequence


class ItemBase(BaseModel, ABC):
    """
    Base class for the schema of the evaluation dataset items.
    """

    model_config = {"frozen": True}

    @property
    @abstractmethod
    def item_id(self) -> ItemID:
        """Provide the unique id of the item."""

    _sample_id: SampleID = PrivateAttr()

    @property
    def _eval_id(self) -> EvalID:
        return EvalID(self.item_id, self._sample_id)

    @abstractmethod
    def format(self, history: ChatSequence) -> ChatSequence | None:
        """Format a chat sequence from the item and previous history. Return `None` if chat finished."""


class InstructItemBase(ItemBase):
    @abstractmethod
    def format_instruction(self) -> str:
        """Format a single instruction (i.e., a single user message) from the item."""

    @override
    @final
    def format(self, history: ChatSequence) -> ChatSequence | None:
        return [prompt.user(self.format_instruction())]

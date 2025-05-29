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

    def is_finished(self, history: ChatSequence) -> bool:
        """
        Check if the conversation should be considered finished.

        This method will be called after `format` method is called.

        Default implementation checks if the history has more than 2 messages.

        Override this method to implement custom logic for terminating the conversation.
        """

        return len(history) > 2

    @abstractmethod
    def format(self, history: ChatSequence) -> ChatSequence:
        """Format a chat sequence from the item and previous history."""


class InstructItemBase(ItemBase):
    _formatted: bool = PrivateAttr(default=False)

    @abstractmethod
    def format_instruction(self) -> str:
        """Format a single instruction (i.e., a single user message) from the item."""

    @override
    @final
    def is_finished(self, history: ChatSequence) -> bool:
        if self._formatted:
            return True

        self._formatted = True
        return False

    @override
    @final
    def format(self, history: ChatSequence) -> ChatSequence:
        return [prompt.user(self.format_instruction())]

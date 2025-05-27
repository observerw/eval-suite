from abc import ABC, abstractmethod

from pydantic import BaseModel, PrivateAttr

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

    @abstractmethod
    def format(self, history: ChatSequence) -> ChatSequence:
        """Provide the prompt representation of the item."""

    _sample_id: SampleID = PrivateAttr()

    @property
    def _eval_id(self) -> EvalID:
        return EvalID(self.item_id, self._sample_id)

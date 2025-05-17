from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType

from pydantic import BaseModel, PrivateAttr

from eval_suite_core.prompt.schema import ChatSequence

# type ItemID = str
# type SampleID = int

ItemID = NewType("ItemID", str)
SampleID = NewType("SampleID", int)


@dataclass
class EvalID:
    item_id: ItemID
    sample_id: SampleID

    def __str__(self) -> str:
        return f"{self.item_id}_{self.sample_id}"

    def __hash__(self) -> int:
        return hash((self.item_id, self.sample_id))


class EvalItemBase(BaseModel, ABC):
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
        return EvalID(item_id=self.item_id, sample_id=self._sample_id)

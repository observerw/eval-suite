from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from eval_suite_core.metric.item import ItemBase
from eval_suite_core.prompt.schema import ChatSequence


class FormatFields(BaseModel):
    model_config = {"frozen": True}


class FormatterBase[Item: ItemBase](BaseModel, ABC):
    name: str

    Fields: ClassVar[type[FormatFields]]

    @abstractmethod
    def provide(self, item: Item, history: ChatSequence) -> FormatFields: ...

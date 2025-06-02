from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel

from eval_suite_core.metric.item import ChatItemBase
from eval_suite_core.prompt.schema import ChatSequence


class FormatFields(BaseModel):
    model_config = {"frozen": True}


class FormatterBase[Item: ChatItemBase](BaseModel, ABC):
    name: str

    Fields: ClassVar[type[FormatFields]]

    @abstractmethod
    def provide(self, item: Item, history: ChatSequence) -> FormatFields: ...


type FormatterCallable = Callable[[ChatItemBase, ChatSequence], dict[str, Any]]

type FormatterValue = dict[str, Any]

type AnyFormatter = FormatterBase[ChatItemBase] | FormatterCallable | FormatterValue

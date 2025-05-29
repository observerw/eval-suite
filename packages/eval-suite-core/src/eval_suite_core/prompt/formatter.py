from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache
from typing import Any, ClassVar

from pydantic import BaseModel

from eval_suite_core.metric.item import ItemBase
from eval_suite_core.prompt.schema import ChatSequence


class FormatFields(BaseModel):
    model_config = {"frozen": True}


class FormatterBase[Item: ItemBase](BaseModel, ABC):
    name: str

    Fields: ClassVar[type[FormatFields]]

    @abstractmethod
    @cache
    def provide(self, item: Item, history: ChatSequence) -> FormatFields: ...


type FormatterCallable = Callable[[ItemBase, ChatSequence], dict[str, Any]]

type FormatterValue = dict[str, Any]

type AnyFormatter = FormatterBase[ItemBase] | FormatterCallable | FormatterValue

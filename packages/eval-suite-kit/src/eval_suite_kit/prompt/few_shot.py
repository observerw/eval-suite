from collections.abc import Sequence
from random import sample
from typing import Any, final, override

from eval_suite_core.metric import ItemBase
from eval_suite_core.prompt import ChatSequence, FormatFields, FormatterBase


class FewShotFormatter[Item: ItemBase](FormatterBase):
    name: str = "few_shot"

    class Fields(FormatterBase.Fields):
        examples: list[Any]

    examples: Sequence[Any] = []
    n_shots: int = 5

    def select(self, item: Item, history: ChatSequence) -> Sequence[Any]:
        """
        Strategy to select examples.

        Default implementation is to randomly select `n_shots` examples.

        Override this method to implement your own selection strategy.
        """

        return sample(self.examples, min(self.n_shots, len(self.examples)))

    @override
    @final
    def provide(self, item: Item, history: ChatSequence) -> FormatFields:
        return self.Fields(examples=[*self.select(item, history)])

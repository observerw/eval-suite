from collections.abc import Callable
from typing import Literal, Self

from pydantic import PrivateAttr, RootModel, SerializeAsAny

from eval_suite.benchmark.config import BaseEvalConfig
from eval_suite.benchmark.schema import EvalID, EvalSchema, InputID
from eval_suite.exception import BaseEvalResultType, EvalException


class _EvalResultBase(EvalSchema):
    type: str
    _eval_id: EvalID = PrivateAttr()

    @property
    def _input_id(self) -> InputID:
        return self._eval_id.input_id


class EvalResultBase(_EvalResultBase):
    result_type: Literal["regular"] = "regular"

    type: str = BaseEvalResultType.success

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        """If a regular eval result can be created from exception, it should be implemented here"""

        if real_exc := exc.exc:
            raise exc from real_exc

        raise exc


class ExceptionEvalResult(_EvalResultBase):
    result_type: Literal["exception"] = "exception"

    type: str = BaseEvalResultType.fail
    message: str | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> Self:
        exc = EvalException.from_exception(exc)
        return cls(type=exc.type, message=exc.message)


# type EvalResultGroups[Result: EvalResultBase] = Mapping[InputID, Sequence[Result]]


class EvalResultGroups[Result: EvalResultBase](dict[InputID, list[Result]]):
    def map[T: EvalResultBase](
        self,
        callable: Callable[[Result], T],
    ) -> "EvalResultGroups[T]":
        """Apply a function to the results and return a new EvalResultGroups"""

        return EvalResultGroups[T](
            {
                input_id: [callable(result) for result in results]
                for input_id, results in self.items()
            }
        )


class _EvalResultGroups(RootModel[dict[InputID, list[_EvalResultBase]]]):
    # This class is not intend to be deserialized, no need to discriminate
    root: dict[InputID, list[SerializeAsAny[_EvalResultBase]]] = {}

    _config: BaseEvalConfig = PrivateAttr()
    _extra_count: int = PrivateAttr(default=0)

    @property
    def extra_count(self) -> int:
        """Extra result count, for tqdm total count"""

        return self._extra_count

    def add_result(self, result: _EvalResultBase):
        if isinstance(result, ExceptionEvalResult):
            self._extra_count += 1

        self.root.setdefault(result._input_id, []).append(result)

    def is_completed(self, input_id: InputID) -> bool:
        """Check if there are enough results (`>= n_samples`) for the given eval_id"""

        result_count = len(
            [
                result
                for result in self.root.get(input_id, [])
                if isinstance(result, EvalResultBase)
            ]
        )

        return result_count >= self._config.n_samples

    def stat(self) -> EvalResultGroups[EvalResultBase]:
        """Filter out incomplete and exception results"""

        return EvalResultGroups(
            {
                input_id: [
                    result  #
                    for result in results
                    if isinstance(result, EvalResultBase)
                ]
                for input_id, results in self.root.items()
                if self.is_completed(input_id)
            }
        )

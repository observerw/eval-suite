from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast

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
        return cls.from_exception(EvalException.from_exception(exc))


type EvalResultGroups[Result: EvalResultBase] = Mapping[InputID, Sequence[Result]]


class _EvalResultGroups(RootModel[dict[InputID, list[_EvalResultBase]]]):
    # This class is not intend to be deserialized, no need to discriminate
    root: dict[InputID, list[SerializeAsAny[_EvalResultBase]]] = {}

    _config: BaseEvalConfig = PrivateAttr()
    _exc_result_count: int = PrivateAttr(default=0)
    """Extra result count, for tqdm total count"""

    def add_result(self, result: _EvalResultBase):
        if isinstance(result, ExceptionEvalResult):
            self._exc_result_count += 1

        self.root.setdefault(result._input_id, []).append(result)

    def is_completed(self, input_id: InputID) -> bool:
        """Check if there are enough results (`>= n_samples`) for the given eval_id"""

        result_count = len(
            [
                result
                for result in self.root[input_id]
                if isinstance(result, EvalResultBase)
            ]
        )

        return result_count >= self._config.n_samples

    @property
    def stat[Result: EvalResultBase](self) -> EvalResultGroups[Result]:
        """Filter out incomplete and exception results"""

        return {
            input_id: [
                cast(Result, result)  #
                for result in results
                if isinstance(result, EvalResultBase)
            ]
            for input_id, results in self.root.items()
            if self.is_completed(input_id)
        }

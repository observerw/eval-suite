from pydantic import PrivateAttr, RootModel, SerializeAsAny

from eval_suite.benchmark.config import BenchmarkConfig
from eval_suite.metric import EvalResultBase
from eval_suite.metric.result import (
    EvalResultGroups,
    ExceptionEvalResult,
    _EvalResultBase,
)
from eval_suite.metric.schema import InputID


class _EvalResultGroups(RootModel[dict[InputID, list[_EvalResultBase]]]):
    # This class is not intend to be deserialized, no need to discriminate
    root: dict[InputID, list[SerializeAsAny[_EvalResultBase]]] = {}

    _config: BenchmarkConfig = PrivateAttr()
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

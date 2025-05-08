import asyncio as aio
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, Self, final

from pydantic import PrivateAttr, RootModel, SerializeAsAny

from eval_suite.exception import BaseEvalResultType, EvalException
from eval_suite.metric.config import BaseEvalConfig
from eval_suite.metric.schema import (
    EvalID,
    EvalInputBase,
    EvalOutputBase,
    EvalSchema,
    InputID,
)


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


class ToResultArgs[Input: EvalInputBase, Output: EvalOutputBase](NamedTuple):
    """Arguments for the result processing."""

    eval_path: Path
    input: Input
    output: Output


class ToResult[Input: EvalInputBase, Output: EvalOutputBase, Result: EvalResultBase]:
    """
    Groups of `to_result*` methods to process the result. Requires at least one of the methods to be implemented.

    The `to_result` will use the latest (in the deepest subclass), most important (according to the order) method.
    """

    _to_result_order = (
        "to_result_batch_async",
        "to_result_batch",
        "to_result_async",
        "to_result",
    )

    _impl: Literal[
        "to_result_batch_async",
        "to_result_batch",
        "to_result_async",
        "to_result",
    ]

    def __init_subclass__(cls) -> None:
        for method in cls._to_result_order:
            if hasattr(cls, method):
                cls._impl = method
                break

    def to_result_sync(self, eval_path: Path, input: Input, output: Output) -> Result:
        """Evaluate the output.

        Args:
            eval_path (Path): Directory path where evaluation artifacts can be stored.
            input (Input): The original input used for generation.
            output (Output): The structured output to evaluate.

        Raises:
            EvalException: If evaluation encounters an error.

        Returns:
            Result: Evaluation result containing metrics and analysis.
        """

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_order)} is implemented in the subclass"
        )

    async def to_result_async(
        self, eval_path: Path, input: Input, output: Output
    ) -> Result:
        """Evaluate the output asynchronously.

        Args:
            eval_path (Path): Directory path where evaluation artifacts can be stored.
            input (Input): The original input used for generation.
            output (Output): The structured output to evaluate.

        Raises:
            EvalException: If evaluation encounters an error.

        Returns:
            Result: Evaluation result containing metrics and analysis.
        """

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_order)} is implemented in the subclass"
        )

    def to_result_batch_sync(
        self, args: Sequence[ToResultArgs[Input, Output]]
    ) -> Sequence[Result | BaseException]:
        """Evaluate a batch of outputs.

        Args:
            eval_paths (Sequence[Path]): Directory paths where evaluation artifacts can be stored.
            inputs (Sequence[Input]): The original inputs used for generation.
            outputs (Sequence[Output]): The structured outputs to evaluate.

        Raises:
            EvalException: If evaluation encounters an error.

        Returns:
            Sequence[Result | BaseException]: List of evaluation results or exceptions if evaluation failed.
        """

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_order)} is implemented in the subclass"
        )

    async def to_result_batch_async(
        self, args: Sequence[ToResultArgs[Input, Output]]
    ) -> Sequence[Result | BaseException]:
        """Evaluate a batch of outputs asynchronously.

        Args:
            eval_paths (Sequence[Path]): Directory paths where evaluation artifacts can be stored.
            inputs (Sequence[Input]): The original inputs used for generation.
            outputs (Sequence[Output]): The structured outputs to evaluate.

        Raises:
            EvalException: If evaluation encounters an error.

        Returns:
            Sequence[Result | BaseException]: List of evaluation results or exceptions if evaluation failed.
        """

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_order)} is implemented in the subclass"
        )

    @final
    async def to_result(
        self,
        args: Sequence[ToResultArgs[Input, Output]],
    ) -> Sequence[Result | BaseException]:
        match self._impl:
            case "to_result":
                with ProcessPoolExecutor() as executor:
                    method = partial(
                        aio.get_event_loop().run_in_executor,
                        executor,
                        self.to_result_sync,
                    )
                    rets = await aio.gather(
                        *[method(ctx.eval_path, ctx.input, ctx.output) for ctx in args],
                        return_exceptions=True,
                    )
            case "to_result_async":
                rets = await aio.gather(
                    *[
                        self.to_result_async(ctx.eval_path, ctx.input, ctx.output)
                        for ctx in args
                    ],
                    return_exceptions=True,
                )
            case "to_result_batch":
                with ProcessPoolExecutor(max_workers=1) as executor:
                    method = partial(
                        aio.get_event_loop().run_in_executor,
                        executor,
                        self.to_result_batch_sync,
                    )
                    rets = await method(args)
            case "to_result_batch_async":
                rets = await self.to_result_batch_async(args)
            case _:
                raise NotImplementedError(
                    f"Missing implementation for any of {self._to_result_order}"
                )

        return rets

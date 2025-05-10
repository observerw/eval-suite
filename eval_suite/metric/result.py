import asyncio as aio
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self, final

from pydantic import PrivateAttr

from eval_suite.exception import BaseEvalResultType, EvalException
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


class ExceptionEvalResult(_EvalResultBase):
    result_type: Literal["exception"] = "exception"

    type: str = BaseEvalResultType.fail
    message: str | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> Self:
        exc = EvalException.from_exception(exc)
        return cls(type=exc.type, message=exc.message)


class EvalResultBase(_EvalResultBase):
    result_type: Literal["regular"] = "regular"

    type: str = BaseEvalResultType.success

    @classmethod
    def from_exception(cls, exc: EvalException) -> Self:
        """If a regular eval result can be created from exception, it should be implemented here"""

        raise exc

    @classmethod
    def merge(
        cls,
        **kwargs: "Sequence[Any | BaseException]",
    ) -> "ToResultList[Self]":
        keys = list(kwargs.keys())
        values = list(kwargs.values())

        # transpose
        values = list([list(row) for row in zip(*values)])

        # select the first exception in each row if any
        values = [
            exc
            if (exc := next((v for v in row if isinstance(v, BaseException)), None))
            else row
            for row in values
        ]

        return [
            cls.model_validate({k: v for k, v in zip(keys, value)})
            if not isinstance(value, BaseException)
            else value
            for value in values
        ]


class EvalResultGroups[Result: EvalResultBase](dict[InputID, list[Result]]):
    def map[T: EvalResultBase](
        self, map: Callable[[Result], T]
    ) -> "EvalResultGroups[T]":
        """Construct a subgroups of results from the current group."""

        return EvalResultGroups(
            {
                input_id: [map(result) for result in results]
                for input_id, results in self.items()
            }
        )


type ToResultList[Result: EvalResultBase] = Sequence[Result | BaseException]


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
        self, args: Iterable[ToResultArgs[Input, Output]]
    ) -> ToResultList[Result]:
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
        self, args: Iterable[ToResultArgs[Input, Output]]
    ) -> ToResultList[Result]:
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
        args: Iterable[ToResultArgs[Input, Output]],
    ) -> ToResultList[Result]:
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
                    rets = await aio.get_event_loop().run_in_executor(
                        executor,
                        self.to_result_batch_sync,
                        args,
                    )
            case "to_result_batch_async":
                rets = await self.to_result_batch_async(args)
            case _:
                raise NotImplementedError(
                    f"Missing implementation for any of {self._to_result_order}"
                )

        return rets

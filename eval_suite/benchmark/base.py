import asyncio as aio
import concurrent
import concurrent.futures
import itertools
import json
import logging
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar, cast

import pydantic
from pydantic import BaseModel
from pydantic._internal._generics import (
    get_model_typevars_map,  # type: ignore[reportPrivateImportUsage]
)
from tqdm import tqdm

from eval_suite.benchmark.cache import EvalCache, EvalCachePool
from eval_suite.benchmark.config import BaseEvalConfig
from eval_suite.benchmark.result import (
    EvalResultBase,
    EvalResultGroups,
    ExceptionEvalResult,
    _EvalResultBase,
    _EvalResultGroups,
)
from eval_suite.benchmark.schema import EvalID, EvalInputBase, EvalOutputBase
from eval_suite.benchmark.stat._base import BaseStat, EvalStatBase
from eval_suite.benchmark.utils import method_resolve
from eval_suite.client import ClientBase, Message


@dataclass
class _InputContext:
    data: Any


@dataclass
class _OutputContext:
    gen: Message
    input: EvalInputBase

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _ResultContext:
    eval_path: Path
    input: EvalInputBase
    output: EvalOutputBase

    def __post_init__(self):
        self.eval_path.mkdir(parents=True, exist_ok=True)

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _StatContext:
    eval_path: Path
    base: BaseStat
    groups: EvalResultGroups

    def __post_init__(self):
        self.eval_path.mkdir(parents=True, exist_ok=True)


class BenchmarkBase[
    Input: EvalInputBase,
    Output: EvalOutputBase,
    Result: EvalResultBase,
    Stat: EvalStatBase,
    Config: BaseEvalConfig,
](BaseModel, ABC):
    dataset: Sequence[Any]
    config: Config
    name: str = f"benchmark-{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    base_path: Path | None = None

    _typevar_map: ClassVar[dict[TypeVar, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        """Some type magic to resolve the generic type parameters into concrete types"""

        super().__init_subclass__(**kwargs)

        if not issubclass(cls, BaseModel) or cls is BaseModel:
            return

        if not (curr_map := get_model_typevars_map(cls)):
            return

        if all_map := cls._typevar_map:
            for typevar, typ in all_map.items():
                if realtype := curr_map.get(typ):
                    all_map[typevar] = realtype
        else:
            cls._typevar_map = curr_map

    @property
    def _Input(self) -> type[Input]:
        return self._typevar_map[Input]

    @property
    def _Stat(self) -> type[BaseModel]:
        return self._typevar_map[Stat]

    @property
    def _Cache(self) -> type[EvalCache[Output, Result]]:
        return EvalCache[Output, Result]

    def __enter__(self) -> Self:
        if path := self.base_path:
            path.mkdir(parents=True, exist_ok=True)
        else:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.base_path = Path(self._tmpdir.name)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool | None:
        if tmpdir := getattr(self, "_tmpdir", None):
            tmpdir.cleanup()
            self._base_path = None

    @property
    def eval_path(self) -> Path:
        if not self.base_path:
            raise ValueError(
                "`base_path` is not set. Consider providing one, or use `with` statement to use a temporary directory."
            )

        return self.base_path / self.name

    def to_input(self, data: Any) -> Input:
        """Convert the dataset item to a task-specific input"""

        return self._Input.model_validate(data)

    @abstractmethod
    def to_output(self, generation: Message, input: Input) -> Output:
        """Convert the generation to a task-specific output"""

    def to_result(self, eval_path: Path, input: Input, output: Output) -> Result:
        """Evaluate the output. Raise `EvalException` on failure"""

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_methods)} is implemented in the subclass"
        )

    async def to_result_async(
        self,
        eval_path: Path,
        input: Input,
        output: Output,
    ) -> Result:
        """Evaluate the output asynchronously. Raise `EvalException` on failure"""

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_methods)} is implemented in the subclass"
        )

    def to_result_batch(
        self,
        eval_paths: Sequence[Path],
        inputs: Sequence[Input],
        outputs: Sequence[Output],
    ) -> Sequence[Result | BaseException]:
        """Evaluate the batch of outputs. Raise `EvalException` on failure"""

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_methods)} is implemented in the subclass"
        )

    async def to_result_batch_async(
        self,
        eval_paths: Sequence[Path],
        inputs: Sequence[Input],
        outputs: Sequence[Output],
    ) -> Sequence[Result | BaseException]:
        """Evaluate the batch of outputs asynchronously. Raise `EvalException` on failure"""

        raise NotImplementedError(
            f"`to_result*` not implemented. Please ensure exactly one of {', '.join(self._to_result_methods)} is implemented in the subclass"
        )

    @abstractmethod
    def to_stat(self, groups: EvalResultGroups[Result], base: BaseStat) -> Stat:
        """Generate a statistic from the results"""

    _to_result_methods = (
        "to_result",
        "to_result_async",
        "to_result_batch",
        "to_result_batch_async",
    )

    def _to_input(self, ctx: _InputContext, sample_id: int) -> EvalInputBase:
        """Convert the dataset item to a task-specific input"""

        # TODO handle exception

        input = self.to_input(ctx.data)
        input._sample_id = sample_id
        return input

    def _to_output(self, ctx: _OutputContext) -> EvalOutputBase:
        """Internal method to convert the generation to a task-specific output with metadata"""

        # TODO handle exception

        output = self.to_output(ctx.gen, cast(Input, ctx.input))
        output._eval_id = ctx.eval_id
        return output

    async def _to_result(
        self,
        ctx: _ResultContext,
        executor: concurrent.futures.Executor,
    ) -> _EvalResultBase:
        """Internal method to evaluate the output with metadata"""
        try:
            result = await aio.get_event_loop().run_in_executor(
                executor,
                self.to_result,
                ctx.eval_path,
                cast(Input, ctx.input),
                cast(Output, ctx.output),
            )
        except BaseException as e:
            result = ExceptionEvalResult.from_exception(e)

        result._eval_id = ctx.eval_id
        return result

    async def _to_result_async(self, ctx: _ResultContext) -> _EvalResultBase:
        """Internal method to evaluate the output asynchronously with metadata"""
        try:
            result = await self.to_result_async(
                ctx.eval_path,
                cast(Input, ctx.input),
                cast(Output, ctx.output),
            )
        except BaseException as e:
            result = ExceptionEvalResult.from_exception(e)

        result._eval_id = ctx.eval_id
        return result

    @staticmethod
    def _transform_results(
        ctx_batch: Sequence[_ResultContext],
        results: Sequence[Result | BaseException],
    ) -> Sequence[_EvalResultBase]:
        """Transform the results to a list of EvalResultUnion"""
        transformed_results = [
            ExceptionEvalResult.from_exception(result)
            if isinstance(result, BaseException)
            else result
            for result in results
        ]

        for ctx, result in zip(ctx_batch, transformed_results):
            result._eval_id = ctx.eval_id

        return transformed_results

    async def _to_result_batch(
        self,
        ctx_batch: Sequence[_ResultContext],
        executor: concurrent.futures.Executor,
    ) -> Sequence[_EvalResultBase]:
        """Internal method to evaluate the batch of outputs with metadata"""
        eval_paths = [ctx.eval_path for ctx in ctx_batch]
        inputs = [ctx.input for ctx in ctx_batch]
        outputs = [ctx.output for ctx in ctx_batch]
        results = await aio.get_event_loop().run_in_executor(
            executor,
            self.to_result_batch,
            eval_paths,
            cast(Sequence[Input], inputs),
            cast(Sequence[Output], outputs),
        )

        return self._transform_results(ctx_batch, results)

    async def _to_result_batch_async(
        self,
        ctx_batch: Sequence[_ResultContext],
    ) -> Sequence[_EvalResultBase]:
        eval_paths = [ctx.eval_path for ctx in ctx_batch]
        inputs = [ctx.input for ctx in ctx_batch]
        outputs = [ctx.output for ctx in ctx_batch]
        results = await self.to_result_batch_async(
            eval_paths,
            cast(Sequence[Input], inputs),
            cast(Sequence[Output], outputs),
        )

        return self._transform_results(ctx_batch, results)

    async def _to_result_batch_impl(
        self,
        ctx_batch: Sequence[_ResultContext],
    ) -> Sequence[_EvalResultBase]:
        """Execute implemented `to_result*` method"""

        # find the first implemented `to_result*` method in the order of mro, means user implementation always
        # takes precedence over the base class implementation
        # if none is found, fall back to `BenchmarkBase.to_result*`
        # which raises `NotImplementedError`
        match method_resolve(
            self.__class__,
            self._to_result_methods,
            base=BenchmarkBase,
        ):
            case "to_result":
                with ProcessPoolExecutor() as executor:
                    # `to_result` is exception-safe, so `gather` will work just fine
                    results = await aio.gather(
                        *[self._to_result(ctx, executor) for ctx in ctx_batch],
                    )
            case "to_result_batch":
                with ProcessPoolExecutor(max_workers=1) as executor:
                    results = await self._to_result_batch(ctx_batch, executor)
            case "to_result_async":
                results = await aio.gather(
                    *[self._to_result_async(ctx) for ctx in ctx_batch],
                )
            case "to_result_batch_async":
                results = await self._to_result_batch_async(ctx_batch)
            case _:
                raise NotImplementedError(
                    f"Missing implementation for {self._to_result_methods}"
                )

        assert len(results) == len(ctx_batch), (
            "Result length mismatch, please ensure the `to_result_batch` method returns the same number of results as the input batch"
        )

        return results

    def _to_stat(self, ctx: _StatContext) -> Stat:
        """Internal method to generate statistics with metadata if needed"""
        return self.to_stat(groups=ctx.groups, base=ctx.base)

    async def run(self, client: ClientBase) -> Stat | None:
        if stat := await BenchmarkExcutor(cast(BenchmarkBase, self), client).run():
            return cast(Stat, stat)


class BenchmarkExcutor:  # Type-free since we don't really care about concrete types
    """Execute the evaluation process using one benchmark and one client"""

    def __init__(
        self,
        benchmark: BenchmarkBase,
        client: ClientBase,
    ) -> None:
        self._ben = benchmark
        self._cli = client

        self._logger = logging.getLogger(f"{self._ben.name}/{self._cli.model}")

        self._eval_path = benchmark.eval_path / benchmark.name / self._cli.model

        self._Cache = self._ben._Cache
        self._cache_pool = EvalCachePool(schema=self._Cache, base_path=self._eval_path)

        self._input_queue = aio.Queue[_InputContext | None](self._cli.config.batch_size)
        self._result_queue = aio.Queue[_ResultContext | None](
            maxsize=self._ben.config.eval_batch_size
        )

        self._group = _EvalResultGroups()
        self._group._config = self._ben.config

        # if overlap is enabled, we can run two tasks in parallel
        self._overlap_sem = aio.Semaphore(value=2 if self._ben.config.overlap else 1)

    async def _generate(
        self,
        inputs: Iterable[EvalInputBase],
    ) -> Sequence[Message | None]:
        input_list = list(inputs)
        gen_batch = await self._cli.generate(
            [str(input) for input in input_list],
            system_prompt=self._ben.config.system_prompt,
        )

        success_count = sum(1 for gen in gen_batch if gen)
        if success_count < len(gen_batch):
            self._logger.warning(
                f"Generation: {success_count}/{len(gen_batch)} successful"
            )

        for input, gen in zip(input_list, gen_batch):
            if not gen:
                continue

            cache = self._cache_pool[input._eval_id]
            cache.gen = gen
            self._cache_pool.update(cache)

        return gen_batch

    def _to_output(self, ctx: _OutputContext) -> EvalOutputBase:
        """`to_output` with cache management"""

        cache = self._cache_pool[ctx.eval_id]

        if output := cache.output:
            return output

        try:
            output = self._ben._to_output(ctx)
            cache.output = output
            self._cache_pool.update(cache)
            return output
        except Exception as e:
            self._logger.warning(
                f"Failed to convert output for {ctx.eval_id}: {str(e)}"
            )
            raise

    async def _to_result_batch_impl(
        self,
        ctx_batch: Sequence[_ResultContext],
    ) -> Sequence[_EvalResultBase]:
        """`to_result_batch_impl` with cache management"""

        results: list[_EvalResultBase] = []
        uncached_ctx_batch: list[_ResultContext] = []

        for ctx in ctx_batch:
            if result := self._cache_pool[ctx.eval_id].result:
                results.append(result)
            else:
                uncached_ctx_batch.append(ctx)

        if not uncached_ctx_batch:
            return results

        uncached_results = await self._ben._to_result_batch_impl(uncached_ctx_batch)

        for ctx, result in zip(uncached_ctx_batch, uncached_results):
            cache = self._cache_pool[ctx.eval_id]
            cache.result = result
            self._cache_pool.update(cache)
            results.append(result)

        return results

    async def _gen_worker(self):
        self._logger.info("Generation worker started")

        async def process_batch(
            input_ctx_batch: list[_InputContext],
        ) -> list[_ResultContext]:
            input_batch = [self._ben.to_input(ctx.data) for ctx in input_ctx_batch]

            async with self._overlap_sem:
                gen_batch = await self._generate(input_batch)

            output_batch = [
                self._to_output(_OutputContext(gen=gen, input=input))  #
                for gen, input in zip(gen_batch, input_batch)
                if gen
            ]

            return [
                _ResultContext(
                    eval_path=self._eval_path / str(input._eval_id),
                    input=input,
                    output=output,
                )
                for input, output in zip(input_batch, output_batch)
            ]

        input_ctx_batch: list[_InputContext] = []
        while True:
            if input_ctx := await self._input_queue.get():
                input_ctx_batch.append(input_ctx)

            if not input_ctx or (len(input_ctx_batch) == self._cli._config.batch_size):
                if input_ctx_batch:
                    self._logger.debug(
                        f"Processing batch of size {len(input_ctx_batch)}"
                    )
                    result_ctx_batch = await process_batch(input_ctx_batch)

                    for result_ctx in result_ctx_batch:
                        await self._result_queue.put(result_ctx)
                        self._input_queue.task_done()

                    input_ctx_batch = []

            if not input_ctx:
                self._input_queue.task_done()
                break

        self._logger.debug(
            f"Generation worker finished with {self._input_queue.qsize()} inputs left"
        )

    async def _eval_worker(self):
        self._logger.info("Evaluation worker started")

        async def process_batch(result_ctx_batch: list[_ResultContext]):
            async with self._overlap_sem:
                return await self._to_result_batch_impl(result_ctx_batch)

        result_ctx_batch: list[_ResultContext] = []
        while True:
            if result_ctx := await self._result_queue.get():
                result_ctx_batch.append(result_ctx)

            if not result_ctx or (
                len(result_ctx_batch) == self._ben.config.eval_batch_size
            ):
                if result_ctx_batch:
                    self._logger.debug(
                        f"Evaluating batch of size {len(result_ctx_batch)}"
                    )
                    results = await process_batch(result_ctx_batch)

                    for result in results:
                        if isinstance(result, ExceptionEvalResult):
                            self._logger.error(
                                f"Eval {result._eval_id} failed: {result.type} - {result.message}"
                            )

                        self._group.add_result(result)
                        self._result_queue.task_done()

                    result_ctx_batch = []

            if not result_ctx:
                self._result_queue.task_done()
                break

        self._logger.debug(
            f"Evaluation worker finished with {self._result_queue.qsize()} results left"
        )

    @property
    def _total_count(self) -> int:
        expected_count = self._ben.config.n_samples * len(self._ben.dataset)
        unexpected_count = self._group._exc_result_count

        return expected_count + unexpected_count

    def _input_stream(self) -> Iterable[EvalInputBase]:
        for sample_id in (
            range(1, max_n)
            # try at most `max_n_samples` samples
            if (max_n := self._ben.config.max_n_samples)
            else itertools.count(1)
        ):
            inputs = (
                self._ben._to_input(
                    _InputContext(data=item),
                    sample_id,
                )
                for item in self._ben.dataset
            )
            inputs = (
                input
                for input in inputs
                # skip item with sufficient samples
                if not self._group.is_completed(input._eval_id.input_id)
            )

            count = 0
            for input in inputs:
                count += 1
                yield input

            if count > 0:
                self._logger.info(f"Generated {count} inputs for sample_id {sample_id}")

            if count == 0:
                self._logger.info(
                    "All inputs have sufficient samples, evaluation complete"
                )
                break

    async def run(self) -> BaseModel | None:
        if (result_path := self._eval_path / self._ben.config.stat_file).exists():
            if self._ben.config.overwrite:
                self._logger.info(
                    f"Result file {result_path} already exists. Overwriting."
                )
                shutil.rmtree(self._eval_path, ignore_errors=True)  # FIXME dangerous
            else:
                self._logger.info(
                    f"Result file {result_path} already exists. Skipping evaluation."
                )
                data = json.loads(result_path.read_text())
                try:
                    return self._ben._Stat.model_validate(data)
                except pydantic.ValidationError:
                    self._logger.warning(
                        f"Result file {result_path} cannot be recovered."
                    )
                    return

        self._logger.info(
            f"Running benchmark {self._ben.name} with model {self._cli.model}"
        )

        try:
            async with aio.TaskGroup() as tg:
                tg.create_task(self._gen_worker())
                tg.create_task(self._eval_worker())

                for input in tqdm(
                    self._input_stream(),
                    desc="Processing",
                    total=self._total_count,
                ):
                    await self._input_queue.put(_InputContext(input))

                print("done")

                await self._input_queue.put(None)
                await self._input_queue.join()
                await self._result_queue.put(None)
                await self._result_queue.join()
        except Exception as e:
            self._logger.error(f"Benchmark execution failed: {str(e)}")
            raise e
        finally:
            self._logger.info("Generating and saving statistics")
            stat = self._ben._to_stat(
                _StatContext(
                    eval_path=self._eval_path,
                    groups=self._group.stat,
                    base=BaseStat.from_groups(self._group),
                )
            )

            (self._eval_path / self._ben.config.stat_file).write_text(
                json.dumps(stat.model_dump(), indent=2)
            )

            if results_file := self._ben.config.results_file:
                (self._eval_path / results_file).write_text(
                    json.dumps(self._group.model_dump(), indent=2)
                )

            self._logger.info("Benchmark execution completed successfully")
            return stat

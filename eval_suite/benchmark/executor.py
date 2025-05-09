import asyncio as aio
import json
import logging
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pydantic
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from eval_suite.benchmark.cache import EvalCachePool
from eval_suite.benchmark.result import EvalResultGroups, _EvalResultGroups
from eval_suite.benchmark.utils import dump_json
from eval_suite.client import ClientBase, Message
from eval_suite.metric.result import (
    EvalResultBase,
    ExceptionEvalResult,
    ToResultArgs,
    _EvalResultBase,
)
from eval_suite.metric.schema import EvalID, EvalInputBase, EvalOutputBase
from eval_suite.metric.stat import BaseStat, EvalStatBase

if TYPE_CHECKING:
    from eval_suite.benchmark import BenchmarkBase


@dataclass
class _InputContext:
    data: Any


@dataclass
class _OutputContext:
    gen: Message | None
    input: EvalInputBase

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _ResultContext:
    eval_path: Path
    input: EvalInputBase
    output: EvalOutputBase

    @property
    def eval_id(self) -> EvalID:
        return self.input._eval_id


@dataclass
class _StatContext:
    eval_path: Path
    base: BaseStat
    groups: EvalResultGroups


class BenchmarkExcutor:  # Type-free since we don't really care about concrete types
    """Execute benchmark with a client"""

    def __init__(self, benchmark: BenchmarkBase, client: ClientBase) -> None:
        self._ben = benchmark
        self._cli = client

        self._logger = logging.getLogger(f"{self._ben.name}/{self._cli.model}")

        self._cache_pool = EvalCachePool(
            cache_schema=self._ben._Cache,
            base_path=self._eval_path / self._ben.config.results_dir,
        )

        self._input_queue = aio.Queue[_InputContext | None](
            maxsize=self._cli.config.batch_size
        )
        self._retry_queue = aio.Queue[EvalID | None]()
        self._result_queue = aio.Queue[_ResultContext | None](
            maxsize=self._ben.config.eval_batch_size
        )

        self._result_group = _EvalResultGroups()
        self._result_group._config = self._ben.config

        # if overlap is enabled, we can run two tasks in parallel
        self._overlap_sem = aio.Semaphore(value=2 if self._ben.config.overlap else 1)

        self._progress = Progress(
            TextColumn("[bold blue] {task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )

    @property
    def _eval_path(self) -> Path:
        if not (base_path := self._ben.base_path):
            raise ValueError(
                "`base_path` is not set. Consider providing one, or use `with` statement to use a temporary directory."
            )

        benchmark_name = self._ben.name
        if self._ben.config.with_timestamp:
            benchmark_name += f"-{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        client_name = self._cli.path_name

        match self._ben.config.output_organize:
            case "model-first":
                return base_path / client_name / benchmark_name
            case "benchmark-first":
                return base_path / benchmark_name / client_name

    def _sample_path(self, eval_id: EvalID) -> Path:
        return self._eval_path / self._ben.config.results_dir / str(eval_id)

    def __enter__(self) -> Self:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool | None:
        self._progress.stop()

    async def _generate(
        self,
        inputs: Iterable[EvalInputBase],
    ) -> Sequence[Message | None]:
        # TODO support stream-generation, see <client/config.py>

        input_list = list(inputs)
        gen_batch = await self._cli.generate(
            [str(input) for input in input_list],
            system_prompt=self._ben.config.system_prompt,
        )

        for input, gen in zip(input_list, gen_batch):
            if not gen:
                continue

            self._cache_pool.update_field(input._eval_id, gen=gen)

        return gen_batch

    def _to_input(self, ctx: _InputContext) -> EvalInputBase:
        # TODO config.exception_level
        return self._ben.to_input(ctx.data)

    def _to_output_cached(
        self, ctx: _OutputContext
    ) -> EvalOutputBase | ExceptionEvalResult:
        """`to_output` with cache management"""

        cache = self._cache_pool[ctx.eval_id]

        if output := cache.output:
            return output

        if not (gen := ctx.gen):
            ret = ExceptionEvalResult(message="Got an empty generation")
        else:
            try:
                ret = self._ben.to_output(gen, ctx.input)
            except BaseException as e:
                ret = ExceptionEvalResult.from_exception(e)

        ret._eval_id = ctx.eval_id
        return ret

    async def _to_result_cached(
        self,
        ctx_batch: Sequence[_ResultContext],
    ) -> Sequence[EvalResultBase | ExceptionEvalResult]:
        # TODO support stream-evaluation

        rets: list[EvalResultBase | ExceptionEvalResult] = []
        uncached_ctx_batch: list[_ResultContext] = []

        for ctx in ctx_batch:
            ctx.eval_path.mkdir(parents=True, exist_ok=True)
            if ret := self._cache_pool[ctx.eval_id].result:
                rets.append(ret)
            else:
                uncached_ctx_batch.append(ctx)

        if not uncached_ctx_batch:
            return rets

        uncached_rets = await self._ben.to_result(
            args=[
                ToResultArgs(
                    eval_path=ctx.eval_path,
                    input=ctx.input,
                    output=ctx.output,
                )
                for ctx in uncached_ctx_batch
            ]
        )

        for ctx, ret in zip(uncached_ctx_batch, uncached_rets):
            if isinstance(ret, BaseException):
                ret = ExceptionEvalResult.from_exception(ret)

            self._cache_pool.update_field(eval_id=ctx.eval_id, result=ret)
            rets.append(ret)

        return rets

    def _to_stat(self, ctx: _StatContext) -> EvalStatBase:
        # TODO config.exception_level
        return self._ben.to_stat(ctx.groups, ctx.base)

    async def _add_result(self, result: _EvalResultBase):
        if isinstance(result, ExceptionEvalResult):
            self._logger.error(
                f"Eval {result._eval_id} failed: {result.type} - {result.message}"
            )
            await self._retry_queue.put(result._eval_id)

        self._result_group.add_result(result)

    async def _gen_worker(self):
        # TODO support stream-generation

        self._logger.info("Generation worker started")

        async def process_batch(
            input_ctx_batch: list[_InputContext],
        ) -> list[_ResultContext]:
            input_batch = [self._ben.to_input(ctx.data) for ctx in input_ctx_batch]

            # ensure sample path exists
            for input in input_batch:
                self._sample_path(input._eval_id).mkdir(parents=True, exist_ok=True)

            async with self._overlap_sem:
                task = self._progress.add_task(
                    description=f"Generating {len(input_batch)} inputs"
                )
                gen_batch = await self._generate(input_batch)
                self._progress.remove_task(task)

            output_batch: list[EvalOutputBase] = []

            for input, gen in zip(input_batch, gen_batch):
                match self._to_output_cached(_OutputContext(gen=gen, input=input)):
                    case ExceptionEvalResult() as exc_result:
                        await self._add_result(exc_result)
                    case _ as output:
                        output_batch.append(output)

            return [
                _ResultContext(
                    eval_path=self._sample_path(input._eval_id),
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
        # TODO support stream-evaluation

        self._logger.info("Evaluation worker started")

        async def process_batch(result_ctx_batch: list[_ResultContext]):
            async with self._overlap_sem:
                task = self._progress.add_task(
                    description=f"Evaluating {len(result_ctx_batch)} results"
                )
                results = await self._to_result_cached(result_ctx_batch)
                self._progress.remove_task(task)

                return results

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

                        await self._add_result(result)
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
        unexpected_count = self._result_group._extra_count

        return expected_count + unexpected_count

    async def _input_stream(self):
        inputs = [
            self._to_input(_InputContext(data=item))  #
            for item in self._ben.dataset
        ]

        def with_sample_id(input: EvalInputBase, sample_id: int) -> EvalInputBase:
            ret = input.model_copy()
            ret._sample_id = sample_id
            return ret

        # n_samples not reached, simply yield
        n_samples = self._ben.config.n_samples
        for sample_id, input in product(range(1, n_samples + 1), inputs):
            yield with_sample_id(input, sample_id)

        input_lookup = {input.input_id: input for input in inputs}
        max_n_samples = self._ben.config.max_n_samples

        # n_samples reached, wait for retry
        while eval_id := await self._retry_queue.get():
            input_id = eval_id.input_id
            prev_sample_id = eval_id.sample_id
            curr_sample_id = prev_sample_id + 1

            if not (max_n_samples and curr_sample_id > max_n_samples):
                yield with_sample_id(input_lookup[input_id], curr_sample_id)

            self._retry_queue.task_done()

    async def run(self) -> BaseModel | None:
        self._eval_path.mkdir(parents=True, exist_ok=True)

        if (stat_path := self._eval_path / self._ben.config.stat_file).exists():
            if self._ben.config.overwrite:
                self._logger.info(
                    f"Result file {stat_path} already exists. Overwriting."
                )
                shutil.rmtree(self._eval_path, ignore_errors=True)  # FIXME dangerous
            else:
                self._logger.info(
                    f"Result file {stat_path} already exists. Skipping evaluation."
                )
                try:
                    data = json.loads(stat_path.read_text())
                    return self._ben._Stat.model_validate(data)
                except pydantic.ValidationError:
                    self._logger.warning(
                        f"Result file {stat_path} cannot be recovered."
                    )
                    return
                except json.JSONDecodeError:
                    self._logger.warning(
                        f"Result file {stat_path} is not a valid JSON."
                    )
                    return

        self._logger.info(
            f"Running benchmark {self._ben.name} with model {self._cli.model}"
        )

        try:
            async with aio.TaskGroup() as tg:
                tg.create_task(self._gen_worker())
                tg.create_task(self._eval_worker())

                task = self._progress.add_task("Processing", total=self._total_count)
                async for input in self._input_stream():
                    await self._input_queue.put(_InputContext(input))
                    self._progress.update(task, advance=1, total=self._total_count)

                await self._input_queue.put(None)
                await self._input_queue.join()
                await self._result_queue.put(None)
                await self._result_queue.join()
        except* Exception as exc_group:
            exc_msg = "\n".join(str(exc) for exc in exc_group.exceptions)
            self._logger.error(f"Benchmark execution failed: {exc_msg}")
            raise exc_group
        finally:  # return a result no matter what
            self._logger.info("Generating and saving statistics")
            stat = self._to_stat(
                _StatContext(
                    eval_path=self._eval_path,
                    groups=self._result_group.stat(),
                    base=BaseStat.from_groups(self._result_group),
                )
            )

            dump_json(
                stat.model_dump(
                    exclude_unset=True,
                    exclude_none=True,
                ),
                self._eval_path / self._ben.config.stat_file,
            )

            if results_file := self._ben.config.results_file:
                dump_json(
                    self._result_group.model_dump(),
                    self._eval_path / results_file,
                )

            if config_file := self._ben.config.config_file:
                dump_json(
                    self._ben.config.model_dump(),
                    self._eval_path / config_file,
                )

            self._logger.info("Benchmark execution completed successfully")
            return stat

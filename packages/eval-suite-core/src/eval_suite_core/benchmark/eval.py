import asyncio as aio
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Self, cast

import ray
from ray.dag import DAGNode, FunctionNode, InputNode, MultiOutputNode

from eval_suite_core.metric.base import (
    AsyncBatchMetricDefault,
    BatchComputeMetricBase,
    BatchIOMetricBase,
    BatchMetricDefault,
    IOMetricBase,
    IOMetricDefault,
    MetricBase,
    MetricDefault,
    MetricID,
    _MetricBase,
)
from eval_suite_core.metric.result import (
    EvalResultBase,
    EvalResultMap,
    ExceptionEvalResult,
    ToResultArgs,
    ToResultArgsBase,
)
from eval_suite_core.utils.ray import RayQueue


@ray.remote(num_cpus=1, max_retries=3)
class SyncEvalWorker:
    metric: MetricDefault

    def __init__(self, metric: MetricDefault):
        self.metric = metric

    def run(
        self, input: ToResultArgsBase, *prec_list: EvalResultBase
    ) -> EvalResultBase:
        return self.metric.to_result(
            eval_path=input.eval_path,
            item=input.item,
            generation=input.generation,
            prec=EvalResultMap.create(prec_list),
        )


@ray.remote(max_retries=3)
class AsyncEvalWorker:
    metric: IOMetricBase

    def __init__(self, metric: IOMetricDefault):
        self.metric = metric

    async def run(
        self, input: ToResultArgsBase, *prec_list: EvalResultBase
    ) -> EvalResultBase:
        return await self.metric.to_result(
            eval_path=input.eval_path,
            item=input.item,
            generation=input.generation,
            prec=EvalResultMap.create(prec_list),
        )


@ray.remote(num_cpus=1, max_retries=3)
class SyncBatchEvalWorker:
    metric: BatchMetricDefault

    def __init__(self, metric: BatchComputeMetricBase):
        self.metric = metric

    def run(self, args_batch: Iterable[ToResultArgs]):
        return self.metric.to_result([*args_batch])


@ray.remote(max_retries=3)
class AsyncBatchEvalWorker:
    metric: AsyncBatchMetricDefault

    def __init__(self, metric: BatchIOMetricBase):
        self.metric = metric

    async def run(self, args_batch: Iterable[ToResultArgs]):
        return await self.metric.to_result([*args_batch])


class ResultEvent(aio.Event):
    type Result = EvalResultBase | BaseException

    args: ToResultArgs
    result: Result | None = None

    @classmethod
    def create(cls, args: ToResultArgs) -> Self:
        instance = cls()
        instance.args = args
        return instance

    def set_result(self, result: Result) -> None:
        self.result = result
        self.set()

    async def wait_result(self) -> Result:
        await self.wait()
        if not (result := self.result):
            raise RuntimeError("Result is not set")

        return result


@ray.remote(max_retries=3)
class BatchEvalReceiver:
    queue: RayQueue[ResultEvent]

    def __init__(self, queue: RayQueue[ResultEvent]):
        self.queue = queue

    async def run(
        self, input: ToResultArgsBase, *prec_list: EvalResultBase
    ) -> EvalResultBase:
        """Reciver do not actually execute the `to_result`,
        it just send the item to the queue and wait for result,
        so that the batch execution can also be a part of the DAG."""

        event = ResultEvent.create(
            ToResultArgs(
                eval_path=input.eval_path,
                item=input.item,
                generation=input.generation,
                prec=EvalResultMap.create(prec_list),
            )
        )
        await self.queue.put(event)

        match await event.wait_result():
            case BaseException() as exc:
                raise exc
            case result:
                return result


async def batch_eval_spawner(
    execute: Callable[
        [Iterable[ToResultArgs]],
        Awaitable[Sequence[EvalResultBase | BaseException]],
    ],
    batch_size: int,
    queue: RayQueue[ResultEvent],
):
    while batch := await queue.get_batch(batch_size=batch_size):
        args_batch = [event.args for event in batch]
        result_batch = await execute([*args_batch])

        for event, result in zip(batch, result_batch):
            event.set_result(result)


@dataclass
class MetricGraph:
    graph: DAGNode
    sink_metrics: list[_MetricBase]

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        ordered_metrics: list[_MetricBase],
        sink_metrics: list[_MetricBase],
    ):
        node_lookup: dict[_MetricBase, FunctionNode] = {}
        spawner_lookup: dict[_MetricBase, Coroutine] = {}

        with InputNode() as input_node:
            # build ray DAG by traversing metrics topologically
            for metric in ordered_metrics:
                prec_nodes = (node_lookup[metric] for metric in metric.prec)

                node: FunctionNode | None = None
                spawner: Coroutine | None = None

                match metric:
                    case MetricBase():
                        node = (
                            SyncEvalWorker.options(**dict(metric.config.ray_options))
                            .bind(metric)
                            .run.bind(input_node, *prec_nodes)
                        )
                    case IOMetricBase():
                        # use `remote` here to avoid actor shutdown after DAG execution
                        node = (
                            AsyncEvalWorker.options(**dict(metric.config.ray_options))
                            .remote(metric)
                            .run.bind(  # type: ignore[assignment]
                                input_node, *prec_nodes
                            )
                        )
                    case BatchComputeMetricBase():
                        queue = RayQueue.create()
                        node = BatchEvalReceiver.remote(queue).run.bind(  # type: ignore[assignment]
                            input_node, *prec_nodes
                        )
                        spawner = batch_eval_spawner(
                            SyncBatchEvalWorker.options(
                                **dict(metric.config.ray_options)
                            )
                            .remote(metric)
                            .run.remote,  # type: ignore[assignment]
                            batch_size=metric.config.batch_size,
                            queue=queue,
                        )
                    case BatchIOMetricBase():
                        queue = RayQueue.create()
                        node = BatchEvalReceiver.remote(queue).run.bind(  # type: ignore[assignment]
                            input_node, *prec_nodes
                        )
                        spawner = batch_eval_spawner(
                            AsyncBatchEvalWorker.options(
                                **dict(metric.config.ray_options)
                            )
                            .remote(metric)
                            .run.remote,  # type: ignore[assignment]
                            batch_size=metric.config.batch_size,
                            queue=queue,
                        )
                    case _:
                        raise NotImplementedError(f"Metric {metric} is not supported")

                if node:
                    node_lookup[metric] = node
                if spawner:
                    spawner_lookup[metric] = spawner

        output_node = MultiOutputNode([node_lookup[metric] for metric in sink_metrics])

        async with aio.TaskGroup() as tg:
            for metric, spawner in spawner_lookup.items():
                name = f"{metric.name}[{metric.id}]-spawner"
                tg.create_task(spawner, name=name)

            yield cls(
                graph=output_node,
                sink_metrics=sink_metrics,
            )

    async def execute(
        self, input: ToResultArgsBase
    ) -> Mapping[MetricID, EvalResultBase | ExceptionEvalResult]:
        results = await self.graph.execute(input)
        results = [
            ExceptionEvalResult.from_exception(result)
            if isinstance(result, BaseException)
            else cast(EvalResultBase, result)
            for result in results
        ]

        return {metric.id: result for metric, result in zip(self.sink_metrics, results)}

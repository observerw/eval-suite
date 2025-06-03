import asyncio as aio
from collections.abc import Coroutine, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Self, cast

import sunray
from sunray.dag import DAGNode, InputNode, MultiOutputNode

from eval_suite_core.metric.base import (
    AnyMetric,
    BatchComputeMetricDefault,
    BatchIOMetricDefault,
    ComputeMetricDefault,
    IOMetricDefault,
    MetricDefault,
)
from eval_suite_core.metric.result import (
    ExceptionResult,
    ResultBase,
    ResultMap,
    ToResultArgs,
    ToResultArgsBase,
)
from eval_suite_core.utils.ray import RayQueue


@dataclass
class MetricWorker(sunray.ActorMixin):
    executor: ThreadPoolExecutor
    metric: MetricDefault

    @sunray.remote_method()
    def run(self, args: ToResultArgsBase, *prec: ResultBase) -> ResultBase:
        return self.executor.submit(
            self.metric.to_result,
            args.eval_path,
            args.item,
            args.generation,
            ResultMap.create(prec),
        ).result()


@dataclass
class ComputeMetricWorker(sunray.ActorMixin):
    metric: ComputeMetricDefault

    @sunray.remote_method()
    def run(self, args: ToResultArgsBase, *prec: ResultBase) -> ResultBase:
        return self.metric.to_result(
            eval_path=args.eval_path,
            item=args.item,
            generation=args.generation,
            prec=ResultMap.create(prec),
        )


@dataclass
class IOMetricWorker(sunray.ActorMixin):
    metric: IOMetricDefault

    @sunray.remote_method()
    async def run(self, args: ToResultArgsBase, *prec: ResultBase) -> ResultBase:
        return await self.metric.to_result(
            eval_path=args.eval_path,
            item=args.item,
            generation=args.generation,
            prec=ResultMap.create(prec),
        )


class ResultEvent(aio.Event):
    type Result = ResultBase | BaseException

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


@dataclass
class BatchComputeMetricWorker(sunray.ActorMixin):
    metric: BatchComputeMetricDefault

    @sunray.remote_method()
    def run(
        self, args_batch: Iterable[ToResultArgs]
    ) -> Iterable[ResultBase | BaseException]:
        return self.metric.to_result([*args_batch])


@dataclass
class BatchComputeMetricReceiver(sunray.ActorMixin):
    queue: RayQueue[ResultEvent]

    async def run(self, args: ToResultArgsBase, *prec: ResultBase) -> ResultBase:
        """
        Reciver do not actually execute the `to_result`,
        it just send the item to the queue and wait for result,
        so that the batch execution can also be a part of the DAG.
        """

        event = ResultEvent.create(
            ToResultArgs(
                eval_path=args.eval_path,
                item=args.item,
                generation=args.generation,
                prec=ResultMap.create(prec),
            )
        )
        await self.queue.put(event)

        match await event.wait_result():
            case BaseException() as exc:
                raise exc
            case result:
                return result


@dataclass
class BatchIOMetricWorker(sunray.ActorMixin):
    queue: aio.Queue[ResultEvent]
    metric: BatchIOMetricDefault

    @sunray.remote_method
    async def receive(self, args: ToResultArgsBase, *prec: ResultBase) -> ResultBase:
        event = ResultEvent.create(
            ToResultArgs(
                eval_path=args.eval_path,
                item=args.item,
                generation=args.generation,
                prec=ResultMap.create(prec),
            )
        )
        await self.queue.put(event)

        match await event.wait_result():
            case BaseException() as exc:
                raise exc
            case result:
                return result

    @sunray.remote_method
    async def run(
        self, args_batch: Iterable[ToResultArgs]
    ) -> Iterable[ResultBase | BaseException]:
        return await self.metric.to_result([*args_batch])


type MetricGraphResult = ResultMap | ExceptionResult


@dataclass
class MetricGraph:
    graph: MultiOutputNode
    sink_metrics: list[AnyMetric]

    @classmethod
    def _create(
        cls,
        ordered_metrics: Sequence[AnyMetric],
        sink_metrics: Sequence[AnyMetric],
        input_node: InputNode[ToResultArgsBase],
        tg: aio.TaskGroup,
        executor: ThreadPoolExecutor,
    ):
        node_lookup: dict[AnyMetric, DAGNode] = {}
        spawner_lookup: dict[AnyMetric, Coroutine] = {}
        for metric in ordered_metrics:
            prec = (node_lookup[metric] for metric in metric.prec)
            raise NotImplementedError

        output_node = MultiOutputNode(node_lookup[metric] for metric in sink_metrics)  # type: ignore
        for metric, spawner in spawner_lookup.items():
            name = f"{metric.name}[{metric.id}]-spawner"
            tg.create_task(spawner, name=name)

        return cls(
            graph=output_node,
            sink_metrics=list(sink_metrics),
        )

    @classmethod
    @asynccontextmanager
    async def create(
        cls, ordered_metrics: Sequence[AnyMetric], sink_metrics: Sequence[AnyMetric]
    ):
        async with aio.TaskGroup() as tg:
            with (
                InputNode[ToResultArgsBase]() as input_node,
                ThreadPoolExecutor() as executor,
            ):
                yield cls._create(
                    ordered_metrics=ordered_metrics,
                    sink_metrics=sink_metrics,
                    input_node=input_node,
                    tg=tg,
                    executor=executor,
                )

    async def execute(self, input: ToResultArgsBase) -> MetricGraphResult:
        results = self.graph.execute(input)
        results = [
            ExceptionResult.from_exception(result)
            if isinstance(result, BaseException)
            else cast(ResultBase, result)
            for result in results
        ]

        raise NotImplementedError

        return ResultMap(
            {metric.id: result for metric, result in zip(self.sink_metrics, results)}
        )

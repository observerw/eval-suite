import asyncio as aio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import cast

import ray
from ray.dag.function_node import FunctionNode
from ray.dag.input_node import InputNode
from ray.util.queue import Queue

from eval_suite_core.benchmark.base import BenchmarkBase
from eval_suite_core.benchmark.eval import (
    async_eval_worker,
    batch_eval_receiver,
    eval_worker,
)
from eval_suite_core.client.base import OfflineClientBase, OnlineClientBase, _ClientBase
from eval_suite_core.metric.base import (
    AsyncBatchMetricBase,
    AsyncMetricBase,
    BatchMetricBase,
    MetricBase,
    MetricID,
    _MetricBase,
)
from eval_suite_core.metric.item import EvalID, EvalItemBase, SampleID
from eval_suite_core.metric.result import (
    EvalResultBase,
    ExceptionEvalResult,
    MetricResultGroups,
    ToResultArgsBase,
)
from eval_suite_core.utils.collections import OrderedSet, queue_counter


@ray.remote
async def online_generate_worker(client: OnlineClientBase, item: EvalItemBase):
    return await client.generate(item.format([]))  # TODO history


@ray.remote
def offline_generate_worker(
    client: OfflineClientBase, item_batch: Sequence[EvalItemBase]
):
    return client.generate(item.format([]) for item in item_batch)  # TODO history


@dataclass
class BenchmarkExecutor:
    def __init__(self, benchmark: BenchmarkBase, client: _ClientBase) -> None:
        self._ben = benchmark
        self._cli = client

        self._metric_result_groups = MetricResultGroups()

        self._batch_queue_map, self._eval_graph = self.create_eval_graph()

        self._item_queue: aio.Queue[EvalItemBase | None] = aio.Queue(
            maxsize=self._cli.config.batch_size
        )
        self._eval_queue: aio.Queue[ToResultArgsBase | None] = aio.Queue(
            maxsize=self._cli.config.batch_size
        )
        self._regenerate_queue: aio.Queue[EvalID | None] = aio.Queue()

    @cached_property
    def ordered_metrics(self) -> list[_MetricBase]:
        """Topologically sort metrics."""

        metrics: OrderedSet[_MetricBase] = OrderedSet()
        stack: list[_MetricBase] = [*self._ben.metrics.values()]

        while stack:
            metric = stack.pop()
            metrics.add(metric)
            stack.extend(metric.prec)

        return list(reversed(metrics))  # reverse to get the real topological order

    def create_eval_graph(self):
        node_lookup: dict[_MetricBase, FunctionNode] = {}
        queue_lookup: dict[MetricID, Queue] = {}

        with InputNode() as input_node:
            # build ray DAG by traversing metrics topologically
            for metric in self.ordered_metrics:
                prec = [
                    (metric.id, node_lookup[metric])  #
                    for metric in metric.prec
                ]
                match metric:
                    case MetricBase():
                        node_lookup[metric] = eval_worker.bind(
                            metric, input_node, *prec
                        )
                    case AsyncMetricBase():
                        node_lookup[metric] = async_eval_worker.bind(metric, input_node)
                    case BatchMetricBase() | AsyncBatchMetricBase():
                        queue = queue_lookup[metric.id] = Queue()
                        node_lookup[metric] = batch_eval_receiver.bind(
                            metric, input_node, queue
                        )

        async def execute(
            input: ToResultArgsBase,
        ) -> Mapping[MetricID, EvalResultBase | ExceptionEvalResult]:
            tail_metrics = list(self._ben.metrics.values())
            nodes = [node_lookup[metric] for metric in tail_metrics]
            results = await aio.gather(
                *(node.execute(input) for node in nodes), return_exceptions=True
            )
            results = [
                ExceptionEvalResult.from_exception(result)
                if isinstance(result, BaseException)
                else cast(EvalResultBase, result)
                for result in results
            ]

            return {metric.id: result for metric, result in zip(tail_metrics, results)}

        return queue_lookup, execute

    async def generate_worker(self, tg: aio.TaskGroup):
        item_batch: list[EvalItemBase] = []

        while item := await self._item_queue.get():
            match self._cli:
                case OnlineClientBase() as cli:
                    raise NotImplementedError
                case OfflineClientBase() as cli:
                    item_batch.append(item)
                    raise NotImplementedError

        raise NotImplementedError("Not implemented yet")

    async def eval_worker(self, tg: aio.TaskGroup):
        async def execute(input: ToResultArgsBase):
            results = await self._eval_graph(input)
            self._metric_result_groups.merge(results)
            self._eval_queue.task_done()

        batch_size_lookup: dict[int, list[MetricID]] = {}

        async for input, count in queue_counter(self._eval_queue):
            tg.create_task(execute(input))
            # TODO when input enough, create a batch_eval_worker task

        self._eval_queue.task_done()
        await self._eval_queue.join()

        raise NotImplementedError("Not implemented yet")

    async def item_stream(self):
        items = [
            self._ben._Item.model_validate(raw_item)  #
            for raw_item in self._ben.dataset
        ]

        def with_sample_id(item: EvalItemBase, sample_id: SampleID) -> EvalItemBase:
            ret = item.model_copy()
            ret._sample_id = sample_id
            return ret

        # n_samples not reached, simply yield
        n_samples = self._ben.config.n_samples
        for sample_id, item in product(range(1, n_samples + 1), items):
            yield with_sample_id(item, SampleID(sample_id))

        item_lookup = {item.item_id: item for item in items}
        max_n_samples = self._ben.config.max_n_samples

        # n_samples reached, wait for retry
        while eval_id := await self._regenerate_queue.get():
            input_id = eval_id.item_id
            prev_sample_id = eval_id.sample_id
            curr_sample_id = SampleID(prev_sample_id + 1)

            # if exceeded max_n_samples, skip
            if not (max_n_samples and curr_sample_id > max_n_samples):
                yield with_sample_id(item_lookup[input_id], curr_sample_id)

            self._regenerate_queue.task_done()

    async def run(self):
        async with aio.TaskGroup() as tg:
            gen_worker = tg.create_task(self.generate_worker(tg))
            eval_worker = tg.create_task(self.eval_worker(tg))

            async for item in self.item_stream():
                await self._item_queue.put(item)

        raise NotImplementedError
